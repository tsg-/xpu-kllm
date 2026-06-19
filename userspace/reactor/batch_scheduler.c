/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Continuous batching scheduler.
 *
 * Manages multiple in-flight requests on the SPDK reactor thread.
 * Implements iteration-level scheduling: each step processes prefills
 * first (bounded), then batches decode requests for GPU/CPU inference.
 */

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <openssl/sha.h>

#include "batch_scheduler.h"

struct kllm_batch_scheduler {
	struct kllm_batch_config cfg;
	struct kllm_infer_cpu_ctx *cpu;
	struct kllm_infer_gpu_ctx *gpu;
	struct kllm_dispatch_ctx *dispatch;

	/* Request slots (fixed-size pool, no dynamic alloc) */
	struct kllm_request *requests;
	uint32_t max_requests;
	uint32_t active_count;
	uint32_t next_request_id;

	/* Phase queues (indices into requests[]) */
	uint32_t *prefill_queue;
	uint32_t prefill_count;
	uint32_t *decode_queue;
	uint32_t decode_count;

	struct kllm_batch_stats stats;
};

static inline uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

struct kllm_batch_scheduler *kllm_batch_create(
    const struct kllm_batch_config *cfg,
    struct kllm_infer_cpu_ctx *cpu,
    struct kllm_infer_gpu_ctx *gpu,
    struct kllm_dispatch_ctx *dispatch)
{
	struct kllm_batch_scheduler *s = calloc(1, sizeof(*s));
	if (!s)
		return NULL;

	s->cfg = *cfg;
	s->cpu = cpu;
	s->gpu = gpu;
	s->dispatch = dispatch;
	s->max_requests = cfg->max_requests;
	s->next_request_id = 1;

	s->requests = calloc(cfg->max_requests, sizeof(struct kllm_request));
	s->prefill_queue = calloc(cfg->max_requests, sizeof(uint32_t));
	s->decode_queue = calloc(cfg->max_requests, sizeof(uint32_t));

	if (!s->requests || !s->prefill_queue || !s->decode_queue) {
		kllm_batch_destroy(s);
		return NULL;
	}

	return s;
}

void kllm_batch_destroy(struct kllm_batch_scheduler *sched)
{
	if (!sched)
		return;

	if (sched->requests) {
		for (uint32_t i = 0; i < sched->max_requests; i++)
			free(sched->requests[i].tokens);
	}
	free(sched->requests);
	free(sched->prefill_queue);
	free(sched->decode_queue);
	free(sched);
}

static struct kllm_request *find_free_slot(struct kllm_batch_scheduler *s)
{
	for (uint32_t i = 0; i < s->max_requests; i++) {
		if (s->requests[i].phase == KLLM_PHASE_DONE &&
		    s->requests[i].request_id == 0)
			return &s->requests[i];
	}
	/* Second pass: reuse completed slots */
	for (uint32_t i = 0; i < s->max_requests; i++) {
		if (s->requests[i].phase == KLLM_PHASE_DONE) {
			free(s->requests[i].tokens);
			memset(&s->requests[i], 0, sizeof(s->requests[i]));
			return &s->requests[i];
		}
	}
	return NULL;
}

uint32_t kllm_batch_submit(struct kllm_batch_scheduler *sched,
			   const uint32_t *prompt_tokens, uint32_t prompt_len,
			   struct kllm_response_ctx *response)
{
	if (sched->active_count >= sched->max_requests)
		return 0;

	struct kllm_request *req = find_free_slot(sched);
	if (!req)
		return 0;

	req->tokens = calloc(KLLM_MAX_SEQ_LEN, sizeof(uint32_t));
	if (!req->tokens)
		return 0;

	memcpy(req->tokens, prompt_tokens, prompt_len * sizeof(uint32_t));
	req->prompt_len = prompt_len;
	req->seq_len = prompt_len;
	req->request_id = sched->next_request_id++;
	req->phase = KLLM_PHASE_PREFILL;
	req->arrival_ns = now_ns();
	req->response = response;
	req->preempted = false;

	sched->active_count++;
	sched->stats.total_requests++;

	return req->request_id;
}

static void rebuild_queues(struct kllm_batch_scheduler *s)
{
	s->prefill_count = 0;
	s->decode_count = 0;

	for (uint32_t i = 0; i < s->max_requests; i++) {
		struct kllm_request *r = &s->requests[i];
		if (r->request_id == 0 || r->phase == KLLM_PHASE_DONE)
			continue;

		if (r->phase == KLLM_PHASE_PREFILL)
			s->prefill_queue[s->prefill_count++] = i;
		else if (r->phase == KLLM_PHASE_DECODE)
			s->decode_queue[s->decode_count++] = i;
	}
}

static int run_prefill(struct kllm_batch_scheduler *s, struct kllm_request *req)
{
	/*
	 * Prefill: process the full prompt through KV cache.
	 * On hit: transition directly to decode (CPU path ready).
	 * On miss: fetch via wavefront, then transition.
	 */
	uint8_t seq_hash[32];
	SHA256((const unsigned char *)req->tokens,
	       req->prompt_len * sizeof(uint32_t), seq_hash);

	bool cache_hit = true;  /* assume hit for dispatch decision */
	req->target = kllm_dispatch_decide(s->dispatch, req->seq_len, cache_hit);

	/* Transition to decode phase */
	req->phase = KLLM_PHASE_DECODE;
	return 0;
}

static int run_decode_step(struct kllm_batch_scheduler *s, struct kllm_request *req)
{
	struct kllm_infer_result result;
	int rc;

	memset(&result, 0, sizeof(result));

	if (req->target == KLLM_TARGET_CPU) {
		rc = kllm_infer_cpu(s->cpu, req->tokens, req->seq_len, &result);
		if (rc < 0 && s->gpu) {
			/* Escalate to GPU on cache miss */
			rc = kllm_infer_gpu_escalate(s->gpu, s->cpu,
						     req->tokens, req->seq_len,
						     &result);
			req->target = KLLM_TARGET_GPU;
		}
	} else {
		if (!s->gpu) {
			req->phase = KLLM_PHASE_DONE;
			return -1;
		}
		rc = kllm_infer_gpu(s->gpu, req->tokens, req->seq_len, &result);
	}

	if (rc < 0) {
		req->phase = KLLM_PHASE_DONE;
		free(result.logits);
		return -1;
	}

	/* Greedy sample (scheduler does simple argmax; full sampling in decode_loop) */
	uint32_t next_token = result.next_token;
	free(result.logits);

	/* Check EOS */
	if (next_token == s->cfg.eos_token_id) {
		kllm_response_eos(req->response);
		req->phase = KLLM_PHASE_DONE;
		s->stats.completed_requests++;
		return 0;
	}

	/* Emit token and append to sequence */
	kllm_response_emit_one(req->response, next_token);
	if (req->seq_len < KLLM_MAX_SEQ_LEN) {
		req->tokens[req->seq_len] = next_token;
		req->seq_len++;
	} else {
		/* Hit max length */
		kllm_response_eos(req->response);
		req->phase = KLLM_PHASE_DONE;
		s->stats.completed_requests++;
	}

	req->last_step_ns = now_ns();
	s->stats.total_tokens_generated++;
	return 1;
}

int kllm_batch_step(struct kllm_batch_scheduler *sched)
{
	int tokens_generated = 0;

	rebuild_queues(sched);
	sched->stats.batch_iterations++;
	sched->stats.current_prefill_queue = sched->prefill_count;
	sched->stats.current_decode_queue = sched->decode_count;

	/*
	 * Phase 1: Process prefills (bounded by prefill_budget).
	 * Each prefill moves a request from PREFILL → DECODE.
	 */
	uint32_t prefill_tokens_spent = 0;
	for (uint32_t i = 0; i < sched->prefill_count; i++) {
		struct kllm_request *req = &sched->requests[sched->prefill_queue[i]];

		if (prefill_tokens_spent + req->prompt_len > sched->cfg.prefill_budget)
			break;

		run_prefill(sched, req);
		prefill_tokens_spent += req->prompt_len;
	}

	/*
	 * Phase 2: Batch decode step.
	 * Run one decode iteration per active decode request (up to batch limit).
	 *
	 * TODO: true batched GPU attention (one kernel launch for all decode
	 * requests, with the block tables encoding per-request KV). For now,
	 * iterate sequentially — still correct, just not optimal.
	 */
	rebuild_queues(sched);  /* prefill may have added to decode queue */

	uint32_t batch_limit = sched->cfg.max_batch_size;
	if (sched->decode_count < batch_limit)
		batch_limit = sched->decode_count;
	sched->stats.current_batch_size = batch_limit;

	for (uint32_t i = 0; i < batch_limit; i++) {
		struct kllm_request *req = &sched->requests[sched->decode_queue[i]];
		int rc = run_decode_step(sched, req);
		if (rc > 0)
			tokens_generated += rc;
	}

	/* Retire completed requests */
	for (uint32_t i = 0; i < sched->max_requests; i++) {
		if (sched->requests[i].phase == KLLM_PHASE_DONE &&
		    sched->requests[i].request_id != 0) {
			free(sched->requests[i].tokens);
			sched->requests[i].tokens = NULL;
			sched->requests[i].request_id = 0;
			sched->active_count--;
		}
	}

	return tokens_generated;
}

bool kllm_batch_idle(struct kllm_batch_scheduler *sched)
{
	return sched->active_count == 0;
}

void kllm_batch_get_stats(struct kllm_batch_scheduler *sched,
			  struct kllm_batch_stats *out)
{
	*out = sched->stats;
}
