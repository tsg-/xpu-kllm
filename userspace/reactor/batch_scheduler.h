/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_BATCH_SCHEDULER_H
#define _KLLM_BATCH_SCHEDULER_H

#include <stdint.h>
#include <stdbool.h>

#include "../compute/infer_cpu.h"
#include "../compute/infer_gpu.h"
#include "../compute/dispatch_policy.h"
#include "kllm_response.h"

/*
 * Continuous batching scheduler for multi-request serving.
 *
 * Multiple clients write to /dev/llm_prompt1 concurrently. Each write
 * produces tokens into the shared hugepage ring. The scheduler:
 *
 * 1. Demultiplexes tokens from the ring into per-request sequences
 *    (each request has a unique request_id encoded as first token)
 * 2. Maintains in-flight request state (prompt phase vs decode phase)
 * 3. Batches decode-phase requests for GPU paged attention
 * 4. Routes short/cached requests to CPU for low-latency serving
 *
 * Scheduling policy (iteration-level):
 * - Prefill requests run first (bounded by prefill budget)
 * - Decode requests batched together (up to max_batch_size)
 * - Preemption: long decode sequences can be preempted by new prefills
 *
 * The scheduler runs on the SPDK reactor thread — no locks needed,
 * all state is reactor-local.
 */

#define KLLM_MAX_BATCH_SIZE	256   /* max requests in one decode batch */
#define KLLM_MAX_SEQ_LEN	32768
#define KLLM_REQUEST_ID_MASK	0xFF000000  /* top 8 bits = request_id sentinel */

enum kllm_request_phase {
	KLLM_PHASE_PREFILL = 0,    /* processing prompt tokens */
	KLLM_PHASE_DECODE,         /* generating tokens */
	KLLM_PHASE_DONE,           /* completed or error */
};

struct kllm_request {
	uint32_t request_id;
	enum kllm_request_phase phase;
	enum kllm_dispatch_target target;

	/* Token sequence (prompt + generated) */
	uint32_t *tokens;
	uint32_t prompt_len;
	uint32_t seq_len;          /* prompt_len + generated so far */

	/* Scheduling metadata */
	uint64_t arrival_ns;       /* timestamp of first token */
	uint64_t last_step_ns;     /* timestamp of last decode step */
	uint32_t priority;         /* lower = higher priority */
	bool     preempted;        /* set when preempted by prefill */

	/* Per-request response path */
	struct kllm_response_ctx *response;
};

struct kllm_batch_config {
	uint32_t max_batch_size;       /* max decode requests per iteration */
	uint32_t prefill_budget;       /* max prefill tokens per iteration */
	uint32_t max_requests;         /* max concurrent in-flight requests */
	uint32_t eos_token_id;
};

struct kllm_batch_stats {
	uint64_t total_requests;
	uint64_t completed_requests;
	uint64_t preemptions;
	uint64_t total_tokens_generated;
	uint64_t batch_iterations;
	uint32_t current_batch_size;
	uint32_t current_prefill_queue;
	uint32_t current_decode_queue;
};

struct kllm_batch_scheduler;

struct kllm_batch_scheduler *kllm_batch_create(
    const struct kllm_batch_config *cfg,
    struct kllm_infer_cpu_ctx *cpu,
    struct kllm_infer_gpu_ctx *gpu,
    struct kllm_dispatch_ctx *dispatch);

void kllm_batch_destroy(struct kllm_batch_scheduler *sched);

/*
 * Submit a new request (from ring consumer when a complete prompt arrives).
 * Returns request_id, or 0 on error (queue full).
 */
uint32_t kllm_batch_submit(struct kllm_batch_scheduler *sched,
			   const uint32_t *prompt_tokens, uint32_t prompt_len,
			   struct kllm_response_ctx *response);

/*
 * Run one scheduler iteration. Called by the reactor poller.
 *
 * Each iteration:
 * 1. Process pending prefills (up to prefill_budget tokens)
 * 2. Batch decode requests and run inference
 * 3. Emit generated tokens to response paths
 * 4. Retire completed requests
 *
 * Returns number of tokens generated in this iteration.
 */
int kllm_batch_step(struct kllm_batch_scheduler *sched);

/* Check if the scheduler has any in-flight work */
bool kllm_batch_idle(struct kllm_batch_scheduler *sched);

void kllm_batch_get_stats(struct kllm_batch_scheduler *sched,
			  struct kllm_batch_stats *out);

#endif /* _KLLM_BATCH_SCHEDULER_H */
