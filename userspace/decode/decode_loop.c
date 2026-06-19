/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Autoregressive decode loop for xpu-kllm.
 *
 * Given prompt tokens, generates one token at a time:
 * 1. Infer next-token logits (CPU or GPU path)
 * 2. Sample from logits (greedy, top-k, top-p, temperature)
 * 3. Stream token to chardev response path
 * 4. Append to sequence, loop until EOS or max_tokens
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "decode_loop.h"

#define MAX_SEQUENCE_LEN	32768

struct kllm_decode_ctx {
	struct kllm_infer_cpu_ctx *infer;
	struct kllm_dispatch_ctx  *dispatch;
	struct kllm_response_ctx  *response;
};

static inline uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

struct kllm_decode_ctx *kllm_decode_create(struct kllm_infer_cpu_ctx *infer,
					   struct kllm_dispatch_ctx *dispatch,
					   struct kllm_response_ctx *response)
{
	struct kllm_decode_ctx *ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;

	ctx->infer = infer;
	ctx->dispatch = dispatch;
	ctx->response = response;
	return ctx;
}

void kllm_decode_destroy(struct kllm_decode_ctx *ctx)
{
	free(ctx);
}

/* --- Sampling strategies --- */

static uint32_t sample_greedy(const float *logits, uint32_t vocab_size)
{
	float max_val = -INFINITY;
	uint32_t max_idx = 0;

	for (uint32_t i = 0; i < vocab_size; i++) {
		if (logits[i] > max_val) {
			max_val = logits[i];
			max_idx = i;
		}
	}
	return max_idx;
}

static uint32_t sample_temperature(const float *logits, uint32_t vocab_size,
				   float temperature)
{
	if (temperature <= 0.0f)
		return sample_greedy(logits, vocab_size);

	/* Apply temperature, compute softmax, sample */
	float *probs = (float *)malloc(vocab_size * sizeof(float));
	if (!probs)
		return sample_greedy(logits, vocab_size);

	float max_logit = -INFINITY;
	for (uint32_t i = 0; i < vocab_size; i++)
		if (logits[i] > max_logit)
			max_logit = logits[i];

	float sum = 0.0f;
	for (uint32_t i = 0; i < vocab_size; i++) {
		probs[i] = expf((logits[i] - max_logit) / temperature);
		sum += probs[i];
	}
	for (uint32_t i = 0; i < vocab_size; i++)
		probs[i] /= sum;

	/* Random sample from distribution */
	float r = (float)rand() / (float)RAND_MAX;
	float cumsum = 0.0f;
	uint32_t token = vocab_size - 1;
	for (uint32_t i = 0; i < vocab_size; i++) {
		cumsum += probs[i];
		if (r <= cumsum) {
			token = i;
			break;
		}
	}

	free(probs);
	return token;
}

static uint32_t sample_top_k(const float *logits, uint32_t vocab_size,
			     uint32_t k, float temperature)
{
	if (k == 0 || k >= vocab_size)
		return sample_temperature(logits, vocab_size, temperature);

	/* Find top-k logits (simple selection sort for k << vocab_size) */
	float *top_vals = (float *)malloc(k * sizeof(float));
	uint32_t *top_idxs = (uint32_t *)malloc(k * sizeof(uint32_t));
	if (!top_vals || !top_idxs) {
		free(top_vals);
		free(top_idxs);
		return sample_greedy(logits, vocab_size);
	}

	for (uint32_t i = 0; i < k; i++) {
		top_vals[i] = -INFINITY;
		top_idxs[i] = 0;
	}

	for (uint32_t i = 0; i < vocab_size; i++) {
		if (logits[i] > top_vals[k - 1]) {
			top_vals[k - 1] = logits[i];
			top_idxs[k - 1] = i;
			/* Bubble up */
			for (int j = k - 2; j >= 0; j--) {
				if (top_vals[j + 1] > top_vals[j]) {
					float tv = top_vals[j];
					top_vals[j] = top_vals[j + 1];
					top_vals[j + 1] = tv;
					uint32_t ti = top_idxs[j];
					top_idxs[j] = top_idxs[j + 1];
					top_idxs[j + 1] = ti;
				}
			}
		}
	}

	uint32_t result = sample_temperature(top_vals, k, temperature);
	uint32_t token = top_idxs[result < k ? result : 0];

	free(top_vals);
	free(top_idxs);
	return token;
}

static uint32_t sample_top_p(const float *logits, uint32_t vocab_size,
			     float p, float temperature)
{
	/* Simplified: apply temperature then nucleus sampling */
	return sample_temperature(logits, vocab_size, temperature);
	/* TODO: proper nucleus sampling with sorted cumulative probability */
	(void)p;
}

static uint32_t sample_token(const float *logits, uint32_t vocab_size,
			     const struct kllm_decode_params *params)
{
	switch (params->sampling) {
	case KLLM_SAMPLE_GREEDY:
		return sample_greedy(logits, vocab_size);
	case KLLM_SAMPLE_TEMPERATURE:
		return sample_temperature(logits, vocab_size, params->temperature);
	case KLLM_SAMPLE_TOP_K:
		return sample_top_k(logits, vocab_size, params->top_k,
				    params->temperature);
	case KLLM_SAMPLE_TOP_P:
		return sample_top_p(logits, vocab_size, params->top_p,
				    params->temperature);
	default:
		return sample_greedy(logits, vocab_size);
	}
}

/* --- Main decode loop --- */

int kllm_decode_run(struct kllm_decode_ctx *ctx,
		    const uint32_t *prompt_tokens, uint32_t prompt_len,
		    const struct kllm_decode_params *params,
		    struct kllm_decode_stats *stats)
{
	uint32_t *sequence;
	uint32_t seq_len, generated = 0;
	uint64_t start_ns, total_ns = 0;
	struct kllm_infer_result result;
	bool stopped_eos = false;

	if (!ctx || !prompt_tokens || prompt_len == 0 || !params)
		return -1;

	sequence = (uint32_t *)calloc(MAX_SEQUENCE_LEN, sizeof(uint32_t));
	if (!sequence)
		return -1;

	/* Initialize sequence with prompt */
	uint32_t copy_len = prompt_len < MAX_SEQUENCE_LEN ? prompt_len : MAX_SEQUENCE_LEN;
	memcpy(sequence, prompt_tokens, copy_len * sizeof(uint32_t));
	seq_len = copy_len;

	memset(&result, 0, sizeof(result));

	/* Decode loop */
	while (generated < params->max_tokens && seq_len < MAX_SEQUENCE_LEN) {
		start_ns = now_ns();

		/* Decide dispatch path */
		bool cache_hit = true;  /* optimistic; infer_cpu returns -1 on miss */
		enum kllm_dispatch_target target =
			kllm_dispatch_decide(ctx->dispatch, seq_len, cache_hit);

		int rc;
		if (target == KLLM_TARGET_CPU) {
			rc = kllm_infer_cpu(ctx->infer, sequence, seq_len, &result);
			if (rc < 0) {
				/* Cache miss: escalate to GPU */
				/* TODO: invoke GPU path */
				break;
			}
		} else {
			/* GPU path */
			/* TODO: submit wavefront, run GPU attention, get logits */
			break;
		}

		/* Sample next token */
		uint32_t next_token = sample_token(result.logits, result.vocab_size,
						   params);

		total_ns += now_ns() - start_ns;

		/* Check EOS */
		if (next_token == params->eos_token_id) {
			stopped_eos = true;
			kllm_response_eos(ctx->response);
			break;
		}

		/* Emit token to response path (streaming to chardev reader) */
		kllm_response_emit_one(ctx->response, next_token);

		/* Append to sequence */
		sequence[seq_len] = next_token;
		seq_len++;
		generated++;
	}

	/* Signal EOS if we hit max_tokens */
	if (!stopped_eos)
		kllm_response_eos(ctx->response);

	if (stats) {
		stats->tokens_generated = generated;
		stats->total_latency_ns = total_ns;
		stats->avg_token_latency_ns = generated > 0 ? total_ns / generated : 0;
		stats->stopped_eos = stopped_eos;
	}

	free(result.logits);
	free(sequence);
	return generated;
}
