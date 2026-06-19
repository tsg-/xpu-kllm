/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_DECODE_LOOP_H
#define _KLLM_DECODE_LOOP_H

#include <stdint.h>
#include <stdbool.h>

#include "../compute/infer_cpu.h"
#include "../compute/infer_gpu.h"
#include "../compute/dispatch_policy.h"
#include "../reactor/kllm_response.h"

/*
 * Autoregressive decode loop.
 *
 * Given an initial token sequence (from the eBPF tokenizer), generates
 * tokens one at a time until EOS or max_tokens is reached.
 *
 * Each iteration:
 * 1. Run inference (CPU or GPU path based on dispatch)
 * 2. Sample next token from logits
 * 3. Emit token to response path (chardev read())
 * 4. Append token to sequence, update KV cache
 * 5. Repeat
 */

enum kllm_sampling_method {
	KLLM_SAMPLE_GREEDY = 0,   /* argmax */
	KLLM_SAMPLE_TOP_K,
	KLLM_SAMPLE_TOP_P,
	KLLM_SAMPLE_TEMPERATURE,
};

struct kllm_decode_params {
	uint32_t max_tokens;       /* max tokens to generate */
	uint32_t eos_token_id;     /* stop on this token */
	enum kllm_sampling_method sampling;
	float temperature;         /* for temperature sampling */
	uint32_t top_k;            /* for top-k sampling */
	float top_p;               /* for top-p (nucleus) sampling */
};

struct kllm_decode_stats {
	uint32_t tokens_generated;
	uint64_t total_latency_ns;
	uint64_t avg_token_latency_ns;
	bool     stopped_eos;
};

struct kllm_decode_ctx;

struct kllm_decode_ctx *kllm_decode_create(struct kllm_infer_cpu_ctx *infer,
					   struct kllm_infer_gpu_ctx *infer_gpu,
					   struct kllm_dispatch_ctx *dispatch,
					   struct kllm_response_ctx *response);
void kllm_decode_destroy(struct kllm_decode_ctx *ctx);

/*
 * Run the decode loop for a request.
 *
 * prompt_tokens: initial token IDs from the eBPF tokenizer
 * prompt_len:    number of prompt tokens
 * params:        generation parameters
 * stats:         output stats (may be NULL)
 *
 * Tokens are streamed to the response path as they are generated.
 * Returns total tokens generated, or -1 on error.
 */
int kllm_decode_run(struct kllm_decode_ctx *ctx,
		    const uint32_t *prompt_tokens, uint32_t prompt_len,
		    const struct kllm_decode_params *params,
		    struct kllm_decode_stats *stats);

#endif /* _KLLM_DECODE_LOOP_H */
