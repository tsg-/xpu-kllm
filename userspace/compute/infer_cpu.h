/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_INFER_CPU_H
#define _KLLM_INFER_CPU_H

#include <stdint.h>
#include "../kvcache/block_format.h"
#include "../kvcache/cache_index.h"
#include "../kvcache/arena_alloc.h"
#include "ace_attention.h"

/*
 * CPU inference path (short-path, ACE attention).
 *
 * Full pipeline for the reactor thread:
 * 1. Hash token prefix → content-addressed KV lookup
 * 2. On hit: run ACE paged attention per layer
 * 3. Produce next-token logits
 *
 * Single-layer version for vertical slice, multi-layer for production.
 */

struct kllm_infer_cpu_ctx {
	struct kllm_cache_index *cache_idx;
	struct kllm_arena *arena;
	struct kllm_model_config model;

	/* Model weights (pre-loaded in hugepage arena) */
	bf16_t *wq;  /* query projection weights per layer */
	bf16_t *wk;  /* key projection weights per layer */
	bf16_t *wv;  /* value projection weights per layer */
	bf16_t *wo;  /* output projection weights per layer */
	bf16_t *lm_head;  /* final projection to vocab logits */

	uint32_t vocab_size;
};

struct kllm_infer_result {
	float *logits;        /* [vocab_size] softmax logits */
	uint32_t vocab_size;
	uint32_t next_token;  /* argmax of logits */
};

/*
 * Run CPU inference for a token sequence.
 * Returns 0 on success with result populated, -1 on error/cache-miss.
 */
int kllm_infer_cpu(struct kllm_infer_cpu_ctx *ctx,
		   const uint32_t *token_ids, uint32_t seq_len,
		   struct kllm_infer_result *result);

/*
 * Single-layer attention (vertical slice).
 * For testing: runs one attention layer against cached KV.
 */
int kllm_infer_cpu_single_layer(struct kllm_infer_cpu_ctx *ctx,
				const uint32_t *token_ids, uint32_t seq_len,
				uint32_t layer_idx,
				bf16_t *output, uint32_t output_dim);

#endif /* _KLLM_INFER_CPU_H */
