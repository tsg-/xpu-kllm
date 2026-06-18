/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_ACE_ATTENTION_H
#define _KLLM_ACE_ATTENTION_H

#include <stdint.h>
#include "../kvcache/block_format.h"

/*
 * ACE BF16 attention kernel for xpu-kllm.
 *
 * Implements single-head and multi-head scaled dot-product attention
 * using ACE outer-product intrinsics (AVX10 BF16, fallback to AMX BF16).
 *
 * This runs on the SPDK reactor thread for short cached prefixes
 * where GPU round-trip latency would dominate.
 *
 * Operations:
 *   scores = Q * K^T / sqrt(head_dim)     [ACE outer product]
 *   weights = softmax(scores)              [scalar]
 *   output = weights * V                   [ACE outer product]
 *
 * All inputs/outputs in BF16. OCP MX block scaling applied inline.
 */

/* BF16 type — stored as uint16_t, arithmetic via conversion */
typedef uint16_t bf16_t;

struct kllm_attention_params {
	uint32_t num_heads;
	uint32_t head_dim;
	uint32_t seq_len;         /* number of KV tokens to attend over */
	uint32_t query_len;       /* number of query tokens (usually 1 for decode) */
	float    scale;           /* 1/sqrt(head_dim) */
};

/*
 * Compute attention for one layer.
 *
 * q:       [query_len, num_heads, head_dim] in BF16
 * k:       [seq_len, num_heads, head_dim] in BF16 (from KV cache block)
 * v:       [seq_len, num_heads, head_dim] in BF16 (from KV cache block)
 * output:  [query_len, num_heads, head_dim] in BF16
 * scales:  OCP MX FP8 scale factors (NULL if not using MX quantization)
 *
 * Returns 0 on success, -1 on error.
 */
int kllm_ace_attention(const bf16_t *q, const bf16_t *k, const bf16_t *v,
		       bf16_t *output, const uint8_t *scales,
		       const struct kllm_attention_params *params);

/*
 * Check if ACE instructions are available on this CPU.
 * Falls back to AMX BF16 if ACE not present.
 */
int kllm_ace_probe(void);

#endif /* _KLLM_ACE_ATTENTION_H */
