/* SPDX-License-Identifier: Apache-2.0 */
/*
 * CPU inference path: full short-path from token IDs to logits.
 *
 * Pipeline per request:
 * 1. SHA-256(token_ids[0..seq_len]) → content-addressed KV lookup
 * 2. Cache hit → extract K/V from hugepage arena block
 * 3. Compute Q = embed(last_token) * Wq  (simplified: use cached embedding)
 * 4. ACE attention: output = softmax(Q * K^T / sqrt(d)) * V
 * 5. Project output → logits via lm_head
 * 6. Return argmax(logits) as next token
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <openssl/sha.h>

#include "infer_cpu.h"

/* BF16 helpers (duplicated from ace_attention.c for inlining) */
static inline float bf16_to_f32(bf16_t val)
{
	uint32_t bits = (uint32_t)val << 16;
	float result;
	memcpy(&result, &bits, sizeof(result));
	return result;
}

static inline bf16_t f32_to_bf16(float val)
{
	uint32_t bits;
	memcpy(&bits, &val, sizeof(bits));
	bits += 0x7FFF + ((bits >> 16) & 1);
	return (bf16_t)(bits >> 16);
}

int kllm_infer_cpu_single_layer(struct kllm_infer_cpu_ctx *ctx,
				const uint32_t *token_ids, uint32_t seq_len,
				uint32_t layer_idx,
				bf16_t *output, uint32_t output_dim)
{
	uint8_t seq_hash[32];
	void *kv_block;
	bf16_t *k_data, *v_data;
	struct kllm_attention_params attn_params;

	if (!ctx || !token_ids || seq_len == 0)
		return -1;

	/* Content-addressed lookup */
	SHA256((const unsigned char *)token_ids,
	       seq_len * sizeof(uint32_t), seq_hash);

	kv_block = kllm_index_lookup(ctx->cache_idx, seq_hash);
	if (!kv_block)
		return -1;  /* cache miss — caller should escalate to GPU */

	/* Extract K/V for requested layer */
	k_data = (bf16_t *)kllm_block_key(kv_block, &ctx->model, layer_idx);
	v_data = (bf16_t *)kllm_block_value(kv_block, &ctx->model, layer_idx);

	/*
	 * For the vertical slice: use the last token's KV entry as a
	 * synthetic "query" (in production, Q comes from the embedding
	 * layer + Wq projection of the new token being generated).
	 */
	uint32_t head_dim = ctx->model.head_dim;
	uint32_t num_heads = ctx->model.num_kv_heads;
	uint32_t q_size = num_heads * head_dim;

	bf16_t *q = (bf16_t *)calloc(q_size, sizeof(bf16_t));
	if (!q)
		return -1;

	/* Synthetic Q: copy last position from K as query (for testing) */
	uint32_t last_pos_offset = (seq_len - 1) * num_heads * head_dim;
	if (last_pos_offset + q_size <= ctx->model.tokens_per_block * q_size) {
		memcpy(q, k_data + last_pos_offset, q_size * sizeof(bf16_t));
	}

	/* Run ACE attention */
	attn_params.num_heads = num_heads;
	attn_params.head_dim = head_dim;
	attn_params.seq_len = seq_len;
	attn_params.query_len = 1;
	attn_params.scale = 1.0f / sqrtf((float)head_dim);

	bf16_t *attn_output = (bf16_t *)calloc(q_size, sizeof(bf16_t));
	if (!attn_output) {
		free(q);
		return -1;
	}

	int rc = kllm_ace_attention(q, k_data, v_data, attn_output, NULL,
				    &attn_params);
	free(q);

	if (rc < 0) {
		free(attn_output);
		return -1;
	}

	/* Copy output (truncate to output_dim if needed) */
	uint32_t copy_dim = output_dim < q_size ? output_dim : q_size;
	memcpy(output, attn_output, copy_dim * sizeof(bf16_t));
	free(attn_output);

	return 0;
}

int kllm_infer_cpu(struct kllm_infer_cpu_ctx *ctx,
		   const uint32_t *token_ids, uint32_t seq_len,
		   struct kllm_infer_result *result)
{
	uint32_t head_dim = ctx->model.head_dim;
	uint32_t num_heads = ctx->model.num_kv_heads;
	uint32_t hidden_dim = num_heads * head_dim;

	bf16_t *hidden = (bf16_t *)calloc(hidden_dim, sizeof(bf16_t));
	if (!hidden)
		return -1;

	/* Run attention for each layer */
	for (uint32_t layer = 0; layer < ctx->model.num_layers; layer++) {
		int rc = kllm_infer_cpu_single_layer(ctx, token_ids, seq_len,
						     layer, hidden, hidden_dim);
		if (rc < 0) {
			free(hidden);
			return -1;
		}
	}

	/* Project to vocabulary logits via lm_head */
	if (!result->logits) {
		result->logits = (float *)calloc(ctx->vocab_size, sizeof(float));
		if (!result->logits) {
			free(hidden);
			return -1;
		}
	}
	result->vocab_size = ctx->vocab_size;

	if (ctx->lm_head) {
		/* lm_head: [vocab_size, hidden_dim] in BF16, matmul with hidden */
		for (uint32_t v = 0; v < ctx->vocab_size; v++) {
			float acc = 0.0f;
			bf16_t *row = ctx->lm_head + v * hidden_dim;
			for (uint32_t d = 0; d < hidden_dim; d++)
				acc += bf16_to_f32(hidden[d]) * bf16_to_f32(row[d]);
			result->logits[v] = acc;
		}
	} else {
		/* No weights loaded: synthetic logits for testing */
		for (uint32_t v = 0; v < ctx->vocab_size; v++)
			result->logits[v] = bf16_to_f32(hidden[v % hidden_dim]);
	}

	/* Argmax for greedy decoding */
	float max_logit = -INFINITY;
	uint32_t max_idx = 0;
	for (uint32_t v = 0; v < ctx->vocab_size; v++) {
		if (result->logits[v] > max_logit) {
			max_logit = result->logits[v];
			max_idx = v;
		}
	}
	result->next_token = max_idx;

	free(hidden);
	return 0;
}
