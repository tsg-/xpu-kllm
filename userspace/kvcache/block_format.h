/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_BLOCK_FORMAT_H
#define _KLLM_BLOCK_FORMAT_H

/*
 * KV cache block format for xpu-kllm.
 *
 * Each block stores key and value tensors for a fixed number of tokens
 * across all layers. Blocks are content-addressed: the block's identity
 * is SHA-256(token prefix it represents).
 *
 * Memory layout (contiguous, hugepage-backed):
 *   [header][K layer 0][V layer 0][K layer 1][V layer 1]...[scales]
 *
 * K/V tensors are stored in BF16 (2 bytes per element).
 * OCP MX block scaling: one FP8 scale factor per 8-element micro-block,
 * stored contiguously after the K/V data.
 */

#include <stdint.h>

#define KLLM_BLOCK_MAGIC	0x4B56424C  /* "KVBL" */
#define KLLM_BLOCK_VERSION	1
#define KLLM_SEQ_HASH_BYTES	32  /* SHA-256 */

/* Configurable model dimensions — set at init, fixed for lifetime */
struct kllm_model_config {
	uint32_t num_layers;
	uint32_t num_kv_heads;
	uint32_t head_dim;
	uint32_t tokens_per_block;  /* how many token positions per block */
};

struct kllm_block_hdr {
	uint32_t magic;
	uint32_t version;
	uint32_t block_size_bytes;  /* total block size including header */
	uint32_t tokens_per_block;

	/* Content address: SHA-256 of the token prefix this block caches */
	uint8_t  seq_hash[KLLM_SEQ_HASH_BYTES];

	/* Model dimensions for this block */
	uint32_t num_layers;
	uint32_t num_kv_heads;
	uint32_t head_dim;
	uint32_t _pad0;

	/* Cache management */
	uint64_t last_access_ns;    /* monotonic timestamp for LRU */
	uint32_t refcount;          /* active references (atomic) */
	uint32_t flags;             /* KLLM_BLOCK_F_* */

	/* OCP MX scaling metadata */
	uint32_t scale_offset;      /* byte offset from block start to scales */
	uint32_t scale_count;       /* number of FP8 scale factors */

	uint8_t  _reserved[32];
} __attribute__((aligned(64)));

#define KLLM_BLOCK_F_VALID	(1 << 0)
#define KLLM_BLOCK_F_DIRTY	(1 << 1)  /* modified since last NVMe-EP write-back */
#define KLLM_BLOCK_F_PINNED	(1 << 2)  /* exempt from eviction */

/*
 * Compute block size for given model config.
 *
 * Per layer: K[tokens_per_block * num_kv_heads * head_dim] in BF16
 *          + V[tokens_per_block * num_kv_heads * head_dim] in BF16
 * Scales: one FP8 per 8 elements across all K/V data.
 */
static inline uint64_t kllm_block_kv_bytes(const struct kllm_model_config *cfg)
{
	uint64_t elems_per_layer = (uint64_t)cfg->tokens_per_block *
				   cfg->num_kv_heads * cfg->head_dim;
	/* K + V, BF16 = 2 bytes each */
	return cfg->num_layers * elems_per_layer * 2 * 2;
}

static inline uint64_t kllm_block_scale_bytes(const struct kllm_model_config *cfg)
{
	uint64_t total_elems = (uint64_t)cfg->num_layers *
			       cfg->tokens_per_block *
			       cfg->num_kv_heads * cfg->head_dim * 2; /* K+V */
	/* One FP8 scale per 8 elements (OCP MX block size) */
	return (total_elems + 7) / 8;
}

static inline uint64_t kllm_block_total_bytes(const struct kllm_model_config *cfg)
{
	return sizeof(struct kllm_block_hdr) +
	       kllm_block_kv_bytes(cfg) +
	       kllm_block_scale_bytes(cfg);
}

/*
 * Accessors into a block's K/V data.
 * Returns pointer to K or V tensor for a given layer.
 * Layout: all layers packed sequentially, K then V per layer.
 */
static inline void *kllm_block_key(void *block, const struct kllm_model_config *cfg,
				   uint32_t layer)
{
	uint8_t *data = (uint8_t *)block + sizeof(struct kllm_block_hdr);
	uint64_t elems_per_layer = (uint64_t)cfg->tokens_per_block *
				   cfg->num_kv_heads * cfg->head_dim;
	uint64_t layer_stride = elems_per_layer * 2 * 2;  /* K+V in BF16 */
	return data + layer * layer_stride;
}

static inline void *kllm_block_value(void *block, const struct kllm_model_config *cfg,
				     uint32_t layer)
{
	uint8_t *data = (uint8_t *)block + sizeof(struct kllm_block_hdr);
	uint64_t elems_per_layer = (uint64_t)cfg->tokens_per_block *
				   cfg->num_kv_heads * cfg->head_dim;
	uint64_t layer_stride = elems_per_layer * 2 * 2;
	uint64_t k_size = elems_per_layer * 2;  /* BF16 */
	return data + layer * layer_stride + k_size;
}

static inline uint8_t *kllm_block_scales(void *block, const struct kllm_model_config *cfg)
{
	struct kllm_block_hdr *hdr = (struct kllm_block_hdr *)block;
	return (uint8_t *)block + hdr->scale_offset;
}

#endif /* _KLLM_BLOCK_FORMAT_H */
