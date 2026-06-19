/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_WEIGHT_LOADER_H
#define _KLLM_WEIGHT_LOADER_H

#include <stdint.h>
#include <stdbool.h>

#include "../gpu/nvme_ep_wavefront.h"
#include "../gpu/gpu_attention.h"

/*
 * Model weight loader via io_uring GPU Direct Storage.
 *
 * Loads model weights from local NVMe directly into VRAM (for GPU)
 * or hugepage arena (for CPU) using io_uring + dma-buf — same
 * mechanism as KV cache fetch, different content.
 *
 * Weights are content-addressed:
 *   key = SHA-256("model_name:layer_N:component")
 *   value = BF16 tensor data
 *
 * Primary source: local NVMe (io_uring + dma-buf P2P)
 * Alternative: RADOS-NKV pool objects (distributed, multi-node)
 *
 * Load order:
 * 1. Embedding table → VRAM/arena
 * 2. Per-layer weights (Wq, Wk, Wv, Wo, FFN) → VRAM/arena
 * 3. LM head → VRAM/arena
 */

struct kllm_weight_config {
	const char *model_name;        /* content-address prefix for NVMe lookup */
	uint32_t num_layers;
	uint32_t hidden_dim;
	uint32_t num_heads;
	uint32_t num_kv_heads;
	uint32_t head_dim;
	uint32_t intermediate_dim;     /* FFN intermediate size */
	uint32_t vocab_size;
	bool     load_to_gpu;          /* true: VRAM, false: hugepage arena */
};

struct kllm_weight_loader;

struct kllm_weight_loader *kllm_weight_loader_create(
    struct kllm_wavefront_ctx *wf_ctx,
    struct kllm_gpu_ctx *gpu_ctx,  /* NULL for CPU-only */
    const struct kllm_weight_config *cfg);

void kllm_weight_loader_destroy(struct kllm_weight_loader *loader);

/*
 * Load all model weights. Blocks until complete.
 * Uses io_uring wavefronts for parallel I/O (one wavefront per layer).
 * Returns 0 on success.
 */
int kllm_weight_loader_load_all(struct kllm_weight_loader *loader);

/* Get pointers to loaded weights (valid after load_all completes) */
void *kllm_weight_get_embedding(struct kllm_weight_loader *loader);
void *kllm_weight_get_layer(struct kllm_weight_loader *loader, uint32_t layer,
			    const char *component); /* "wq", "wk", "wv", "wo", "ffn_up", "ffn_down" */
void *kllm_weight_get_lm_head(struct kllm_weight_loader *loader);

/* Load progress (for status reporting) */
struct kllm_weight_progress {
	uint32_t layers_loaded;
	uint32_t layers_total;
	uint64_t bytes_loaded;
	uint64_t bytes_total;
	bool     complete;
};

void kllm_weight_loader_progress(struct kllm_weight_loader *loader,
				 struct kllm_weight_progress *out);

#endif /* _KLLM_WEIGHT_LOADER_H */
