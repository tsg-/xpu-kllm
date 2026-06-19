/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_NVME_EP_WAVEFRONT_H
#define _KLLM_NVME_EP_WAVEFRONT_H

#include <stdint.h>

/*
 * NVMe wavefront construction for GPU Direct Storage.
 *
 * The reactor hashes token sequences using the same content-addressing
 * scheme as the CPU cache, then constructs wavefronts — batched NVMe
 * read commands submitted via io_uring with dma-buf registered VRAM targets.
 *
 * Each wavefront:
 * - Targets local NVMe blocks keyed by SHA-256(token prefix)
 * - Batched io_uring submission (amortizes syscall cost)
 * - Data lands directly in GPU VRAM via dma-buf P2P DMA
 *
 * This module runs on the SPDK reactor and prepares wavefronts
 * on behalf of the GPU. The GPU kernel triggers fetch via a
 * completion queue notification.
 *
 * Alternative backends (same wavefront interface):
 * - RADOS-NKV: object keys map to RADOS pool objects (multi-node)
 * - rocm-xio: GPU-initiated fetch via AMD ROCm I/O extensions
 */

#define KLLM_WAVEFRONT_MAX_BLOCKS	64  /* max KV blocks per wavefront */
#define KLLM_SEQ_HASH_BYTES		32

/* Single block fetch descriptor within a wavefront */
struct kllm_wf_block_desc {
	uint8_t  obj_key[KLLM_SEQ_HASH_BYTES];  /* content-address key (NVMe LBA index) */
	uint64_t vram_dest_addr;                  /* GPU VRAM destination */
	uint32_t block_size;                      /* bytes to transfer */
	uint32_t layer_start;                     /* first layer in block */
	uint32_t layer_count;                     /* layers in this block */
	uint32_t _pad;
};

/* Wavefront: batch of KV block fetches, one doorbell ring */
struct kllm_wavefront {
	uint32_t num_blocks;
	uint32_t flags;
	uint64_t sequence_id;                     /* for completion matching */
	struct kllm_wf_block_desc blocks[KLLM_WAVEFRONT_MAX_BLOCKS];
};

#define KLLM_WF_FLAG_WEIGHT_LOAD	(1 << 0)  /* loading model weights */
#define KLLM_WF_FLAG_KV_FETCH		(1 << 1)  /* fetching KV cache blocks */
#define KLLM_WF_FLAG_PREFETCH		(1 << 2)  /* speculative prefetch */

/* Wavefront completion status */
struct kllm_wf_completion {
	uint64_t sequence_id;
	uint32_t blocks_completed;
	uint32_t blocks_failed;
	uint64_t total_bytes;
	uint64_t latency_ns;
};

struct kllm_wavefront_ctx;

/*
 * Create wavefront context.
 * nvme_ep_dev: path to NVMe device (e.g., /dev/nvme0n1 or SPDK bdev name)
 */
struct kllm_wavefront_ctx *kllm_wavefront_create(const char *nvme_ep_dev);
void kllm_wavefront_destroy(struct kllm_wavefront_ctx *ctx);

/*
 * Submit a wavefront for execution.
 * Non-blocking: returns immediately, completion via poll.
 */
int kllm_wavefront_submit(struct kllm_wavefront_ctx *ctx,
			  const struct kllm_wavefront *wf);

/* Poll for completed wavefronts. Returns count of completions filled. */
int kllm_wavefront_poll(struct kllm_wavefront_ctx *ctx,
			struct kllm_wf_completion *completions, int max_completions);

/*
 * Construct a KV fetch wavefront from token sequence.
 * Hashes the sequence, looks up which blocks are needed,
 * fills the wavefront descriptor with content-address keys.
 */
int kllm_wavefront_build_kv_fetch(struct kllm_wavefront *wf,
				  const uint32_t *token_ids, uint32_t seq_len,
				  uint64_t vram_base_addr, uint32_t block_size);

/* Construct a weight-load wavefront (startup path) */
int kllm_wavefront_build_weight_load(struct kllm_wavefront *wf,
				     const char *model_path,
				     uint32_t layer_start, uint32_t layer_count,
				     uint64_t vram_dest_addr);

#endif /* _KLLM_NVME_EP_WAVEFRONT_H */
