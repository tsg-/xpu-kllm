/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_INFER_GPU_H
#define _KLLM_INFER_GPU_H

#include <stdint.h>
#include "../gpu/gpu_attention.h"
#include "../gpu/nvme_ep_wavefront.h"
#include "../gpu/p2p_dma.h"
#include "../kvcache/block_format.h"
#include "infer_cpu.h"

/*
 * GPU inference path (long-context or cache-miss escalation).
 *
 * When the CPU path fails (cache miss or sequence too long), the
 * reactor escalates to GPU:
 *
 * 1. Build NVMe-EP wavefront for KV blocks needed
 * 2. Submit wavefront → data lands in VRAM via P2P DMA
 * 3. Run GPU paged attention kernel
 * 4. Copy logits back to host
 * 5. Return result in same format as CPU path
 *
 * Mid-sequence escalation: if a request starts on CPU (cache hit,
 * short prefix) but grows past the CPU threshold or hits a cache miss
 * during autoregressive decode, it seamlessly transfers to GPU without
 * restarting the sequence.
 */

struct kllm_infer_gpu_ctx {
	struct kllm_gpu_ctx *gpu;
	struct kllm_wavefront_ctx *wf;
	struct kllm_p2p_ctx *p2p;
	struct kllm_model_config model;

	/* VRAM regions for KV blocks and working memory */
	struct kllm_p2p_region kv_region;
	struct kllm_p2p_region logits_region;

	/* Host-side logits buffer for GPU→CPU copy */
	float *host_logits;
	uint32_t vocab_size;

	/* Block table (maps sequence positions to VRAM block addresses) */
	uint64_t *block_table;
	uint32_t max_blocks;
};

struct kllm_infer_gpu_config {
	struct kllm_gpu_ctx *gpu;
	struct kllm_wavefront_ctx *wf;
	uint32_t gpu_id;
	struct kllm_model_config model;
	uint32_t vocab_size;
	uint32_t max_seq_len;
};

struct kllm_infer_gpu_ctx *kllm_infer_gpu_create(
    const struct kllm_infer_gpu_config *cfg);
void kllm_infer_gpu_destroy(struct kllm_infer_gpu_ctx *ctx);

/*
 * Run GPU inference for a token sequence.
 *
 * Fetches KV data via NVMe-EP wavefronts, runs paged attention on GPU,
 * copies logits back to host. Result format matches kllm_infer_result
 * for seamless interop with the decode loop.
 *
 * Returns 0 on success, -1 on error.
 */
int kllm_infer_gpu(struct kllm_infer_gpu_ctx *ctx,
		   const uint32_t *token_ids, uint32_t seq_len,
		   struct kllm_infer_result *result);

/*
 * Mid-sequence escalation: transfer from CPU to GPU.
 *
 * Called when CPU path fails mid-decode (cache miss or seq_len exceeded
 * threshold). Copies any existing CPU-side KV into VRAM, then continues
 * generation on GPU.
 *
 * cpu_ctx: the CPU context with partial KV state
 * token_ids: full sequence so far (prompt + generated)
 * seq_len: current sequence length
 * result: output logits
 */
int kllm_infer_gpu_escalate(struct kllm_infer_gpu_ctx *ctx,
			    struct kllm_infer_cpu_ctx *cpu_ctx,
			    const uint32_t *token_ids, uint32_t seq_len,
			    struct kllm_infer_result *result);

#endif /* _KLLM_INFER_GPU_H */
