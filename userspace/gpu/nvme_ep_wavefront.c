/* SPDX-License-Identifier: Apache-2.0 */
/*
 * NVMe-EP wavefront construction and submission.
 *
 * Builds batched NVMe read commands for KV cache blocks addressed
 * by SHA-256 of token prefix. Submitted to the rocm-xio NVMe-EP
 * device via SPDK NVMe driver.
 *
 * In production: SPDK NVMe bdev → NVMe-oF/RDMA → RADOS-NKV gateway.
 * For development: mock with local NVMe or SPDK malloc bdev.
 */

#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>

#include "nvme_ep_wavefront.h"

struct kllm_wavefront_ctx {
	char *device_path;
	uint64_t next_seq_id;
	/* TODO: SPDK NVMe controller handle, qpair, etc. */
};

struct kllm_wavefront_ctx *kllm_wavefront_create(const char *nvme_ep_dev)
{
	struct kllm_wavefront_ctx *ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;

	ctx->device_path = strdup(nvme_ep_dev);
	ctx->next_seq_id = 1;

	/*
	 * TODO: open SPDK NVMe controller, allocate queue pair.
	 * In production this is:
	 *   spdk_nvme_probe() → spdk_nvme_ctrlr_alloc_io_qpair()
	 * For now, just store the device path.
	 */

	return ctx;
}

void kllm_wavefront_destroy(struct kllm_wavefront_ctx *ctx)
{
	if (!ctx)
		return;
	free(ctx->device_path);
	free(ctx);
}

int kllm_wavefront_submit(struct kllm_wavefront_ctx *ctx,
			  const struct kllm_wavefront *wf)
{
	if (!ctx || !wf || wf->num_blocks == 0)
		return -1;

	/*
	 * TODO: translate wavefront into SPDK NVMe read commands.
	 *
	 * For each block in the wavefront:
	 * 1. Map obj_key to NVMe namespace LBA (via RADOS-NKV lookup)
	 * 2. Set up scatter-gather with vram_dest_addr as DMA target
	 * 3. Submit as single NVMe read command (or fused batch)
	 *
	 * One doorbell ring covers the entire wavefront.
	 * Completion notifies via SPDK callback → reactor event.
	 *
	 * spdk_nvme_ns_cmd_read() per block, with shared completion cb.
	 */

	return 0;  /* placeholder: success */
}

int kllm_wavefront_poll(struct kllm_wavefront_ctx *ctx,
			struct kllm_wf_completion *completions, int max_completions)
{
	/*
	 * TODO: poll SPDK NVMe completion queue.
	 * spdk_nvme_qpair_process_completions(qpair, max_completions)
	 */
	(void)ctx;
	(void)completions;
	(void)max_completions;
	return 0;
}

/*
 * Content-addressing: hash the token sequence to derive the RADOS
 * object key(s) for the KV cache blocks covering this sequence.
 *
 * Scheme: for a sequence of length N with tokens_per_block = B,
 * block i covers tokens [i*B, (i+1)*B).
 * Object key for block i = SHA-256(tokens[0..(i+1)*B]).
 *
 * This enables prefix sharing: two sequences with common prefix
 * share the same KV cache blocks for the shared portion.
 */
int kllm_wavefront_build_kv_fetch(struct kllm_wavefront *wf,
				  const uint32_t *token_ids, uint32_t seq_len,
				  uint64_t vram_base_addr, uint32_t block_size)
{
	uint32_t tokens_per_block = 128;  /* matches block_format.h default */
	uint32_t num_blocks_needed = (seq_len + tokens_per_block - 1) / tokens_per_block;

	if (num_blocks_needed > KLLM_WAVEFRONT_MAX_BLOCKS)
		num_blocks_needed = KLLM_WAVEFRONT_MAX_BLOCKS;

	memset(wf, 0, sizeof(*wf));
	wf->num_blocks = num_blocks_needed;
	wf->flags = KLLM_WF_FLAG_KV_FETCH;

	for (uint32_t i = 0; i < num_blocks_needed; i++) {
		struct kllm_wf_block_desc *desc = &wf->blocks[i];
		uint32_t prefix_end = (i + 1) * tokens_per_block;
		if (prefix_end > seq_len)
			prefix_end = seq_len;

		/* SHA-256 of token prefix up to this block boundary */
		SHA256((const unsigned char *)token_ids,
		       prefix_end * sizeof(uint32_t),
		       desc->obj_key);

		desc->vram_dest_addr = vram_base_addr + (uint64_t)i * block_size;
		desc->block_size = block_size;
		desc->layer_start = 0;
		desc->layer_count = 0;  /* all layers */
	}

	return num_blocks_needed;
}

int kllm_wavefront_build_weight_load(struct kllm_wavefront *wf,
				     const char *model_path,
				     uint32_t layer_start, uint32_t layer_count,
				     uint64_t vram_dest_addr)
{
	if (layer_count > KLLM_WAVEFRONT_MAX_BLOCKS)
		layer_count = KLLM_WAVEFRONT_MAX_BLOCKS;

	memset(wf, 0, sizeof(*wf));
	wf->num_blocks = layer_count;
	wf->flags = KLLM_WF_FLAG_WEIGHT_LOAD;

	for (uint32_t i = 0; i < layer_count; i++) {
		struct kllm_wf_block_desc *desc = &wf->blocks[i];

		/* Object key for weights: SHA-256("model_path:layer_N") */
		char key_input[512];
		int len = snprintf(key_input, sizeof(key_input), "%s:layer_%u",
				   model_path, layer_start + i);
		SHA256((const unsigned char *)key_input, len, desc->obj_key);

		desc->vram_dest_addr = vram_dest_addr;  /* caller manages offsets */
		desc->layer_start = layer_start + i;
		desc->layer_count = 1;
	}

	return layer_count;
}
