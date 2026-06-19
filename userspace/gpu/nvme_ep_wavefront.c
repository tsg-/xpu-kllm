/* SPDX-License-Identifier: Apache-2.0 */
/*
 * NVMe wavefront construction and submission.
 *
 * Builds batched NVMe read commands for KV cache blocks addressed
 * by SHA-256 of token prefix. Submitted via io_uring with dma-buf
 * registered VRAM targets for GPU Direct Storage.
 *
 * Primary data path: local NVMe → io_uring → dma-buf → P2P DMA → GPU VRAM.
 * Alternative: RADOS-NKV objects via SPDK NVMe-oF or rocm-xio.
 * For development: mock with SPDK malloc bdev or /tmp ramdisk.
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
	 * TODO: open NVMe device and set up io_uring with dma-buf registration.
	 *
	 * Production sequence:
	 *   nvme_fd = open(nvme_dev, O_RDONLY | O_DIRECT)
	 *   io_uring_queue_init(depth, &ring, IORING_SETUP_SQPOLL)
	 *
	 *   // Register dma-buf VRAM as fixed buffer:
	 *   struct io_uring_regbuf_desc desc = {
	 *       .type = IO_REGBUF_TYPE_DMABUF,
	 *       .dmabuf_fd = vram_dma_buf_fd,
	 *       .target_fd = nvme_fd,  // lazy: kernel defers attach until first I/O
	 *   };
	 *   io_uring_register_buffers_ext(&ring, &desc, 1)
	 *
	 *   // Acquire dma-token for fencing:
	 *   token = io_dmabuf_token_get(desc.dmabuf_fd)
	 *   // ~310ns software overhead per I/O vs ~20µs IOCTL path
	 *
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
	 * TODO: translate wavefront into io_uring NVMe read SQEs.
	 *
	 * For each block in the wavefront:
	 * 1. Map obj_key → NVMe LBA (via content-address index)
	 * 2. Prepare io_uring SQE with fixed buffer (dma-buf VRAM target)
	 * 3. io_uring_prep_read_fixed(sqe, nvme_fd, vram_addr, size, lba, buf_idx)
	 *
	 * Single io_uring_submit() covers the entire wavefront batch.
	 * Completion via io_uring CQE reap → reactor event.
	 */

	return 0;  /* placeholder: success */
}

int kllm_wavefront_poll(struct kllm_wavefront_ctx *ctx,
			struct kllm_wf_completion *completions, int max_completions)
{
	/*
	 * TODO: reap io_uring CQEs.
	 * io_uring_peek_batch_cqe(&ring, cqes, max_completions)
	 */
	(void)ctx;
	(void)completions;
	(void)max_completions;
	return 0;
}

/*
 * Content-addressing: hash the token sequence to derive the NVMe
 * LBA key(s) for the KV cache blocks covering this sequence.
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
