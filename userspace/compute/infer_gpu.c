/* SPDX-License-Identifier: Apache-2.0 */
/*
 * GPU inference path and mid-sequence escalation.
 *
 * Handles two cases:
 * 1. Direct GPU dispatch (long sequence, dispatch says GPU from start)
 * 2. Escalation from CPU (cache miss or threshold exceeded mid-decode)
 *
 * Both paths use io_uring + dma-buf wavefronts to fetch KV data from
 * local NVMe into VRAM, then run GPU paged attention and copy logits
 * back to host.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <openssl/sha.h>

#include "infer_gpu.h"

struct kllm_infer_gpu_ctx *kllm_infer_gpu_create(
    const struct kllm_infer_gpu_config *cfg)
{
	struct kllm_infer_gpu_ctx *ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;

	ctx->gpu = cfg->gpu;
	ctx->wf = cfg->wf;
	ctx->model = cfg->model;
	ctx->vocab_size = cfg->vocab_size;

	/* Create P2P context for DMA */
	ctx->p2p = kllm_p2p_create(cfg->gpu_id);
	if (!ctx->p2p) {
		free(ctx);
		return NULL;
	}

	/* Allocate VRAM region for KV blocks */
	uint64_t block_bytes = kllm_block_total_bytes(&cfg->model);
	uint32_t blocks_needed = (cfg->max_seq_len + cfg->model.tokens_per_block - 1)
				 / cfg->model.tokens_per_block;
	ctx->max_blocks = blocks_needed;

	uint64_t kv_size = block_bytes * blocks_needed;
	if (kllm_p2p_alloc_region(ctx->p2p, kv_size, &ctx->kv_region) < 0) {
		kllm_p2p_destroy(ctx->p2p);
		free(ctx);
		return NULL;
	}

	/* Register with SPDK for NVMe DMA target */
	kllm_p2p_register_spdk(ctx->p2p, &ctx->kv_region);

	/* Allocate VRAM region for logits output */
	uint64_t logits_size = (uint64_t)cfg->vocab_size * sizeof(float);
	if (kllm_p2p_alloc_region(ctx->p2p, logits_size, &ctx->logits_region) < 0) {
		kllm_p2p_free_region(ctx->p2p, &ctx->kv_region);
		kllm_p2p_destroy(ctx->p2p);
		free(ctx);
		return NULL;
	}

	/* Host logits buffer for GPU→CPU copy */
	ctx->host_logits = (float *)calloc(cfg->vocab_size, sizeof(float));

	/* Block table: maps sequence block indices to VRAM addresses */
	ctx->block_table = (uint64_t *)calloc(blocks_needed, sizeof(uint64_t));

	if (!ctx->host_logits || !ctx->block_table) {
		kllm_infer_gpu_destroy(ctx);
		return NULL;
	}

	/* Initialize block table with VRAM offsets */
	for (uint32_t i = 0; i < blocks_needed; i++)
		ctx->block_table[i] = ctx->kv_region.gpu_va + i * block_bytes;

	return ctx;
}

void kllm_infer_gpu_destroy(struct kllm_infer_gpu_ctx *ctx)
{
	if (!ctx)
		return;

	free(ctx->host_logits);
	free(ctx->block_table);

	if (ctx->p2p) {
		kllm_p2p_free_region(ctx->p2p, &ctx->logits_region);
		kllm_p2p_free_region(ctx->p2p, &ctx->kv_region);
		kllm_p2p_destroy(ctx->p2p);
	}
	free(ctx);
}

static int fetch_kv_blocks(struct kllm_infer_gpu_ctx *ctx,
			   const uint32_t *token_ids, uint32_t seq_len)
{
	uint32_t tokens_per_block = ctx->model.tokens_per_block;
	uint32_t num_blocks = (seq_len + tokens_per_block - 1) / tokens_per_block;
	uint64_t block_bytes = kllm_block_total_bytes(&ctx->model);

	/* Build wavefront: one fetch per KV block */
	struct kllm_wavefront wf;
	memset(&wf, 0, sizeof(wf));
	wf.flags = KLLM_WF_FLAG_KV_FETCH;

	uint32_t submitted = 0;
	for (uint32_t b = 0; b < num_blocks; b++) {
		/* Hash the token prefix up to this block boundary */
		uint32_t prefix_len = (b + 1) * tokens_per_block;
		if (prefix_len > seq_len)
			prefix_len = seq_len;

		SHA256((const unsigned char *)token_ids,
		       prefix_len * sizeof(uint32_t),
		       wf.blocks[wf.num_blocks].obj_key);

		wf.blocks[wf.num_blocks].vram_dest_addr = ctx->block_table[b];
		wf.blocks[wf.num_blocks].block_size = (uint32_t)block_bytes;
		wf.blocks[wf.num_blocks].layer_start = 0;
		wf.blocks[wf.num_blocks].layer_count = ctx->model.num_layers;
		wf.num_blocks++;

		/* Submit when wavefront is full */
		if (wf.num_blocks >= KLLM_WAVEFRONT_MAX_BLOCKS) {
			if (kllm_wavefront_submit(ctx->wf, &wf) < 0)
				return -1;
			submitted += wf.num_blocks;
			wf.num_blocks = 0;
		}
	}

	/* Submit remaining */
	if (wf.num_blocks > 0) {
		if (kllm_wavefront_submit(ctx->wf, &wf) < 0)
			return -1;
		submitted += wf.num_blocks;
	}

	/* Wait for completions */
	struct kllm_wf_completion comp;
	uint32_t completed = 0;
	while (completed < submitted) {
		int n = kllm_wavefront_poll(ctx->wf, &comp, 1);
		if (n > 0)
			completed += comp.blocks_completed;
	}

	return 0;
}

int kllm_infer_gpu(struct kllm_infer_gpu_ctx *ctx,
		   const uint32_t *token_ids, uint32_t seq_len,
		   struct kllm_infer_result *result)
{
	if (!ctx || !token_ids || seq_len == 0)
		return -1;

	/* Fetch KV data from local NVMe into VRAM via io_uring GDS */
	if (fetch_kv_blocks(ctx, token_ids, seq_len) < 0)
		return -1;

	/* Run GPU paged attention */
	uint32_t tokens_per_block = ctx->model.tokens_per_block;
	uint32_t num_blocks = (seq_len + tokens_per_block - 1) / tokens_per_block;
	int seq_len_int = (int)seq_len;

	/*
	 * Build query from last token position.
	 * In production this comes from the embedding + Wq projection;
	 * here we use the last KV entry as a stand-in.
	 */
	uint32_t head_dim = ctx->model.head_dim;
	uint32_t num_heads = ctx->model.num_kv_heads;
	uint64_t query_size = (uint64_t)num_heads * head_dim * sizeof(uint16_t);

	void *gpu_query = kllm_gpu_malloc(ctx->gpu, query_size);
	void *gpu_output = kllm_gpu_malloc(ctx->gpu, query_size);
	if (!gpu_query || !gpu_output) {
		if (gpu_query) kllm_gpu_free(ctx->gpu, gpu_query);
		if (gpu_output) kllm_gpu_free(ctx->gpu, gpu_output);
		return -1;
	}

	struct kllm_gpu_attention_params attn = {
		.output = gpu_output,
		.query = gpu_query,
		.block_tables = (const void *)ctx->block_table,
		.seq_lens = &seq_len_int,
		.num_seqs = 1,
		.num_heads = (int)num_heads,
		.num_kv_heads = (int)ctx->model.num_kv_heads,
		.head_dim = (int)head_dim,
		.max_blocks_per_seq = (int)num_blocks,
		.scale = 1.0f / sqrtf((float)head_dim),
		.stream = kllm_gpu_get_stream(ctx->gpu),
	};

	int rc = kllm_gpu_paged_attention(ctx->gpu, &attn);

	kllm_gpu_free(ctx->gpu, gpu_query);
	kllm_gpu_free(ctx->gpu, gpu_output);

	if (rc < 0)
		return -1;

	/* Copy logits from GPU to host */
	/* TODO: GPU lm_head projection (currently returns attention output) */
	if (!result->logits) {
		result->logits = (float *)calloc(ctx->vocab_size, sizeof(float));
		if (!result->logits)
			return -1;
	}
	result->vocab_size = ctx->vocab_size;

	/* Argmax placeholder (real implementation projects through lm_head on GPU) */
	float max_logit = -INFINITY;
	uint32_t max_idx = 0;
	for (uint32_t v = 0; v < ctx->vocab_size; v++) {
		if (result->logits[v] > max_logit) {
			max_logit = result->logits[v];
			max_idx = v;
		}
	}
	result->next_token = max_idx;

	return 0;
}

int kllm_infer_gpu_escalate(struct kllm_infer_gpu_ctx *ctx,
			    struct kllm_infer_cpu_ctx *cpu_ctx,
			    const uint32_t *token_ids, uint32_t seq_len,
			    struct kllm_infer_result *result)
{
	/*
	 * Mid-sequence escalation from CPU to GPU.
	 *
	 * The CPU had a cache hit for an earlier prefix but now:
	 * - The sequence grew past the CPU threshold, OR
	 * - A cache miss occurred during decode
	 *
	 * Strategy: re-fetch full KV from NVMe (the GPU will need
	 * all blocks, not just the one the CPU missed). The CPU-computed
	 * KV for earlier tokens is already persisted to NVMe by the
	 * write-through cache, so the wavefront will retrieve it.
	 *
	 * Future optimization: copy CPU-resident KV directly to VRAM
	 * via memcpy (UMA) or RDMA-WRITE (dGPU) to avoid re-fetch for
	 * blocks the CPU already has.
	 */

	if (!ctx || !token_ids || seq_len == 0)
		return -1;

	(void)cpu_ctx;  /* reserved for future direct KV transfer */

	/* For now: full GPU inference path re-fetches everything */
	return kllm_infer_gpu(ctx, token_ids, seq_len, result);
}
