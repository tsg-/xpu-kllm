/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Unified GPU attention dispatch — selects backend at runtime.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

#include "gpu_attention.h"

struct kllm_gpu_ctx {
	enum kllm_gpu_backend backend;
	void *stream;      /* native stream/queue */
	void *lib_handle;  /* dlopen'd backend library (if dynamic) */
};

/* Probe GPU availability by checking for runtime libraries */
static enum kllm_gpu_backend detect_gpu(void)
{
	void *handle;

	/* Try ROCm first (AMD) */
	handle = dlopen("libamdhip64.so", RTLD_LAZY);
	if (handle) {
		dlclose(handle);
		return KLLM_GPU_ROCM;
	}

	/* Try CUDA (NVIDIA) */
	handle = dlopen("libcuda.so.1", RTLD_LAZY);
	if (handle) {
		dlclose(handle);
		return KLLM_GPU_CUDA;
	}

	/* Try Level Zero (Intel Xe) */
	handle = dlopen("libze_loader.so.1", RTLD_LAZY);
	if (handle) {
		dlclose(handle);
		return KLLM_GPU_XE;
	}

	return KLLM_GPU_NONE;
}

struct kllm_gpu_ctx *kllm_gpu_create(void)
{
	struct kllm_gpu_ctx *ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;

	ctx->backend = detect_gpu();

	/*
	 * TODO: create native stream/queue based on backend.
	 * - ROCm:  hipStreamCreate(&stream)
	 * - CUDA:  cudaStreamCreate(&stream)
	 * - Xe:    new sycl::queue(...)
	 */

	return ctx;
}

void kllm_gpu_destroy(struct kllm_gpu_ctx *ctx)
{
	if (!ctx)
		return;
	/* TODO: destroy native stream/queue */
	free(ctx);
}

enum kllm_gpu_backend kllm_gpu_get_backend(struct kllm_gpu_ctx *ctx)
{
	return ctx->backend;
}

int kllm_gpu_paged_attention(struct kllm_gpu_ctx *ctx,
			     const struct kllm_gpu_attention_params *p)
{
	void *stream = p->stream ? p->stream : ctx->stream;

	switch (ctx->backend) {
	case KLLM_GPU_ROCM:
		return kllm_gpu_paged_attention_rocm(
			p->output, p->query, p->block_tables, p->seq_lens,
			p->num_seqs, p->num_heads, p->num_kv_heads,
			p->head_dim, p->max_blocks_per_seq, p->scale, stream);

	case KLLM_GPU_CUDA:
		return kllm_gpu_paged_attention_cuda(
			p->output, p->query, p->block_tables, p->seq_lens,
			p->num_seqs, p->num_heads, p->num_kv_heads,
			p->head_dim, p->max_blocks_per_seq, p->scale, stream);

	case KLLM_GPU_XE:
		return kllm_gpu_paged_attention_xe(
			p->output, p->query, p->block_tables, p->seq_lens,
			p->num_seqs, p->num_heads, p->num_kv_heads,
			p->head_dim, p->max_blocks_per_seq, p->scale, stream);

	case KLLM_GPU_NONE:
	default:
		return -1;
	}
}

void *kllm_gpu_malloc(struct kllm_gpu_ctx *ctx, uint64_t size)
{
	(void)ctx;
	(void)size;
	/* TODO: hipMalloc / cudaMalloc / sycl::malloc_device */
	return NULL;
}

void kllm_gpu_free(struct kllm_gpu_ctx *ctx, void *ptr)
{
	(void)ctx;
	(void)ptr;
	/* TODO: hipFree / cudaFree / sycl::free */
}

int kllm_gpu_memcpy_h2d(struct kllm_gpu_ctx *ctx, void *dst,
			const void *src, uint64_t size)
{
	(void)ctx;
	(void)dst;
	(void)src;
	(void)size;
	/* TODO: hipMemcpy / cudaMemcpy / queue.memcpy */
	return -1;
}

void *kllm_gpu_get_stream(struct kllm_gpu_ctx *ctx)
{
	return ctx->stream;
}
