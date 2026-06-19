/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_GPU_ATTENTION_H
#define _KLLM_GPU_ATTENTION_H

#include <stdint.h>

/*
 * Unified GPU attention dispatch.
 *
 * Supports three backends:
 * - Xe/SYCL   (Intel Data Center GPU Max / Battlemage)
 * - CUDA      (NVIDIA A100/H100+)
 * - ROCm/HIP  (AMD MI300+)
 *
 * The backend is selected at init time based on detected GPU hardware.
 * All backends expose the same C API; the kernel implementations differ.
 */

enum kllm_gpu_backend {
	KLLM_GPU_NONE = 0,
	KLLM_GPU_XE,
	KLLM_GPU_CUDA,
	KLLM_GPU_ROCM,
};

struct kllm_gpu_attention_params {
	void *output;             /* device ptr: [num_seqs, num_heads, head_dim] bf16 */
	const void *query;        /* device ptr */
	const void *block_tables; /* device ptr: block_table_entry array */
	const int *seq_lens;      /* device ptr */
	int num_seqs;
	int num_heads;
	int num_kv_heads;
	int head_dim;
	int max_blocks_per_seq;
	float scale;
	void *stream;             /* backend-specific: hipStream_t / cudaStream_t / sycl::queue* */
};

struct kllm_gpu_ctx;

/* Probe for available GPU and select backend */
struct kllm_gpu_ctx *kllm_gpu_create(void);
void kllm_gpu_destroy(struct kllm_gpu_ctx *ctx);

/* Get detected backend */
enum kllm_gpu_backend kllm_gpu_get_backend(struct kllm_gpu_ctx *ctx);

/* Run paged attention on the detected GPU */
int kllm_gpu_paged_attention(struct kllm_gpu_ctx *ctx,
			     const struct kllm_gpu_attention_params *params);

/* Allocate device memory (VRAM) */
void *kllm_gpu_malloc(struct kllm_gpu_ctx *ctx, uint64_t size);
void kllm_gpu_free(struct kllm_gpu_ctx *ctx, void *ptr);

/* Copy host→device */
int kllm_gpu_memcpy_h2d(struct kllm_gpu_ctx *ctx, void *dst,
			const void *src, uint64_t size);

/* Get the native stream/queue for P2P registration */
void *kllm_gpu_get_stream(struct kllm_gpu_ctx *ctx);

/* Backend-specific launchers (called by kllm_gpu_paged_attention) */
extern int kllm_gpu_paged_attention_rocm(
    void *output, const void *query, const void *block_tables,
    const int *seq_lens, int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int max_blocks_per_seq, float scale, void *stream);

extern int kllm_gpu_paged_attention_cuda(
    void *output, const void *query, const void *block_tables,
    const int *seq_lens, int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int max_blocks_per_seq, float scale, void *stream);

extern int kllm_gpu_paged_attention_xe(
    void *output, const void *query, const void *block_tables,
    const int *seq_lens, int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int max_blocks_per_seq, float scale, void *sycl_queue_ptr);

#endif /* _KLLM_GPU_ATTENTION_H */
