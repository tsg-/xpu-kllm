/* SPDX-License-Identifier: Apache-2.0 */
/*
 * GPU paged attention kernel for xpu-kllm (CUDA / NVIDIA).
 *
 * Same algorithm as the HIP version: paged KV cache with block tables,
 * online softmax, GQA support. KV pages arrive via NVMe-EP + GDS
 * directly into VRAM.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

#define BLOCK_SIZE      128
#define WARP_SIZE       32      /* NVIDIA warp size */
#define HEAD_DIM        128

struct block_table_entry {
    void *k_ptr;
    void *v_ptr;
    int   valid;
};

__global__ void paged_attention_kernel_cuda(
    __nv_bfloat16 *__restrict__ output,
    const __nv_bfloat16 *__restrict__ query,
    const struct block_table_entry *__restrict__ block_tables,
    const int *__restrict__ seq_lens,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const float scale)
{
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (seq_idx >= num_seqs)
        return;

    const int seq_len = seq_lens[seq_idx];
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const __nv_bfloat16 *q = query + (seq_idx * num_heads + head_idx) * head_dim;

    float max_score = -1e20f;
    float exp_sum = 0.0f;

    /* Per-thread partial accumulator for output vector */
    float acc[HEAD_DIM / WARP_SIZE] = {0};

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const struct block_table_entry *entry =
            &block_tables[seq_idx * max_blocks_per_seq + block_idx];

        if (!entry->valid || !entry->k_ptr)
            continue;

        const __nv_bfloat16 *k_block = (const __nv_bfloat16 *)entry->k_ptr +
                                        kv_head_idx * BLOCK_SIZE * head_dim;
        const __nv_bfloat16 *v_block = (const __nv_bfloat16 *)entry->v_ptr +
                                        kv_head_idx * BLOCK_SIZE * head_dim;

        int block_start = block_idx * BLOCK_SIZE;
        int block_tokens = min(BLOCK_SIZE, seq_len - block_start);

        for (int t = 0; t < block_tokens; t++) {
            const __nv_bfloat16 *k_vec = k_block + t * head_dim;
            float dot = 0.0f;

            for (int d = tid; d < head_dim; d += WARP_SIZE) {
                float q_val = __bfloat162float(q[d]);
                float k_val = __bfloat162float(k_vec[d]);
                dot += q_val * k_val;
            }

            /* Warp reduction */
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, offset);

            float score = dot * scale;

            /* Online softmax (lane 0 tracks state) */
            if (tid == 0) {
                float old_max = max_score;
                max_score = fmaxf(max_score, score);
                float correction = expf(old_max - max_score);
                exp_sum = exp_sum * correction + expf(score - max_score);
                for (int i = 0; i < HEAD_DIM / WARP_SIZE; i++)
                    acc[i] *= correction;
            }

            max_score = __shfl_sync(0xffffffff, max_score, 0);
            float weight = expf(score - max_score);
            weight = __shfl_sync(0xffffffff, weight, 0);

            const __nv_bfloat16 *v_vec = v_block + t * head_dim;
            for (int d = tid; d < head_dim; d += WARP_SIZE) {
                int acc_idx = d / WARP_SIZE;
                if (acc_idx < HEAD_DIM / WARP_SIZE)
                    acc[acc_idx] += weight * __bfloat162float(v_vec[d]);
            }
        }
    }

    /* Normalize and write */
    exp_sum = __shfl_sync(0xffffffff, exp_sum, 0);
    float inv_sum = (exp_sum > 0.0f) ? (1.0f / exp_sum) : 0.0f;

    __nv_bfloat16 *out = output + (seq_idx * num_heads + head_idx) * head_dim;
    for (int d = tid; d < head_dim; d += WARP_SIZE) {
        int acc_idx = d / WARP_SIZE;
        float val = (acc_idx < HEAD_DIM / WARP_SIZE) ? acc[acc_idx] * inv_sum : 0.0f;
        out[d] = __float2bfloat16(val);
    }
}

extern "C" int kllm_gpu_paged_attention_cuda(
    void *output,
    const void *query,
    const void *block_tables,
    const int *seq_lens,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    float scale,
    cudaStream_t stream)
{
    dim3 grid(num_seqs, num_heads);
    dim3 block(WARP_SIZE);

    paged_attention_kernel_cuda<<<grid, block, 0, stream>>>(
        (__nv_bfloat16 *)output,
        (const __nv_bfloat16 *)query,
        (const struct block_table_entry *)block_tables,
        seq_lens,
        num_seqs, num_heads, num_kv_heads,
        head_dim, max_blocks_per_seq, scale);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
