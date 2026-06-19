/* SPDX-License-Identifier: Apache-2.0 */
/*
 * GPU paged attention kernel for xpu-kllm (HIP/ROCm).
 *
 * Adapted from vLLM's PagedAttention V2 concept:
 * - KV cache stored in fixed-size pages (blocks)
 * - Pages are non-contiguous in VRAM (paged, not sequential)
 * - Block table maps logical positions → physical page addresses
 *
 * Key difference from vLLM: our KV pages arrive via NVMe-EP P2P DMA
 * directly into VRAM. The block table is populated as wavefront
 * completions land, not by a host-side scheduler.
 *
 * Kernel handles: multi-query attention with GQA (grouped-query).
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#define BLOCK_SIZE      128     /* tokens per KV block */
#define WARP_SIZE       64      /* AMD wavefront size */
#define MAX_SEQ_LEN     32768
#define HEAD_DIM        128

/*
 * Block table: maps (seq_idx, block_idx) → physical VRAM pointer.
 * Populated by SPDK reactor as NVMe-EP wavefronts complete.
 */
struct block_table_entry {
    void *k_ptr;   /* physical K block in VRAM */
    void *v_ptr;   /* physical V block in VRAM */
    int   valid;   /* 1 if this block has been fetched */
};

/*
 * Paged attention kernel — one wavefront per query head.
 *
 * Each wavefront computes attention for one (query_pos, head) pair
 * across all KV blocks in the sequence.
 *
 * output:       [num_seqs, num_heads, head_dim]
 * query:        [num_seqs, num_heads, head_dim]
 * block_tables: [num_seqs, max_blocks_per_seq] → block_table_entry
 * seq_lens:     [num_seqs] actual sequence lengths
 * scale:        1/sqrt(head_dim)
 */
__global__ void paged_attention_v2_kernel(
    hip_bfloat16 *__restrict__ output,
    const hip_bfloat16 *__restrict__ query,
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

    /* GQA: map query head to KV head */
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    /* Query vector for this head */
    const hip_bfloat16 *q = query + (seq_idx * num_heads + head_idx) * head_dim;

    /* Phase 1: compute attention scores per block */
    float max_score = -1e20f;
    float exp_sum = 0.0f;
    float acc[HEAD_DIM / WARP_SIZE] = {0};  /* partial output per thread */

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const struct block_table_entry *entry =
            &block_tables[seq_idx * max_blocks_per_seq + block_idx];

        if (!entry->valid || !entry->k_ptr)
            continue;

        const hip_bfloat16 *k_block = (const hip_bfloat16 *)entry->k_ptr +
                                       kv_head_idx * BLOCK_SIZE * head_dim;
        const hip_bfloat16 *v_block = (const hip_bfloat16 *)entry->v_ptr +
                                       kv_head_idx * BLOCK_SIZE * head_dim;

        /* Tokens in this block */
        int block_start = block_idx * BLOCK_SIZE;
        int block_tokens = min(BLOCK_SIZE, seq_len - block_start);

        /* Compute Q*K^T for each token in this block */
        for (int t = 0; t < block_tokens; t++) {
            const hip_bfloat16 *k_vec = k_block + t * head_dim;
            float dot = 0.0f;

            /* Vectorized dot product across wavefront */
            for (int d = tid; d < head_dim; d += WARP_SIZE) {
                float q_val = __bfloat162float(q[d]);
                float k_val = __bfloat162float(k_vec[d]);
                dot += q_val * k_val;
            }

            /* Warp-level reduction for dot product */
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_down(dot, offset, WARP_SIZE);

            float score = dot * scale;

            /* Online softmax update */
            if (tid == 0) {
                float old_max = max_score;
                max_score = fmaxf(max_score, score);
                float exp_correction = expf(old_max - max_score);
                exp_sum = exp_sum * exp_correction + expf(score - max_score);

                /* Correct accumulated output for new max */
                for (int i = 0; i < HEAD_DIM / WARP_SIZE; i++)
                    acc[i] *= exp_correction;
            }

            /* Broadcast max_score and weight to wavefront */
            max_score = __shfl(max_score, 0, WARP_SIZE);
            float weight = expf(score - max_score);
            weight = __shfl(weight, 0, WARP_SIZE);

            /* Accumulate weighted V */
            const hip_bfloat16 *v_vec = v_block + t * head_dim;
            for (int d = tid; d < head_dim; d += WARP_SIZE) {
                int acc_idx = (d / WARP_SIZE);
                if (acc_idx < HEAD_DIM / WARP_SIZE)
                    acc[acc_idx] += weight * __bfloat162float(v_vec[d]);
            }
        }
    }

    /* Normalize by softmax sum and write output */
    exp_sum = __shfl(exp_sum, 0, WARP_SIZE);
    float inv_sum = (exp_sum > 0.0f) ? (1.0f / exp_sum) : 0.0f;

    hip_bfloat16 *out = output + (seq_idx * num_heads + head_idx) * head_dim;
    for (int d = tid; d < head_dim; d += WARP_SIZE) {
        int acc_idx = (d / WARP_SIZE);
        float val = (acc_idx < HEAD_DIM / WARP_SIZE) ? acc[acc_idx] * inv_sum : 0.0f;
        out[d] = __float2bfloat16(val);
    }
}

/*
 * Host-side launcher.
 */
extern "C" int kllm_gpu_paged_attention(
    void *output,       /* device ptr: [num_seqs, num_heads, head_dim] bf16 */
    const void *query,  /* device ptr: [num_seqs, num_heads, head_dim] bf16 */
    const void *block_tables,  /* device ptr: block_table_entry array */
    const int *seq_lens,       /* device ptr */
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    float scale,
    hipStream_t stream)
{
    dim3 grid(num_seqs, num_heads);
    dim3 block(WARP_SIZE);

    hipLaunchKernelGGL(paged_attention_v2_kernel,
                       grid, block, 0, stream,
                       (hip_bfloat16 *)output,
                       (const hip_bfloat16 *)query,
                       (const struct block_table_entry *)block_tables,
                       seq_lens,
                       num_seqs, num_heads, num_kv_heads,
                       head_dim, max_blocks_per_seq, scale);

    return hipGetLastError() == hipSuccess ? 0 : -1;
}
