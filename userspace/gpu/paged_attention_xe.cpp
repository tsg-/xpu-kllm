/* SPDX-License-Identifier: Apache-2.0 */
/*
 * GPU paged attention kernel for xpu-kllm (Intel Xe / Data Center GPU Max).
 *
 * Implemented in SYCL targeting Level Zero backend. Same paged KV cache
 * algorithm, adapted for Intel's subgroup (EU thread) execution model.
 *
 * Intel Xe specifics:
 * - Subgroup size: 16 or 32 (vs 32 NVIDIA, 64 AMD)
 * - BF16 via hardware XMX (Xe Matrix Extensions)
 * - Unified Shared Memory (USM) for host↔device transfers
 * - dma-buf import via Level Zero for P2P from RNIC
 */

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

namespace kllm {

constexpr int BLOCK_SIZE = 128;
constexpr int HEAD_DIM = 128;
constexpr int SUBGROUP_SIZE = 16;  /* Intel Xe EU thread width */

struct block_table_entry {
    void *k_ptr;
    void *v_ptr;
    int   valid;
};

using bf16 = sycl::ext::oneapi::bfloat16;

/*
 * Paged attention kernel — one work-group per (seq, head) pair.
 * Each subgroup handles a slice of the head dimension.
 */
class PagedAttentionKernel {
public:
    PagedAttentionKernel(bf16 *output, const bf16 *query,
                         const block_table_entry *block_tables,
                         const int *seq_lens,
                         int num_seqs, int num_heads, int num_kv_heads,
                         int head_dim, int max_blocks_per_seq, float scale)
        : output_(output), query_(query), block_tables_(block_tables),
          seq_lens_(seq_lens), num_seqs_(num_seqs), num_heads_(num_heads),
          num_kv_heads_(num_kv_heads), head_dim_(head_dim),
          max_blocks_per_seq_(max_blocks_per_seq), scale_(scale) {}

    void operator()(sycl::nd_item<2> item) const {
        const int seq_idx = item.get_global_id(0);
        const int head_idx = item.get_global_id(1);
        auto sg = item.get_sub_group();
        const int sg_lid = sg.get_local_linear_id();

        if (seq_idx >= num_seqs_)
            return;

        const int seq_len = seq_lens_[seq_idx];
        const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int kv_head_idx = head_idx / (num_heads_ / num_kv_heads_);

        const bf16 *q = query_ + (seq_idx * num_heads_ + head_idx) * head_dim_;

        float max_score = -1e20f;
        float exp_sum = 0.0f;
        float acc[HEAD_DIM / SUBGROUP_SIZE] = {0};

        for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
            const auto &entry = block_tables_[seq_idx * max_blocks_per_seq_ + block_idx];

            if (!entry.valid || !entry.k_ptr)
                continue;

            const bf16 *k_block = static_cast<const bf16 *>(entry.k_ptr) +
                                  kv_head_idx * BLOCK_SIZE * head_dim_;
            const bf16 *v_block = static_cast<const bf16 *>(entry.v_ptr) +
                                  kv_head_idx * BLOCK_SIZE * head_dim_;

            int block_start = block_idx * BLOCK_SIZE;
            int block_tokens = sycl::min(BLOCK_SIZE, seq_len - block_start);

            for (int t = 0; t < block_tokens; t++) {
                const bf16 *k_vec = k_block + t * head_dim_;
                float dot = 0.0f;

                /* Strided dot product across subgroup */
                for (int d = sg_lid; d < head_dim_; d += SUBGROUP_SIZE) {
                    float q_val = static_cast<float>(q[d]);
                    float k_val = static_cast<float>(k_vec[d]);
                    dot += q_val * k_val;
                }

                /* Subgroup reduction */
                dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());

                float score = dot * scale_;

                /* Online softmax */
                if (sg_lid == 0) {
                    float old_max = max_score;
                    max_score = sycl::fmax(max_score, score);
                    float correction = sycl::exp(old_max - max_score);
                    exp_sum = exp_sum * correction + sycl::exp(score - max_score);
                    for (int i = 0; i < HEAD_DIM / SUBGROUP_SIZE; i++)
                        acc[i] *= correction;
                }

                /* Broadcast from lane 0 */
                max_score = sycl::group_broadcast(sg, max_score, 0);
                float weight = sycl::exp(score - max_score);
                weight = sycl::group_broadcast(sg, weight, 0);

                /* Accumulate weighted V */
                const bf16 *v_vec = v_block + t * head_dim_;
                for (int d = sg_lid; d < head_dim_; d += SUBGROUP_SIZE) {
                    int acc_idx = d / SUBGROUP_SIZE;
                    if (acc_idx < HEAD_DIM / SUBGROUP_SIZE)
                        acc[acc_idx] += weight * static_cast<float>(v_vec[d]);
                }
            }
        }

        /* Normalize and write output */
        exp_sum = sycl::group_broadcast(sg, exp_sum, 0);
        float inv_sum = (exp_sum > 0.0f) ? (1.0f / exp_sum) : 0.0f;

        bf16 *out = output_ + (seq_idx * num_heads_ + head_idx) * head_dim_;
        for (int d = sg_lid; d < head_dim_; d += SUBGROUP_SIZE) {
            int acc_idx = d / SUBGROUP_SIZE;
            float val = (acc_idx < HEAD_DIM / SUBGROUP_SIZE) ? acc[acc_idx] * inv_sum : 0.0f;
            out[d] = bf16(val);
        }
    }

private:
    bf16 *output_;
    const bf16 *query_;
    const block_table_entry *block_tables_;
    const int *seq_lens_;
    int num_seqs_, num_heads_, num_kv_heads_;
    int head_dim_, max_blocks_per_seq_;
    float scale_;
};

}  // namespace kllm

extern "C" int kllm_gpu_paged_attention_xe(
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
    void *sycl_queue_ptr)
{
    auto *queue = static_cast<sycl::queue *>(sycl_queue_ptr);

    using namespace kllm;
    using bf16 = sycl::ext::oneapi::bfloat16;

    try {
        queue->submit([&](sycl::handler &cgh) {
            auto kernel = PagedAttentionKernel(
                static_cast<bf16 *>(output),
                static_cast<const bf16 *>(query),
                static_cast<const block_table_entry *>(block_tables),
                seq_lens,
                num_seqs, num_heads, num_kv_heads,
                head_dim, max_blocks_per_seq, scale);

            sycl::range<2> global(num_seqs, num_heads);
            sycl::range<2> local(1, 1);

            cgh.parallel_for(
                sycl::nd_range<2>(global, local),
                kernel);
        });
        queue->wait();
    } catch (const sycl::exception &e) {
        return -1;
    }

    return 0;
}
