/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_DISPATCH_POLICY_H
#define _KLLM_DISPATCH_POLICY_H

#include <stdint.h>
#include <stdbool.h>

/*
 * CPU vs GPU dispatch policy.
 *
 * Decision factors:
 * 1. Sequence length — short prefixes go CPU (ACE), long go GPU
 * 2. KV cache hit rate — hits stay on CPU, misses need NVMe-EP (GPU path)
 * 3. Current GPU queue depth — if GPU is saturated, prefer CPU for short
 *
 * The threshold is empirically tunable: ACE on Xeon beats GPU round-trip
 * for prefixes that fit in L3 (typically ≤512 tokens for 70B model KV).
 */

enum kllm_dispatch_target {
	KLLM_TARGET_CPU = 0,  /* ACE attention in SPDK reactor */
	KLLM_TARGET_GPU = 1,  /* hand off to GPU via NVMe-EP wavefronts */
};

struct kllm_dispatch_config {
	uint32_t cpu_seq_threshold;   /* max seq_len for CPU path (default: 512) */
	float    cache_hit_threshold; /* min hit rate to stay CPU (default: 0.8) */
	uint32_t gpu_queue_max;       /* GPU queue depth before overflow to CPU */
};

struct kllm_dispatch_state {
	uint64_t total_requests;
	uint64_t cpu_dispatches;
	uint64_t gpu_dispatches;
	uint64_t cache_hits;
	uint64_t cache_misses;
	uint32_t gpu_queue_depth;
};

struct kllm_dispatch_ctx;

struct kllm_dispatch_ctx *kllm_dispatch_create(const struct kllm_dispatch_config *cfg);
void kllm_dispatch_destroy(struct kllm_dispatch_ctx *ctx);

/*
 * Decide where to run attention for this request.
 *
 * seq_len:   total sequence length (prompt + generated so far)
 * cache_hit: whether the KV cache had a hit for this prefix
 */
enum kllm_dispatch_target kllm_dispatch_decide(struct kllm_dispatch_ctx *ctx,
					       uint32_t seq_len, bool cache_hit);

/* Notify dispatch that GPU completed a request (updates queue depth) */
void kllm_dispatch_gpu_complete(struct kllm_dispatch_ctx *ctx);

void kllm_dispatch_get_state(struct kllm_dispatch_ctx *ctx,
			     struct kllm_dispatch_state *out);

#endif /* _KLLM_DISPATCH_POLICY_H */
