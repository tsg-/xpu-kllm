/* SPDX-License-Identifier: Apache-2.0 */
/*
 * CPU vs GPU dispatch policy for xpu-kllm.
 */

#include <stdlib.h>
#include <stdatomic.h>
#include "dispatch_policy.h"

struct kllm_dispatch_ctx {
	struct kllm_dispatch_config cfg;
	struct kllm_dispatch_state state;
};

struct kllm_dispatch_ctx *kllm_dispatch_create(const struct kllm_dispatch_config *cfg)
{
	struct kllm_dispatch_ctx *ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;

	ctx->cfg = *cfg;

	/* Defaults */
	if (ctx->cfg.cpu_seq_threshold == 0)
		ctx->cfg.cpu_seq_threshold = 512;
	if (ctx->cfg.cache_hit_threshold == 0.0f)
		ctx->cfg.cache_hit_threshold = 0.8f;
	if (ctx->cfg.gpu_queue_max == 0)
		ctx->cfg.gpu_queue_max = 64;

	return ctx;
}

void kllm_dispatch_destroy(struct kllm_dispatch_ctx *ctx)
{
	free(ctx);
}

enum kllm_dispatch_target kllm_dispatch_decide(struct kllm_dispatch_ctx *ctx,
					       uint32_t seq_len, bool cache_hit)
{
	enum kllm_dispatch_target target;

	ctx->state.total_requests++;

	if (cache_hit)
		ctx->state.cache_hits++;
	else
		ctx->state.cache_misses++;

	/*
	 * Decision logic:
	 * 1. If sequence is short AND cache hit → CPU (fast path)
	 * 2. If sequence exceeds threshold → GPU (memory-bound on CPU)
	 * 3. If cache miss → GPU (needs NVMe-EP fetch anyway)
	 * 4. If GPU is saturated AND seq is short → CPU (overflow)
	 */
	if (seq_len <= ctx->cfg.cpu_seq_threshold && cache_hit) {
		target = KLLM_TARGET_CPU;
	} else if (seq_len > ctx->cfg.cpu_seq_threshold) {
		target = KLLM_TARGET_GPU;
	} else if (!cache_hit) {
		/* Cache miss: GPU will fetch via NVMe-EP wavefront */
		target = KLLM_TARGET_GPU;
	} else {
		target = KLLM_TARGET_GPU;
	}

	/* GPU overflow: if GPU queue is saturated and seq is CPU-eligible */
	if (target == KLLM_TARGET_GPU &&
	    ctx->state.gpu_queue_depth >= ctx->cfg.gpu_queue_max &&
	    seq_len <= ctx->cfg.cpu_seq_threshold) {
		target = KLLM_TARGET_CPU;
	}

	if (target == KLLM_TARGET_CPU) {
		ctx->state.cpu_dispatches++;
	} else {
		ctx->state.gpu_dispatches++;
		ctx->state.gpu_queue_depth++;
	}

	return target;
}

void kllm_dispatch_gpu_complete(struct kllm_dispatch_ctx *ctx)
{
	if (ctx->state.gpu_queue_depth > 0)
		ctx->state.gpu_queue_depth--;
}

void kllm_dispatch_get_state(struct kllm_dispatch_ctx *ctx,
			     struct kllm_dispatch_state *out)
{
	*out = ctx->state;
}
