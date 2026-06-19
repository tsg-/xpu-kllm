/* SPDX-License-Identifier: Apache-2.0 */
/*
 * xpu-kllm SPDK reactor: polls the hugepage token ring, dispatches
 * inference requests to CPU (ACE) or GPU path.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "spdk/env.h"
#include "spdk/event.h"
#include "spdk/thread.h"
#include "spdk/log.h"

#include "../include/kllm_ring_user.h"
#include "../kvcache/cache_index.h"
#include "../kvcache/arena_alloc.h"
#include "../compute/ace_attention.h"
#include "../compute/dispatch_policy.h"

#define KLLM_CHARDEV_PATH	"/dev/llm_prompt1"
#define KLLM_RING_SIZE		(2 * 1024 * 1024)  /* 2MB */
#define KLLM_POLL_BATCH		256
#define KLLM_ARENA_SIZE_GB	4
#define KLLM_INDEX_CAPACITY	(64 * 1024)  /* 64K entries */

struct kllm_reactor_ctx {
	int			chardev_fd;
	struct kllm_ring_hdr	*ring;
	struct spdk_poller	*poller;
	uint64_t		tokens_consumed;
	uint64_t		prompts_dispatched;

	/* KV cache */
	struct kllm_arena	*arena;
	struct kllm_cache_index	*cache_idx;

	/* Dispatch policy */
	struct kllm_dispatch_ctx *dispatch;

	/* Model config */
	struct kllm_model_config model;
};

static struct kllm_reactor_ctx g_ctx;

/*
 * Token ring poller — called on every reactor iteration.
 * Drains available tokens from the ring in batches.
 */
static int kllm_ring_poll(void *arg)
{
	struct kllm_reactor_ctx *ctx = arg;
	uint32_t batch[KLLM_POLL_BATCH];
	uint32_t count;

	count = kllm_ring_consume_batch(ctx->ring, batch, KLLM_POLL_BATCH);
	if (count == 0)
		return SPDK_POLLER_IDLE;

	ctx->tokens_consumed += count;

	/*
	 * Dispatch: compute SHA-256 of token prefix for content-addressed
	 * KV cache lookup, then decide CPU vs GPU.
	 *
	 * TODO: accumulate tokens into per-request sequence buffers.
	 * For now, treat each batch as a complete prompt (vertical slice).
	 */
	uint8_t seq_hash[32];
	/* TODO: SHA-256(batch, count * sizeof(uint32_t)) → seq_hash */
	memset(seq_hash, 0, sizeof(seq_hash));

	void *kv_block = kllm_index_lookup(ctx->cache_idx, seq_hash);
	bool cache_hit = (kv_block != NULL);

	enum kllm_dispatch_target target =
		kllm_dispatch_decide(ctx->dispatch, count, cache_hit);

	if (target == KLLM_TARGET_CPU && kv_block) {
		/* Short path: ACE attention on reactor thread */
		/* TODO: extract Q from current token embedding,
		 * run kllm_ace_attention against cached K/V */
		ctx->prompts_dispatched++;
	} else {
		/* Long path: hand off to GPU via io_uring GDS */
		/* TODO: construct wavefront, submit io_uring SQEs */
		ctx->prompts_dispatched++;
	}

	return SPDK_POLLER_BUSY;
}

static int kllm_map_token_ring(struct kllm_reactor_ctx *ctx)
{
	int fd;
	void *map;
	struct kllm_ring_hdr *hdr;

	fd = open(KLLM_CHARDEV_PATH, O_RDWR);
	if (fd < 0) {
		SPDK_ERRLOG("failed to open %s: %m\n", KLLM_CHARDEV_PATH);
		return -1;
	}

	map = mmap(NULL, KLLM_RING_SIZE, PROT_READ | PROT_WRITE,
		   MAP_SHARED, fd, 0);
	if (map == MAP_FAILED) {
		SPDK_ERRLOG("failed to mmap token ring: %m\n");
		close(fd);
		return -1;
	}

	hdr = (struct kllm_ring_hdr *)map;
	if (hdr->magic != KLLM_TOKEN_RING_MAGIC) {
		SPDK_ERRLOG("token ring magic mismatch: 0x%x\n", hdr->magic);
		munmap(map, KLLM_RING_SIZE);
		close(fd);
		return -1;
	}

	ctx->chardev_fd = fd;
	ctx->ring = hdr;

	SPDK_NOTICELOG("mapped token ring: %u slots (%lu bytes)\n",
		       hdr->slot_count, (unsigned long)KLLM_RING_SIZE);
	return 0;
}

static int kllm_init_kvcache(struct kllm_reactor_ctx *ctx)
{
	struct kllm_arena_config arena_cfg = {
		.total_bytes = (uint64_t)KLLM_ARENA_SIZE_GB * 1024 * 1024 * 1024,
		.hugepage_size_mb = 2,
		.numa_node = -1,
		.model = ctx->model,
	};

	ctx->arena = kllm_arena_create(&arena_cfg);
	if (!ctx->arena) {
		SPDK_ERRLOG("failed to create KV cache arena\n");
		return -1;
	}

	struct kllm_index_config idx_cfg = {
		.capacity = KLLM_INDEX_CAPACITY,
		.arena = ctx->arena,
	};

	ctx->cache_idx = kllm_index_create(&idx_cfg);
	if (!ctx->cache_idx) {
		SPDK_ERRLOG("failed to create cache index\n");
		kllm_arena_destroy(ctx->arena);
		return -1;
	}

	struct kllm_dispatch_config disp_cfg = {
		.cpu_seq_threshold = 512,
		.cache_hit_threshold = 0.8f,
		.gpu_queue_max = 64,
	};

	ctx->dispatch = kllm_dispatch_create(&disp_cfg);
	if (!ctx->dispatch) {
		SPDK_ERRLOG("failed to create dispatch policy\n");
		kllm_index_destroy(ctx->cache_idx);
		kllm_arena_destroy(ctx->arena);
		return -1;
	}

	return 0;
}

static void kllm_reactor_start(void *arg)
{
	struct kllm_reactor_ctx *ctx = &g_ctx;
	int rc;

	/* Default model config (Llama-70B-like) */
	ctx->model.num_layers = 80;
	ctx->model.num_kv_heads = 8;
	ctx->model.head_dim = 128;
	ctx->model.tokens_per_block = 128;

	/* Probe CPU ISA for ACE/AMX support */
	int isa = kllm_ace_probe();
	SPDK_NOTICELOG("CPU ISA level: %d (0=none, 1=avx512bf16, 2=amx, 3=ace)\n", isa);

	rc = kllm_map_token_ring(ctx);
	if (rc) {
		spdk_app_stop(-1);
		return;
	}

	rc = kllm_init_kvcache(ctx);
	if (rc) {
		spdk_app_stop(-1);
		return;
	}

	ctx->poller = spdk_poller_register(kllm_ring_poll, ctx, 0);
	if (!ctx->poller) {
		SPDK_ERRLOG("failed to register ring poller\n");
		spdk_app_stop(-1);
		return;
	}

	SPDK_NOTICELOG("kllm reactor started, polling token ring\n");
}

static void kllm_reactor_shutdown(void)
{
	struct kllm_reactor_ctx *ctx = &g_ctx;

	if (ctx->poller)
		spdk_poller_unregister(&ctx->poller);

	SPDK_NOTICELOG("kllm reactor shutdown: consumed %lu tokens, %lu prompts\n",
		       ctx->tokens_consumed, ctx->prompts_dispatched);

	struct kllm_dispatch_state dstate;
	if (ctx->dispatch) {
		kllm_dispatch_get_state(ctx->dispatch, &dstate);
		SPDK_NOTICELOG("dispatch: cpu=%lu gpu=%lu hits=%lu misses=%lu\n",
			       dstate.cpu_dispatches, dstate.gpu_dispatches,
			       dstate.cache_hits, dstate.cache_misses);
		kllm_dispatch_destroy(ctx->dispatch);
	}

	if (ctx->cache_idx)
		kllm_index_destroy(ctx->cache_idx);
	if (ctx->arena)
		kllm_arena_destroy(ctx->arena);

	if (ctx->ring) {
		munmap(ctx->ring, KLLM_RING_SIZE);
		ctx->ring = NULL;
	}
	if (ctx->chardev_fd >= 0) {
		close(ctx->chardev_fd);
		ctx->chardev_fd = -1;
	}

	spdk_app_stop(0);
}

int main(int argc, char **argv)
{
	struct spdk_app_opts opts;
	int rc;

	spdk_app_opts_init(&opts, sizeof(opts));
	opts.name = "kllm_reactor";
	opts.shutdown_cb = kllm_reactor_shutdown;

	rc = spdk_app_start(&opts, kllm_reactor_start, NULL);
	spdk_app_fini();
	return rc;
}
