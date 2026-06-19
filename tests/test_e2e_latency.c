// SPDX-License-Identifier: Apache-2.0
/*
 * E2E latency test: prompt → tokenize → ring → reactor → attention → response
 *
 * Two modes:
 * 1. Full path (--chardev): writes to /dev/llm_prompt1, measures time to
 *    first generated token on the response path.
 * 2. Synthetic (default): exercises dispatch → cache lookup → ACE attention
 *    with pre-tokenized input and a small model config. No kernel required.
 *
 * Usage:
 *   ./test_e2e_latency                   # synthetic mode
 *   ./test_e2e_latency --chardev         # full chardev path (needs kllm.ko)
 *   ./test_e2e_latency --iterations 100  # repeat for statistics
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <openssl/sha.h>

#include "../userspace/include/kllm_ring_user.h"
#include "../userspace/kvcache/block_format.h"
#include "../userspace/kvcache/arena_alloc.h"
#include "../userspace/kvcache/cache_index.h"
#include "../userspace/compute/ace_attention.h"
#include "../userspace/compute/dispatch_policy.h"

#define RING_SIZE		(2 * 1024 * 1024)
#define CHARDEV_PATH		"/dev/llm_prompt1"
#define DEFAULT_ITERATIONS	10

/* Small model config for testing (1-layer, 4 KV heads, dim 64) */
#define TEST_NUM_LAYERS		1
#define TEST_NUM_KV_HEADS	4
#define TEST_HEAD_DIM		64
#define TEST_TOKENS_PER_BLOCK	16

static inline uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

struct latency_stats {
	uint64_t min_ns;
	uint64_t max_ns;
	uint64_t sum_ns;
	uint32_t count;
};

static void stats_init(struct latency_stats *s)
{
	s->min_ns = UINT64_MAX;
	s->max_ns = 0;
	s->sum_ns = 0;
	s->count = 0;
}

static void stats_add(struct latency_stats *s, uint64_t ns)
{
	if (ns < s->min_ns) s->min_ns = ns;
	if (ns > s->max_ns) s->max_ns = ns;
	s->sum_ns += ns;
	s->count++;
}

static void stats_print(const char *label, struct latency_stats *s)
{
	if (s->count == 0) {
		printf("  %s: no samples\n", label);
		return;
	}
	printf("  %s: min=%luus avg=%luus max=%luus (n=%u)\n",
	       label,
	       (unsigned long)(s->min_ns / 1000),
	       (unsigned long)(s->sum_ns / s->count / 1000),
	       (unsigned long)(s->max_ns / 1000),
	       s->count);
}

/*
 * Synthetic test: exercise dispatch → cache lookup → ACE attention
 * with pre-computed token IDs and a dummy model.
 */
static int test_synthetic_path(int iterations)
{
	struct latency_stats dispatch_lat, lookup_lat, attention_lat, total_lat;
	struct kllm_model_config model = {
		.num_layers = TEST_NUM_LAYERS,
		.num_kv_heads = TEST_NUM_KV_HEADS,
		.head_dim = TEST_HEAD_DIM,
		.tokens_per_block = TEST_TOKENS_PER_BLOCK,
	};

	stats_init(&dispatch_lat);
	stats_init(&lookup_lat);
	stats_init(&attention_lat);
	stats_init(&total_lat);

	/* Compute block size from model config */
	uint64_t block_size = kllm_block_total_bytes(&model);

	/* Allocate arena */
	struct kllm_arena_config acfg = {
		.total_bytes = block_size * 256,  /* 256 blocks */
		.hugepage_size_mb = 2,
		.numa_node = -1,
		.model = model,
	};
	struct kllm_arena *arena = kllm_arena_create(&acfg);
	if (!arena) {
		fprintf(stderr, "FAIL: arena alloc (may need hugepages)\n");
		fprintf(stderr, "  Try: echo 64 > /proc/sys/vm/nr_hugepages\n");
		return -1;
	}

	/* Create cache index */
	struct kllm_index_config icfg = {
		.capacity = 256,
		.arena = arena,
	};
	struct kllm_cache_index *cache_idx = kllm_index_create(&icfg);
	if (!cache_idx) {
		fprintf(stderr, "FAIL: cache index alloc\n");
		kllm_arena_destroy(arena);
		return -1;
	}

	/* Pre-warm: insert a synthetic KV entry for our test sequence */
	uint32_t test_tokens[] = {100, 200, 300, 400};
	uint32_t test_len = 4;

	/* Compute content-address hash for token prefix */
	uint8_t seq_hash[KLLM_INDEX_HASH_BYTES];
	SHA256((const unsigned char *)test_tokens,
	       test_len * sizeof(uint32_t), seq_hash);

	/* Insert into index — returns pointer to block in arena */
	void *block_ptr = kllm_index_insert(cache_idx, seq_hash);
	if (!block_ptr) {
		fprintf(stderr, "FAIL: index insert\n");
		kllm_index_destroy(cache_idx);
		kllm_arena_destroy(arena);
		return -1;
	}

	/* Fill KV block with synthetic data */
	struct kllm_block_hdr *hdr = (struct kllm_block_hdr *)block_ptr;
	hdr->magic = KLLM_BLOCK_MAGIC;
	hdr->version = KLLM_BLOCK_VERSION;
	hdr->block_size_bytes = (uint32_t)block_size;
	hdr->tokens_per_block = TEST_TOKENS_PER_BLOCK;
	hdr->num_layers = model.num_layers;
	hdr->num_kv_heads = model.num_kv_heads;
	hdr->head_dim = model.head_dim;
	hdr->flags = KLLM_BLOCK_F_VALID;
	memcpy(hdr->seq_hash, seq_hash, KLLM_SEQ_HASH_BYTES);

	/* Fill K/V data with synthetic BF16 values (~1.0) */
	bf16_t *k_data = (bf16_t *)kllm_block_key(block_ptr, &model, 0);
	bf16_t *v_data = (bf16_t *)kllm_block_value(block_ptr, &model, 0);
	uint64_t kv_elems = (uint64_t)TEST_TOKENS_PER_BLOCK * TEST_NUM_KV_HEADS * TEST_HEAD_DIM;
	for (uint64_t i = 0; i < kv_elems; i++) {
		k_data[i] = 0x3F80;  /* ~1.0 in BF16 */
		v_data[i] = 0x3F00;  /* ~0.5 in BF16 */
	}

	/* Build dispatch context */
	struct kllm_dispatch_config dcfg = {
		.cpu_seq_threshold = 512,
		.cache_hit_threshold = 0.8f,
		.gpu_queue_max = 64,
	};
	struct kllm_dispatch_ctx *dispatch = kllm_dispatch_create(&dcfg);

	printf("synthetic path (%d iterations):\n", iterations);
	printf("  model: %u layers, %u kv_heads, dim %u, %u tokens/block\n",
	       model.num_layers, model.num_kv_heads, model.head_dim,
	       model.tokens_per_block);
	printf("  block size: %lu bytes\n", (unsigned long)block_size);

	/* Allocate query and output buffers */
	uint32_t num_heads = TEST_NUM_KV_HEADS;  /* MHA for test (no GQA) */
	uint64_t q_elems = num_heads * TEST_HEAD_DIM;
	bf16_t *query = (bf16_t *)calloc(q_elems, sizeof(bf16_t));
	bf16_t *output = (bf16_t *)calloc(q_elems, sizeof(bf16_t));
	for (uint64_t i = 0; i < q_elems; i++)
		query[i] = 0x3F80;  /* ~1.0 in BF16 */

	for (int i = 0; i < iterations; i++) {
		uint64_t t0, t1, t2, t3;

		t0 = now_ns();

		/* Step 1: dispatch decision */
		enum kllm_dispatch_target target =
			kllm_dispatch_decide(dispatch, test_len, true);
		t1 = now_ns();

		if (target != KLLM_TARGET_CPU) {
			printf("  WARN: dispatch chose GPU for %u tokens\n", test_len);
			continue;
		}

		/* Step 2: cache lookup */
		void *found = kllm_index_lookup(cache_idx, seq_hash);
		t2 = now_ns();
		if (!found) {
			printf("  FAIL: cache miss on pre-warmed entry\n");
			free(query);
			free(output);
			kllm_dispatch_destroy(dispatch);
			kllm_index_destroy(cache_idx);
			kllm_arena_destroy(arena);
			return -1;
		}

		/* Step 3: ACE attention kernel */
		struct kllm_attention_params params = {
			.num_heads = num_heads,
			.head_dim = TEST_HEAD_DIM,
			.seq_len = test_len,
			.query_len = 1,
			.scale = 1.0f / 8.0f,  /* 1/sqrt(64) */
		};

		bf16_t *cached_k = (bf16_t *)kllm_block_key(found, &model, 0);
		bf16_t *cached_v = (bf16_t *)kllm_block_value(found, &model, 0);

		kllm_ace_attention(query, cached_k, cached_v, output, NULL, &params);
		t3 = now_ns();

		stats_add(&dispatch_lat, t1 - t0);
		stats_add(&lookup_lat, t2 - t1);
		stats_add(&attention_lat, t3 - t2);
		stats_add(&total_lat, t3 - t0);
	}

	printf("results:\n");
	stats_print("dispatch decision", &dispatch_lat);
	stats_print("cache lookup", &lookup_lat);
	stats_print("ACE attention", &attention_lat);
	stats_print("total (dispatch+lookup+attention)", &total_lat);

	free(query);
	free(output);
	kllm_dispatch_destroy(dispatch);
	kllm_index_destroy(cache_idx);
	kllm_arena_destroy(arena);
	return 0;
}

/*
 * Full chardev test: write prompt → eBPF tokenizes → ring → read response.
 * Measures wall-clock from write() to first token on read().
 */
static int test_chardev_path(int iterations)
{
	struct latency_stats ttft;  /* time to first token */
	int fd;
	void *ring_map;
	struct kllm_ring_hdr *ring;

	fd = open(CHARDEV_PATH, O_RDWR);
	if (fd < 0) {
		perror("open " CHARDEV_PATH);
		fprintf(stderr, "Full chardev test requires kllm.ko loaded\n");
		return -1;
	}

	ring_map = mmap(NULL, RING_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (ring_map == MAP_FAILED) {
		perror("mmap token ring");
		close(fd);
		return -1;
	}
	ring = (struct kllm_ring_hdr *)ring_map;

	if (ring->magic != KLLM_TOKEN_RING_MAGIC) {
		fprintf(stderr, "ring magic mismatch\n");
		munmap(ring_map, RING_SIZE);
		close(fd);
		return -1;
	}

	stats_init(&ttft);
	printf("chardev path (%d iterations):\n", iterations);

	const char *prompt = "Hello world";
	uint32_t response_token;

	for (int i = 0; i < iterations; i++) {
		uint64_t t0 = now_ns();

		ssize_t written = write(fd, prompt, strlen(prompt));
		if (written < 0) {
			perror("write prompt");
			break;
		}

		/* Poll for first response token (blocking read) */
		ssize_t nread = read(fd, &response_token, sizeof(response_token));
		uint64_t t1 = now_ns();

		if (nread < 0) {
			perror("read response");
			break;
		}
		if (nread == 0) {
			printf("  iter %d: got EOS immediately (no generation)\n", i);
			continue;
		}

		stats_add(&ttft, t1 - t0);
	}

	printf("results:\n");
	stats_print("time-to-first-token", &ttft);

	munmap(ring_map, RING_SIZE);
	close(fd);
	return 0;
}

int main(int argc, char **argv)
{
	int use_chardev = 0;
	int iterations = DEFAULT_ITERATIONS;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--chardev") == 0)
			use_chardev = 1;
		else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
			iterations = atoi(argv[++i]);
	}

	printf("=== xpu-kllm E2E latency test ===\n");
	printf("mode: %s\n\n", use_chardev ? "chardev (full path)" : "synthetic");

	int rc;
	if (use_chardev)
		rc = test_chardev_path(iterations);
	else
		rc = test_synthetic_path(iterations);

	return rc < 0 ? 1 : 0;
}
