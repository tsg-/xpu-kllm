// SPDX-License-Identifier: Apache-2.0
/*
 * Tokenizer benchmark: eBPF BPE vs userspace BPE.
 *
 * Measures tokens/sec and per-prompt latency for both paths:
 * 1. Kernel path: write() to /dev/llm_prompt1 → eBPF tokenizes → ring
 * 2. Userspace path: pure C BPE tokenizer (reference implementation)
 *
 * Reports speedup factor, throughput, and latency percentiles.
 *
 * Usage:
 *   ./bench_tokenizer                       # synthetic prompts
 *   ./bench_tokenizer --file corpus.txt     # real text corpus
 *   ./bench_tokenizer --iterations 1000
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "../userspace/include/kllm_ring_user.h"

#define RING_SIZE		(2 * 1024 * 1024)
#define CHARDEV_PATH		"/dev/llm_prompt1"
#define DEFAULT_ITERATIONS	100
#define MAX_PROMPT_LEN		4096
#define MAX_LATENCIES		10000

static inline uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* Simple userspace BPE tokenizer (reference, unoptimized) */
struct bpe_merge {
	uint32_t left;
	uint32_t right;
	uint32_t result;
	uint32_t priority;
};

struct userspace_tokenizer {
	uint32_t *byte_to_token;    /* 256 entries: byte → initial token */
	struct bpe_merge *merges;
	uint32_t num_merges;
};

static struct userspace_tokenizer *create_synthetic_tokenizer(void)
{
	struct userspace_tokenizer *tok = calloc(1, sizeof(*tok));
	if (!tok) return NULL;

	tok->byte_to_token = calloc(256, sizeof(uint32_t));
	if (!tok->byte_to_token) { free(tok); return NULL; }

	/* Identity mapping: byte value = token ID */
	for (int i = 0; i < 256; i++)
		tok->byte_to_token[i] = (uint32_t)i;

	/* Synthetic merges (e.g., "He" → 256, "ll" → 257, "lo" → 258) */
	tok->num_merges = 3;
	tok->merges = calloc(tok->num_merges, sizeof(struct bpe_merge));
	if (!tok->merges) { free(tok->byte_to_token); free(tok); return NULL; }

	tok->merges[0] = (struct bpe_merge){ 'H', 'e', 256, 0 };
	tok->merges[1] = (struct bpe_merge){ 'l', 'l', 257, 1 };
	tok->merges[2] = (struct bpe_merge){ 'l', 'o', 258, 2 };

	return tok;
}

static void destroy_tokenizer(struct userspace_tokenizer *tok)
{
	if (!tok) return;
	free(tok->byte_to_token);
	free(tok->merges);
	free(tok);
}

static uint32_t userspace_tokenize(struct userspace_tokenizer *tok,
				   const char *text, uint32_t len,
				   uint32_t *out_tokens, uint32_t max_tokens)
{
	/* Phase 1: byte-level tokenization */
	uint32_t *buf = calloc(len, sizeof(uint32_t));
	if (!buf) return 0;

	uint32_t tok_len = 0;
	for (uint32_t i = 0; i < len && tok_len < max_tokens; i++)
		buf[tok_len++] = tok->byte_to_token[(unsigned char)text[i]];

	/* Phase 2: iterative BPE merges */
	for (uint32_t m = 0; m < tok->num_merges; m++) {
		struct bpe_merge *merge = &tok->merges[m];
		uint32_t write_pos = 0;

		for (uint32_t i = 0; i < tok_len; i++) {
			if (i + 1 < tok_len &&
			    buf[i] == merge->left && buf[i + 1] == merge->right) {
				buf[write_pos++] = merge->result;
				i++;  /* skip merged pair */
			} else {
				buf[write_pos++] = buf[i];
			}
		}
		tok_len = write_pos;
	}

	uint32_t copy_len = tok_len < max_tokens ? tok_len : max_tokens;
	memcpy(out_tokens, buf, copy_len * sizeof(uint32_t));
	free(buf);
	return copy_len;
}

/* Latency percentile computation */
static int cmp_u64(const void *a, const void *b)
{
	uint64_t va = *(const uint64_t *)a;
	uint64_t vb = *(const uint64_t *)b;
	return (va > vb) - (va < vb);
}

struct bench_result {
	uint64_t total_ns;
	uint64_t total_tokens;
	uint64_t latencies[MAX_LATENCIES];
	uint32_t num_samples;
};

static void print_result(const char *label, struct bench_result *r)
{
	if (r->num_samples == 0) {
		printf("  %s: no samples\n", label);
		return;
	}

	qsort(r->latencies, r->num_samples, sizeof(uint64_t), cmp_u64);

	double throughput = (double)r->total_tokens * 1e9 / (double)r->total_ns;
	uint64_t p50 = r->latencies[r->num_samples / 2];
	uint64_t p99 = r->latencies[(uint32_t)(r->num_samples * 0.99)];
	uint64_t p999 = r->latencies[(uint32_t)(r->num_samples * 0.999)];

	printf("  %s:\n", label);
	printf("    throughput:  %.0f tokens/sec\n", throughput);
	printf("    p50 latency: %lu us\n", (unsigned long)(p50 / 1000));
	printf("    p99 latency: %lu us\n", (unsigned long)(p99 / 1000));
	printf("    p99.9:       %lu us\n", (unsigned long)(p999 / 1000));
	printf("    total:       %lu tokens in %.1f ms\n",
	       (unsigned long)r->total_tokens,
	       (double)r->total_ns / 1e6);
}

/* Benchmark: userspace tokenizer */
static int bench_userspace(const char **prompts, int num_prompts,
			   int iterations, struct bench_result *result)
{
	struct userspace_tokenizer *tok = create_synthetic_tokenizer();
	if (!tok) return -1;

	uint32_t tokens[MAX_PROMPT_LEN];
	memset(result, 0, sizeof(*result));

	uint64_t start = now_ns();
	for (int iter = 0; iter < iterations; iter++) {
		for (int p = 0; p < num_prompts; p++) {
			uint64_t t0 = now_ns();
			uint32_t n = userspace_tokenize(tok, prompts[p],
							(uint32_t)strlen(prompts[p]),
							tokens, MAX_PROMPT_LEN);
			uint64_t t1 = now_ns();

			result->total_tokens += n;
			if (result->num_samples < MAX_LATENCIES)
				result->latencies[result->num_samples++] = t1 - t0;
		}
	}
	result->total_ns = now_ns() - start;

	destroy_tokenizer(tok);
	return 0;
}

/* Benchmark: eBPF kernel tokenizer */
static int bench_kernel(const char **prompts, int num_prompts,
			int iterations, struct bench_result *result)
{
	int fd = open(CHARDEV_PATH, O_RDWR);
	if (fd < 0) {
		perror("open " CHARDEV_PATH);
		return -1;
	}

	void *map = mmap(NULL, RING_SIZE, PROT_READ | PROT_WRITE,
			 MAP_SHARED, fd, 0);
	if (map == MAP_FAILED) {
		perror("mmap");
		close(fd);
		return -1;
	}

	struct kllm_ring_hdr *ring = (struct kllm_ring_hdr *)map;
	if (ring->magic != KLLM_TOKEN_RING_MAGIC) {
		fprintf(stderr, "ring magic mismatch\n");
		munmap(map, RING_SIZE);
		close(fd);
		return -1;
	}

	uint32_t tokens[MAX_PROMPT_LEN];
	memset(result, 0, sizeof(*result));

	uint64_t start = now_ns();
	for (int iter = 0; iter < iterations; iter++) {
		for (int p = 0; p < num_prompts; p++) {
			uint64_t t0 = now_ns();

			ssize_t w = write(fd, prompts[p], strlen(prompts[p]));
			if (w < 0) break;

			/* Consume tokens produced by eBPF */
			uint32_t n = kllm_ring_consume_batch(ring, tokens,
							     MAX_PROMPT_LEN);
			uint64_t t1 = now_ns();

			result->total_tokens += n;
			if (result->num_samples < MAX_LATENCIES)
				result->latencies[result->num_samples++] = t1 - t0;
		}
	}
	result->total_ns = now_ns() - start;

	munmap(map, RING_SIZE);
	close(fd);
	return 0;
}

int main(int argc, char **argv)
{
	int iterations = DEFAULT_ITERATIONS;
	int has_chardev = 0;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
			iterations = atoi(argv[++i]);
	}

	/* Test prompts (varying lengths) */
	const char *prompts[] = {
		"Hello world",
		"The quick brown fox jumps over the lazy dog",
		"In a hole in the ground there lived a hobbit",
		"To be or not to be, that is the question",
		"Four score and seven years ago our fathers brought forth on this continent a new nation",
	};
	int num_prompts = sizeof(prompts) / sizeof(prompts[0]);

	printf("=== xpu-kllm tokenizer benchmark ===\n");
	printf("prompts: %d, iterations: %d\n\n", num_prompts, iterations);

	/* Always run userspace benchmark */
	struct bench_result userspace_result;
	printf("benchmarking userspace BPE...\n");
	if (bench_userspace(prompts, num_prompts, iterations, &userspace_result) == 0)
		print_result("userspace BPE", &userspace_result);

	/* Try kernel benchmark (only if chardev exists) */
	if (access(CHARDEV_PATH, W_OK) == 0)
		has_chardev = 1;

	if (has_chardev) {
		struct bench_result kernel_result;
		printf("\nbenchmarking eBPF kernel BPE...\n");
		if (bench_kernel(prompts, num_prompts, iterations, &kernel_result) == 0) {
			print_result("eBPF kernel BPE", &kernel_result);

			/* Speedup */
			if (userspace_result.total_ns > 0 && kernel_result.total_ns > 0) {
				double speedup = (double)userspace_result.total_ns /
						 (double)kernel_result.total_ns;
				printf("\n  speedup: %.2fx %s\n", speedup > 1.0 ? speedup : 1.0/speedup,
				       speedup > 1.0 ? "(kernel faster)" : "(userspace faster)");
			}
		}
	} else {
		printf("\n  [skip kernel benchmark: %s not available]\n", CHARDEV_PATH);
		printf("  Load kllm.ko to enable kernel path benchmarking.\n");
	}

	return 0;
}
