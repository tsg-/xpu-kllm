/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Hugepage-backed slab allocator for KV cache blocks.
 *
 * No dynamic allocation after init. All blocks are fixed-size, carved
 * from a pre-mapped hugepage region. Free slots tracked via bitmap.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <numaif.h>
#include <errno.h>

#include "arena_alloc.h"

struct kllm_arena {
	uint32_t magic;
	void     *base;           /* mmap'd hugepage region */
	uint64_t region_bytes;    /* total mmap size */
	uint64_t slot_size;       /* bytes per KV block slot (aligned) */
	uint64_t total_slots;
	int      numa_node;

	/* Bitmap: 1 = free, 0 = allocated */
	uint64_t *bitmap;
	uint64_t bitmap_words;

	/* Free list hint: first word with a free bit */
	uint64_t hint;

	struct kllm_arena_stats stats;
	struct kllm_model_config model;
};

/* Round up to next multiple of alignment */
static inline uint64_t align_up(uint64_t val, uint64_t align)
{
	return (val + align - 1) & ~(align - 1);
}

struct kllm_arena *kllm_arena_create(const struct kllm_arena_config *cfg)
{
	struct kllm_arena *arena;
	uint64_t slot_size, total_slots, bitmap_bytes;
	void *base;
	int mmap_flags;
	unsigned long nodemask;

	arena = calloc(1, sizeof(*arena));
	if (!arena)
		return NULL;

	/* Slot size: block format size, rounded up to 4KB for page alignment */
	slot_size = kllm_block_total_bytes(&cfg->model);
	slot_size = align_up(slot_size, 4096);

	total_slots = cfg->total_bytes / slot_size;
	if (total_slots == 0) {
		free(arena);
		return NULL;
	}

	/* Map hugepage region */
	mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
	if (cfg->hugepage_size_mb == 1024)
		mmap_flags |= MAP_HUGETLB | (30 << MAP_HUGE_SHIFT);  /* 1GB */
	else if (cfg->hugepage_size_mb == 2)
		mmap_flags |= MAP_HUGETLB | (21 << MAP_HUGE_SHIFT);  /* 2MB */

	base = mmap(NULL, cfg->total_bytes, PROT_READ | PROT_WRITE,
		    mmap_flags, -1, 0);
	if (base == MAP_FAILED) {
		/* Fallback: try without HUGETLB for development */
		base = mmap(NULL, cfg->total_bytes, PROT_READ | PROT_WRITE,
			    MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
		if (base == MAP_FAILED) {
			free(arena);
			return NULL;
		}
	}

	/* Bind to NUMA node if requested */
	if (cfg->numa_node >= 0) {
		nodemask = 1UL << cfg->numa_node;
		mbind(base, cfg->total_bytes, MPOL_BIND, &nodemask,
		      cfg->numa_node + 2, MPOL_MF_MOVE);
	}

	/* Allocate bitmap (1 bit per slot) */
	bitmap_bytes = align_up((total_slots + 63) / 64 * 8, 64);
	arena->bitmap = calloc(1, bitmap_bytes);
	if (!arena->bitmap) {
		munmap(base, cfg->total_bytes);
		free(arena);
		return NULL;
	}

	/* Mark all slots as free (bit = 1) */
	arena->bitmap_words = (total_slots + 63) / 64;
	for (uint64_t i = 0; i < arena->bitmap_words; i++)
		arena->bitmap[i] = ~0ULL;

	/* Clear bits beyond total_slots */
	uint64_t remainder = total_slots % 64;
	if (remainder > 0)
		arena->bitmap[arena->bitmap_words - 1] = (1ULL << remainder) - 1;

	arena->magic = KLLM_ARENA_MAGIC;
	arena->base = base;
	arena->region_bytes = cfg->total_bytes;
	arena->slot_size = slot_size;
	arena->total_slots = total_slots;
	arena->numa_node = cfg->numa_node;
	arena->hint = 0;
	arena->model = cfg->model;

	arena->stats.total_slots = total_slots;
	arena->stats.used_slots = 0;

	return arena;
}

void kllm_arena_destroy(struct kllm_arena *arena)
{
	if (!arena)
		return;
	if (arena->base)
		munmap(arena->base, arena->region_bytes);
	free(arena->bitmap);
	free(arena);
}

void *kllm_arena_alloc(struct kllm_arena *arena)
{
	/* Scan bitmap from hint for first free bit */
	for (uint64_t w = arena->hint; w < arena->bitmap_words; w++) {
		if (arena->bitmap[w] == 0)
			continue;
		int bit = __builtin_ctzll(arena->bitmap[w]);
		uint64_t slot = w * 64 + bit;
		if (slot >= arena->total_slots)
			return NULL;

		/* Mark allocated */
		arena->bitmap[w] &= ~(1ULL << bit);
		arena->hint = w;
		arena->stats.used_slots++;
		arena->stats.alloc_count++;

		return (uint8_t *)arena->base + slot * arena->slot_size;
	}

	/* Wrap around from start */
	for (uint64_t w = 0; w < arena->hint; w++) {
		if (arena->bitmap[w] == 0)
			continue;
		int bit = __builtin_ctzll(arena->bitmap[w]);
		uint64_t slot = w * 64 + bit;
		if (slot >= arena->total_slots)
			return NULL;

		arena->bitmap[w] &= ~(1ULL << bit);
		arena->hint = w;
		arena->stats.used_slots++;
		arena->stats.alloc_count++;

		return (uint8_t *)arena->base + slot * arena->slot_size;
	}

	return NULL;  /* arena full */
}

void kllm_arena_free(struct kllm_arena *arena, void *block)
{
	uint64_t offset = (uint8_t *)block - (uint8_t *)arena->base;
	uint64_t slot = offset / arena->slot_size;
	uint64_t w = slot / 64;
	int bit = slot % 64;

	arena->bitmap[w] |= (1ULL << bit);
	arena->stats.used_slots--;
	arena->stats.free_count++;

	if (w < arena->hint)
		arena->hint = w;
}

uint32_t kllm_arena_slot_index(struct kllm_arena *arena, void *block)
{
	uint64_t offset = (uint8_t *)block - (uint8_t *)arena->base;
	return (uint32_t)(offset / arena->slot_size);
}

void *kllm_arena_slot_ptr(struct kllm_arena *arena, uint32_t index)
{
	if (index >= arena->total_slots)
		return NULL;
	return (uint8_t *)arena->base + (uint64_t)index * arena->slot_size;
}

void kllm_arena_get_stats(struct kllm_arena *arena, struct kllm_arena_stats *out)
{
	*out = arena->stats;
}

uint64_t kllm_arena_phys_addr(struct kllm_arena *arena, void *block)
{
	/*
	 * For hugepage-backed mappings, the physical address is stable
	 * and can be read from /proc/self/pagemap. This is needed for
	 * SPDK DMA registration and RDMA MR setup.
	 *
	 * In production, we'd use SPDK's spdk_mem_register() which handles
	 * the VA→PA translation internally. This is a placeholder.
	 */
	(void)arena;
	(void)block;
	return 0;  /* TODO: implement via pagemap or SPDK mem registration */
}
