/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_ARENA_ALLOC_H
#define _KLLM_ARENA_ALLOC_H

#include <stdint.h>
#include <stdbool.h>
#include "block_format.h"

/*
 * Hugepage-backed slab allocator for KV cache blocks.
 *
 * Design:
 * - Pre-allocates N gigabytes of 1GB hugepages at startup
 * - Divides into fixed-size slots (one KV block per slot)
 * - Free list is a simple bitmap — no malloc ever
 * - NUMA-aware: one arena per NUMA node
 * - All operations are O(1) amortized
 */

#define KLLM_ARENA_MAGIC	0x41524E41  /* "ARNA" */
#define KLLM_MAX_NUMA_NODES	8

struct kllm_arena_config {
	uint64_t total_bytes;       /* total hugepage memory to allocate */
	uint32_t hugepage_size_mb;  /* 2 or 1024 (2MB or 1GB pages) */
	int      numa_node;         /* -1 for any, or specific node */
	struct kllm_model_config model;
};

struct kllm_arena_stats {
	uint64_t total_slots;
	uint64_t used_slots;
	uint64_t alloc_count;
	uint64_t free_count;
	uint64_t eviction_count;
};

struct kllm_arena;

/* Lifecycle */
struct kllm_arena *kllm_arena_create(const struct kllm_arena_config *cfg);
void kllm_arena_destroy(struct kllm_arena *arena);

/* Allocation — returns pointer to block within hugepage region, or NULL */
void *kllm_arena_alloc(struct kllm_arena *arena);
void kllm_arena_free(struct kllm_arena *arena, void *block);

/* Slot index (for external indexing) */
uint32_t kllm_arena_slot_index(struct kllm_arena *arena, void *block);
void *kllm_arena_slot_ptr(struct kllm_arena *arena, uint32_t index);

/* Stats */
void kllm_arena_get_stats(struct kllm_arena *arena, struct kllm_arena_stats *out);

/* Physical address for DMA registration (stable after alloc) */
uint64_t kllm_arena_phys_addr(struct kllm_arena *arena, void *block);

#endif /* _KLLM_ARENA_ALLOC_H */
