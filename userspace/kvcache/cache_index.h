/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_CACHE_INDEX_H
#define _KLLM_CACHE_INDEX_H

#include <stdint.h>
#include <stdbool.h>
#include "arena_alloc.h"

/*
 * Content-addressed KV cache index.
 *
 * Maps SHA-256(token prefix) → arena slot index.
 * Lock-free reads via atomic slot entries. LRU eviction when full.
 *
 * Implementation: open-addressing hash table with linear probing.
 * Power-of-2 size for fast modulo. Each entry stores the full hash
 * for comparison (avoids false positives from truncated hashes).
 */

#define KLLM_INDEX_HASH_BYTES	32  /* SHA-256 */

struct kllm_cache_entry {
	uint8_t  hash[KLLM_INDEX_HASH_BYTES];
	uint32_t slot_index;     /* arena slot, or UINT32_MAX if empty */
	uint32_t flags;
	uint64_t last_access_ns; /* for LRU */
};

struct kllm_cache_index;

struct kllm_index_config {
	uint32_t capacity;       /* must be power of 2 */
	struct kllm_arena *arena;
};

/* Lifecycle */
struct kllm_cache_index *kllm_index_create(const struct kllm_index_config *cfg);
void kllm_index_destroy(struct kllm_cache_index *idx);

/*
 * Lookup: returns arena slot pointer if found, NULL on miss.
 * Updates LRU timestamp on hit.
 */
void *kllm_index_lookup(struct kllm_cache_index *idx,
			const uint8_t hash[KLLM_INDEX_HASH_BYTES]);

/*
 * Insert: associates hash with a block. If index is full, evicts LRU entry.
 * Returns the arena slot pointer for the new block.
 */
void *kllm_index_insert(struct kllm_cache_index *idx,
			const uint8_t hash[KLLM_INDEX_HASH_BYTES]);

/* Evict: remove entry and free arena slot. Returns true if found. */
bool kllm_index_evict(struct kllm_cache_index *idx,
		      const uint8_t hash[KLLM_INDEX_HASH_BYTES]);

/* Stats */
struct kllm_index_stats {
	uint64_t hits;
	uint64_t misses;
	uint64_t evictions;
	uint64_t insertions;
	uint32_t occupied;
	uint32_t capacity;
};

void kllm_index_get_stats(struct kllm_cache_index *idx,
			  struct kllm_index_stats *out);

#endif /* _KLLM_CACHE_INDEX_H */
