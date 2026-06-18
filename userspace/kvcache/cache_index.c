/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Content-addressed KV cache index.
 *
 * Open-addressing hash table mapping SHA-256 → arena slot.
 * LRU eviction on insert when table is at capacity.
 */

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>

#include "cache_index.h"

#define EMPTY_SLOT	UINT32_MAX
#define LOAD_FACTOR_MAX	0.85  /* evict when occupancy exceeds this */

struct kllm_cache_index {
	struct kllm_cache_entry *entries;
	uint32_t capacity;
	uint32_t mask;
	uint32_t occupied;
	struct kllm_arena *arena;
	struct kllm_index_stats stats;
};

static inline uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline uint32_t hash_to_bucket(const uint8_t hash[KLLM_INDEX_HASH_BYTES],
				      uint32_t mask)
{
	/* Use first 4 bytes of SHA-256 as bucket index */
	uint32_t h;
	memcpy(&h, hash, sizeof(h));
	return h & mask;
}

static inline bool hash_eq(const uint8_t a[KLLM_INDEX_HASH_BYTES],
			   const uint8_t b[KLLM_INDEX_HASH_BYTES])
{
	return memcmp(a, b, KLLM_INDEX_HASH_BYTES) == 0;
}

struct kllm_cache_index *kllm_index_create(const struct kllm_index_config *cfg)
{
	struct kllm_cache_index *idx;

	if ((cfg->capacity & (cfg->capacity - 1)) != 0)
		return NULL;  /* must be power of 2 */

	idx = calloc(1, sizeof(*idx));
	if (!idx)
		return NULL;

	idx->entries = calloc(cfg->capacity, sizeof(struct kllm_cache_entry));
	if (!idx->entries) {
		free(idx);
		return NULL;
	}

	idx->capacity = cfg->capacity;
	idx->mask = cfg->capacity - 1;
	idx->arena = cfg->arena;

	/* Mark all entries empty */
	for (uint32_t i = 0; i < cfg->capacity; i++)
		idx->entries[i].slot_index = EMPTY_SLOT;

	idx->stats.capacity = cfg->capacity;
	return idx;
}

void kllm_index_destroy(struct kllm_cache_index *idx)
{
	if (!idx)
		return;
	free(idx->entries);
	free(idx);
}

void *kllm_index_lookup(struct kllm_cache_index *idx,
			const uint8_t hash[KLLM_INDEX_HASH_BYTES])
{
	uint32_t bucket = hash_to_bucket(hash, idx->mask);

	for (uint32_t i = 0; i < idx->capacity; i++) {
		uint32_t pos = (bucket + i) & idx->mask;
		struct kllm_cache_entry *e = &idx->entries[pos];

		if (e->slot_index == EMPTY_SLOT) {
			idx->stats.misses++;
			return NULL;
		}
		if (hash_eq(e->hash, hash)) {
			e->last_access_ns = now_ns();
			idx->stats.hits++;
			return kllm_arena_slot_ptr(idx->arena, e->slot_index);
		}
	}

	idx->stats.misses++;
	return NULL;
}

/* Find the LRU entry (oldest last_access_ns) */
static uint32_t find_lru_entry(struct kllm_cache_index *idx)
{
	uint64_t oldest = UINT64_MAX;
	uint32_t victim = 0;

	for (uint32_t i = 0; i < idx->capacity; i++) {
		if (idx->entries[i].slot_index == EMPTY_SLOT)
			continue;
		if (idx->entries[i].last_access_ns < oldest) {
			oldest = idx->entries[i].last_access_ns;
			victim = i;
		}
	}
	return victim;
}

void *kllm_index_insert(struct kllm_cache_index *idx,
			const uint8_t hash[KLLM_INDEX_HASH_BYTES])
{
	/* Check if we need to evict */
	if ((double)idx->occupied / idx->capacity >= LOAD_FACTOR_MAX) {
		uint32_t victim = find_lru_entry(idx);
		struct kllm_cache_entry *ve = &idx->entries[victim];
		kllm_arena_free(idx->arena, kllm_arena_slot_ptr(idx->arena, ve->slot_index));
		ve->slot_index = EMPTY_SLOT;
		idx->occupied--;
		idx->stats.evictions++;
	}

	/* Allocate arena block */
	void *block = kllm_arena_alloc(idx->arena);
	if (!block)
		return NULL;

	uint32_t slot = kllm_arena_slot_index(idx->arena, block);

	/* Find insertion position */
	uint32_t bucket = hash_to_bucket(hash, idx->mask);
	for (uint32_t i = 0; i < idx->capacity; i++) {
		uint32_t pos = (bucket + i) & idx->mask;
		struct kllm_cache_entry *e = &idx->entries[pos];

		if (e->slot_index == EMPTY_SLOT) {
			memcpy(e->hash, hash, KLLM_INDEX_HASH_BYTES);
			e->slot_index = slot;
			e->last_access_ns = now_ns();
			e->flags = 0;
			idx->occupied++;
			idx->stats.insertions++;
			return block;
		}
	}

	/* Shouldn't reach here after eviction, but safety */
	kllm_arena_free(idx->arena, block);
	return NULL;
}

bool kllm_index_evict(struct kllm_cache_index *idx,
		      const uint8_t hash[KLLM_INDEX_HASH_BYTES])
{
	uint32_t bucket = hash_to_bucket(hash, idx->mask);

	for (uint32_t i = 0; i < idx->capacity; i++) {
		uint32_t pos = (bucket + i) & idx->mask;
		struct kllm_cache_entry *e = &idx->entries[pos];

		if (e->slot_index == EMPTY_SLOT)
			return false;
		if (hash_eq(e->hash, hash)) {
			kllm_arena_free(idx->arena,
					kllm_arena_slot_ptr(idx->arena, e->slot_index));
			e->slot_index = EMPTY_SLOT;
			idx->occupied--;
			idx->stats.evictions++;
			return true;
		}
	}
	return false;
}

void kllm_index_get_stats(struct kllm_cache_index *idx,
			  struct kllm_index_stats *out)
{
	*out = idx->stats;
	out->occupied = idx->occupied;
}
