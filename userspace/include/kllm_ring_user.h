/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_RING_USER_H
#define _KLLM_RING_USER_H

/*
 * Userspace mirror of kllm_ring.h — SPDK reactor uses this to consume
 * token IDs from the hugepage ring buffer produced by the kernel eBPF
 * tokenizer.
 *
 * The ring is mapped via mmap(/dev/llm_prompt1, ...) at offset 0.
 */

#include <stdint.h>
#include <stdatomic.h>

#define KLLM_TOKEN_RING_MAGIC	0x4B4C4C4D
#define KLLM_TOKEN_RING_VERSION	1

struct kllm_ring_hdr {
	uint32_t magic;
	uint32_t version;
	uint32_t slot_count;
	uint32_t _pad0;

	/* Producer (kernel) — cache-line aligned */
	_Alignas(64) uint32_t head;
	uint32_t _pad1[15];

	/* Consumer (SPDK) — cache-line aligned */
	_Alignas(64) uint32_t tail;
	uint32_t _pad2[15];

	/* Sequence fencing */
	_Alignas(64) uint32_t seq_id;
	uint32_t _pad3[15];
};

#define KLLM_RING_HDR_SIZE sizeof(struct kllm_ring_hdr)

static inline uint32_t *kllm_ring_slots(struct kllm_ring_hdr *hdr)
{
	return (uint32_t *)((uint8_t *)hdr + KLLM_RING_HDR_SIZE);
}

static inline uint32_t kllm_ring_mask(struct kllm_ring_hdr *hdr)
{
	return hdr->slot_count - 1;
}

/* Consumer (SPDK reactor): how many tokens are ready to read */
static inline uint32_t kllm_ring_available(struct kllm_ring_hdr *hdr)
{
	uint32_t head = atomic_load_explicit(
		(_Atomic uint32_t *)&hdr->head, memory_order_acquire);
	uint32_t tail = hdr->tail;
	return (head - tail) & kllm_ring_mask(hdr);
}

/* Consumer (SPDK reactor): read one token, advance tail */
static inline uint32_t kllm_ring_consume(struct kllm_ring_hdr *hdr)
{
	uint32_t tail = hdr->tail;
	uint32_t token_id = kllm_ring_slots(hdr)[tail];

	atomic_thread_fence(memory_order_acquire);
	atomic_store_explicit((_Atomic uint32_t *)&hdr->tail,
			      (tail + 1) & kllm_ring_mask(hdr),
			      memory_order_release);
	return token_id;
}

/* Consumer: batch read up to max_tokens, returns count consumed */
static inline uint32_t kllm_ring_consume_batch(struct kllm_ring_hdr *hdr,
					       uint32_t *out, uint32_t max_tokens)
{
	uint32_t avail = kllm_ring_available(hdr);
	uint32_t count = avail < max_tokens ? avail : max_tokens;
	uint32_t tail = hdr->tail;
	uint32_t mask = kllm_ring_mask(hdr);
	uint32_t *slots = kllm_ring_slots(hdr);

	for (uint32_t i = 0; i < count; i++) {
		out[i] = slots[(tail + i) & mask];
	}

	atomic_thread_fence(memory_order_acquire);
	atomic_store_explicit((_Atomic uint32_t *)&hdr->tail,
			      (tail + count) & mask,
			      memory_order_release);
	return count;
}

#endif /* _KLLM_RING_USER_H */
