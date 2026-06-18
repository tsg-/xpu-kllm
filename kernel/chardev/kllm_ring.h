/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _KLLM_RING_H
#define _KLLM_RING_H

#include <linux/types.h>

/*
 * Hugepage-backed token ring buffer.
 *
 * Layout: single contiguous 2MB hugepage (or compound page allocation),
 * split into a control header and token slots.
 *
 * Producer: eBPF tokenizer (in kernel write() path)
 * Consumer: SPDK reactor (mmaps this region from userspace)
 *
 * Lock-free SPSC (single-producer single-consumer) via atomic head/tail.
 * Producer and consumer on separate cache lines to avoid false sharing.
 */

#define KLLM_TOKEN_RING_MAGIC	0x4B4C4C4D  /* "KLLM" */
#define KLLM_TOKEN_RING_VERSION	1

/* Control header at the start of the ring page(s) */
struct kllm_ring_hdr {
	u32	magic;
	u32	version;
	u32	slot_count;	/* number of u32 token slots */
	u32	_pad0;

	/* Producer (eBPF/kernel) — own cache line */
	u32	head __aligned(64);
	u32	_pad1[15];

	/* Consumer (SPDK/userspace) — own cache line */
	u32	tail __aligned(64);
	u32	_pad2[15];

	/* Sequence ID for fencing (incremented per prompt submission) */
	u32	seq_id __aligned(64);
	u32	_pad3[15];
} __packed;

#define KLLM_RING_HDR_SIZE	sizeof(struct kllm_ring_hdr)

/*
 * Token slots start immediately after the header.
 * With a 2MB hugepage: (2MB - 256B header) / 4B per token ≈ 524,224 slots.
 */
static inline u32 *kllm_ring_slots(struct kllm_ring_hdr *hdr)
{
	return (u32 *)((u8 *)hdr + KLLM_RING_HDR_SIZE);
}

static inline u32 kllm_ring_mask(struct kllm_ring_hdr *hdr)
{
	return hdr->slot_count - 1;
}

/* Producer helpers (kernel side) */
static inline u32 kllm_ring_free(struct kllm_ring_hdr *hdr)
{
	u32 head = READ_ONCE(hdr->head);
	u32 tail = READ_ONCE(hdr->tail);
	return (tail - head - 1) & kllm_ring_mask(hdr);
}

static inline int kllm_ring_produce(struct kllm_ring_hdr *hdr, u32 token_id)
{
	u32 head = hdr->head;
	u32 next = (head + 1) & kllm_ring_mask(hdr);

	if (next == READ_ONCE(hdr->tail))
		return -1;  /* full */

	kllm_ring_slots(hdr)[head] = token_id;
	smp_wmb();
	WRITE_ONCE(hdr->head, next);
	return 0;
}

/* Consumer helpers (userspace/SPDK side — mirrored in userspace header) */
static inline u32 kllm_ring_available(struct kllm_ring_hdr *hdr)
{
	u32 head = READ_ONCE(hdr->head);
	u32 tail = hdr->tail;
	return (head - tail) & kllm_ring_mask(hdr);
}

static inline u32 kllm_ring_consume(struct kllm_ring_hdr *hdr)
{
	u32 tail = hdr->tail;
	u32 token_id;

	token_id = kllm_ring_slots(hdr)[tail];
	smp_rmb();
	WRITE_ONCE(hdr->tail, (tail + 1) & kllm_ring_mask(hdr));
	return token_id;
}

#endif /* _KLLM_RING_H */
