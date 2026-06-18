// SPDX-License-Identifier: GPL-2.0
/*
 * eBPF BPE tokenizer for xpu-kllm
 *
 * Implements byte-pair encoding (BPE) with:
 * - Fixed vocabulary stored in BPF_MAP_TYPE_HASH
 * - Iterative merge passes via bpf_loop() (bounded to 128 iterations)
 * - Token IDs written directly to the hugepage ring buffer
 *
 * This program is invoked by the kllm chardev on write() via struct_ops.
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define MAX_PROMPT_LEN		4096
#define MAX_MERGE_PASSES	128
#define MAX_SEQ_LEN		2048
#define VOCAB_SIZE		65536

/* Merge table entry: pair of token IDs → merged token ID */
struct merge_key {
	__u32 left;
	__u32 right;
};

struct merge_val {
	__u32 merged_id;
	__u32 priority;  /* lower = merge first (rank in BPE) */
};

/* Token ring header — must match struct kllm_ring_hdr in kllm_ring.h */
struct ring_hdr {
	__u32 magic;
	__u32 version;
	__u32 slot_count;
	__u32 _pad0;
	__u32 head __attribute__((aligned(64)));
	__u32 _pad1[15];
	__u32 tail __attribute__((aligned(64)));
	__u32 _pad2[15];
	__u32 seq_id __attribute__((aligned(64)));
	__u32 _pad3[15];
};

/*
 * BPE merge table: (left_token, right_token) → merged_token + priority.
 * Loaded from userspace at program attach time.
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, VOCAB_SIZE);
	__type(key, struct merge_key);
	__type(value, struct merge_val);
} merge_table SEC(".maps");

/*
 * Byte-to-token initial mapping: single byte → token ID.
 * For tiktoken/GPT-style BPE, the first 256 entries are byte tokens.
 */
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 256);
	__type(key, __u32);
	__type(value, __u32);
} byte_to_token SEC(".maps");

/*
 * Per-CPU scratch space for the token sequence being merged.
 * Avoids stack overflow (BPF stack limit: 512 bytes).
 */
struct token_seq {
	__u32 tokens[MAX_SEQ_LEN];
	__u32 len;
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct token_seq);
} scratch SEC(".maps");

/* State passed to bpf_loop callback */
struct merge_state {
	struct token_seq *seq;
	int merged_any;
};

/*
 * Single merge pass: scan the sequence for the highest-priority merge.
 * Called iteratively via bpf_loop().
 */
static int do_merge_pass(u32 index, void *ctx)
{
	struct merge_state *state = ctx;
	struct token_seq *seq = state->seq;
	struct merge_key key;
	struct merge_val *val;
	__u32 best_pos = 0;
	__u32 best_priority = ~0U;
	__u32 best_merged = 0;
	int found = 0;

	if (seq->len < 2)
		return 1;  /* stop: nothing to merge */

	/* Find highest-priority (lowest rank) mergeable pair */
	for (__u32 i = 0; i < seq->len - 1 && i < MAX_SEQ_LEN - 1; i++) {
		key.left = seq->tokens[i];
		key.right = seq->tokens[i + 1];

		val = bpf_map_lookup_elem(&merge_table, &key);
		if (val && val->priority < best_priority) {
			best_priority = val->priority;
			best_merged = val->merged_id;
			best_pos = i;
			found = 1;
		}
	}

	if (!found)
		return 1;  /* stop: no more merges possible */

	/* Apply the merge: replace pair at best_pos with merged token */
	seq->tokens[best_pos] = best_merged;

	/* Shift remaining tokens left by 1 */
	for (__u32 i = best_pos + 1; i < seq->len - 1 && i < MAX_SEQ_LEN - 1; i++) {
		seq->tokens[i] = seq->tokens[i + 1];
	}
	seq->len--;

	state->merged_any = 1;
	return 0;  /* continue: try another pass */
}

/*
 * Tokenize a prompt buffer into token IDs.
 *
 * This is the struct_ops entry point invoked by the kllm chardev on write().
 * Arguments:
 *   buf      - pointer to prompt bytes (in kernel memory)
 *   len      - length of prompt
 *   ring_hdr - pointer to the hugepage ring buffer header
 *
 * Returns: number of tokens produced, or negative error.
 */
SEC("struct_ops/kllm_tokenize")
int BPF_PROG(kllm_tokenize, const char *buf, __u32 len, struct ring_hdr *ring_hdr)
{
	__u32 zero = 0;
	struct token_seq *seq;
	struct merge_state state;
	__u32 ring_mask, head, slot_count;
	__u32 *slots;

	if (len == 0 || len > MAX_PROMPT_LEN)
		return -1;

	seq = bpf_map_lookup_elem(&scratch, &zero);
	if (!seq)
		return -1;

	/* Phase 1: convert bytes to initial token IDs */
	seq->len = 0;
	for (__u32 i = 0; i < len && i < MAX_PROMPT_LEN; i++) {
		__u32 byte_idx = (unsigned char)buf[i];
		__u32 *tok = bpf_map_lookup_elem(&byte_to_token, &byte_idx);
		if (!tok)
			return -1;
		if (seq->len >= MAX_SEQ_LEN)
			break;
		seq->tokens[seq->len] = *tok;
		seq->len++;
	}

	/* Phase 2: iterative BPE merges */
	state.seq = seq;
	state.merged_any = 0;
	bpf_loop(MAX_MERGE_PASSES, do_merge_pass, &state, 0);

	/* Phase 3: write token IDs to hugepage ring */
	if (!ring_hdr)
		return seq->len;

	slot_count = ring_hdr->slot_count;
	if (slot_count == 0)
		return -1;
	ring_mask = slot_count - 1;
	head = ring_hdr->head;
	slots = (__u32 *)((char *)ring_hdr + sizeof(struct ring_hdr));

	for (__u32 i = 0; i < seq->len && i < MAX_SEQ_LEN; i++) {
		__u32 next = (head + 1) & ring_mask;
		if (next == ring_hdr->tail)
			break;  /* ring full */
		slots[head] = seq->tokens[i];
		head = next;
	}

	/* Memory fence + update head atomically */
	__sync_synchronize();
	ring_hdr->head = head;

	/* Bump sequence ID so consumer knows a new prompt landed */
	__sync_fetch_and_add(&ring_hdr->seq_id, 1);

	return seq->len;
}

char _license[] SEC("license") = "GPL";
