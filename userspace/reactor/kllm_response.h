/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_RESPONSE_H
#define _KLLM_RESPONSE_H

#include <stdint.h>

/*
 * Response path: write generated token IDs back to the chardev
 * so userspace clients can read() them.
 *
 * Two options:
 * 1. ioctl/write to chardev fd (kernel copies to response ring)
 * 2. Shared mmap'd response ring (zero-copy, SPDK writes directly)
 *
 * We use option 1 initially (simpler), with option 2 for production.
 */

struct kllm_response_ctx;

struct kllm_response_ctx *kllm_response_create(int chardev_fd);
void kllm_response_destroy(struct kllm_response_ctx *ctx);

/* Write a batch of generated tokens back to the chardev response path */
int kllm_response_emit(struct kllm_response_ctx *ctx,
		       const uint32_t *tokens, uint32_t count);

/* Write a single token (streaming decode) */
int kllm_response_emit_one(struct kllm_response_ctx *ctx, uint32_t token);

/* Signal end-of-sequence to the chardev (client sees EOF on read) */
int kllm_response_eos(struct kllm_response_ctx *ctx);

#endif /* _KLLM_RESPONSE_H */
