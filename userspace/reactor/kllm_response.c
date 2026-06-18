/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Response path: push generated tokens back to the chardev for
 * userspace readers to consume via read()/poll().
 *
 * Uses an ioctl (KLLM_IOC_EMIT_TOKENS) to the chardev fd, which
 * copies tokens into the kernel-side response ring and wakes readers.
 */

#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <string.h>
#include <errno.h>

#include "kllm_response.h"

/* Must match kernel module ioctl definitions */
#define KLLM_IOC_MAGIC		'K'
#define KLLM_IOC_EMIT_TOKENS	_IOW(KLLM_IOC_MAGIC, 1, struct kllm_emit_cmd)
#define KLLM_IOC_EOS		_IO(KLLM_IOC_MAGIC, 2)

struct kllm_emit_cmd {
	uint32_t count;
	uint32_t tokens[256];  /* max batch size per ioctl call */
};

struct kllm_response_ctx {
	int fd;
};

struct kllm_response_ctx *kllm_response_create(int chardev_fd)
{
	struct kllm_response_ctx *ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;
	ctx->fd = chardev_fd;
	return ctx;
}

void kllm_response_destroy(struct kllm_response_ctx *ctx)
{
	free(ctx);
}

int kllm_response_emit(struct kllm_response_ctx *ctx,
		       const uint32_t *tokens, uint32_t count)
{
	struct kllm_emit_cmd cmd;
	uint32_t sent = 0;

	while (sent < count) {
		uint32_t batch = count - sent;
		if (batch > 256)
			batch = 256;

		cmd.count = batch;
		memcpy(cmd.tokens, tokens + sent, batch * sizeof(uint32_t));

		if (ioctl(ctx->fd, KLLM_IOC_EMIT_TOKENS, &cmd) < 0)
			return -errno;

		sent += batch;
	}

	return 0;
}

int kllm_response_emit_one(struct kllm_response_ctx *ctx, uint32_t token)
{
	return kllm_response_emit(ctx, &token, 1);
}

int kllm_response_eos(struct kllm_response_ctx *ctx)
{
	if (ioctl(ctx->fd, KLLM_IOC_EOS) < 0)
		return -errno;
	return 0;
}
