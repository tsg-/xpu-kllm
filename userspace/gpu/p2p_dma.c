/* SPDX-License-Identifier: Apache-2.0 */
/*
 * P2P DMA implementation: GPU VRAM ↔ RNIC for zero-copy KV fetch.
 *
 * Uses DRM (for dGPU dma-buf export) or KFD (for ROCm/AMD GPUs).
 * Placeholder implementation — actual GPU interaction requires:
 * - ROCm: HSA/KFD APIs for VRAM allocation + dma-buf export
 * - NVIDIA: CUDA driver API + GDS for dma-buf
 *
 * The UMA path is trivial: system malloc + ibv_reg_mr.
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "p2p_dma.h"

struct kllm_p2p_ctx {
	uint32_t gpu_id;
	int drm_fd;          /* /dev/dri/renderD128+gpu_id */
	int is_uma;
};

struct kllm_p2p_ctx *kllm_p2p_create(uint32_t gpu_id)
{
	struct kllm_p2p_ctx *ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;

	ctx->gpu_id = gpu_id;

	/* Try to open DRM render node for this GPU */
	char path[64];
	snprintf(path, sizeof(path), "/dev/dri/renderD%u", 128 + gpu_id);
	ctx->drm_fd = open(path, O_RDWR);
	if (ctx->drm_fd < 0) {
		/* No dGPU: assume UMA */
		ctx->is_uma = 1;
		ctx->drm_fd = -1;
	}

	return ctx;
}

void kllm_p2p_destroy(struct kllm_p2p_ctx *ctx)
{
	if (!ctx)
		return;
	if (ctx->drm_fd >= 0)
		close(ctx->drm_fd);
	free(ctx);
}

int kllm_p2p_alloc_region(struct kllm_p2p_ctx *ctx,
			  uint64_t size_bytes,
			  struct kllm_p2p_region *out)
{
	memset(out, 0, sizeof(*out));
	out->size_bytes = size_bytes;
	out->gpu_id = ctx->gpu_id;

	if (ctx->is_uma) {
		/*
		 * UMA path: allocate hugepage-backed system memory.
		 * Both GPU and RNIC can access this directly.
		 */
		void *ptr = mmap(NULL, size_bytes, PROT_READ | PROT_WRITE,
				 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
				 -1, 0);
		if (ptr == MAP_FAILED) {
			/* Fallback without hugepages */
			ptr = mmap(NULL, size_bytes, PROT_READ | PROT_WRITE,
				   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
			if (ptr == MAP_FAILED)
				return -1;
		}
		out->gpu_va = (uint64_t)ptr;
		out->bus_addr = 0;  /* will be resolved by RDMA MR registration */
		out->dma_buf_fd = -1;
		out->flags = KLLM_P2P_FLAG_UMA | KLLM_P2P_FLAG_PINNED;
		return 0;
	}

	/*
	 * dGPU path: allocate VRAM via DRM/KFD and export as dma-buf.
	 *
	 * ROCm (AMD):
	 *   hsaKmtAllocMemory(gpu_node, size, HSA_MEM_FLAGS_VRAM, &ptr)
	 *   hsaKmtShareMemory(ptr, size, &dma_buf_fd)
	 *
	 * DRM generic:
	 *   struct drm_mode_create_dumb → DRM_IOCTL_PRIME_HANDLE_TO_FD
	 *
	 * Placeholder: return error until GPU SDK is integrated.
	 */

	return -1;  /* TODO: implement with ROCm KFD or DRM */
}

void kllm_p2p_free_region(struct kllm_p2p_ctx *ctx,
			  struct kllm_p2p_region *region)
{
	if (region->flags & KLLM_P2P_FLAG_UMA) {
		if (region->gpu_va)
			munmap((void *)region->gpu_va, region->size_bytes);
	} else {
		if (region->dma_buf_fd >= 0)
			close(region->dma_buf_fd);
		/* TODO: free VRAM allocation */
	}
	memset(region, 0, sizeof(*region));
}

int kllm_p2p_register_spdk(struct kllm_p2p_ctx *ctx,
			   struct kllm_p2p_region *region)
{
	/*
	 * Register the P2P region with SPDK's memory subsystem so that
	 * NVMe completions can target this address directly.
	 *
	 * For UMA: spdk_mem_register(gpu_va, size)
	 * For dGPU: spdk_mem_register_dma_buf(dma_buf_fd, size)
	 *
	 * After registration, the physical/bus address is known to SPDK
	 * and can be used in NVMe PRP/SGL entries.
	 */
	(void)ctx;
	(void)region;
	return 0;  /* TODO */
}

int kllm_p2p_register_rdma(struct kllm_p2p_ctx *ctx,
			   struct kllm_p2p_region *region,
			   uint32_t *rkey_out)
{
	/*
	 * Register region with RNIC for RDMA-WRITE access.
	 *
	 * For UMA: ibv_reg_mr(pd, gpu_va, size, IBV_ACCESS_REMOTE_WRITE)
	 * For dGPU: ibv_reg_dmabuf_mr(pd, dma_buf_fd, size, ...)
	 *
	 * The rkey is shared with the storage node so it can RDMA-WRITE
	 * KV blocks directly into this region.
	 */
	(void)ctx;
	(void)region;
	if (rkey_out)
		*rkey_out = 0;
	return 0;  /* TODO */
}

int kllm_p2p_check_topology(uint32_t gpu_id, const char *nic_pci_addr)
{
	/*
	 * Check PCIe topology: is the GPU on the same root complex / switch
	 * as the RNIC? If not, P2P goes through the CPU root complex
	 * (still works but lower bandwidth).
	 *
	 * Implementation: read /sys/bus/pci/devices/.../numa_node and
	 * compare. Or use nvidia-smi topo / rocm-smi --showtopo.
	 */
	(void)gpu_id;
	(void)nic_pci_addr;
	return 1;  /* assume supported for now */
}
