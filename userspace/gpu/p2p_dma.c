/* SPDX-License-Identifier: Apache-2.0 */
/*
 * P2P DMA implementation: GPU VRAM data path via dma-buf.
 *
 * Primary: local NVMe → io_uring → dma-buf → PCIe P2P → GPU VRAM
 * Alternative: RNIC RDMA-WRITE into dma-buf exported VRAM (remote)
 *
 * Uses DRM (for dGPU dma-buf export) or KFD (for AMD GPUs).
 * Placeholder implementation — actual GPU interaction requires:
 * - Intel Xe: Level Zero / DRM for VRAM allocation + dma-buf export
 * - NVIDIA: CUDA driver API + GDS for dma-buf
 * - ROCm: HSA/KFD APIs for VRAM allocation + dma-buf export
 *
 * The UMA path is trivial: hugepage mmap, same virtual address space.
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
	 * Intel Xe (Level Zero):
	 *   zeMemAllocDevice(..., &ptr)
	 *   zeMemGetIpcHandle(...) or DRM_IOCTL_PRIME_HANDLE_TO_FD
	 *
	 * NVIDIA (CUDA):
	 *   cuMemAlloc(&ptr, size) + cuMemExportToShareableHandle(dma_buf_fd)
	 *
	 * ROCm (AMD):
	 *   hsaKmtAllocMemory(gpu_node, size, HSA_MEM_FLAGS_VRAM, &ptr)
	 *   hsaKmtShareMemory(ptr, size, &dma_buf_fd)
	 *
	 * Placeholder: return error until GPU SDK is integrated.
	 */

	return -1;  /* TODO: implement with Level Zero / CUDA / KFD */
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
	 * Register the P2P region for io_uring fixed-buffer I/O.
	 *
	 * For UMA: io_uring_register_buffers(ring, &iov, 1)
	 * For dGPU (io_uring + dma-buf, primary path):
	 *   struct io_uring_regbuf_desc desc = {
	 *       .type = IO_REGBUF_TYPE_DMABUF,
	 *       .dmabuf_fd = region->dma_buf_fd,
	 *       .target_fd = nvme_fd,  // lazy attach on first I/O
	 *   };
	 *   io_uring_register_buffers_ext(ring, &desc, 1)
	 *
	 * Kernel uses ITER_DMABUF_MAP to iterate scatter-gather list from
	 * the dma-buf attachment. dma-token (percpu_ref + RCU map) provides
	 * fencing for concurrent I/O. IORING_OP_READ_FIXED then references
	 * the registered buffer slot index.
	 *
	 * After registration, NVMe reads land directly in VRAM.
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
	 * Register dma-buf with RNIC for RDMA access (alternative path).
	 *
	 * For RNIC/RDMA path:
	 *   ibv_reg_dmabuf_mr(pd, region->dma_buf_fd, size, ...)
	 *   Returns rkey for remote RDMA-WRITE into VRAM.
	 *
	 * Primary path uses io_uring fixed buffers instead (see
	 * kllm_p2p_register_spdk). Both share the dma-buf export.
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
	 * as the NVMe device (or RNIC for RDMA path)? If not, P2P goes
	 * through the CPU root complex (still works but lower bandwidth).
	 *
	 * Implementation: read /sys/bus/pci/devices/.../numa_node and
	 * compare. Or use nvidia-smi topo / rocm-smi --showtopo.
	 */
	(void)gpu_id;
	(void)nic_pci_addr;
	return 1;  /* assume supported for now */
}
