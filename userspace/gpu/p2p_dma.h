/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_P2P_DMA_H
#define _KLLM_P2P_DMA_H

#include <stdint.h>

/*
 * P2P DMA: direct data path into GPU VRAM without host bounce.
 *
 * Primary path: local NVMe → io_uring → dma-buf → PCIe P2P → GPU VRAM
 * Alternative:  RNIC RDMA-WRITE into dma-buf exported VRAM (remote NVMe-oF)
 *
 * Implementation (io_uring GDS):
 * 1. Export GPU VRAM region as dma-buf (via DRM or KFD)
 * 2. Register dma-buf fd with io_uring as fixed buffer
 * 3. io_uring NVMe read SQEs target the dma-buf directly
 * 4. NVMe controller P2P DMAs into VRAM via PCIe
 *
 * Implementation (RNIC RDMA, alternative):
 * 1. Export GPU VRAM region as dma-buf
 * 2. Register dma-buf with RNIC for MR (ibv_reg_dmabuf_mr)
 * 3. Remote NVMe-oF target RDMA-WRITEs into VRAM
 *
 * For UMA (integrated GPU / shared memory): same API, but the
 * dma-buf points to system DRAM. No code change needed.
 */

struct kllm_p2p_region {
	uint64_t gpu_va;        /* GPU virtual address */
	uint64_t bus_addr;      /* PCIe bus address for P2P */
	uint64_t size_bytes;
	int      dma_buf_fd;    /* exported dma-buf file descriptor */
	uint32_t gpu_id;        /* which GPU owns this region */
	uint32_t flags;
};

#define KLLM_P2P_FLAG_UMA	(1 << 0)  /* unified memory, no PCIe P2P needed */
#define KLLM_P2P_FLAG_PINNED	(1 << 1)  /* region is pinned (won't be evicted) */

struct kllm_p2p_ctx;

/*
 * Create P2P context for a GPU.
 * gpu_id: device index (e.g., 0 for /dev/dri/renderD128)
 */
struct kllm_p2p_ctx *kllm_p2p_create(uint32_t gpu_id);
void kllm_p2p_destroy(struct kllm_p2p_ctx *ctx);

/*
 * Allocate and export a VRAM region for P2P DMA.
 * The returned dma_buf_fd can be passed to the RNIC for MR registration.
 */
int kllm_p2p_alloc_region(struct kllm_p2p_ctx *ctx,
			  uint64_t size_bytes,
			  struct kllm_p2p_region *out);

/* Free a previously allocated P2P region */
void kllm_p2p_free_region(struct kllm_p2p_ctx *ctx,
			  struct kllm_p2p_region *region);

/*
 * Register a P2P region with io_uring as a fixed buffer.
 * After this call, NVMe read completions can land directly in VRAM.
 */
int kllm_p2p_register_spdk(struct kllm_p2p_ctx *ctx,
			   struct kllm_p2p_region *region);

/*
 * Register a P2P region's dma-buf fd with io_uring for fixed-buffer I/O.
 * Returns the buffer index for use in io_uring SQEs.
 */
int kllm_p2p_register_rdma(struct kllm_p2p_ctx *ctx,
			   struct kllm_p2p_region *region,
			   uint32_t *rkey_out);

/* Check if P2P between GPU and NIC is supported on this topology */
int kllm_p2p_check_topology(uint32_t gpu_id, const char *nic_pci_addr);

#endif /* _KLLM_P2P_DMA_H */
