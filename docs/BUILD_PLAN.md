# xpu-kllm Build Plan

## Context

Build a kernel-integrated LLM serving engine. The serving endpoint is a character device (`/dev/llm_prompt1`). Write a prompt, get a token stream back. No vLLM, no userspace serving framework. The kernel module tokenizes via eBPF, hands token IDs to SPDK via a hugepage ring buffer, SPDK decides CPU vs GPU attention based on sequence length and cache hit rate, GPU closes the loop.

Two solved problems: hugepages kill the allocation problem (pre-allocated arena, no runtime malloc), ACE kills the FP16 problem (BF16 outer products in SPDK reactor thread, userspace, no verifier constraints).

## Architecture

```
/dev/llm_prompt1 (chardev, misc_register)
  │
  ├─ write(prompt) → eBPF BPE tokenizer
  │    - fixed vocabulary in bpf_map
  │    - bounded merge steps via bpf_loop()
  │    - token IDs → hugepage-backed ring buffer (pre-allocated at module load)
  │    - zero dynamic allocation in hot path
  │
  ├─ SPDK reactor (polls hugepage ring)
  │    ├─ SHORT PATH (sequence ≤ threshold, KV cache hit):
  │    │    - paged attention via ACE outer-product intrinsics
  │    │    - AVX10 BF16/INT8, OCP MX block scaling inline
  │    │    - KV blocks in hugepage DRAM
  │    │    - no GPU round-trip
  │    │
  │    └─ LONG PATH (sequence > threshold or cache miss):
  │         - hand off to GPU
  │         - GPU hashes token sequence (same content-addressing scheme)
  │         - constructs wavefronts via rocm-xio NVMe-EP device
  │         - pulls KV blocks from RADOS-NKV (content-addressed by seq hash)
  │         - one doorbell ring per wavefront, zero GPU coordination
  │         - weights load same way via NVMe-EP wavefronts
  │
  └─ P2P DMA for KV Retrieve:
       - SPDK exports VRAM dma-buf
       - RNIC RDMA-WRITEs directly into GPU memory
       - no host bounce buffer
       - dGPU path: P2P DMA
       - UMA path: same code, shared DRAM

Response: token stream back through chardev read()/poll()
```

## Key Design Principles

1. **No dynamic allocation in hot path** — all memory from hugepage arenas pre-registered at startup
2. **No intermediate frameworks** — no io_uring bridge, no Wasm layer. Hugepage ring is the kernel↔SPDK interface.
3. **Content-addressed KV cache** — same hash scheme on CPU and GPU, zero coordination
4. **P2P everywhere** — RNIC→VRAM for KV, NVMe-EP→VRAM for weights, no host bounce
5. **CPU attention is the fast path** — ACE makes short-prefix reconstruction competitive, GPU is the overflow path

## Epics

### Epic 1: eBPF Tokenizer + Chardev (start here)

Prove BPE tokenization runs in eBPF, token IDs land in hugepage ring.

| Task | What | Standalone? |
|------|------|-------------|
| 1.1 | `/dev/llm_prompt1` chardev via `misc_register()` with write()/read()/poll() | Yes |
| 1.2 | Hugepage ring buffer: pre-allocated at module load, mmap'd to userspace | Yes |
| 1.3 | BPF struct_ops `llm_tokenizer_ops` invoked on chardev write | Yes |
| 1.4 | BPE merge table as BPF_MAP_TYPE_HASH, iterative merges via bpf_loop() | Yes |
| 1.5 | Token IDs written to hugepage ring (producer: eBPF, consumer: SPDK) | Needs 1.2+1.4 |
| 1.6 | Userspace test harness: validate against HuggingFace tokenizer | Yes |
| 1.7 | Benchmark: tokens/sec vs userspace BPE | Yes |

**Risk:** BPF verifier rejects merge logic. Mitigation: bpf_loop() 128-iteration bound, pre-sorted merge table. Fallback: tokenize in SPDK reactor, chardev passes raw bytes through ring.

### Epic 2: SPDK Reactor + Hugepage Ring Consumer

SPDK reactor polls the hugepage ring, dispatches to CPU or GPU path.

| Task | What | Standalone? |
|------|------|-------------|
| 2.1 | SPDK app skeleton: custom reactor thread, hugepage mempool init | Yes |
| 2.2 | Ring consumer: poll hugepage ring for token batches, zero-copy read | Needs 1.2 |
| 2.3 | Dispatch policy: sequence length + cache hit rate → CPU vs GPU decision | Yes (logic) |
| 2.4 | Response path: generated tokens written back to chardev response ring | Needs 1.1 |
| 2.5 | E2E latency test: write prompt → chardev → tokenize → ring → reactor callback | Full chain |

### Epic 3: ACE CPU Attention (short path)

Paged attention on SPDK reactor using ACE intrinsics against hugepage KV blocks.

| Task | What | Standalone? |
|------|------|-------------|
| 3.1 | KV block format: BF16 K/V per layer, sequence hash header, OCP MX scale factors | Yes |
| 3.2 | Hugepage KV arena: slab allocator, fixed-size blocks, NUMA-aware | Yes |
| 3.3 | Content-addressed index: SHA-256(token prefix) → arena slot, LRU eviction | Yes |
| 3.4 | ACE attention kernel: tiled GEMM via outer-product intrinsics (Q*K^T, attn*V) | Yes |
| 3.5 | OCP MX block scaling: inline dequant during matmul, per-8-element scale factors | Yes |
| 3.6 | Full short-path: token IDs → KV lookup → ACE attention → next-token logits | Needs 2.2+3.2+3.4 |

### Epic 4: GPU Path + P2P DMA + NVMe-EP Wavefronts

GPU attention for long sequences, KV cache via NVMe-EP from RADOS-NKV, P2P DMA.

| Task | What | Standalone? |
|------|------|-------------|
| 4.1 | rocm-xio NVMe-EP wavefront construction: batch KV block fetches by seq hash | Needs HW |
| 4.2 | RADOS-NKV content-addressed store: object key = SHA-256(token prefix) | Yes (mock) |
| 4.3 | dma-buf export from SPDK: expose VRAM region for RNIC RDMA-WRITE target | Needs GPU |
| 4.4 | P2P DMA path: RNIC writes KV blocks directly into VRAM, no host bounce | Needs 4.3 |
| 4.5 | GPU paged attention kernel (reference vLLM PagedAttention, adapted) | Needs GPU |
| 4.6 | Weight loading via NVMe-EP wavefronts into VRAM at startup | Needs 4.1 |
| 4.7 | UMA fallback: same code path, shared DRAM instead of P2P | Yes |

### Epic 5: Full Pipeline + Production

End-to-end orchestration, multi-request batching, decode loop, observability.

| Task | What | Standalone? |
|------|------|-------------|
| 5.1 | Autoregressive decode loop: sample next token, feed back, stream to chardev | Full chain |
| 5.2 | Multi-request batching: concurrent chardev writers, continuous batching | Full chain |
| 5.3 | CPU↔GPU handoff: mid-sequence escalation when cache miss rate exceeds threshold | Full chain |
| 5.4 | eBPF tracing: per-request latency breakdown (tokenize, KV lookup, attention, decode) | Yes |
| 5.5 | Chardev response interface: read()/poll() for streaming token output | Needs 1.1 |

## Dependency Graph

```
Epic 1 (chardev + tokenizer + ring)
    │
    ├──→ Epic 2 (SPDK reactor + dispatch)
    │        │
    │        ├──→ Epic 3 (ACE CPU attention)
    │        │        │
    │        └──→ Epic 4 (GPU + P2P + NVMe-EP)
    │                 │
    └─────────────────┴──→ Epic 5 (full pipeline)
```

Parallelizable: Epic 1 tasks, Epic 3 tasks 3.1-3.5, Epic 4 task 4.2 (RADOS mock).

## Vertical Slice (thinnest E2E proof)

```
echo "Hello world" > /dev/llm_prompt1
  → eBPF tokenizer (1000-entry merge table, hardcoded)
  → hugepage ring buffer
  → SPDK reactor reads token batch
  → KV cache hit (pre-warmed synthetic data in hugepage arena)
  → ACE BF16 single-layer attention (8 heads, dim 64)
  → next-token logits
  → token written to response ring
  → cat /dev/llm_prompt1 reads response
```

Scope: single transformer layer, pre-loaded KV, CPU-only (no GPU path), single request, hardcoded merge table, known-correct reference output.

**~3-4 weeks, 2 engineers (1 kernel, 1 userspace).**

## Prerequisites

| Need | For | Minimum | Fallback |
|------|-----|---------|----------|
| Linux kernel | chardev, eBPF, hugepage ring | 6.4+ (struct_ops, bpf_loop) | 6.1 (limited) |
| SPDK | reactor, hugepage mempool, NVMe | 24.01 | 23.09 |
| libbpf + clang | BPF compilation | 1.2+ / 15+ | — |
| Intel GNR+ (ACE) | BF16 outer-product intrinsics | Granite Rapids | AMX BF16 on SPR |
| AMD MI300+ | rocm-xio NVMe-EP | MI300X | Software NVMe-oF |
| RADOS / Ceph | KV cache persistence | Pacific 16.2+ | Local NVMe + mock |
| Mellanox CX-7+ | RDMA-WRITE to VRAM (P2P) | ConnectX-7 | Host-staged DMA |
| ROCm | GPU attention kernels | ROCm 6.0+ | CUDA 12+ path |

## Top Risks

1. **BPF verifier rejects tokenizer** — instruction count / loop bounds. Mitigation: bpf_loop(), early exit, fallback to reactor-side tokenization.
2. **P2P DMA RNIC→VRAM latency/bandwidth** — PCIe topology dependent, may not beat host-staged for small blocks. Mitigation: batch wavefronts, measure break-even block size.
3. **ACE memory-bandwidth wall** — CPU attention is DRAM-bound beyond L3. Mitigation: strict sequence-length threshold, arena sized to L3 for hot prefixes.
4. **NVMe-EP wavefront latency** — RADOS round-trip on cache miss. Mitigation: aggressive prefetch based on prefix prediction, size arena for working set.
5. **rocm-xio maturity** — NVMe-EP device driver may not be production-ready. Mitigation: software NVMe-oF fallback (higher latency, same API).

## Verification

1. Epic 1: `echo "test prompt" > /dev/llm_prompt1` produces correct token IDs matching HuggingFace reference tokenizer
2. Epic 2: p99 latency from write() to reactor callback < 5μs (hugepage ring, no io_uring overhead)
3. Epic 3: ACE attention output matches PyTorch reference for known input within BF16 tolerance
4. Vertical slice: single-layer generates correct next-token prediction for canned input
5. Full pipeline: generation quality matches reference on same model+prompt, throughput > baseline
