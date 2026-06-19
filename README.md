# xpu-kllm

Kernel-integrated LLM serving engine. The inference endpoint is a character
device — `echo "prompt" > /dev/llm_prompt1` replaces vLLM.

## Architecture

```
 userspace write()
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  /dev/llm_prompt1  (chardev, misc_register)             │
│       │                                                 │
│       ▼                                                 │
│  eBPF BPE tokenizer (struct_ops, bpf_loop merges)       │
│       │                                                 │
│       ▼                                                 │
│  hugepage SPSC ring (2MB, cache-line aligned head/tail) │
└───────────────────────────┬─────────────────────────────┘
                            │ mmap
                            ▼
┌─────────────────────────────────────────────────────────┐
│  SPDK reactor (polls ring, zero-copy consume)           │
│       │                                                 │
│       ├─── dispatch policy (seq_len + cache hit rate)   │
│       │                                                 │
│       ├─► CPU path (≤512 tokens, cache hit)             │
│       │     ACE/AMX/AVX-512 BF16 paged attention        │
│       │                                                 │
│       └─► GPU path (long context or cache miss)         │
│             io_uring + dma-buf GPU Direct Storage        │
│             local NVMe → VRAM (no host bounce)          │
│             paged attention (Intel Xe / CUDA / ROCm)    │
│                                                         │
│       ▼                                                 │
│  decode loop (greedy / top-k / top-p / temperature)     │
│       │                                                 │
│       ▼                                                 │
│  response path → ioctl → chardev read()/poll()          │
└─────────────────────────────────────────────────────────┘
```

## Key properties

- **No vLLM, no Python** — kernel owns ingestion, SPDK reactor owns inference
- **Zero dynamic allocation** — all memory from pre-registered hugepage arenas
- **GPU Direct Storage** — io_uring + dma-buf for NVMe→VRAM without host bounce (also supports RADOS-NKV, RNIC RDMA)
- **Content-addressed KV cache** — SHA-256(token prefix) → arena slot, same scheme on CPU and GPU
- **Three GPU backends** — Intel Xe (PVC/BMG), CUDA (A100/H100+), ROCm (MI300+), detected at runtime via dlopen
- **Mid-sequence escalation** — starts on CPU, transparently moves to GPU on cache miss
- **Continuous batching** — multiple concurrent writers, iteration-level scheduling
- **OCP MX quantization** — FP8 E4M3 scale per 8-element micro-block for KV cache compression

## Storage I/O path

Primary: KV cache blocks and model weights live on local NVMe. On a cache
miss or weight load, the reactor submits io_uring SQEs targeting dma-buf
exported GPU VRAM regions — the NVMe controller DMAs directly into VRAM
via P2P PCIe, bypassing host DRAM entirely.

```
cache miss → SHA-256(prefix) → LBA lookup
  → io_uring_prep_read_fixed(nvme_fd, dmabuf_vram, ...)
  → NVMe controller P2P DMA → GPU VRAM
  → paged attention kernel launches on arrival
```

For UMA (integrated GPU / shared memory), the same code path works — the
dma-buf simply points to system DRAM.

Alternative storage backends (selectable at build time):
- **RADOS-NKV** — distributed object store for multi-node KV persistence
- **RNIC RDMA-WRITE** — remote NVMe-oF targets inject directly into VRAM
- **rocm-xio** — AMD GPU-initiated I/O for ROCm deployments

## Directory layout

```
kernel/chardev/     chardev module (write/read/poll/mmap/ioctl, ring alloc)
bpf/                eBPF BPE tokenizer + tracing probes
userspace/
  include/          shared ring protocol (C11 atomics)
  reactor/          SPDK app, ring consumer, batch scheduler, response path
  kvcache/          hugepage arena, content-addressed index, block format
  compute/          ACE attention, dispatch, CPU inference, GPU inference
  gpu/              paged attention (SYCL/Xe, CUDA, HIP), io_uring GDS, P2P DMA
  weights/          model weight loader (NVMe → VRAM direct)
  decode/           autoregressive decode loop + sampling
  tools/            kllm_trace (eBPF latency breakdown)
tests/              E2E latency test, tokenizer benchmark, tokenizer validation
docs/               BUILD_PLAN.md
```

## Building

Requires Linux 6.4+, clang 15+, libbpf, SPDK 24.01+.

```bash
# Kernel module
make -C kernel/

# eBPF programs
make -C bpf/

# Userspace (links against SPDK, OpenSSL)
make -C userspace/

# GPU backends (conditional on toolchain availability)
make -C userspace/gpu/

# Tests
make -C tests/
```

GPU backends build when their toolchain is detected:
- Intel Xe: `icpx` with `-fsycl` (intel_gpu_pvc)
- CUDA: `nvcc` (sm_80)
- ROCm: `hipcc` (gfx942)

## Running

```bash
# Load the kernel module
sudo insmod kernel/chardev/kllm.ko

# Start the SPDK reactor (polls the token ring)
sudo userspace/kllm_reactor

# Send a prompt
echo "Hello world" > /dev/llm_prompt1

# Read generated tokens
cat /dev/llm_prompt1
```

## Testing

```bash
# Synthetic E2E latency (no kernel module needed)
./tests/test_e2e_latency

# Full chardev path (requires kllm.ko)
./tests/test_e2e_latency --chardev

# Tokenizer benchmark
./tests/bench_tokenizer --iterations 1000

# Tokenizer correctness (requires kllm.ko)
./tests/test_ebpf_tokenizer
```

## Tracing

```bash
# Live per-request latency breakdown
sudo ./userspace/tools/kllm_trace

# JSON output for dashboards
sudo ./userspace/tools/kllm_trace --json

# Aggregate summary only
sudo ./userspace/tools/kllm_trace --summary
```

## Hardware requirements

| Component | Minimum | Optimal |
|-----------|---------|---------|
| CPU | AVX-512 BF16 (Cooper Lake+) | ACE BF16 (Granite Rapids+) |
| GPU | Any Xe/CUDA/ROCm GPU | Intel PVC / H100 / MI300X |
| Storage | Local NVMe (any), or RADOS-NKV | NVMe with CMB or P2P support |
| Memory | 2MB hugepages | 1GB hugepages, 4+ GB arena |

## License

Apache-2.0
