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
│             NVMe-EP wavefront → RADOS-NKV               │
│             P2P DMA (RNIC → VRAM, no host bounce)       │
│             paged attention (ROCm / CUDA / Xe)          │
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
- **Content-addressed KV cache** — SHA-256(token prefix) → arena slot, same scheme on CPU and GPU
- **Three GPU backends** — ROCm (MI300+), CUDA (A100/H100+), Intel Xe (PVC/BMG), detected at runtime via dlopen
- **Mid-sequence escalation** — starts on CPU, transparently moves to GPU on cache miss
- **Continuous batching** — multiple concurrent writers, iteration-level scheduling
- **OCP MX quantization** — FP8 E4M3 scale per 8-element micro-block for KV cache compression

## Directory layout

```
kernel/chardev/     chardev module (write/read/poll/mmap/ioctl, ring alloc)
bpf/                eBPF BPE tokenizer + tracing probes
userspace/
  include/          shared ring protocol (C11 atomics)
  reactor/          SPDK app, ring consumer, batch scheduler, response path
  kvcache/          hugepage arena, content-addressed index, block format
  compute/          ACE attention, dispatch, CPU inference, GPU inference
  gpu/              paged attention (HIP/CUDA/SYCL), NVMe-EP wavefronts, P2P DMA
  weights/          model weight loader via NVMe-EP
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
- ROCm: `hipcc` (gfx942)
- CUDA: `nvcc` (sm_80)
- Intel Xe: `icpx` with `-fsycl` (intel_gpu_pvc)

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
| GPU | Any ROCm/CUDA/Xe GPU | MI300X / H100 / PVC |
| NIC | Any RDMA NIC | CX-7 for P2P DMA |
| Storage | Local NVMe | NVMe-EP (B70) + RADOS-NKV |
| Memory | 2MB hugepages | 1GB hugepages, 4+ GB arena |

## License

Apache-2.0
