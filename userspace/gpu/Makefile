CC ?= gcc
HIPCC ?= hipcc
NVCC ?= nvcc
ICPX ?= icpx

CFLAGS := -Wall -O2 -fPIC -I..
LDFLAGS := -ldl -lm

# Common C sources (always built)
COMMON_SRCS := gpu_attention.c nvme_ep_wavefront.c p2p_dma.c
COMMON_OBJS := $(COMMON_SRCS:.c=.o)

# Backend detection
HAS_ROCM := $(shell which $(HIPCC) 2>/dev/null && echo yes || echo no)
HAS_CUDA := $(shell which $(NVCC) 2>/dev/null && echo yes || echo no)
HAS_XE   := $(shell which $(ICPX) 2>/dev/null && echo yes || echo no)

BACKEND_OBJS :=

ifeq ($(HAS_ROCM),yes)
BACKEND_OBJS += paged_attention_hip.o
endif

ifeq ($(HAS_CUDA),yes)
BACKEND_OBJS += paged_attention_cuda.o
endif

ifeq ($(HAS_XE),yes)
BACKEND_OBJS += paged_attention_xe.o
endif

all: libkllm_gpu.a

libkllm_gpu.a: $(COMMON_OBJS) $(BACKEND_OBJS)
	ar rcs $@ $^

# Common C objects
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ROCm/HIP backend
paged_attention_hip.o: paged_attention.hip
	$(HIPCC) -O2 -fPIC --offload-arch=gfx942 -c $< -o $@

# CUDA backend
paged_attention_cuda.o: paged_attention.cu
	$(NVCC) -O2 --compiler-options -fPIC -arch=sm_80 -c $< -o $@

# Intel Xe / SYCL backend
paged_attention_xe.o: paged_attention_xe.cpp
	$(ICPX) -O2 -fPIC -fsycl -fsycl-targets=intel_gpu_pvc -c $< -o $@

clean:
	rm -f *.o libkllm_gpu.a

.PHONY: all clean
