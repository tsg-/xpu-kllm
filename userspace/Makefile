CC ?= gcc
CFLAGS := -Wall -O2 -mavx512bf16 -mamx-bf16 -I include -I .
LDFLAGS := -lm -lnuma

# SPDK build flags (adjust SPDK_DIR as needed)
SPDK_DIR ?= /usr/local/lib/spdk
SPDK_CFLAGS := $(shell pkg-config --cflags spdk_event 2>/dev/null)
SPDK_LDFLAGS := $(shell pkg-config --libs spdk_event spdk_env_dpdk 2>/dev/null)

KVCACHE_SRCS := kvcache/arena_alloc.c kvcache/cache_index.c
COMPUTE_SRCS := compute/ace_attention.c compute/dispatch_policy.c
REACTOR_SRCS := reactor/kllm_reactor.c

KVCACHE_OBJS := $(KVCACHE_SRCS:.c=.o)
COMPUTE_OBJS := $(COMPUTE_SRCS:.c=.o)

# Library: libkllm.a (KV cache + compute, used by reactor and tests)
all: libkllm.a kllm_reactor

libkllm.a: $(KVCACHE_OBJS) $(COMPUTE_OBJS)
	ar rcs $@ $^

kllm_reactor: $(REACTOR_SRCS) libkllm.a
	$(CC) $(CFLAGS) $(SPDK_CFLAGS) -o $@ $(REACTOR_SRCS) -L. -lkllm $(SPDK_LDFLAGS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(KVCACHE_OBJS) $(COMPUTE_OBJS) libkllm.a kllm_reactor

.PHONY: all clean
