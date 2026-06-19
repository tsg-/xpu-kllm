CLANG ?= clang
LLVM_STRIP ?= llvm-strip
BPFTOOL ?= bpftool

# Kernel headers for vmlinux.h generation
VMLINUX_H ?= vmlinux.h
ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/')

CFLAGS := -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH)
INCLUDES := -I. -I../kernel/chardev

PROGS := tokenizer.bpf.o kllm_trace.bpf.o

all: $(VMLINUX_H) $(PROGS)

$(VMLINUX_H):
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

%.bpf.o: %.bpf.c $(VMLINUX_H)
	$(CLANG) $(CFLAGS) $(INCLUDES) -c $< -o $@
	$(LLVM_STRIP) -g $@

clean:
	rm -f *.bpf.o $(VMLINUX_H)

.PHONY: all clean
