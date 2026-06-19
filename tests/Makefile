CC ?= gcc
CFLAGS := -Wall -O2 -I../userspace/include -I../userspace
LDFLAGS := -lm -lcrypto

LIBKLLM := ../userspace/libkllm.a

TESTS := test_ebpf_tokenizer test_e2e_latency

all: $(TESTS)

test_ebpf_tokenizer: test_ebpf_tokenizer.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_e2e_latency: test_e2e_latency.c $(LIBKLLM)
	$(CC) $(CFLAGS) -o $@ $< -L../userspace -lkllm $(LDFLAGS)

run: $(TESTS)
	@for t in $(TESTS); do \
		echo "=== $$t ==="; \
		./$$t || exit 1; \
	done

clean:
	rm -f $(TESTS)

.PHONY: all run clean
