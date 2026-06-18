CC ?= gcc
CFLAGS := -Wall -O2 -I../userspace/include
LDFLAGS :=

TESTS := test_ebpf_tokenizer

all: $(TESTS)

test_ebpf_tokenizer: test_ebpf_tokenizer.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TESTS)
	@for t in $(TESTS); do \
		echo "=== $$t ==="; \
		./$$t || exit 1; \
	done

clean:
	rm -f $(TESTS)

.PHONY: all run clean
