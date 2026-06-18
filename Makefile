SUBDIRS := kernel bpf userspace tests

all:
	@for dir in $(SUBDIRS); do \
		if [ -f $$dir/Makefile ]; then $(MAKE) -C $$dir all; fi; \
	done

clean:
	@for dir in $(SUBDIRS); do \
		if [ -f $$dir/Makefile ]; then $(MAKE) -C $$dir clean; fi; \
	done

.PHONY: all clean
