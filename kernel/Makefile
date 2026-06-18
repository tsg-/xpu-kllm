SUBDIRS := chardev bpf

all clean:
	@for dir in $(SUBDIRS); do \
		if [ -d $$dir ]; then $(MAKE) -C $$dir $@; fi; \
	done

.PHONY: all clean
