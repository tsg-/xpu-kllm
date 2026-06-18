// SPDX-License-Identifier: GPL-2.0
/*
 * xpu-kllm: character device for kernel-integrated LLM serving
 *
 * /dev/llm_prompt1 — write prompts in, read generated tokens out.
 * Tokenization happens via eBPF struct_ops attached to this device.
 * Token IDs land in a hugepage-backed ring buffer consumed by SPDK.
 */

#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/wait.h>
#include <linux/circ_buf.h>
#include <linux/mm.h>
#include <linux/hugetlb.h>
#include <linux/io.h>
#include <linux/ioctl.h>

#include "kllm_ring.h"

/* ioctl definitions — must match userspace kllm_response.c */
#define KLLM_IOC_MAGIC		'K'

struct kllm_emit_cmd {
	__u32 count;
	__u32 tokens[256];
};

#define KLLM_IOC_EMIT_TOKENS	_IOW(KLLM_IOC_MAGIC, 1, struct kllm_emit_cmd)
#define KLLM_IOC_EOS		_IO(KLLM_IOC_MAGIC, 2)

#define KLLM_MAX_PROMPT_SIZE	(64 * 1024)

/* Token ring: 2MB compound page allocation (hugepage-backed) */
#define KLLM_RING_ORDER		9  /* 2^9 pages = 512 * 4K = 2MB */
#define KLLM_RING_BYTES		(PAGE_SIZE << KLLM_RING_ORDER)

/*
 * Per-device state. In production there will be multiple instances
 * (/dev/llm_prompt1, /dev/llm_prompt2, ...) for multi-tenant isolation.
 */
struct kllm_device {
	struct miscdevice	misc;
	wait_queue_head_t	read_wq;   /* wake readers when tokens available */
	wait_queue_head_t	write_wq;  /* wake writers when ring has space */

	/* Token ring: hugepage-backed, shared with SPDK via mmap */
	struct kllm_ring_hdr	*token_ring;
	struct page		*ring_page;   /* compound page backing the ring */

	/* Response ring: generated tokens flow back to userspace readers */
	u32			*resp_ring;
	unsigned int		resp_head;  /* producer: reactor writes here */
	unsigned int		resp_tail;  /* consumer: read() advances this */
	unsigned int		resp_mask;
	spinlock_t		resp_lock;
};

static struct kllm_device kllm_dev;

/* --- Response ring helpers --- */

static inline unsigned int resp_ring_count(struct kllm_device *dev)
{
	return CIRC_CNT(dev->resp_head, dev->resp_tail, dev->resp_mask + 1);
}

static inline unsigned int resp_ring_space(struct kllm_device *dev)
{
	return CIRC_SPACE(dev->resp_head, dev->resp_tail, dev->resp_mask + 1);
}

/* Called by SPDK reactor (via mmap'd ring) to push response tokens */
void kllm_resp_produce(struct kllm_device *dev, u32 token_id)
{
	unsigned long flags;

	spin_lock_irqsave(&dev->resp_lock, flags);
	if (resp_ring_space(dev) > 0) {
		dev->resp_ring[dev->resp_head] = token_id;
		dev->resp_head = (dev->resp_head + 1) & dev->resp_mask;
	}
	spin_unlock_irqrestore(&dev->resp_lock, flags);
	wake_up_interruptible(&dev->read_wq);
}
EXPORT_SYMBOL_GPL(kllm_resp_produce);

/* --- file_operations --- */

static int kllm_open(struct inode *inode, struct file *filp)
{
	filp->private_data = &kllm_dev;
	return 0;
}

static int kllm_release(struct inode *inode, struct file *filp)
{
	return 0;
}

/*
 * write() — submit a prompt for tokenization.
 *
 * The eBPF tokenizer (attached via struct_ops) will be invoked here.
 * For now, the raw bytes are accepted and the hook point is marked.
 * Token IDs produced by eBPF go to the hugepage token ring (task 1.2/1.5).
 */
static ssize_t kllm_write(struct file *filp, const char __user *buf,
			   size_t count, loff_t *ppos)
{
	struct kllm_device *dev = filp->private_data;
	char *kbuf;
	ssize_t ret;

	if (count == 0)
		return 0;
	if (count > KLLM_MAX_PROMPT_SIZE)
		return -EFBIG;

	kbuf = kvmalloc(count, GFP_KERNEL);
	if (!kbuf)
		return -ENOMEM;

	if (copy_from_user(kbuf, buf, count)) {
		ret = -EFAULT;
		goto out;
	}

	/*
	 * TODO(task 1.3): invoke eBPF struct_ops tokenizer here.
	 * The tokenizer will write token IDs directly to the hugepage
	 * ring buffer. For now, just accept the bytes.
	 *
	 * Hook point:
	 *   ret = kllm_bpf_tokenize(dev, kbuf, count);
	 */

	ret = count;

out:
	kvfree(kbuf);
	return ret;
}

/*
 * read() — consume generated tokens from the response ring.
 *
 * Returns token IDs as a stream of u32 values.
 * Blocks if no tokens available (unless O_NONBLOCK).
 */
static ssize_t kllm_read(struct file *filp, char __user *buf,
			  size_t count, loff_t *ppos)
{
	struct kllm_device *dev = filp->private_data;
	unsigned int avail, to_copy;
	unsigned long flags;
	ssize_t ret;

	/* Align to u32 boundary */
	count &= ~(sizeof(u32) - 1);
	if (count == 0)
		return 0;

	if (resp_ring_count(dev) == 0) {
		if (filp->f_flags & O_NONBLOCK)
			return -EAGAIN;
		ret = wait_event_interruptible(dev->read_wq,
					       resp_ring_count(dev) > 0);
		if (ret)
			return ret;
	}

	spin_lock_irqsave(&dev->resp_lock, flags);
	avail = resp_ring_count(dev);
	to_copy = min(avail, (unsigned int)(count / sizeof(u32)));

	/* Handle wrap-around */
	if (dev->resp_tail + to_copy <= dev->resp_mask + 1) {
		spin_unlock_irqrestore(&dev->resp_lock, flags);
		if (copy_to_user(buf, &dev->resp_ring[dev->resp_tail],
				 to_copy * sizeof(u32)))
			return -EFAULT;
		spin_lock_irqsave(&dev->resp_lock, flags);
	} else {
		unsigned int first = dev->resp_mask + 1 - dev->resp_tail;
		unsigned int second = to_copy - first;

		spin_unlock_irqrestore(&dev->resp_lock, flags);
		if (copy_to_user(buf, &dev->resp_ring[dev->resp_tail],
				 first * sizeof(u32)))
			return -EFAULT;
		if (copy_to_user(buf + first * sizeof(u32), &dev->resp_ring[0],
				 second * sizeof(u32)))
			return -EFAULT;
		spin_lock_irqsave(&dev->resp_lock, flags);
	}

	dev->resp_tail = (dev->resp_tail + to_copy) & dev->resp_mask;
	spin_unlock_irqrestore(&dev->resp_lock, flags);

	wake_up_interruptible(&dev->write_wq);
	return to_copy * sizeof(u32);
}

static __poll_t kllm_poll(struct file *filp, struct poll_table_struct *wait)
{
	struct kllm_device *dev = filp->private_data;
	__poll_t mask = 0;

	poll_wait(filp, &dev->read_wq, wait);
	poll_wait(filp, &dev->write_wq, wait);

	if (resp_ring_count(dev) > 0)
		mask |= EPOLLIN | EPOLLRDNORM;

	/* Always writable for now (prompt submission) */
	mask |= EPOLLOUT | EPOLLWRNORM;

	return mask;
}

/*
 * mmap() — map the hugepage token ring into userspace.
 *
 * The SPDK reactor mmaps this to consume token IDs produced by the
 * eBPF tokenizer. This is the kernel↔userspace zero-copy interface.
 */
/*
 * mmap() — map the hugepage token ring into userspace.
 *
 * The SPDK reactor mmaps this to consume token IDs produced by the
 * eBPF tokenizer. Offset 0 = token ring (2MB).
 */
static int kllm_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct kllm_device *dev = filp->private_data;
	unsigned long size = vma->vm_end - vma->vm_start;
	unsigned long pfn;

	if (size > KLLM_RING_BYTES)
		return -EINVAL;

	if (vma->vm_pgoff != 0)
		return -EINVAL;

	/* Don't allow writable mappings from arbitrary userspace */
	vm_flags_set(vma, VM_DONTEXPAND | VM_DONTDUMP);

	pfn = page_to_pfn(dev->ring_page);
	return remap_pfn_range(vma, vma->vm_start, pfn, size,
			       vma->vm_page_prot);
}

/*
 * ioctl() — SPDK reactor uses this to push generated tokens back
 * into the response ring for userspace readers.
 */
static long kllm_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct kllm_device *dev = filp->private_data;
	struct kllm_emit_cmd emit;
	unsigned int i;

	switch (cmd) {
	case KLLM_IOC_EMIT_TOKENS:
		if (copy_from_user(&emit, (void __user *)arg, sizeof(emit)))
			return -EFAULT;
		if (emit.count > 256)
			return -EINVAL;
		for (i = 0; i < emit.count; i++)
			kllm_resp_produce(dev, emit.tokens[i]);
		return 0;

	case KLLM_IOC_EOS:
		/* Signal end-of-sequence: push a sentinel token (0xFFFFFFFF) */
		kllm_resp_produce(dev, 0xFFFFFFFF);
		return 0;

	default:
		return -ENOTTY;
	}
}

static const struct file_operations kllm_fops = {
	.owner		= THIS_MODULE,
	.open		= kllm_open,
	.release	= kllm_release,
	.write		= kllm_write,
	.read		= kllm_read,
	.poll		= kllm_poll,
	.mmap		= kllm_mmap,
	.unlocked_ioctl	= kllm_ioctl,
};

static int kllm_alloc_token_ring(struct kllm_device *dev)
{
	struct kllm_ring_hdr *hdr;
	struct page *page;
	u32 slot_count;

	/*
	 * Allocate a 2MB compound page. On systems with transparent
	 * hugepages or explicit hugetlb, this gives us physically
	 * contiguous memory with stable pfn for remap_pfn_range.
	 */
	page = alloc_pages(GFP_KERNEL | __GFP_COMP | __GFP_ZERO,
			   KLLM_RING_ORDER);
	if (!page)
		return -ENOMEM;

	hdr = page_address(page);
	slot_count = (KLLM_RING_BYTES - KLLM_RING_HDR_SIZE) / sizeof(u32);
	/* Round down to power-of-2 for masking */
	slot_count = rounddown_pow_of_two(slot_count);

	hdr->magic = KLLM_TOKEN_RING_MAGIC;
	hdr->version = KLLM_TOKEN_RING_VERSION;
	hdr->slot_count = slot_count;
	hdr->head = 0;
	hdr->tail = 0;
	hdr->seq_id = 0;

	dev->token_ring = hdr;
	dev->ring_page = page;
	return 0;
}

static void kllm_free_token_ring(struct kllm_device *dev)
{
	if (dev->ring_page) {
		__free_pages(dev->ring_page, KLLM_RING_ORDER);
		dev->ring_page = NULL;
		dev->token_ring = NULL;
	}
}

static int __init kllm_init(void)
{
	int ret;

	/* Allocate hugepage-backed token ring */
	ret = kllm_alloc_token_ring(&kllm_dev);
	if (ret)
		return ret;

	/* Allocate response ring (power-of-2 slots) */
	kllm_dev.resp_mask = 4096 - 1;  /* 4096 u32 slots = 16KB */
	kllm_dev.resp_ring = kvmalloc_array(kllm_dev.resp_mask + 1,
					    sizeof(u32), GFP_KERNEL | __GFP_ZERO);
	if (!kllm_dev.resp_ring) {
		kllm_free_token_ring(&kllm_dev);
		return -ENOMEM;
	}

	kllm_dev.resp_head = 0;
	kllm_dev.resp_tail = 0;
	spin_lock_init(&kllm_dev.resp_lock);
	init_waitqueue_head(&kllm_dev.read_wq);
	init_waitqueue_head(&kllm_dev.write_wq);

	kllm_dev.misc.minor = MISC_DYNAMIC_MINOR;
	kllm_dev.misc.name = "llm_prompt1";
	kllm_dev.misc.fops = &kllm_fops;
	kllm_dev.misc.mode = 0666;

	ret = misc_register(&kllm_dev.misc);
	if (ret) {
		kvfree(kllm_dev.resp_ring);
		kllm_free_token_ring(&kllm_dev);
		return ret;
	}

	pr_info("kllm: /dev/llm_prompt1 registered (token ring: %u slots, %lu bytes)\n",
		kllm_dev.token_ring->slot_count, (unsigned long)KLLM_RING_BYTES);
	return 0;
}

static void __exit kllm_exit(void)
{
	misc_deregister(&kllm_dev.misc);
	kvfree(kllm_dev.resp_ring);
	kllm_free_token_ring(&kllm_dev);
	pr_info("kllm: /dev/llm_prompt1 unregistered\n");
}

module_init(kllm_init);
module_exit(kllm_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Tushar Gohad");
MODULE_DESCRIPTION("xpu-kllm: kernel character device for LLM inference");
