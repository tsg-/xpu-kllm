// SPDX-License-Identifier: GPL-2.0
/*
 * eBPF tracing for xpu-kllm: per-request latency breakdown.
 *
 * Attaches to key points in the request lifecycle and records
 * timestamps for each stage:
 *
 *   chardev_write → tokenize → ring_produce → reactor_consume →
 *   dispatch → cache_lookup → attention → sample → response_emit
 *
 * Exposed via BPF ring buffer to userspace trace consumer.
 *
 * Usage:
 *   Load with libbpf, attach fentry/fexit probes to kllm functions.
 *   Read events from the ring buffer for per-request flamegraph data.
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define KLLM_TRACE_EVENT_MAX	16
#define KLLM_MAX_TRACKED	1024

enum kllm_trace_stage {
	KLLM_STAGE_CHARDEV_WRITE = 0,
	KLLM_STAGE_TOKENIZE_START,
	KLLM_STAGE_TOKENIZE_END,
	KLLM_STAGE_RING_PRODUCE,
	KLLM_STAGE_REACTOR_CONSUME,
	KLLM_STAGE_DISPATCH_DECIDE,
	KLLM_STAGE_CACHE_LOOKUP,
	KLLM_STAGE_ATTENTION_START,
	KLLM_STAGE_ATTENTION_END,
	KLLM_STAGE_SAMPLE,
	KLLM_STAGE_RESPONSE_EMIT,
	KLLM_STAGE_GPU_SUBMIT,
	KLLM_STAGE_GPU_COMPLETE,
	KLLM_STAGE_EOS,
	KLLM_STAGE_MAX,
};

struct kllm_trace_event {
	__u64 timestamp_ns;
	__u32 request_id;
	__u32 stage;
	__u32 seq_len;
	__u32 token_count;
	__u64 latency_ns;      /* delta from previous stage */
};

/* Per-request in-flight tracking */
struct kllm_request_trace {
	__u64 stage_ts[KLLM_STAGE_MAX];
	__u32 last_stage;
	__u32 seq_len;
};

/* Ring buffer for events (128KB) */
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 128 * 1024);
} trace_events SEC(".maps");

/* Per-request state (keyed by request_id) */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, KLLM_MAX_TRACKED);
	__type(key, __u32);
	__type(value, struct kllm_request_trace);
} request_state SEC(".maps");

/* Global stats (per-CPU for lock-free aggregation) */
struct kllm_trace_stats {
	__u64 total_requests;
	__u64 total_tokens;
	__u64 total_latency_ns;
	__u64 max_latency_ns;
	__u64 stage_totals[KLLM_STAGE_MAX];
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct kllm_trace_stats);
} global_stats SEC(".maps");

static __always_inline void emit_trace(
    __u32 request_id, __u32 stage, __u32 seq_len, __u32 token_count)
{
	__u64 now = bpf_ktime_get_ns();
	__u64 delta = 0;

	/* Look up in-flight state */
	struct kllm_request_trace *rt = bpf_map_lookup_elem(&request_state, &request_id);
	if (rt) {
		if (rt->last_stage < KLLM_STAGE_MAX)
			delta = now - rt->stage_ts[rt->last_stage];
		if (stage < KLLM_STAGE_MAX)
			rt->stage_ts[stage] = now;
		rt->last_stage = stage;
		rt->seq_len = seq_len;
	}

	/* Emit event to ring buffer */
	struct kllm_trace_event *evt =
		bpf_ringbuf_reserve(&trace_events, sizeof(*evt), 0);
	if (!evt)
		return;

	evt->timestamp_ns = now;
	evt->request_id = request_id;
	evt->stage = stage;
	evt->seq_len = seq_len;
	evt->token_count = token_count;
	evt->latency_ns = delta;

	bpf_ringbuf_submit(evt, 0);

	/* Update per-CPU stats */
	__u32 zero = 0;
	struct kllm_trace_stats *stats =
		bpf_map_lookup_elem(&global_stats, &zero);
	if (stats && stage < KLLM_STAGE_MAX)
		stats->stage_totals[stage] += delta;
}

static __always_inline void start_request(__u32 request_id)
{
	struct kllm_request_trace rt = {};
	rt.stage_ts[KLLM_STAGE_CHARDEV_WRITE] = bpf_ktime_get_ns();
	rt.last_stage = KLLM_STAGE_CHARDEV_WRITE;
	bpf_map_update_elem(&request_state, &request_id, &rt, BPF_ANY);

	__u32 zero = 0;
	struct kllm_trace_stats *stats =
		bpf_map_lookup_elem(&global_stats, &zero);
	if (stats)
		__sync_fetch_and_add(&stats->total_requests, 1);
}

static __always_inline void end_request(__u32 request_id)
{
	struct kllm_request_trace *rt =
		bpf_map_lookup_elem(&request_state, &request_id);
	if (rt) {
		__u64 now = bpf_ktime_get_ns();
		__u64 total = now - rt->stage_ts[KLLM_STAGE_CHARDEV_WRITE];

		__u32 zero = 0;
		struct kllm_trace_stats *stats =
			bpf_map_lookup_elem(&global_stats, &zero);
		if (stats) {
			__sync_fetch_and_add(&stats->total_latency_ns, total);
			if (total > stats->max_latency_ns)
				stats->max_latency_ns = total;
		}
	}
	bpf_map_delete_elem(&request_state, &request_id);
}

/*
 * Tracepoint / fentry probes.
 *
 * These attach to kllm kernel module and userspace USDT probes.
 * For the kernel chardev path, use fentry on llm_prompt_write.
 * For userspace (reactor), use USDT or uprobe.
 */

SEC("fentry/llm_prompt_write")
int BPF_PROG(trace_chardev_write, struct file *f, const char *buf,
	     size_t count, loff_t *pos)
{
	__u32 pid = bpf_get_current_pid_tgid() >> 32;
	start_request(pid);
	emit_trace(pid, KLLM_STAGE_CHARDEV_WRITE, 0, (u32)count);
	return 0;
}

SEC("fexit/llm_prompt_write")
int BPF_PROG(trace_chardev_write_exit, struct file *f, const char *buf,
	     size_t count, loff_t *pos, ssize_t ret)
{
	__u32 pid = bpf_get_current_pid_tgid() >> 32;
	emit_trace(pid, KLLM_STAGE_TOKENIZE_END, 0, 0);
	return 0;
}

/*
 * Userspace probes (uprobe/USDT) for reactor-side tracing.
 *
 * These would be attached to specific functions in the reactor binary.
 * The actual attachment is done by the trace consumer tool, not this BPF
 * program — we just define the handlers here.
 */

SEC("uprobe")
int trace_reactor_consume(struct pt_regs *ctx)
{
	__u32 pid = bpf_get_current_pid_tgid() >> 32;
	emit_trace(pid, KLLM_STAGE_REACTOR_CONSUME, 0, 0);
	return 0;
}

SEC("uprobe")
int trace_dispatch_decide(struct pt_regs *ctx)
{
	__u32 pid = bpf_get_current_pid_tgid() >> 32;
	emit_trace(pid, KLLM_STAGE_DISPATCH_DECIDE, 0, 0);
	return 0;
}

SEC("uprobe")
int trace_attention_start(struct pt_regs *ctx)
{
	__u32 pid = bpf_get_current_pid_tgid() >> 32;
	emit_trace(pid, KLLM_STAGE_ATTENTION_START, 0, 0);
	return 0;
}

SEC("uprobe")
int trace_attention_end(struct pt_regs *ctx)
{
	__u32 pid = bpf_get_current_pid_tgid() >> 32;
	emit_trace(pid, KLLM_STAGE_ATTENTION_END, 0, 0);
	return 0;
}

SEC("uprobe")
int trace_response_emit(struct pt_regs *ctx)
{
	__u32 pid = bpf_get_current_pid_tgid() >> 32;
	__u32 zero = 0;
	struct kllm_trace_stats *stats =
		bpf_map_lookup_elem(&global_stats, &zero);
	if (stats)
		__sync_fetch_and_add(&stats->total_tokens, 1);
	emit_trace(pid, KLLM_STAGE_RESPONSE_EMIT, 0, 1);
	return 0;
}

SEC("uprobe")
int trace_eos(struct pt_regs *ctx)
{
	__u32 pid = bpf_get_current_pid_tgid() >> 32;
	emit_trace(pid, KLLM_STAGE_EOS, 0, 0);
	end_request(pid);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
