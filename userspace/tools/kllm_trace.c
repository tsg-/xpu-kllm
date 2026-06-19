/* SPDX-License-Identifier: Apache-2.0 */
/*
 * kllm_trace: per-request latency breakdown tool.
 *
 * Loads the kllm_trace.bpf.o program, attaches probes, and consumes
 * trace events from the BPF ring buffer. Outputs per-stage latency
 * breakdown and live statistics.
 *
 * Usage:
 *   ./kllm_trace                # live trace
 *   ./kllm_trace --summary      # aggregate stats only
 *   ./kllm_trace --json         # JSON output for dashboards
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>

/* libbpf headers */
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

static volatile int running = 1;

static const char *stage_names[] = {
	[0]  = "chardev_write",
	[1]  = "tokenize_start",
	[2]  = "tokenize_end",
	[3]  = "ring_produce",
	[4]  = "reactor_consume",
	[5]  = "dispatch_decide",
	[6]  = "cache_lookup",
	[7]  = "attention_start",
	[8]  = "attention_end",
	[9]  = "sample",
	[10] = "response_emit",
	[11] = "gpu_submit",
	[12] = "gpu_complete",
	[13] = "eos",
};

#define STAGE_MAX 14

struct trace_event {
	uint64_t timestamp_ns;
	uint32_t request_id;
	uint32_t stage;
	uint32_t seq_len;
	uint32_t token_count;
	uint64_t latency_ns;
};

struct trace_stats {
	uint64_t total_requests;
	uint64_t total_tokens;
	uint64_t total_latency_ns;
	uint64_t max_latency_ns;
	uint64_t stage_totals[STAGE_MAX];
};

static int output_json;
static int summary_only;

static void sig_handler(int sig)
{
	(void)sig;
	running = 0;
}

static int handle_event(void *ctx, void *data, size_t size)
{
	struct trace_event *evt = (struct trace_event *)data;
	(void)ctx;
	(void)size;

	if (summary_only)
		return 0;

	const char *name = (evt->stage < STAGE_MAX) ?
			   stage_names[evt->stage] : "unknown";

	if (output_json) {
		printf("{\"ts\":%lu,\"req\":%u,\"stage\":\"%s\","
		       "\"seq_len\":%u,\"delta_us\":%.1f}\n",
		       (unsigned long)evt->timestamp_ns,
		       evt->request_id, name, evt->seq_len,
		       (double)evt->latency_ns / 1000.0);
	} else {
		printf("[%12lu] req=%-6u %-20s seq=%-5u delta=%7.1fus\n",
		       (unsigned long)(evt->timestamp_ns / 1000),
		       evt->request_id, name, evt->seq_len,
		       (double)evt->latency_ns / 1000.0);
	}

	return 0;
}

static void print_summary(int stats_fd)
{
	struct trace_stats stats;
	uint32_t key = 0;

	memset(&stats, 0, sizeof(stats));
	bpf_map_lookup_elem(stats_fd, &key, &stats);

	printf("\n=== kllm trace summary ===\n");
	printf("  total requests:  %lu\n", (unsigned long)stats.total_requests);
	printf("  total tokens:    %lu\n", (unsigned long)stats.total_tokens);
	if (stats.total_requests > 0) {
		printf("  avg latency:     %.1f ms\n",
		       (double)stats.total_latency_ns / stats.total_requests / 1e6);
		printf("  max latency:     %.1f ms\n",
		       (double)stats.max_latency_ns / 1e6);
		if (stats.total_tokens > 0)
			printf("  tokens/sec:      %.0f\n",
			       (double)stats.total_tokens * 1e9 / stats.total_latency_ns);
	}

	printf("\n  per-stage breakdown (avg per request):\n");
	for (int i = 0; i < STAGE_MAX; i++) {
		if (stats.stage_totals[i] == 0)
			continue;
		double avg_us = (double)stats.stage_totals[i] /
				stats.total_requests / 1000.0;
		printf("    %-20s %8.1f us\n", stage_names[i], avg_us);
	}
}

int main(int argc, char **argv)
{
	struct bpf_object *obj;
	struct ring_buffer *rb;
	int ringbuf_fd, stats_fd;
	int rc;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--json") == 0)
			output_json = 1;
		else if (strcmp(argv[i], "--summary") == 0)
			summary_only = 1;
	}

	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load BPF object */
	obj = bpf_object__open_file("kllm_trace.bpf.o", NULL);
	if (!obj) {
		fprintf(stderr, "failed to open kllm_trace.bpf.o\n");
		return 1;
	}

	rc = bpf_object__load(obj);
	if (rc) {
		fprintf(stderr, "failed to load BPF object: %d\n", rc);
		bpf_object__close(obj);
		return 1;
	}

	/* Attach all programs */
	rc = bpf_object__attach_skeleton(bpf_object__open_skeleton(obj, NULL));
	/* Note: in practice, use skeleton auto-generated from .bpf.c */

	/* Get map FDs */
	ringbuf_fd = bpf_object__find_map_fd_by_name(obj, "trace_events");
	stats_fd = bpf_object__find_map_fd_by_name(obj, "global_stats");
	if (ringbuf_fd < 0 || stats_fd < 0) {
		fprintf(stderr, "failed to find BPF maps\n");
		bpf_object__close(obj);
		return 1;
	}

	/* Create ring buffer consumer */
	rb = ring_buffer__new(ringbuf_fd, handle_event, NULL, NULL);
	if (!rb) {
		fprintf(stderr, "failed to create ring buffer\n");
		bpf_object__close(obj);
		return 1;
	}

	if (!summary_only)
		printf("tracing kllm requests (Ctrl+C to stop)...\n\n");

	while (running) {
		rc = ring_buffer__poll(rb, 100);
		if (rc < 0 && rc != -EINTR)
			break;
	}

	print_summary(stats_fd);

	ring_buffer__free(rb);
	bpf_object__close(obj);
	return 0;
}
