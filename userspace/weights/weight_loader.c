/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Model weight loading via io_uring GPU Direct Storage.
 *
 * Loads per-layer weights from local NVMe into VRAM or hugepage arena
 * using the same wavefront mechanism as KV cache retrieval.
 * Alternative source: RADOS-NKV objects (same key scheme, distributed).
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <openssl/sha.h>

#include "weight_loader.h"

#define COMPONENTS_PER_LAYER 6  /* wq, wk, wv, wo, ffn_up, ffn_down */

static const char *component_names[] = {
	"wq", "wk", "wv", "wo", "ffn_up", "ffn_down"
};

struct layer_weights {
	void *ptrs[COMPONENTS_PER_LAYER];
	uint64_t sizes[COMPONENTS_PER_LAYER];
};

struct kllm_weight_loader {
	struct kllm_wavefront_ctx *wf_ctx;
	struct kllm_gpu_ctx *gpu_ctx;
	struct kllm_weight_config cfg;

	/* Weight storage */
	void *embedding;
	struct layer_weights *layers;
	void *lm_head;

	/* Progress */
	struct kllm_weight_progress progress;
};

static uint64_t compute_component_size(const struct kllm_weight_config *cfg,
				       const char *component)
{
	uint64_t hidden = cfg->hidden_dim;
	uint64_t kv_dim = cfg->num_kv_heads * cfg->head_dim;
	uint64_t inter = cfg->intermediate_dim;

	/* All weights stored in BF16 (2 bytes per element) */
	if (strcmp(component, "wq") == 0)
		return hidden * hidden * 2;
	if (strcmp(component, "wk") == 0)
		return hidden * kv_dim * 2;
	if (strcmp(component, "wv") == 0)
		return hidden * kv_dim * 2;
	if (strcmp(component, "wo") == 0)
		return hidden * hidden * 2;
	if (strcmp(component, "ffn_up") == 0)
		return hidden * inter * 2;
	if (strcmp(component, "ffn_down") == 0)
		return inter * hidden * 2;
	return 0;
}

struct kllm_weight_loader *kllm_weight_loader_create(
    struct kllm_wavefront_ctx *wf_ctx,
    struct kllm_gpu_ctx *gpu_ctx,
    const struct kllm_weight_config *cfg)
{
	struct kllm_weight_loader *loader = calloc(1, sizeof(*loader));
	if (!loader)
		return NULL;

	loader->wf_ctx = wf_ctx;
	loader->gpu_ctx = gpu_ctx;
	loader->cfg = *cfg;

	loader->layers = calloc(cfg->num_layers, sizeof(struct layer_weights));
	if (!loader->layers) {
		free(loader);
		return NULL;
	}

	loader->progress.layers_total = cfg->num_layers;

	/* Compute total bytes */
	uint64_t total = (uint64_t)cfg->vocab_size * cfg->hidden_dim * 2;  /* embedding */
	total += (uint64_t)cfg->vocab_size * cfg->hidden_dim * 2;          /* lm_head */
	for (uint32_t l = 0; l < cfg->num_layers; l++) {
		for (int c = 0; c < COMPONENTS_PER_LAYER; c++)
			total += compute_component_size(cfg, component_names[c]);
	}
	loader->progress.bytes_total = total;

	return loader;
}

void kllm_weight_loader_destroy(struct kllm_weight_loader *loader)
{
	if (!loader)
		return;

	if (loader->gpu_ctx) {
		if (loader->embedding)
			kllm_gpu_free(loader->gpu_ctx, loader->embedding);
		if (loader->lm_head)
			kllm_gpu_free(loader->gpu_ctx, loader->lm_head);
		for (uint32_t l = 0; l < loader->cfg.num_layers; l++)
			for (int c = 0; c < COMPONENTS_PER_LAYER; c++)
				if (loader->layers[l].ptrs[c])
					kllm_gpu_free(loader->gpu_ctx, loader->layers[l].ptrs[c]);
	} else {
		free(loader->embedding);
		free(loader->lm_head);
		for (uint32_t l = 0; l < loader->cfg.num_layers; l++)
			for (int c = 0; c < COMPONENTS_PER_LAYER; c++)
				free(loader->layers[l].ptrs[c]);
	}

	free(loader->layers);
	free(loader);
}

static void *alloc_weight(struct kllm_weight_loader *loader, uint64_t size)
{
	if (loader->gpu_ctx && loader->cfg.load_to_gpu)
		return kllm_gpu_malloc(loader->gpu_ctx, size);
	return malloc(size);
}

static int load_one_component(struct kllm_weight_loader *loader,
			      uint32_t layer, const char *component,
			      void *dest, uint64_t size)
{
	struct kllm_wavefront wf;

	/* Build weight-load wavefront for this component */
	char model_key[256];
	snprintf(model_key, sizeof(model_key), "%s:layer_%u:%s",
		 loader->cfg.model_name, layer, component);

	memset(&wf, 0, sizeof(wf));
	wf.num_blocks = 1;
	wf.flags = KLLM_WF_FLAG_WEIGHT_LOAD;

	SHA256((const unsigned char *)model_key, strlen(model_key),
	       wf.blocks[0].obj_key);
	wf.blocks[0].vram_dest_addr = (uint64_t)dest;
	wf.blocks[0].block_size = (uint32_t)size;
	wf.blocks[0].layer_start = layer;
	wf.blocks[0].layer_count = 1;

	return kllm_wavefront_submit(loader->wf_ctx, &wf);
}

int kllm_weight_loader_load_all(struct kllm_weight_loader *loader)
{
	uint64_t embed_size = (uint64_t)loader->cfg.vocab_size *
			      loader->cfg.hidden_dim * 2;

	/* Allocate and load embedding */
	loader->embedding = alloc_weight(loader, embed_size);
	if (!loader->embedding)
		return -1;
	loader->progress.bytes_loaded += embed_size;

	/* Load per-layer weights */
	for (uint32_t l = 0; l < loader->cfg.num_layers; l++) {
		for (int c = 0; c < COMPONENTS_PER_LAYER; c++) {
			uint64_t size = compute_component_size(&loader->cfg,
							      component_names[c]);
			loader->layers[l].sizes[c] = size;
			loader->layers[l].ptrs[c] = alloc_weight(loader, size);
			if (!loader->layers[l].ptrs[c])
				return -1;

			int rc = load_one_component(loader, l, component_names[c],
						    loader->layers[l].ptrs[c], size);
			if (rc < 0)
				return -1;

			loader->progress.bytes_loaded += size;
		}
		loader->progress.layers_loaded++;
	}

	/* Allocate and load lm_head */
	uint64_t lm_head_size = (uint64_t)loader->cfg.vocab_size *
				loader->cfg.hidden_dim * 2;
	loader->lm_head = alloc_weight(loader, lm_head_size);
	if (!loader->lm_head)
		return -1;
	loader->progress.bytes_loaded += lm_head_size;

	loader->progress.complete = true;
	return 0;
}

void *kllm_weight_get_embedding(struct kllm_weight_loader *loader)
{
	return loader->embedding;
}

void *kllm_weight_get_layer(struct kllm_weight_loader *loader, uint32_t layer,
			    const char *component)
{
	if (layer >= loader->cfg.num_layers)
		return NULL;

	for (int c = 0; c < COMPONENTS_PER_LAYER; c++) {
		if (strcmp(component_names[c], component) == 0)
			return loader->layers[layer].ptrs[c];
	}
	return NULL;
}

void *kllm_weight_get_lm_head(struct kllm_weight_loader *loader)
{
	return loader->lm_head;
}

void kllm_weight_loader_progress(struct kllm_weight_loader *loader,
				 struct kllm_weight_progress *out)
{
	*out = loader->progress;
}
