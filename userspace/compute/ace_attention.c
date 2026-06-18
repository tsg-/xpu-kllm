/* SPDX-License-Identifier: Apache-2.0 */
/*
 * ACE/AMX BF16 attention kernel for xpu-kllm.
 *
 * Computes scaled dot-product attention using:
 * - ACE outer-product intrinsics (preferred, Granite Rapids+)
 * - AMX BF16 tile operations (fallback, Sapphire Rapids+)
 * - AVX-512 BF16 DPBF16PS (last resort fallback)
 *
 * Runs in SPDK reactor context — no kernel, no verifier constraints.
 * Memory accessed: hugepage-backed KV arena (pre-pinned, stable VA).
 *
 * Design: tiled computation to fit in L2 cache per tile.
 * Tile sizes chosen to match AMX tile register dimensions (16x16).
 */

#include <immintrin.h>
#include <cpuid.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "ace_attention.h"

/* Tile sizes for blocked matmul */
#define TILE_M	16
#define TILE_N	16
#define TILE_K	32  /* BF16 pairs → 16 FP32 accumulators */

/* CPU feature detection */
enum kllm_isa_level {
	ISA_NONE = 0,
	ISA_AVX512_BF16,  /* VDPBF16PS: dot-product BF16 */
	ISA_AMX_BF16,     /* TDPBF16PS: tile dot-product BF16 */
	ISA_ACE_BF16,     /* TOPBF16: tile outer-product BF16 (future) */
};

static enum kllm_isa_level g_isa_level = ISA_NONE;

int kllm_ace_probe(void)
{
	uint32_t eax, ebx, ecx, edx;

	/* Check AVX-512 BF16 (CPUID.7.1:EAX[5]) */
	__cpuid_count(7, 1, eax, ebx, ecx, edx);
	if (eax & (1 << 5))
		g_isa_level = ISA_AVX512_BF16;

	/* Check AMX-BF16 (CPUID.7.0:EDX[22]) */
	__cpuid_count(7, 0, eax, ebx, ecx, edx);
	if (edx & (1 << 22))
		g_isa_level = ISA_AMX_BF16;

	/*
	 * ACE detection (future CPUID leaf — not yet defined in public ISA).
	 * When available, ACE provides TOPBF16 (tile outer-product BF16)
	 * at 16x compute density over VNNI.
	 * For now, treat AMX_BF16 as highest available.
	 */

	return g_isa_level;
}

/* --- BF16 conversion utilities --- */

static inline float bf16_to_f32(bf16_t val)
{
	uint32_t bits = (uint32_t)val << 16;
	float result;
	memcpy(&result, &bits, sizeof(result));
	return result;
}

static inline bf16_t f32_to_bf16(float val)
{
	uint32_t bits;
	memcpy(&bits, &val, sizeof(bits));
	/* Round to nearest even */
	bits += 0x7FFF + ((bits >> 16) & 1);
	return (bf16_t)(bits >> 16);
}

/* --- Scalar fallback (for correctness reference and non-SIMD systems) --- */

static void attention_scalar(const bf16_t *q, const bf16_t *k, const bf16_t *v,
			     bf16_t *output, const struct kllm_attention_params *p)
{
	float scale = p->scale;
	uint32_t head_dim = p->head_dim;
	uint32_t seq_len = p->seq_len;
	uint32_t num_heads = p->num_heads;
	uint32_t query_len = p->query_len;

	/* Per-head, per-query-position */
	for (uint32_t h = 0; h < num_heads; h++) {
		for (uint32_t qi = 0; qi < query_len; qi++) {
			const bf16_t *q_vec = q + (qi * num_heads + h) * head_dim;

			/* Compute attention scores: Q * K^T */
			float scores[seq_len];
			float max_score = -INFINITY;

			for (uint32_t si = 0; si < seq_len; si++) {
				const bf16_t *k_vec = k + (si * num_heads + h) * head_dim;
				float dot = 0.0f;
				for (uint32_t d = 0; d < head_dim; d++)
					dot += bf16_to_f32(q_vec[d]) * bf16_to_f32(k_vec[d]);
				scores[si] = dot * scale;
				if (scores[si] > max_score)
					max_score = scores[si];
			}

			/* Softmax */
			float sum = 0.0f;
			for (uint32_t si = 0; si < seq_len; si++) {
				scores[si] = expf(scores[si] - max_score);
				sum += scores[si];
			}
			for (uint32_t si = 0; si < seq_len; si++)
				scores[si] /= sum;

			/* Weighted sum of values */
			bf16_t *out_vec = output + (qi * num_heads + h) * head_dim;
			for (uint32_t d = 0; d < head_dim; d++) {
				float acc = 0.0f;
				for (uint32_t si = 0; si < seq_len; si++) {
					const bf16_t *v_vec = v + (si * num_heads + h) * head_dim;
					acc += scores[si] * bf16_to_f32(v_vec[d]);
				}
				out_vec[d] = f32_to_bf16(acc);
			}
		}
	}
}

/* --- AVX-512 BF16 path --- */

#ifdef __AVX512BF16__
static void attention_avx512_bf16(const bf16_t *q, const bf16_t *k, const bf16_t *v,
				  bf16_t *output, const struct kllm_attention_params *p)
{
	float scale = p->scale;
	uint32_t head_dim = p->head_dim;
	uint32_t seq_len = p->seq_len;
	uint32_t num_heads = p->num_heads;
	uint32_t query_len = p->query_len;

	for (uint32_t h = 0; h < num_heads; h++) {
		for (uint32_t qi = 0; qi < query_len; qi++) {
			const bf16_t *q_vec = q + (qi * num_heads + h) * head_dim;

			/* Q * K^T using VDPBF16PS (dot product BF16 pairs → FP32) */
			float scores[seq_len];
			float max_score = -INFINITY;

			for (uint32_t si = 0; si < seq_len; si++) {
				const bf16_t *k_vec = k + (si * num_heads + h) * head_dim;
				__m512 acc = _mm512_setzero_ps();

				uint32_t d = 0;
				for (; d + 32 <= head_dim; d += 32) {
					__m512bh a = (__m512bh)_mm512_loadu_si512(q_vec + d);
					__m512bh b = (__m512bh)_mm512_loadu_si512(k_vec + d);
					acc = _mm512_dpbf16_ps(acc, a, b);
				}

				/* Horizontal sum */
				scores[si] = _mm512_reduce_add_ps(acc) * scale;

				/* Handle remainder */
				for (; d < head_dim; d++)
					scores[si] += bf16_to_f32(q_vec[d]) *
						      bf16_to_f32(k_vec[d]) * scale;

				if (scores[si] > max_score)
					max_score = scores[si];
			}

			/* Softmax (scalar — not the bottleneck) */
			float sum = 0.0f;
			for (uint32_t si = 0; si < seq_len; si++) {
				scores[si] = expf(scores[si] - max_score);
				sum += scores[si];
			}
			for (uint32_t si = 0; si < seq_len; si++)
				scores[si] /= sum;

			/* attn * V (VDPBF16PS for the accumulation) */
			bf16_t *out_vec = output + (qi * num_heads + h) * head_dim;
			for (uint32_t d = 0; d < head_dim; d++) {
				float acc = 0.0f;
				for (uint32_t si = 0; si < seq_len; si++) {
					const bf16_t *v_vec = v + (si * num_heads + h) * head_dim;
					acc += scores[si] * bf16_to_f32(v_vec[d]);
				}
				out_vec[d] = f32_to_bf16(acc);
			}
		}
	}
}
#endif /* __AVX512BF16__ */

/* --- AMX BF16 path (tile operations) --- */

#ifdef __AMX_BF16__
/*
 * AMX tiled attention for the Q*K^T computation.
 * Uses TDPBF16PS for the matmul, then scalar softmax, then TDPBF16PS for attn*V.
 *
 * AMX tile layout:
 *   tmm0-tmm3: accumulator tiles (16x16 FP32)
 *   tmm4-tmm5: source A tiles (16x32 BF16)
 *   tmm6-tmm7: source B tiles (16x32 BF16, transposed)
 */
static void attention_amx_bf16(const bf16_t *q, const bf16_t *k, const bf16_t *v,
			       bf16_t *output, const struct kllm_attention_params *p)
{
	/*
	 * AMX requires tile configuration via LDTILECFG.
	 * For full implementation, we'd configure tiles for the specific
	 * dimensions and use _tile_dpbf16ps for the matmul blocks.
	 *
	 * Placeholder: fall through to AVX-512 BF16 or scalar.
	 * The tile configuration and TDPBF16PS calls would go here.
	 */
#ifdef __AVX512BF16__
	attention_avx512_bf16(q, k, v, output, p);
#else
	attention_scalar(q, k, v, output, p);
#endif
}
#endif /* __AMX_BF16__ */

/* --- Public API --- */

int kllm_ace_attention(const bf16_t *q, const bf16_t *k, const bf16_t *v,
		       bf16_t *output, const uint8_t *scales,
		       const struct kllm_attention_params *params)
{
	if (!q || !k || !v || !output || !params)
		return -1;
	if (params->head_dim == 0 || params->seq_len == 0 || params->num_heads == 0)
		return -1;

	/*
	 * OCP MX block scaling: if scales are provided, dequantize K/V
	 * inline during the matmul. For now, the non-scaled path is
	 * implemented; MX scaling is task 3.5.
	 */
	(void)scales;

	switch (g_isa_level) {
#ifdef __AMX_BF16__
	case ISA_ACE_BF16:
	case ISA_AMX_BF16:
		attention_amx_bf16(q, k, v, output, params);
		break;
#endif
#ifdef __AVX512BF16__
	case ISA_AVX512_BF16:
		attention_avx512_bf16(q, k, v, output, params);
		break;
#endif
	default:
		attention_scalar(q, k, v, output, params);
		break;
	}

	return 0;
}
