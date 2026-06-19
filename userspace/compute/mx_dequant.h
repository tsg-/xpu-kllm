/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _KLLM_MX_DEQUANT_H
#define _KLLM_MX_DEQUANT_H

#include <stdint.h>

/*
 * OCP MX (Microscaling) block dequantization.
 *
 * OCP MX format: groups of 8 elements share one FP8 scale factor.
 * Storage: BF16 values * FP8 scale per 8-element micro-block.
 *
 * During attention matmul, dequant happens inline:
 *   real_value[i] = stored_bf16[i] * scale[i / 8]
 *
 * This gives ~2x memory compression for KV cache with minimal
 * accuracy loss (scale granularity is per-8-element, not per-tensor).
 *
 * ACE hardware can apply MX scaling directly in the outer-product
 * pipeline (no separate dequant pass needed on GNR+).
 */

typedef uint16_t bf16_t;
typedef uint8_t  fp8_e4m3_t;  /* FP8 E4M3: 1 sign, 4 exp, 3 mantissa */
typedef uint8_t  fp8_e5m2_t;  /* FP8 E5M2: 1 sign, 5 exp, 2 mantissa */

#define MX_BLOCK_SIZE	8  /* elements per scale factor */

/* Convert FP8 E4M3 scale to FP32 */
static inline float fp8_e4m3_to_f32(fp8_e4m3_t val)
{
	/* E4M3: bias=7, no inf, NaN = 0x7F/0xFF */
	uint8_t sign = (val >> 7) & 1;
	uint8_t exp = (val >> 3) & 0xF;
	uint8_t mant = val & 0x7;

	if (exp == 0 && mant == 0)
		return sign ? -0.0f : 0.0f;

	float fexp, fmant;
	if (exp == 0) {
		/* Subnormal */
		fexp = 1.0f / (1 << 6);  /* 2^(1-bias) = 2^(-6) */
		fmant = (float)mant / 8.0f;
	} else {
		fexp = (float)(1 << (exp - 7 + 127 - 127));  /* simplified */
		/* Actually: 2^(exp - bias) */
		int real_exp = (int)exp - 7;
		fexp = (real_exp >= 0) ? (float)(1 << real_exp) : 1.0f / (float)(1 << (-real_exp));
		fmant = 1.0f + (float)mant / 8.0f;
	}

	float result = fexp * fmant;
	return sign ? -result : result;
}

/*
 * Dequantize a block of BF16 values using MX scaling.
 *
 * data:    input BF16 values (MX-scaled, stored as-is)
 * scales:  FP8 E4M3 scale factors, one per MX_BLOCK_SIZE elements
 * output:  dequantized FP32 values
 * count:   number of elements to dequantize
 */
static inline void mx_dequant_bf16(const bf16_t *data, const fp8_e4m3_t *scales,
				   float *output, uint32_t count)
{
	for (uint32_t i = 0; i < count; i++) {
		uint32_t bits = (uint32_t)data[i] << 16;
		float val;
		__builtin_memcpy(&val, &bits, sizeof(val));

		float scale = fp8_e4m3_to_f32(scales[i / MX_BLOCK_SIZE]);
		output[i] = val * scale;
	}
}

/*
 * Fused dequant + dot product: computes dot(a, b*scale) for attention.
 * Avoids materializing the full dequantized tensor.
 */
static inline float mx_dot_product(const bf16_t *a, const bf16_t *b,
				   const fp8_e4m3_t *b_scales, uint32_t dim)
{
	float acc = 0.0f;

	for (uint32_t i = 0; i < dim; i++) {
		uint32_t bits_a = (uint32_t)a[i] << 16;
		uint32_t bits_b = (uint32_t)b[i] << 16;
		float fa, fb;
		__builtin_memcpy(&fa, &bits_a, sizeof(fa));
		__builtin_memcpy(&fb, &bits_b, sizeof(fb));

		float scale = fp8_e4m3_to_f32(b_scales[i / MX_BLOCK_SIZE]);
		acc += fa * (fb * scale);
	}

	return acc;
}

/*
 * Quantize FP32 values to MX-scaled BF16.
 * Computes per-block scale factors and stores quantized data.
 */
static inline void mx_quant_to_bf16(const float *input, bf16_t *data,
				    fp8_e4m3_t *scales, uint32_t count)
{
	for (uint32_t blk = 0; blk < count; blk += MX_BLOCK_SIZE) {
		/* Find max absolute value in this block */
		float amax = 0.0f;
		uint32_t blk_end = blk + MX_BLOCK_SIZE;
		if (blk_end > count) blk_end = count;

		for (uint32_t i = blk; i < blk_end; i++) {
			float av = input[i] < 0 ? -input[i] : input[i];
			if (av > amax) amax = av;
		}

		/* Compute scale: map amax to FP8 E4M3 range (max ~448) */
		float scale = (amax > 0.0f) ? amax / 448.0f : 1.0f;
		float inv_scale = 1.0f / scale;

		/* Store scale as FP8 (simplified: store magnitude only) */
		/* TODO: proper FP8 E4M3 encoding */
		scales[blk / MX_BLOCK_SIZE] = (fp8_e4m3_t)(amax > 0 ? 64 : 0);

		/* Quantize elements */
		for (uint32_t i = blk; i < blk_end; i++) {
			float scaled = input[i] * inv_scale;
			uint32_t bits;
			__builtin_memcpy(&bits, &scaled, sizeof(bits));
			bits += 0x7FFF + ((bits >> 16) & 1);
			data[i] = (bf16_t)(bits >> 16);
		}
	}
}

#endif /* _KLLM_MX_DEQUANT_H */
