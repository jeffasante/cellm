// Author: Jeffrey Asante (https://jeffasante.github.io/)
use std::f32;
use rayon::prelude::*;
use half::f16;

#[inline(always)]
fn unpack_i4(packed_row: &[u8], idx: usize) -> f32 {
    let byte = packed_row[idx / 2];
    let nibble = if idx % 2 == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    };
    (nibble as i8 - 8) as f32
}

pub fn rms_norm_f32(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());

    let n = x.len();
    let mut mean_sq = 0.0f32;

    // SIMD-accelerated mean square computation on aarch64
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 16 <= n {
            let v0 = vld1q_f32(x.as_ptr().add(i));
            let v1 = vld1q_f32(x.as_ptr().add(i + 4));
            let v2 = vld1q_f32(x.as_ptr().add(i + 8));
            let v3 = vld1q_f32(x.as_ptr().add(i + 12));
            sum0 = vmlaq_f32(sum0, v0, v0);
            sum1 = vmlaq_f32(sum1, v1, v1);
            sum2 = vmlaq_f32(sum2, v2, v2);
            sum3 = vmlaq_f32(sum3, v3, v3);
            i += 16;
        }
        let partial = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        mean_sq = vgetq_lane_f32(partial, 0) + vgetq_lane_f32(partial, 1) +
                  vgetq_lane_f32(partial, 2) + vgetq_lane_f32(partial, 3);
        while i < n {
            mean_sq += x[i] * x[i];
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for &v in x {
            mean_sq += v * v;
        }
    }

    mean_sq /= n as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();

    // SIMD-accelerated output computation on aarch64
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let scale_vec = vdupq_n_f32(inv_rms);
        let mut i = 0usize;
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let wv = vld1q_f32(weight.as_ptr().add(i));
            let result = vmulq_f32(vmulq_f32(xv, scale_vec), wv);
            vst1q_f32(out.as_mut_ptr().add(i), result);
            i += 4;
        }
        while i < n {
            out[i] = x[i] * inv_rms * weight[i];
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Skip Rayon for small vectors where dispatch overhead dominates.
        if n < 2048 {
            for i in 0..n {
                out[i] = x[i] * inv_rms * weight[i];
            }
        } else {
            out.par_iter_mut().zip(x.par_iter()).zip(weight.par_iter()).for_each(|((o, &xi), &wi)| {
                *o = xi * inv_rms * wi;
            });
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod accelerate {
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        pub fn cblas_sgemv(
            order: i32,
            trans_a: i32,
            m: i32,
            n: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            x: *const f32,
            incx: i32,
            beta: f32,
            y: *mut f32,
            incy: i32,
        );
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
    pub const CBLAS_ROW_MAJOR: i32 = 101;
    pub const CBLAS_NO_TRANS: i32 = 111;
}

pub fn matmul_f32(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        use accelerate::*;
        // Use Accelerate BLAS for matrix-matrix products (prefill, n>1) where
        // tiling and AMX are effective. For n==1 (decode), parallel NEON loops
        // are faster because BLAS sgemv doesn't parallelize well for these sizes.
        if m >= 1 && k >= 64 && n > 1 {
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a.as_ptr(),
                    k as i32,
                    b.as_ptr(),
                    n as i32,
                    0.0,
                    out.as_mut_ptr(),
                    n as i32,
                );
            }
            return;
        }
    }

    if n == 1 {
        // Matrix-vector product - parallelize across rows with NEON on aarch64.
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row = &a[i * k..(i + 1) * k];

            #[cfg(target_arch = "aarch64")]
            {
                let mut acc;
                let mut kk = 0;
                unsafe {
                    use std::arch::aarch64::*;
                    let mut sum0 = vdupq_n_f32(0.0);
                    let mut sum1 = vdupq_n_f32(0.0);
                    let mut sum2 = vdupq_n_f32(0.0);
                    let mut sum3 = vdupq_n_f32(0.0);

                    while kk + 16 <= k {
                        let av0 = vld1q_f32(row.as_ptr().add(kk));
                        let av1 = vld1q_f32(row.as_ptr().add(kk + 4));
                        let av2 = vld1q_f32(row.as_ptr().add(kk + 8));
                        let av3 = vld1q_f32(row.as_ptr().add(kk + 12));
                        let bv0 = vld1q_f32(b.as_ptr().add(kk));
                        let bv1 = vld1q_f32(b.as_ptr().add(kk + 4));
                        let bv2 = vld1q_f32(b.as_ptr().add(kk + 8));
                        let bv3 = vld1q_f32(b.as_ptr().add(kk + 12));
                        sum0 = vmlaq_f32(sum0, av0, bv0);
                        sum1 = vmlaq_f32(sum1, av1, bv1);
                        sum2 = vmlaq_f32(sum2, av2, bv2);
                        sum3 = vmlaq_f32(sum3, av3, bv3);
                        kk += 16;
                    }
                    let res = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
                    acc = vgetq_lane_f32(res, 0) + vgetq_lane_f32(res, 1) + vgetq_lane_f32(res, 2) + vgetq_lane_f32(res, 3);
                }
                while kk < k {
                    acc += row[kk] * b[kk];
                    kk += 1;
                }
                *o = acc;
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += row[kk] * b[kk];
                }
                *o = acc;
            }
        });
    } else {
        // Matrix-matrix product - parallelize across output rows.
        out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
            let a_row = &a[i * k..(i + 1) * k];
            for kk in 0..k {
                let av = a_row[kk];
                let b_row = &b[kk * n..(kk + 1) * n];
                for j in 0..n {
                    out_row[j] += av * b_row[j];
                }
            }
        });
    }
}

pub fn matmul_i8_f32(
    a_i8: &[i8],
    a_scales_f16: &[u16],
    m: usize,
    k: usize,
    b: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(a_i8.len(), m * k);
    debug_assert_eq!(a_scales_f16.len(), m);
    debug_assert_eq!(out.len(), m);

    // For small batch sizes (decode), avoid parallelization overhead
    if m < 8 {
        for i in 0..m {
            let row = &a_i8[i * k..(i + 1) * k];
            let scale = f16::from_bits(a_scales_f16[i]).to_f32();

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let mut sum0 = vdupq_n_f32(0.0);
                let mut sum1 = vdupq_n_f32(0.0);
                let mut sum2 = vdupq_n_f32(0.0);
                let mut sum3 = vdupq_n_f32(0.0);
                let mut i_inner = 0;

                while i_inner + 16 <= k {
                    let wv = vld1q_s8(row.as_ptr().add(i_inner));
                    let xv0 = vld1q_f32(b.as_ptr().add(i_inner));
                    let xv1 = vld1q_f32(b.as_ptr().add(i_inner + 4));
                    let xv2 = vld1q_f32(b.as_ptr().add(i_inner + 8));
                    let xv3 = vld1q_f32(b.as_ptr().add(i_inner + 12));

                    let wv16_low = vmovl_s8(vget_low_s8(wv));
                    let wv16_high = vmovl_s8(vget_high_s8(wv));

                    let w_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wv16_low)));
                    let w_f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wv16_low)));
                    let w_f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wv16_high)));
                    let w_f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wv16_high)));

                    sum0 = vmlaq_f32(sum0, w_f0, xv0);
                    sum1 = vmlaq_f32(sum1, w_f1, xv1);
                    sum2 = vmlaq_f32(sum2, w_f2, xv2);
                    sum3 = vmlaq_f32(sum3, w_f3, xv3);
                    i_inner += 16;
                }
                let res = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
                let mut acc = vgetq_lane_f32(res, 0) + vgetq_lane_f32(res, 1) + vgetq_lane_f32(res, 2) + vgetq_lane_f32(res, 3);
                while i_inner < k {
                    acc += (row[i_inner] as f32) * b[i_inner];
                    i_inner += 1;
                }
                out[i] = acc * scale;
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += (row[kk] as f32) * b[kk];
                }
                out[i] = acc * scale;
            }
        }
    } else {
        // Parallel path for larger batches (prefill)
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row = &a_i8[i * k..(i + 1) * k];
            let scale = f16::from_bits(a_scales_f16[i]).to_f32();

            #[cfg(target_arch = "aarch64")]
            {
                let mut dot = 0.0f32;
                let mut i_inner = 0;
                unsafe {
                    use std::arch::aarch64::*;
                    let mut sum0 = vdupq_n_f32(0.0);
                    let mut sum1 = vdupq_n_f32(0.0);
                    let mut sum2 = vdupq_n_f32(0.0);
                    let mut sum3 = vdupq_n_f32(0.0);

                    while i_inner + 16 <= k {
                        let wv = vld1q_s8(row.as_ptr().add(i_inner));
                        let xv0 = vld1q_f32(b.as_ptr().add(i_inner));
                        let xv1 = vld1q_f32(b.as_ptr().add(i_inner + 4));
                        let xv2 = vld1q_f32(b.as_ptr().add(i_inner + 8));
                        let xv3 = vld1q_f32(b.as_ptr().add(i_inner + 12));

                        let wv16_low = vmovl_s8(vget_low_s8(wv));
                        let wv16_high = vmovl_s8(vget_high_s8(wv));

                        let w_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wv16_low)));
                        let w_f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wv16_low)));
                        let w_f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wv16_high)));
                        let w_f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wv16_high)));

                        sum0 = vmlaq_f32(sum0, w_f0, xv0);
                        sum1 = vmlaq_f32(sum1, w_f1, xv1);
                        sum2 = vmlaq_f32(sum2, w_f2, xv2);
                        sum3 = vmlaq_f32(sum3, w_f3, xv3);
                        i_inner += 16;
                    }
                    let res = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
                    dot = vgetq_lane_f32(res, 0) + vgetq_lane_f32(res, 1) + vgetq_lane_f32(res, 2) + vgetq_lane_f32(res, 3);
                }
                while i_inner < k {
                    dot += (row[i_inner] as f32) * b[i_inner];
                    i_inner += 1;
                }
                *o = dot * scale;
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += (row[kk] as f32) * b[kk];
                }
                *o = acc * scale;
            }
        });
    }
}

pub fn matmul_f16_f32(
    a_f16: &[u16],
    m: usize,
    k: usize,
    b: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(a_f16.len(), m * k);
    debug_assert_eq!(out.len(), m);

    out.par_iter_mut().with_min_len(32).enumerate().for_each(|(i, o)| {
        let row = &a_f16[i * k..(i + 1) * k];

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let mut dot = 0.0f32;
            let mut i_inner = 0;
            unsafe {
                use std::arch::aarch64::*;
                let mut sum0 = vdupq_n_f32(0.0);
                let mut sum1 = vdupq_n_f32(0.0);
                let mut sum2 = vdupq_n_f32(0.0);
                let mut sum3 = vdupq_n_f32(0.0);

                // Convert 4 f16 bit-patterns (u16) → float32x4_t via stable NEON bit manipulation.
                // vcvt_f32_f16 (FCVTL) is correct but requires the unstable stdarch_neon_f16 feature.
                // This manual version correctly handles normal numbers and ±0.
                // Subnormals → 0, infinities/NaNs → undefined (not present in model weights).
                //   sign  = bit 15 of f16 → bit 31 of f32  (shift left 16, mask 0x80000000)
                //   exp   = f16 exp + bias_adjust, then << 13 to f32 position
                //   mant  = f16 mant << 13 to f32 mantissa position
                //   zero  = if exp==0 force to ±0 (mask out normal bits)
                // All via: f32_bits = sign | ((w & 0x7fff + 0x1c000) << 13) & (exp_nonzero_mask)
                #[inline(always)]
                unsafe fn f16x4_to_f32x4(row: *const u16) -> float32x4_t {
                    let h = vld1_u16(row);
                    let w = vmovl_u16(h);
                    let sign = vandq_u32(vshlq_n_u32(w, 16), vdupq_n_u32(0x80000000u32));
                    let normal = vshlq_n_u32(
                        vaddq_u32(vandq_u32(w, vdupq_n_u32(0x7fff)), vdupq_n_u32(0x1c000u32)),
                        13,
                    );
                    let not_zero = vtstq_u32(w, vdupq_n_u32(0x7c00));
                    vreinterpretq_f32_u32(vorrq_u32(sign, vandq_u32(normal, not_zero)))
                }

                while i_inner + 16 <= k {
                    let xv0 = vld1q_f32(b.as_ptr().add(i_inner));
                    let xv1 = vld1q_f32(b.as_ptr().add(i_inner + 4));
                    let xv2 = vld1q_f32(b.as_ptr().add(i_inner + 8));
                    let xv3 = vld1q_f32(b.as_ptr().add(i_inner + 12));

                    let wf0 = f16x4_to_f32x4(row.as_ptr().add(i_inner));
                    let wf1 = f16x4_to_f32x4(row.as_ptr().add(i_inner + 4));
                    let wf2 = f16x4_to_f32x4(row.as_ptr().add(i_inner + 8));
                    let wf3 = f16x4_to_f32x4(row.as_ptr().add(i_inner + 12));

                    sum0 = vmlaq_f32(sum0, wf0, xv0);
                    sum1 = vmlaq_f32(sum1, wf1, xv1);
                    sum2 = vmlaq_f32(sum2, wf2, xv2);
                    sum3 = vmlaq_f32(sum3, wf3, xv3);
                    i_inner += 16;
                }
                let res = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
                dot = vgetq_lane_f32(res, 0) + vgetq_lane_f32(res, 1) + vgetq_lane_f32(res, 2) + vgetq_lane_f32(res, 3);
            }
            while i_inner < k {
                dot += f16::from_bits(row[i_inner]).to_f32() * b[i_inner];
                i_inner += 1;
            }
            *o = dot;
        }

        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += f16::from_bits(row[kk]).to_f32() * b[kk];
            }
            *o = acc;
        }
    });
}

pub fn matmul_i4_f32(
    a_i4: &[u8],
    a_scales_f16: &[u16],
    m: usize,
    k: usize,
    gs: usize,
    b: &[f32],
    out: &mut [f32],
) {
    let row_stride = k.div_ceil(2);
    let spr = a_scales_f16.len() / m;
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let row = &a_i4[i * row_stride..(i + 1) * row_stride];
        let rs = &a_scales_f16[i * spr..(i + 1) * spr];
        let mut dot = 0.0f32;

        // Precompute scales for this row to avoid repeated f16->f32 conversions
        let num_groups = (k + gs - 1) / gs;
        let mut scales_buf = [0.0f32; 512];
        let scales_to_copy = num_groups.min(512);
        for g in 0..scales_to_copy {
            scales_buf[g] = f16::from_bits(rs[g]).to_f32();
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;

            // Accumulate in 4 f32 vectors for better ILP
            let mut sum0 = vdupq_n_f32(0.0);
            let mut sum1 = vdupq_n_f32(0.0);
            let mut sum2 = vdupq_n_f32(0.0);
            let mut sum3 = vdupq_n_f32(0.0);

            let mut j = 0usize;

            // Process 8 bytes (16 i4 values) at a time
            // Layout: byte[j/2] contains element j (low nibble) and j+1 (high nibble)
            while j + 16 <= k {
                let byte_base = j / 2;

                // Load 8 bytes = 16 i4 values
                let bytes = vld1_u8(row.as_ptr().add(byte_base));

                // Extract nibbles: low nibble = even indices, high nibble = odd indices
                let mask_lo = vdup_n_u8(0x0f);
                let lo_nibbles = vand_u8(bytes, mask_lo);   // elements j, j+2, j+4, ..., j+14
                let hi_nibbles = vshr_n_u8(bytes, 4);       // elements j+1, j+3, j+5, ..., j+15

                // Convert u8 to i8 by subtracting 8 (zero point)
                let zero_pt = vdup_n_s8(8);
                let lo_i8 = vsub_s8(vreinterpret_s8_u8(lo_nibbles), zero_pt);
                let hi_i8 = vsub_s8(vreinterpret_s8_u8(hi_nibbles), zero_pt);

                // Widen i8 to i16
                let lo_i16 = vmovl_s8(lo_i8);  // int16x8_t
                let hi_i16 = vmovl_s8(hi_i8);  // int16x8_t

                // Widen i16 to i32 and convert to f32
                let lo_f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_i16)));   // 4 floats: j, j+2, j+4, j+6
                let lo_f32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_i16)));  // 4 floats: j+8, j+10, j+12, j+14
                let hi_f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_i16)));   // 4 floats: j+1, j+3, j+5, j+7
                let hi_f32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_i16)));  // 4 floats: j+9, j+11, j+13, j+15

                // Load 16 b values and deinterleave
                let b_0 = vld1q_f32(b.as_ptr().add(j));      // b[j], b[j+1], b[j+2], b[j+3]
                let b_1 = vld1q_f32(b.as_ptr().add(j + 4));  // b[j+4], b[j+5], b[j+6], b[j+7]
                let b_2 = vld1q_f32(b.as_ptr().add(j + 8));  // b[j+8], b[j+9], b[j+10], b[j+11]
                let b_3 = vld1q_f32(b.as_ptr().add(j + 12)); // b[j+12], b[j+13], b[j+14], b[j+15]

                // Deinterleave: even indices go with lo, odd indices go with hi
                let b_lo_0 = vuzp1q_f32(b_0, b_1);  // b[j], b[j+2], b[j+4], b[j+6]
                let b_hi_0 = vuzp2q_f32(b_0, b_1);  // b[j+1], b[j+3], b[j+5], b[j+7]
                let b_lo_1 = vuzp1q_f32(b_2, b_3);  // b[j+8], b[j+10], b[j+12], b[j+14]
                let b_hi_1 = vuzp2q_f32(b_2, b_3);  // b[j+9], b[j+11], b[j+13], b[j+15]

                // Get scale for this chunk
                let scale_idx = j / gs;
                let scale = scales_buf[scale_idx.min(511)];
                let scale_vec = vdupq_n_f32(scale);

                // Apply scale and accumulate
                let lo_scaled_0 = vmulq_f32(lo_f32_0, scale_vec);
                let lo_scaled_1 = vmulq_f32(lo_f32_1, scale_vec);
                let hi_scaled_0 = vmulq_f32(hi_f32_0, scale_vec);
                let hi_scaled_1 = vmulq_f32(hi_f32_1, scale_vec);

                sum0 = vmlaq_f32(sum0, lo_scaled_0, b_lo_0);
                sum1 = vmlaq_f32(sum1, lo_scaled_1, b_lo_1);
                sum2 = vmlaq_f32(sum2, hi_scaled_0, b_hi_0);
                sum3 = vmlaq_f32(sum3, hi_scaled_1, b_hi_1);

                j += 16;
            }

            // Reduce sums
            let partial = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
            dot = vgetq_lane_f32(partial, 0) + vgetq_lane_f32(partial, 1) +
                  vgetq_lane_f32(partial, 2) + vgetq_lane_f32(partial, 3);

            // Handle remaining elements with scalar code
            while j < k {
                let b_idx = j / 2;
                let n = if j % 2 == 0 { row[b_idx] & 0xf } else { row[b_idx] >> 4 };
                let q = (n as i8) - 8;
                let scale = scales_buf[(j / gs).min(511)];
                dot += (q as f32) * scale * b[j];
                j += 1;
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            for j in 0..k {
                let b_idx = j / 2;
                let n = if j % 2 == 0 { row[b_idx] & 0xf } else { row[b_idx] >> 4 };
                let q = (n as i8) - 8;
                let scale = scales_buf[(j / gs).min(511)];
                dot += (q as f32) * scale * b[j];
            }
        }

        *o = dot;
    });
}

pub fn softmax_f32_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 {
        return;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

pub fn rope_non_interleaved_inplace_f32(x: &mut [f32], _n_heads: usize, head_dim: usize, rotary_dim: usize, pos: usize, theta: f32) {
    let half = rotary_dim / 2;
    // Skip Rayon for small head counts where dispatch overhead dominates.
    if x.len() < 2048 {
        for head in x.chunks_exact_mut(head_dim) {
            for i in 0..half {
                let inv_freq = theta.powf(-(2.0 * i as f32) / rotary_dim as f32);
                let angle = pos as f32 * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = head[i];
                let x1 = head[half + i];
                head[i] = x0 * cos - x1 * sin;
                head[half + i] = x1 * cos + x0 * sin;
            }
        }
    } else {
        x.par_chunks_exact_mut(head_dim).for_each(|head| {
            for i in 0..half {
                let inv_freq = theta.powf(-(2.0 * i as f32) / rotary_dim as f32);
                let angle = pos as f32 * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = head[i];
                let x1 = head[half + i];
                head[i] = x0 * cos - x1 * sin;
                head[half + i] = x1 * cos + x0 * sin;
            }
        });
    }
}

pub fn rope_interleaved_inplace_f32(x: &mut [f32], _n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;
    x.par_chunks_exact_mut(head_dim).for_each(|head| {
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = pos as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = head[2 * i];
            let x1 = head[2 * i + 1];
            head[2 * i] = x0 * cos - x1 * sin;
            head[2 * i + 1] = x1 * cos + x0 * sin;
        }
    });
}

pub fn attention_single_token_gqa_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let qkv_ratio = n_heads / n_kv_heads;

    // Parallelize across attention heads.
    out.par_chunks_exact_mut(head_dim).enumerate().for_each(|(h, out_h)| {
        let kv_h = h / qkv_ratio;
        let qh = &q[h * head_dim..(h + 1) * head_dim];

        // We need a thread-local score buffer.
        // For simplicity in this kernel, we'll allocate it, but ideally it's passed in.
        let mut scores = vec![0.0f32; seq];
        for t in 0..seq {
            let kt_base = (t * n_kv_heads + kv_h) * head_dim;
            let kt = &k[kt_base..kt_base + head_dim];
            let mut dot = 0.0f32;

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let mut sumv = vdupq_n_f32(0.0);
                let mut i = 0;
                while i + 4 <= head_dim {
                    let qv = vld1q_f32(qh.as_ptr().add(i));
                    let kv = vld1q_f32(kt.as_ptr().add(i));
                    sumv = vmlaq_f32(sumv, qv, kv);
                    i += 4;
                }
                dot = vgetq_lane_f32(sumv, 0) + vgetq_lane_f32(sumv, 1) + vgetq_lane_f32(sumv, 2) + vgetq_lane_f32(sumv, 3);
                while i < head_dim {
                    dot += qh[i] * kt[i];
                    i += 1;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..head_dim {
                    dot += qh[i] * kt[i];
                }
            }
            scores[t] = dot * scale;
        }

        softmax_f32_inplace(&mut scores);

        out_h.fill(0.0);
        for t in 0..seq {
            let vt_base = (t * n_kv_heads + kv_h) * head_dim;
            let vt = &v[vt_base..vt_base + head_dim];
            let w = scores[t];

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let wv = vdupq_n_f32(w);
                let mut i = 0;
                while i + 4 <= head_dim {
                    let ov = vld1q_f32(out_h.as_ptr().add(i));
                    let vv = vld1q_f32(vt.as_ptr().add(i));
                    vst1q_f32(out_h.as_mut_ptr().add(i), vmlaq_f32(ov, wv, vv));
                    i += 4;
                }
                while i < head_dim {
                    out_h[i] += w * vt[i];
                    i += 1;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..head_dim {
                    out_h[i] += w * vt[i];
                }
            }
        }
    });
}
