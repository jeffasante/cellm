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

/// Optimized INT8 GEMV (matrix-vector) for single-token decode.
/// Keeps weights quantized and fuses dequantization into the dot product.
/// Weight layout: [out_dim, in_dim] row-major INT8 with per-row f16 scales.
pub fn gemv_i8_f32(
    weight_i8: &[i8],
    scales_f16: &[u16],
    input: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    debug_assert_eq!(weight_i8.len(), out_dim * in_dim);
    debug_assert_eq!(scales_f16.len(), out_dim);
    debug_assert_eq!(input.len(), in_dim);
    debug_assert_eq!(out.len(), out_dim);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;

        // Process multiple output rows in parallel for better ILP
        let mut row = 0usize;

        // Process 4 rows at a time
        while row + 4 <= out_dim {
            let w0 = &weight_i8[row * in_dim..(row + 1) * in_dim];
            let w1 = &weight_i8[(row + 1) * in_dim..(row + 2) * in_dim];
            let w2 = &weight_i8[(row + 2) * in_dim..(row + 3) * in_dim];
            let w3 = &weight_i8[(row + 3) * in_dim..(row + 4) * in_dim];

            let s0 = f16::from_bits(scales_f16[row]).to_f32();
            let s1 = f16::from_bits(scales_f16[row + 1]).to_f32();
            let s2 = f16::from_bits(scales_f16[row + 2]).to_f32();
            let s3 = f16::from_bits(scales_f16[row + 3]).to_f32();

            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let mut i = 0usize;
            while i + 16 <= in_dim {
                // Load input vector (shared across all 4 rows)
                let xv0 = vld1q_f32(input.as_ptr().add(i));
                let xv1 = vld1q_f32(input.as_ptr().add(i + 4));
                let xv2 = vld1q_f32(input.as_ptr().add(i + 8));
                let xv3 = vld1q_f32(input.as_ptr().add(i + 12));

                // Load and process weight row 0
                let wv0 = vld1q_s8(w0.as_ptr().add(i));
                let w0_16_low = vmovl_s8(vget_low_s8(wv0));
                let w0_16_high = vmovl_s8(vget_high_s8(wv0));
                let w0_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_16_low)));
                let w0_f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_16_low)));
                let w0_f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_16_high)));
                let w0_f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_16_high)));
                acc0 = vmlaq_f32(acc0, w0_f0, xv0);
                acc0 = vmlaq_f32(acc0, w0_f1, xv1);
                acc0 = vmlaq_f32(acc0, w0_f2, xv2);
                acc0 = vmlaq_f32(acc0, w0_f3, xv3);

                // Load and process weight row 1
                let wv1 = vld1q_s8(w1.as_ptr().add(i));
                let w1_16_low = vmovl_s8(vget_low_s8(wv1));
                let w1_16_high = vmovl_s8(vget_high_s8(wv1));
                let w1_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1_16_low)));
                let w1_f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1_16_low)));
                let w1_f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1_16_high)));
                let w1_f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1_16_high)));
                acc1 = vmlaq_f32(acc1, w1_f0, xv0);
                acc1 = vmlaq_f32(acc1, w1_f1, xv1);
                acc1 = vmlaq_f32(acc1, w1_f2, xv2);
                acc1 = vmlaq_f32(acc1, w1_f3, xv3);

                // Load and process weight row 2
                let wv2 = vld1q_s8(w2.as_ptr().add(i));
                let w2_16_low = vmovl_s8(vget_low_s8(wv2));
                let w2_16_high = vmovl_s8(vget_high_s8(wv2));
                let w2_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w2_16_low)));
                let w2_f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w2_16_low)));
                let w2_f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w2_16_high)));
                let w2_f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w2_16_high)));
                acc2 = vmlaq_f32(acc2, w2_f0, xv0);
                acc2 = vmlaq_f32(acc2, w2_f1, xv1);
                acc2 = vmlaq_f32(acc2, w2_f2, xv2);
                acc2 = vmlaq_f32(acc2, w2_f3, xv3);

                // Load and process weight row 3
                let wv3 = vld1q_s8(w3.as_ptr().add(i));
                let w3_16_low = vmovl_s8(vget_low_s8(wv3));
                let w3_16_high = vmovl_s8(vget_high_s8(wv3));
                let w3_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w3_16_low)));
                let w3_f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w3_16_low)));
                let w3_f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w3_16_high)));
                let w3_f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w3_16_high)));
                acc3 = vmlaq_f32(acc3, w3_f0, xv0);
                acc3 = vmlaq_f32(acc3, w3_f1, xv1);
                acc3 = vmlaq_f32(acc3, w3_f2, xv2);
                acc3 = vmlaq_f32(acc3, w3_f3, xv3);

                i += 16;
            }

            // Reduce and apply scale
            let r0 = vaddvq_f32(acc0) * s0;
            let r1 = vaddvq_f32(acc1) * s1;
            let r2 = vaddvq_f32(acc2) * s2;
            let r3 = vaddvq_f32(acc3) * s3;

            // Handle remaining elements
            let mut tail0 = 0.0f32;
            let mut tail1 = 0.0f32;
            let mut tail2 = 0.0f32;
            let mut tail3 = 0.0f32;
            while i < in_dim {
                let x = input[i];
                tail0 += (w0[i] as f32) * x;
                tail1 += (w1[i] as f32) * x;
                tail2 += (w2[i] as f32) * x;
                tail3 += (w3[i] as f32) * x;
                i += 1;
            }

            out[row] = r0 + tail0 * s0;
            out[row + 1] = r1 + tail1 * s1;
            out[row + 2] = r2 + tail2 * s2;
            out[row + 3] = r3 + tail3 * s3;

            row += 4;
        }

        // Handle remaining rows
        while row < out_dim {
            let w = &weight_i8[row * in_dim..(row + 1) * in_dim];
            let scale = f16::from_bits(scales_f16[row]).to_f32();

            let mut acc = vdupq_n_f32(0.0);
            let mut i = 0usize;
            while i + 16 <= in_dim {
                let xv0 = vld1q_f32(input.as_ptr().add(i));
                let xv1 = vld1q_f32(input.as_ptr().add(i + 4));
                let xv2 = vld1q_f32(input.as_ptr().add(i + 8));
                let xv3 = vld1q_f32(input.as_ptr().add(i + 12));

                let wv = vld1q_s8(w.as_ptr().add(i));
                let w16_low = vmovl_s8(vget_low_s8(wv));
                let w16_high = vmovl_s8(vget_high_s8(wv));
                let wf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_low)));
                let wf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_low)));
                let wf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_high)));
                let wf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_high)));

                acc = vmlaq_f32(acc, wf0, xv0);
                acc = vmlaq_f32(acc, wf1, xv1);
                acc = vmlaq_f32(acc, wf2, xv2);
                acc = vmlaq_f32(acc, wf3, xv3);
                i += 16;
            }

            let mut sum = vaddvq_f32(acc);
            while i < in_dim {
                sum += (w[i] as f32) * input[i];
                i += 1;
            }
            out[row] = sum * scale;
            row += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for row in 0..out_dim {
            let w = &weight_i8[row * in_dim..(row + 1) * in_dim];
            let scale = f16::from_bits(scales_f16[row]).to_f32();
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += (w[i] as f32) * input[i];
            }
            out[row] = sum * scale;
        }
    }
}

/// Quantize an f32 activation vector to per-tensor symmetric INT8.
///
/// Returns the scale `s` such that `x[i] ~= q[i] as f32 * s`. When the input is
/// all zeros the scale is 0 and every quantized value is 0, which keeps the
/// downstream `acc * w_scale * x_scale` product exact.
pub fn quantize_activation_i8(x: &[f32], q: &mut [i8]) -> f32 {
    debug_assert_eq!(x.len(), q.len());

    let mut amax = 0.0f32;
    for &v in x {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    if amax == 0.0 || !amax.is_finite() {
        q.fill(0);
        return 0.0;
    }

    let scale = amax / 127.0;
    let inv = 1.0 / scale;
    for (qi, &v) in q.iter_mut().zip(x.iter()) {
        // round-half-away-from-zero, clamped to the symmetric int8 range
        *qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// `acc += sdot(a, b)` — ARMv8.2 SDOT via inline asm.
///
/// The `vdotq_s32` intrinsic is still unstable on stable Rust, so the instruction
/// is emitted directly. Callers must have verified `dotprod` at runtime.
///
/// `dotprod` is not in the baseline feature set for every aarch64 target (notably
/// `aarch64-apple-ios`), so the attribute is required for the assembler to accept
/// `sdot` at all — without it this fails to build rather than falling back.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn sdot_s32(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut out = acc;
    std::arch::asm!(
        "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
        inout(vreg) out,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    out
}

/// SDOT-accelerated INT8 x INT8 GEMV over a contiguous row range.
///
/// `weight_i8` is the full `[out_dim, in_dim]` row-major matrix; `row_off` is the
/// index of the first row covered by `out` / `scales_f16`.
///
/// # Safety
/// The caller must have verified `dotprod` via [`has_i8_dotprod`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn gemv_i8_dot_rows(
    weight_i8: &[i8],
    scales_f16: &[u16],
    xq: &[i8],
    x_scale: f32,
    out: &mut [f32],
    row_off: usize,
    in_dim: usize,
) {
    use std::arch::aarch64::*;

    let wp = weight_i8.as_ptr();
    let xp = xq.as_ptr();
    let n_rows = out.len();
    let mut r = 0usize;

    // 4 rows at a time: one activation load feeds four independent SDOT chains.
    while r + 4 <= n_rows {
        let b0 = (row_off + r) * in_dim;
        let (b1, b2, b3) = (b0 + in_dim, b0 + 2 * in_dim, b0 + 3 * in_dim);

        let mut a0 = vdupq_n_s32(0);
        let mut a1 = vdupq_n_s32(0);
        let mut a2 = vdupq_n_s32(0);
        let mut a3 = vdupq_n_s32(0);

        let mut i = 0usize;
        while i + 16 <= in_dim {
            let xv = vld1q_s8(xp.add(i));
            a0 = sdot_s32(a0, vld1q_s8(wp.add(b0 + i)), xv);
            a1 = sdot_s32(a1, vld1q_s8(wp.add(b1 + i)), xv);
            a2 = sdot_s32(a2, vld1q_s8(wp.add(b2 + i)), xv);
            a3 = sdot_s32(a3, vld1q_s8(wp.add(b3 + i)), xv);
            i += 16;
        }

        let mut acc = [
            vaddvq_s32(a0),
            vaddvq_s32(a1),
            vaddvq_s32(a2),
            vaddvq_s32(a3),
        ];
        while i < in_dim {
            let xi = *xp.add(i) as i32;
            acc[0] += *wp.add(b0 + i) as i32 * xi;
            acc[1] += *wp.add(b1 + i) as i32 * xi;
            acc[2] += *wp.add(b2 + i) as i32 * xi;
            acc[3] += *wp.add(b3 + i) as i32 * xi;
            i += 1;
        }

        for j in 0..4 {
            let ws = f16::from_bits(scales_f16[r + j]).to_f32();
            out[r + j] = acc[j] as f32 * ws * x_scale;
        }
        r += 4;
    }

    while r < n_rows {
        let base = (row_off + r) * in_dim;
        let mut a0 = vdupq_n_s32(0);
        let mut i = 0usize;
        while i + 16 <= in_dim {
            a0 = sdot_s32(a0, vld1q_s8(wp.add(base + i)), vld1q_s8(xp.add(i)));
            i += 16;
        }
        let mut acc = vaddvq_s32(a0);
        while i < in_dim {
            acc += *wp.add(base + i) as i32 * *xp.add(i) as i32;
            i += 1;
        }
        out[r] = acc as f32 * f16::from_bits(scales_f16[r]).to_f32() * x_scale;
        r += 1;
    }
}

#[cfg(target_arch = "aarch64")]
thread_local! {
    /// Reusable INT8 activation buffer, so a per-token GEMV never allocates.
    static ACT_SCRATCH: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Size rayon's global pool for single-stream decode.
///
/// Decode GEMVs are memory-bandwidth bound, so one core already reaches roughly
/// half of peak throughput and extra workers mostly add fork/join barriers. On
/// Apple silicon the efficiency cores also straggle, stalling every join. Using
/// only the performance cores measured fastest end to end.
///
/// No-op if `RAYON_NUM_THREADS` is set or the pool is already initialised.
pub fn init_decode_thread_pool() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }

    let threads = decode_thread_count();
    // An existing global pool is fine; this is a best-effort default.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
}

/// Number of worker threads to use for decode.
fn decode_thread_count() -> usize {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // hw.perflevel0 is the performance-core cluster on Apple silicon.
        if let Some(p) = sysctl_usize("hw.perflevel0.logicalcpu") {
            if p >= 2 {
                return p.min(4);
            }
        }
    }
    (num_cpus_available() / 2).clamp(2, 4)
}

fn num_cpus_available() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn sysctl_usize(name: &str) -> Option<usize> {
    use std::ffi::CString;

    let cname = CString::new(name).ok()?;
    let mut value: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    // SAFETY: `cname` is NUL-terminated and `value`/`len` describe a valid i32 buffer.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            &mut value as *mut i32 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && value > 0 {
        Some(value as usize)
    } else {
        None
    }
}

/// True when the CPU can run the W8A8 SDOT path.
#[inline]
pub fn has_i8_dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// W8A8 INT8 GEMV: quantizes the activation vector to INT8 and uses ARM SDOT
/// integer dot products, avoiding the int8 -> f32 widening of [`gemv_i8_f32`].
///
/// Falls back to [`gemv_i8_f32`] when SDOT is unavailable.
pub fn gemv_i8_w8a8(
    weight_i8: &[i8],
    scales_f16: &[u16],
    input: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    debug_assert_eq!(weight_i8.len(), out_dim * in_dim);
    debug_assert_eq!(scales_f16.len(), out_dim);
    debug_assert_eq!(input.len(), in_dim);
    debug_assert_eq!(out.len(), out_dim);

    #[cfg(target_arch = "aarch64")]
    {
        if has_i8_dotprod() {
            // The activation scratch is thread-local and stays borrowed across the
            // rayon region below. If a caller invokes this kernel from inside its
            // own parallel loop, work stealing can land a nested task on this same
            // worker thread, which would re-enter the borrow. Fall back to a local
            // buffer in that case instead of panicking.
            let handled = ACT_SCRATCH.with(|cell| match cell.try_borrow_mut() {
                Ok(mut xq) => {
                    if xq.len() < in_dim {
                        xq.resize(in_dim, 0);
                    }
                    gemv_i8_w8a8_with_scratch(
                        weight_i8,
                        scales_f16,
                        input,
                        out,
                        out_dim,
                        in_dim,
                        &mut xq[..in_dim],
                    );
                    true
                }
                Err(_) => false,
            });
            if !handled {
                let mut xq = vec![0i8; in_dim];
                gemv_i8_w8a8_with_scratch(
                    weight_i8, scales_f16, input, out, out_dim, in_dim, &mut xq,
                );
            }
            return;
        }
    }

    gemv_i8_f32(weight_i8, scales_f16, input, out, out_dim, in_dim);
}

#[cfg(target_arch = "aarch64")]
fn gemv_i8_w8a8_with_scratch(
    weight_i8: &[i8],
    scales_f16: &[u16],
    input: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    xq: &mut [i8],
) {
    let x_scale = quantize_activation_i8(input, xq);
    if x_scale == 0.0 {
        out.fill(0.0);
        return;
    }

    // This kernel is memory-bandwidth bound (one weight byte per MAC), so
    // a single core already reaches ~half of peak. Extra threads only pay
    // off when each task is large enough to dwarf the fork/join cost, and
    // oversubscribing lets work stealing balance the P/E core mix.
    let threads = rayon::current_num_threads().max(1);
    const MIN_MACS_PER_TASK: usize = 512 * 1024;
    let rows_per_task = out_dim
        .div_ceil(threads * 8)
        .max(MIN_MACS_PER_TASK.div_ceil(in_dim.max(1)))
        .next_multiple_of(4);

    // Below this the rayon fork/join costs more than the work it saves.
    const PAR_MAC_THRESHOLD: usize = MIN_MACS_PER_TASK * 2;
    if out_dim <= rows_per_task || out_dim * in_dim < PAR_MAC_THRESHOLD {
        unsafe {
            gemv_i8_dot_rows(weight_i8, scales_f16, xq, x_scale, out, 0, in_dim);
        }
    } else {
        let xq: &[i8] = xq;
        out.par_chunks_mut(rows_per_task)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let row_off = ci * rows_per_task;
                let s = &scales_f16[row_off..row_off + chunk.len()];
                unsafe {
                    gemv_i8_dot_rows(weight_i8, s, xq, x_scale, chunk, row_off, in_dim);
                }
            });
    }
}

/// Batched W8A8 INT8 GEMM: the same computation as calling [`gemv_i8_w8a8`]
/// once per token, but reading each weight byte once for the whole batch
/// instead of once per token.
///
/// A GEMV touches one weight byte per multiply-accumulate, so it is limited by
/// memory bandwidth rather than arithmetic. Prefilling a 747-token prompt this
/// way streams the entire model 747 times — hundreds of gigabytes to do work
/// that only needs the weights once. Holding `n_tokens` activation columns in
/// registers while a weight row is loaded amortises that read across the whole
/// batch and turns the loop arithmetic-bound, where the cores are idle today.
///
/// Each token keeps its own activation scale, and accumulation stays in i32 in
/// the same order as the single-token kernel, so results are bit-identical to
/// the per-token path rather than merely close.
///
/// `input` is `[n_tokens, in_dim]` row-major; `out` is `[n_tokens, out_dim]`.
/// Falls back to a per-token loop when SDOT is unavailable.
pub fn gemm_i8_w8a8(
    weight_i8: &[i8],
    scales_f16: &[u16],
    input: &[f32],
    out: &mut [f32],
    n_tokens: usize,
    out_dim: usize,
    in_dim: usize,
) {
    debug_assert_eq!(weight_i8.len(), out_dim * in_dim);
    debug_assert_eq!(scales_f16.len(), out_dim);
    debug_assert_eq!(input.len(), n_tokens * in_dim);
    debug_assert_eq!(out.len(), n_tokens * out_dim);

    if n_tokens == 0 {
        return;
    }
    if n_tokens == 1 {
        gemv_i8_w8a8(weight_i8, scales_f16, input, out, out_dim, in_dim);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if has_i8_dotprod() {
            // Quantize every token up front: the row loop below reuses these
            // columns for all `out_dim` rows, so this cost is paid once per
            // batch rather than once per row.
            let mut xq = vec![0i8; n_tokens * in_dim];
            let mut x_scales = vec![0.0f32; n_tokens];
            for t in 0..n_tokens {
                x_scales[t] = quantize_activation_i8(
                    &input[t * in_dim..(t + 1) * in_dim],
                    &mut xq[t * in_dim..(t + 1) * in_dim],
                );
            }

            // Split by weight rows, not tokens: every task then walks a
            // disjoint, contiguous slice of the weight matrix, and each task's
            // share of the weights is read exactly once for all tokens.
            let threads = rayon::current_num_threads().max(1);
            let rows_per_task = out_dim.div_ceil(threads).max(4).next_multiple_of(4);

            let xq: &[i8] = &xq;
            let x_scales: &[f32] = &x_scales;

            // `out` is token-major but the work is split row-major, so hand
            // each task the row range and let it stride across tokens.
            let row_chunks: Vec<(usize, usize)> = (0..out_dim)
                .step_by(rows_per_task)
                .map(|r0| (r0, (r0 + rows_per_task).min(out_dim)))
                .collect();

            let out_ptr = SendPtr(out.as_mut_ptr());
            row_chunks.into_par_iter().for_each(|(r0, r1)| {
                let out_ptr = &out_ptr;
                // SAFETY: row ranges are disjoint, so the `(t, r)` cells written
                // here are written by no other task.
                unsafe {
                    gemm_i8_dot_tile(
                        weight_i8,
                        &scales_f16[r0..r1],
                        xq,
                        x_scales,
                        out_ptr.0,
                        r0,
                        r1,
                        n_tokens,
                        out_dim,
                        in_dim,
                    );
                }
            });
            return;
        }
    }

    for t in 0..n_tokens {
        gemv_i8_w8a8(
            weight_i8,
            scales_f16,
            &input[t * in_dim..(t + 1) * in_dim],
            &mut out[t * out_dim..(t + 1) * out_dim],
            out_dim,
            in_dim,
        );
    }
}

/// Deinterleave an INT8 activation vector into even and odd halves.
///
/// A packed i4 weight byte holds element `2i` in its low nibble and `2i+1` in
/// its high nibble. Masking and shifting a 16-byte load therefore yields two
/// vectors of *strided* weights, which only line up with the activations if the
/// activations are strided the same way. Doing that shuffle on the activation
/// once per GEMV — rather than on every weight row — keeps the inner loop pure
/// load-and-SDOT, which matters because the weights are the bandwidth cost.
///
/// `even[i] = xq[2i]`, `odd[i] = xq[2i + 1]`.
#[cfg(target_arch = "aarch64")]
fn deinterleave_i8(xq: &[i8], even: &mut [i8], odd: &mut [i8]) {
    use std::arch::aarch64::*;

    let half = xq.len() / 2;
    debug_assert!(even.len() >= half && odd.len() >= half);

    let mut i = 0usize;
    // SAFETY: each iteration reads 32 bytes from `xq` and writes 16 to each
    // half, all bounds-checked by the `i + 16 <= half` guard.
    unsafe {
        let xp = xq.as_ptr();
        while i + 16 <= half {
            let pair = vld2q_s8(xp.add(i * 2));
            vst1q_s8(even.as_mut_ptr().add(i), pair.0);
            vst1q_s8(odd.as_mut_ptr().add(i), pair.1);
            i += 16;
        }
    }
    while i < half {
        even[i] = xq[i * 2];
        odd[i] = xq[i * 2 + 1];
        i += 1;
    }
}

/// SDOT-accelerated INT4 x INT8 GEMV over a contiguous row range.
///
/// Weights are the `"i4"` on-disk layout: byte-packed two-per-byte, element `2i`
/// in the low nibble and `2i+1` in the high nibble, stored biased by +8 so the
/// nibble range `0..15` maps to `-7..=7`. Scales are per group of `group_size`
/// consecutive input elements, laid out `[out_dim, groups_per_row]`.
///
/// The `+8` bias is removed algebraically rather than per element. Since
/// `sum((n_j - 8) * x_j) == sum(n_j * x_j) - 8 * sum(x_j)` over a group, and the
/// group activation sums are the same for every output row, the correction costs
/// one multiply per group instead of a `vsub` in the inner loop. That keeps the
/// loop at two SDOTs per 16 weight bytes — one for the even lane, one for the odd.
///
/// # Safety
/// The caller must have verified `dotprod` via [`has_i8_dotprod`], must have
/// checked `in_dim % group_size == 0` and `group_size % 32 == 0`, and
/// `xq_even`/`xq_odd` must hold at least `in_dim / 2` entries each.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_i4_dot_rows(
    weight_i4: &[u8],
    scales_f16: &[u16],
    xq_even: &[i8],
    xq_odd: &[i8],
    x_scale: f32,
    x_group_sums: &[i32],
    out: &mut [f32],
    row_off: usize,
    in_dim: usize,
    group_size: usize,
) {
    use std::arch::aarch64::*;

    let row_stride = in_dim / 2;
    let bytes_per_group = group_size / 2;
    let groups_per_row = in_dim / group_size;
    let wp = weight_i4.as_ptr();
    let ep = xq_even.as_ptr();
    let op = xq_odd.as_ptr();
    let lo_mask = vdupq_n_u8(0x0f);

    for r in 0..out.len() {
        let base = (row_off + r) * row_stride;
        let srow = r * groups_per_row;
        let mut dot = 0.0f32;

        for g in 0..groups_per_row {
            let gb = g * bytes_per_group;
            let mut acc_e = vdupq_n_s32(0);
            let mut acc_o = vdupq_n_s32(0);

            // 16 packed bytes = 32 weights per iteration.
            let mut b = 0usize;
            while b + 16 <= bytes_per_group {
                let packed = vld1q_u8(wp.add(base + gb + b));
                let lo = vreinterpretq_s8_u8(vandq_u8(packed, lo_mask));
                let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(packed));
                acc_e = sdot_s32(acc_e, lo, vld1q_s8(ep.add(gb + b)));
                acc_o = sdot_s32(acc_o, hi, vld1q_s8(op.add(gb + b)));
                b += 16;
            }

            let acc = vaddvq_s32(acc_e) + vaddvq_s32(acc_o);
            // Undo the +8 storage bias for the whole group at once.
            let gs = f16::from_bits(scales_f16[srow + g]).to_f32();
            dot += (acc - 8 * x_group_sums[g]) as f32 * gs;
        }

        out[r] = dot * x_scale;
    }
}

/// W4A8 INT4 GEMV: quantizes the activation to INT8 and dots it against packed
/// 4-bit weights with ARM SDOT, without materialising the weights as f32.
///
/// `weight_i4` is `[out_dim, in_dim / 2]` row-major and `scales_f16` is
/// `[out_dim, in_dim / group_size]`, matching the converter's `--quantize-int4`
/// output. A single scale per row leaves only 15 levels to cover a whole
/// 1024-wide row, which measurably degrades output; grouping keeps the step size
/// local to a slice of the row and costs one f16 per 64 weights.
///
/// Falls back to a scalar dequant-and-dot loop when SDOT is unavailable or the
/// dimensions do not divide evenly.
pub fn gemv_i4_w4a8(
    weight_i4: &[u8],
    scales_f16: &[u16],
    input: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) {
    debug_assert_eq!(input.len(), in_dim);
    debug_assert_eq!(out.len(), out_dim);

    #[cfg(target_arch = "aarch64")]
    {
        if has_i8_dotprod() && group_size % 32 == 0 && in_dim % group_size == 0 {
            let half = in_dim / 2;
            let mut xq = vec![0i8; in_dim];
            let x_scale = quantize_activation_i8(input, &mut xq);
            if x_scale == 0.0 {
                out.fill(0.0);
                return;
            }

            let mut even = vec![0i8; half];
            let mut odd = vec![0i8; half];
            deinterleave_i8(&xq, &mut even, &mut odd);
            let x_group_sums: Vec<i32> = xq
                .chunks(group_size)
                .map(|g| g.iter().map(|&v| v as i32).sum())
                .collect();

            let threads = rayon::current_num_threads().max(1);
            const MIN_MACS_PER_TASK: usize = 512 * 1024;
            let rows_per_task = out_dim
                .div_ceil(threads * 8)
                .max(MIN_MACS_PER_TASK.div_ceil(in_dim.max(1)))
                .next_multiple_of(4);
            const PAR_MAC_THRESHOLD: usize = MIN_MACS_PER_TASK * 2;

            if out_dim <= rows_per_task || out_dim * in_dim < PAR_MAC_THRESHOLD {
                unsafe {
                    gemv_i4_dot_rows(
                        weight_i4,
                        scales_f16,
                        &even,
                        &odd,
                        x_scale,
                        &x_group_sums,
                        out,
                        0,
                        in_dim,
                        group_size,
                    );
                }
            } else {
                let groups_per_row = in_dim / group_size;
                let (even, odd, sums) = (&even[..], &odd[..], &x_group_sums[..]);
                out.par_chunks_mut(rows_per_task)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let row_off = ci * rows_per_task;
                        let s = &scales_f16[row_off * groups_per_row
                            ..(row_off + chunk.len()) * groups_per_row];
                        unsafe {
                            gemv_i4_dot_rows(
                                weight_i4, s, even, odd, x_scale, sums, chunk, row_off, in_dim,
                                group_size,
                            );
                        }
                    });
            }
            return;
        }
    }

    gemv_i4_f32_ref(weight_i4, scales_f16, input, out, out_dim, in_dim, group_size);
}

/// Scalar reference for [`gemv_i4_w4a8`]: dequantizes in f32 with no activation
/// quantization. Used on non-SDOT hardware and as the test oracle.
pub fn gemv_i4_f32_ref(
    weight_i4: &[u8],
    scales_f16: &[u16],
    input: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) {
    debug_assert_eq!(out.len(), out_dim);
    let row_stride = in_dim.div_ceil(2);
    let groups_per_row = in_dim.div_ceil(group_size);
    out.par_iter_mut().enumerate().for_each(|(r, o)| {
        let base = r * row_stride;
        let srow = r * groups_per_row;
        let mut acc = 0.0f32;
        for j in 0..in_dim {
            let byte = weight_i4[base + j / 2];
            let nibble = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            let scale = f16::from_bits(scales_f16[srow + j / group_size]).to_f32();
            acc += (nibble as i32 - 8) as f32 * scale * input[j];
        }
        *o = acc;
    });
}

/// Raw pointer wrapper so disjoint row ranges can be written from rayon tasks.
///
/// Needed because `out` is token-major while the parallel split is row-major,
/// so the usual `par_chunks_mut` split does not line up with the work.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);

// SAFETY: each task writes only cells in its own disjoint row range.
#[cfg(target_arch = "aarch64")]
unsafe impl Send for SendPtr {}
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for SendPtr {}

/// Core GEMM tile: 4 weight rows x 4 tokens per iteration.
///
/// The register block is what makes this worth doing. Loading four weight rows
/// once and dotting them against four token columns yields 16 SDOT chains per
/// pair of loads, so each weight byte fetched from memory does four times the
/// work it does in the GEMV. Accumulation order within a row matches
/// [`gemv_i8_dot_rows`] exactly, keeping results bit-identical.
///
/// # Safety
/// Caller must have verified `dotprod`, and `out` must be a valid
/// `[n_tokens, out_dim]` buffer whose rows `r0..r1` are exclusively owned.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_i8_dot_tile(
    weight_i8: &[i8],
    scales_f16: &[u16],
    xq: &[i8],
    x_scales: &[f32],
    out: *mut f32,
    r0: usize,
    r1: usize,
    n_tokens: usize,
    out_dim: usize,
    in_dim: usize,
) {
    use std::arch::aarch64::*;

    let wp = weight_i8.as_ptr();
    let xp = xq.as_ptr();

    let mut r = r0;
    while r < r1 {
        let rows = (r1 - r).min(4);
        let wb = [
            r * in_dim,
            (r + 1).min(out_dim - 1) * in_dim,
            (r + 2).min(out_dim - 1) * in_dim,
            (r + 3).min(out_dim - 1) * in_dim,
        ];

        let mut t = 0usize;
        while t < n_tokens {
            let toks = (n_tokens - t).min(4);
            let xb = [
                t * in_dim,
                (t + 1).min(n_tokens - 1) * in_dim,
                (t + 2).min(n_tokens - 1) * in_dim,
                (t + 3).min(n_tokens - 1) * in_dim,
            ];

            // 4 rows x 4 tokens of independent SDOT accumulators.
            let mut acc = [[vdupq_n_s32(0); 4]; 4];

            let mut i = 0usize;
            while i + 16 <= in_dim {
                let w0 = vld1q_s8(wp.add(wb[0] + i));
                let w1 = vld1q_s8(wp.add(wb[1] + i));
                let w2 = vld1q_s8(wp.add(wb[2] + i));
                let w3 = vld1q_s8(wp.add(wb[3] + i));

                let x0 = vld1q_s8(xp.add(xb[0] + i));
                let x1 = vld1q_s8(xp.add(xb[1] + i));
                let x2 = vld1q_s8(xp.add(xb[2] + i));
                let x3 = vld1q_s8(xp.add(xb[3] + i));

                acc[0][0] = sdot_s32(acc[0][0], w0, x0);
                acc[0][1] = sdot_s32(acc[0][1], w0, x1);
                acc[0][2] = sdot_s32(acc[0][2], w0, x2);
                acc[0][3] = sdot_s32(acc[0][3], w0, x3);

                acc[1][0] = sdot_s32(acc[1][0], w1, x0);
                acc[1][1] = sdot_s32(acc[1][1], w1, x1);
                acc[1][2] = sdot_s32(acc[1][2], w1, x2);
                acc[1][3] = sdot_s32(acc[1][3], w1, x3);

                acc[2][0] = sdot_s32(acc[2][0], w2, x0);
                acc[2][1] = sdot_s32(acc[2][1], w2, x1);
                acc[2][2] = sdot_s32(acc[2][2], w2, x2);
                acc[2][3] = sdot_s32(acc[2][3], w2, x3);

                acc[3][0] = sdot_s32(acc[3][0], w3, x0);
                acc[3][1] = sdot_s32(acc[3][1], w3, x1);
                acc[3][2] = sdot_s32(acc[3][2], w3, x2);
                acc[3][3] = sdot_s32(acc[3][3], w3, x3);

                i += 16;
            }

            let mut sums = [[0i32; 4]; 4];
            for (ri, accr) in acc.iter().enumerate() {
                for (ti, a) in accr.iter().enumerate() {
                    sums[ri][ti] = vaddvq_s32(*a);
                }
            }

            // Tail elements, matching the GEMV's scalar remainder.
            while i < in_dim {
                for (ri, s) in sums.iter_mut().enumerate().take(rows) {
                    let wv = *wp.add(wb[ri] + i) as i32;
                    for (ti, sv) in s.iter_mut().enumerate().take(toks) {
                        *sv += wv * *xp.add(xb[ti] + i) as i32;
                    }
                }
                i += 1;
            }

            for ri in 0..rows {
                let ws = f16::from_bits(scales_f16[r + ri - r0]).to_f32();
                for ti in 0..toks {
                    let v = sums[ri][ti] as f32 * ws * x_scales[t + ti];
                    *out.add((t + ti) * out_dim + (r + ri)) = v;
                }
            }

            t += 4;
        }

        r += 4;
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn affine_i4_group_dot_neon(packed: *const u8, x: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::*;

    let mask = vdup_n_u8(0x0f);
    let mut acc = vdupq_n_f32(0.0);
    let pairs = len / 2;
    let mut pair = 0usize;

    // Four independent accumulators: a single one serialises on FMA latency
    // rather than issue rate, which costs several times the throughput here.
    let mask16 = vdupq_n_u8(0x0f);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    while pair + 16 <= pairs {
        let bytes = vld1q_u8(packed.add(pair));
        let lo_n = vandq_u8(bytes, mask16);
        let hi_n = vshrq_n_u8::<4>(bytes);

        let lo_16a = vmovl_u8(vget_low_u8(lo_n));
        let lo_16b = vmovl_high_u8(lo_n);
        let hi_16a = vmovl_u8(vget_low_u8(hi_n));
        let hi_16b = vmovl_high_u8(hi_n);

        let l0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16a)));
        let l1 = vcvtq_f32_u32(vmovl_high_u16(lo_16a));
        let l2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16b)));
        let l3 = vcvtq_f32_u32(vmovl_high_u16(lo_16b));
        let h0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16a)));
        let h1 = vcvtq_f32_u32(vmovl_high_u16(hi_16a));
        let h2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16b)));
        let h3 = vcvtq_f32_u32(vmovl_high_u16(hi_16b));

        let base = x.add(pair * 2);
        let a0 = vld2q_f32(base);
        let a1 = vld2q_f32(base.add(8));
        let a2 = vld2q_f32(base.add(16));
        let a3 = vld2q_f32(base.add(24));

        acc0 = vfmaq_f32(acc0, l0, a0.0);
        acc1 = vfmaq_f32(acc1, h0, a0.1);
        acc2 = vfmaq_f32(acc2, l1, a1.0);
        acc3 = vfmaq_f32(acc3, h1, a1.1);
        acc0 = vfmaq_f32(acc0, l2, a2.0);
        acc1 = vfmaq_f32(acc1, h2, a2.1);
        acc2 = vfmaq_f32(acc2, l3, a3.0);
        acc3 = vfmaq_f32(acc3, h3, a3.1);
        pair += 16;
    }

    acc = vaddq_f32(acc, vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

    while pair + 4 <= pairs {
        let bits = std::ptr::read_unaligned(packed.add(pair) as *const u32) as u64;
        let bytes = vcreate_u8(bits);
        let low = vand_u8(bytes, mask);
        let high = vshr_n_u8::<4>(bytes);
        let low_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(low))));
        let high_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(high))));
        let activations = vld2q_f32(x.add(pair * 2));
        acc = vfmaq_f32(acc, low_f32, activations.0);
        acc = vfmaq_f32(acc, high_f32, activations.1);
        pair += 4;
    }

    let mut dot = vaddvq_f32(acc);
    while pair < pairs {
        let byte = *packed.add(pair);
        dot += ((byte & 0x0f) as f32) * *x.add(pair * 2);
        dot += ((byte >> 4) as f32) * *x.add(pair * 2 + 1);
        pair += 1;
    }
    if len % 2 != 0 {
        dot += ((*packed.add(pairs) & 0x0f) as f32) * *x.add(len - 1);
    }
    dot
}

/// Matrix-vector product for unsigned affine Q4 weights.
///
/// Each row stores two 4-bit values per byte. Scales and biases are f32,
/// row-major per group, and dequantization is `q * scale + bias`.
pub fn matmul_affine_i4_f32(
    weights: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    x: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    let groups_per_row = cols.div_ceil(group_size);
    let packed_per_group = group_size.div_ceil(2);
    let packed_per_row = groups_per_row * packed_per_group;
    debug_assert_eq!(weights.len(), rows * packed_per_row);
    debug_assert_eq!(scales.len(), rows * groups_per_row);
    debug_assert_eq!(biases.len(), rows * groups_per_row);

    // The affine bias contribution is bias * sum(x), which is identical for
    // every output row. Compute it once per activation group instead of once
    // per quantized value and output row.
    let x_sums: Vec<f32> = x
        .chunks(group_size)
        .map(|group| group.iter().copied().sum())
        .collect();

    out.par_iter_mut().enumerate().for_each(|(row_idx, value)| {
        let packed = &weights[row_idx * packed_per_row..(row_idx + 1) * packed_per_row];
        let params = row_idx * groups_per_row;
        let mut dot = 0.0f32;
        for group in 0..groups_per_row {
            let start = group * group_size;
            let len = (cols - start).min(group_size);
            let scale = scales[params + group];
            let bias = biases[params + group];
            let group_packed = &packed[group * packed_per_group..];

            #[cfg(target_arch = "aarch64")]
            let quantized_dot = unsafe {
                affine_i4_group_dot_neon(group_packed.as_ptr(), x.as_ptr().add(start), len)
            };

            #[cfg(not(target_arch = "aarch64"))]
            let quantized_dot = {
                let mut sum = 0.0f32;
                for col in 0..len {
                    let byte = group_packed[col / 2];
                    let q = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    sum += q as f32 * x[start + col];
                }
                sum
            };

            dot += scale * quantized_dot + bias * x_sums[group];
        }
        *value = dot;
    });
}

/// W4A8 form of [`gemm_affine_i4_f32`]: quantizes activations to int8 and uses
/// SDOT, trading a small numeric error for ~4x the arithmetic throughput.
///
/// Affine nibbles are unsigned 0..=15, so they reinterpret as non-negative `i8`
/// exactly and need none of the `+8` bias correction the symmetric path applies.
/// The affine `bias * sum(x)` term is kept in f32 against the unquantized
/// activations, so only the `q . x` product carries quantization error.
///
/// Returns false when the shapes or hardware rule out the fast path, leaving
/// `out` untouched so the caller can fall back.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn gemm_affine_i4_w4a8(
    weights: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    x: &[f32],
    out: &mut [f32],
    n_tokens: usize,
) -> bool {
    use std::arch::aarch64::*;

    if !has_i8_dotprod() || group_size % 32 != 0 || cols % group_size != 0 || n_tokens == 0 {
        return false;
    }

    let groups_per_row = cols / group_size;
    let bytes_per_group = group_size / 2;
    let packed_per_row = groups_per_row * bytes_per_group;
    let half = cols / 2;

    // Pad the token axis to a multiple of 4 so the inner tile never needs a
    // remainder path; padded rows carry scale 0 and are discarded.
    let tiles = n_tokens.div_ceil(4);
    let padded = tiles * 4;

    let mut even = vec![0i8; padded * half];
    let mut odd = vec![0i8; padded * half];
    let mut x_scales = vec![0.0f32; padded];
    let mut x_sums = vec![0.0f32; padded * groups_per_row];
    let mut xq = vec![0i8; cols];

    for t in 0..n_tokens {
        let row = &x[t * cols..(t + 1) * cols];
        x_scales[t] = quantize_activation_i8(row, &mut xq);
        deinterleave_i8(
            &xq,
            &mut even[t * half..(t + 1) * half],
            &mut odd[t * half..(t + 1) * half],
        );
        for (g, chunk) in row.chunks(group_size).enumerate() {
            x_sums[t * groups_per_row + g] = chunk.iter().copied().sum();
        }
    }

    let mut cols_out: Vec<Vec<f32>> = (0..rows).map(|_| vec![0.0f32; n_tokens]).collect();
    let lo_mask = unsafe { vdupq_n_u8(0x0f) };

    cols_out.par_iter_mut().enumerate().for_each(|(row_idx, dst)| {
        let base = row_idx * packed_per_row;
        let params = row_idx * groups_per_row;

        for tile in 0..tiles {
            let t0 = tile * 4;
            let mut qdot = [0.0f32; 4];

            for g in 0..groups_per_row {
                let gb = g * bytes_per_group;
                let mut acc_e = [unsafe { vdupq_n_s32(0) }; 4];
                let mut acc_o = [unsafe { vdupq_n_s32(0) }; 4];

                let mut b = 0usize;
                while b + 16 <= bytes_per_group {
                    unsafe {
                        // One weight load and unpack feeds eight SDOTs.
                        let packed = vld1q_u8(weights.as_ptr().add(base + gb + b));
                        let lo = vreinterpretq_s8_u8(vandq_u8(packed, lo_mask));
                        let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(packed));

                        for k in 0..4 {
                            let off = (t0 + k) * half + gb + b;
                            acc_e[k] = sdot_s32(acc_e[k], lo, vld1q_s8(even.as_ptr().add(off)));
                            acc_o[k] = sdot_s32(acc_o[k], hi, vld1q_s8(odd.as_ptr().add(off)));
                        }
                    }
                    b += 16;
                }

                let scale = scales[params + g];
                for k in 0..4 {
                    let acc = unsafe { vaddvq_s32(acc_e[k]) + vaddvq_s32(acc_o[k]) };
                    qdot[k] += acc as f32 * scale;
                }
            }

            for k in 0..4 {
                let t = t0 + k;
                if t >= n_tokens {
                    break;
                }
                let mut bias_term = 0.0f32;
                for g in 0..groups_per_row {
                    bias_term += biases[params + g] * x_sums[t * groups_per_row + g];
                }
                dst[t] = qdot[k] * x_scales[t] + bias_term;
            }
        }
    });

    for (row_idx, col) in cols_out.iter().enumerate() {
        for (t, v) in col.iter().enumerate() {
            out[t * rows + row_idx] = *v;
        }
    }
    true
}

/// Batched form of [`matmul_affine_i4_f32`] over `n_tokens` activation rows.
///
/// Parallelising over rows rather than tokens keeps each weight row resident
/// while every token dots against it. Prefers the W4A8 SDOT path and falls back
/// to f32 NEON when the shape or hardware does not support it.
#[allow(clippy::too_many_arguments)]
pub fn gemm_affine_i4_f32(
    weights: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    x: &[f32],
    out: &mut [f32],
    n_tokens: usize,
) {
    debug_assert_eq!(x.len(), n_tokens * cols);
    debug_assert_eq!(out.len(), n_tokens * rows);
    if n_tokens == 0 {
        return;
    }
    if n_tokens == 1 {
        matmul_affine_i4_f32(weights, scales, biases, rows, cols, group_size, x, out);
        return;
    }

    // Prefill has enough tokens per weight load to amortise activation
    // quantization, which decode's single row does not.
    #[cfg(target_arch = "aarch64")]
    if gemm_affine_i4_w4a8(
        weights, scales, biases, rows, cols, group_size, x, out, n_tokens,
    ) {
        return;
    }

    let groups_per_row = cols.div_ceil(group_size);
    let packed_per_group = group_size.div_ceil(2);
    let packed_per_row = groups_per_row * packed_per_group;

    let x_sums: Vec<f32> = (0..n_tokens)
        .flat_map(|t| {
            x[t * cols..(t + 1) * cols]
                .chunks(group_size)
                .map(|g| g.iter().copied().sum::<f32>())
                .collect::<Vec<_>>()
        })
        .collect();

    // out is [n_tokens, rows]; each row index touches a strided set of slots,
    // so accumulate per row into a scratch column and scatter once.
    let mut cols_out: Vec<Vec<f32>> = (0..rows).map(|_| vec![0.0f32; n_tokens]).collect();

    cols_out.par_iter_mut().enumerate().for_each(|(row_idx, dst)| {
        let packed = &weights[row_idx * packed_per_row..(row_idx + 1) * packed_per_row];
        let params = row_idx * groups_per_row;
        for group in 0..groups_per_row {
            let start = group * group_size;
            let len = (cols - start).min(group_size);
            let scale = scales[params + group];
            let bias = biases[params + group];
            let group_packed = &packed[group * packed_per_group..];

            for (t, slot) in dst.iter_mut().enumerate() {
                let xt = &x[t * cols..(t + 1) * cols];

                #[cfg(target_arch = "aarch64")]
                let quantized_dot = unsafe {
                    affine_i4_group_dot_neon(group_packed.as_ptr(), xt.as_ptr().add(start), len)
                };

                #[cfg(not(target_arch = "aarch64"))]
                let quantized_dot = {
                    let mut sum = 0.0f32;
                    for col in 0..len {
                        let byte = group_packed[col / 2];
                        let q = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                        sum += q as f32 * xt[start + col];
                    }
                    sum
                };

                *slot += scale * quantized_dot + bias * x_sums[t * groups_per_row + group];
            }
        }
    });

    for (row_idx, dst) in cols_out.iter().enumerate() {
        for (t, v) in dst.iter().enumerate() {
            out[t * rows + row_idx] = *v;
        }
    }
}

#[cfg(test)]
mod w4a8_tests {
    use super::{gemv_i4_f32_ref, gemv_i4_w4a8};
    use half::f16;

    fn build(out_dim: usize, in_dim: usize, gs: usize) -> (Vec<u8>, Vec<u16>, Vec<f32>) {
        let row_stride = in_dim.div_ceil(2);
        let mut seed = 0x9e3779b9u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 24) & 0xff
        };
        // Two nibbles per byte, each already biased by +8 into 1..=15.
        let w: Vec<u8> = (0..out_dim * row_stride)
            .map(|_| {
                let lo = (next() % 15) + 1;
                let hi = (next() % 15) + 1;
                (lo | (hi << 4)) as u8
            })
            .collect();
        let groups = out_dim * in_dim.div_ceil(gs);
        let scales: Vec<u16> = (0..groups)
            .map(|g| f16::from_f32(0.002 + (g % 5) as f32 * 0.0007).to_bits())
            .collect();
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i * 53 % 197) as f32 - 98.0) / 80.0)
            .collect();
        (w, scales, x)
    }

    #[test]
    fn w4a8_matches_f32_reference_within_activation_quant_error() {
        for &(out_dim, in_dim, gs) in &[
            (4usize, 64usize, 64usize),
            (8, 128, 32),
            (128, 1024, 64),
            (68, 96, 32),
        ] {
            let (w, scales, x) = build(out_dim, in_dim, gs);
            let mut got = vec![0.0f32; out_dim];
            let mut want = vec![0.0f32; out_dim];
            gemv_i4_w4a8(&w, &scales, &x, &mut got, out_dim, in_dim, gs);
            gemv_i4_f32_ref(&w, &scales, &x, &mut want, out_dim, in_dim, gs);

            let norm = want.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            assert!(
                err / norm < 0.02,
                "{out_dim}x{in_dim} gs={gs}: relative error {} too high\ngot  {:?}\nwant {:?}",
                err / norm,
                &got[..got.len().min(4)],
                &want[..want.len().min(4)]
            );
        }
    }

    #[test]
    fn w4a8_zero_activation_yields_zero() {
        let (w, scales, _) = build(16, 64, 64);
        let x = vec![0.0f32; 64];
        let mut got = vec![1.0f32; 16];
        gemv_i4_w4a8(&w, &scales, &x, &mut got, 16, 64, 64);
        assert!(got.iter().all(|&v| v == 0.0), "got {got:?}");
    }
}

#[cfg(test)]
mod w8a8_tests {
    use super::{gemv_i8_f32, gemv_i8_w8a8};
    use half::f16;

    fn build(out_dim: usize, in_dim: usize) -> (Vec<i8>, Vec<u16>, Vec<f32>) {
        let mut w = vec![0i8; out_dim * in_dim];
        let mut seed = 0x12345678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) & 0xff) as i32 - 128
        };
        for v in w.iter_mut() {
            *v = next().clamp(-127, 127) as i8;
        }
        let scales: Vec<u16> = (0..out_dim)
            .map(|r| f16::from_f32(0.001 + (r % 7) as f32 * 0.0003).to_bits())
            .collect();
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i * 37 % 211) as f32 - 105.0) / 90.0)
            .collect();
        (w, scales, x)
    }

    #[test]
    fn w8a8_matches_f32_reference_within_activation_quant_error() {
        for &(out_dim, in_dim) in &[(4usize, 64usize), (7, 33), (128, 1024), (65, 96)] {
            let (w, scales, x) = build(out_dim, in_dim);
            let mut got = vec![0.0f32; out_dim];
            let mut want = vec![0.0f32; out_dim];
            gemv_i8_w8a8(&w, &scales, &x, &mut got, out_dim, in_dim);
            gemv_i8_f32(&w, &scales, &x, &mut want, out_dim, in_dim);

            let norm = want.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            assert!(
                err / norm < 0.02,
                "{out_dim}x{in_dim}: relative error {} too large",
                err / norm
            );
        }
    }

    #[test]
    fn w8a8_zero_activation_yields_zeros() {
        let (w, scales, _) = build(16, 64);
        let x = vec![0.0f32; 64];
        let mut out = vec![9.0f32; 16];
        gemv_i8_w8a8(&w, &scales, &x, &mut out, 16, 64);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    /// The batched kernel must be bit-identical to the per-token one.
    ///
    /// Prefill and decode share weights and must agree exactly: if batching
    /// changed the arithmetic even slightly, a prompt would decode differently
    /// depending on how it was chunked, which is far harder to debug than an
    /// outright failure.
    #[test]
    fn gemm_is_bit_identical_to_per_token_gemv() {
        use super::gemm_i8_w8a8;

        for &(n_tokens, out_dim, in_dim) in &[
            (1usize, 16usize, 64usize),
            (2, 16, 64),
            (3, 7, 33),   // rows, tokens and in_dim all non-multiples of 4/16
            (5, 65, 96),
            (8, 128, 1024),
            (17, 36, 128), // token count not a multiple of the 4-wide tile
        ] {
            let (w, scales, _) = build(out_dim, in_dim);

            // Distinct per-token activations, so a mixed-up token index shows up.
            let mut x = vec![0.0f32; n_tokens * in_dim];
            for t in 0..n_tokens {
                for i in 0..in_dim {
                    x[t * in_dim + i] =
                        (((i * 37 + t * 101) % 211) as f32 - 105.0) / (90.0 + t as f32);
                }
            }

            let mut got = vec![0.0f32; n_tokens * out_dim];
            gemm_i8_w8a8(&w, &scales, &x, &mut got, n_tokens, out_dim, in_dim);

            let mut want = vec![0.0f32; n_tokens * out_dim];
            for t in 0..n_tokens {
                gemv_i8_w8a8(
                    &w,
                    &scales,
                    &x[t * in_dim..(t + 1) * in_dim],
                    &mut want[t * out_dim..(t + 1) * out_dim],
                    out_dim,
                    in_dim,
                );
            }

            for t in 0..n_tokens {
                for r in 0..out_dim {
                    let idx = t * out_dim + r;
                    assert_eq!(
                        got[idx].to_bits(),
                        want[idx].to_bits(),
                        "{n_tokens}x{out_dim}x{in_dim}: token {t} row {r}: \
                         gemm {} != gemv {}",
                        got[idx],
                        want[idx]
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod affine_i4_tests {
    use super::matmul_affine_i4_f32;

    fn reference(
        weights: &[u8],
        scales: &[f32],
        biases: &[f32],
        rows: usize,
        cols: usize,
        group_size: usize,
        x: &[f32],
    ) -> Vec<f32> {
        let groups = cols.div_ceil(group_size);
        let packed_per_group = group_size.div_ceil(2);
        let packed_per_row = groups * packed_per_group;
        let mut out = vec![0.0; rows];
        for row in 0..rows {
            for group in 0..groups {
                let start = group * group_size;
                let len = (cols - start).min(group_size);
                let packed = &weights[row * packed_per_row + group * packed_per_group..];
                for col in 0..len {
                    let byte = packed[col / 2];
                    let q = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    out[row] += (q as f32 * scales[row * groups + group]
                        + biases[row * groups + group])
                        * x[start + col];
                }
            }
        }
        out
    }

    #[test]
    fn affine_i4_matches_scalar_for_multiple_and_partial_groups() {
        let rows: usize = 3;
        let cols: usize = 13;
        let group_size: usize = 8;
        let groups = cols.div_ceil(group_size);
        let packed_per_group = group_size.div_ceil(2);
        let mut weights = vec![0u8; rows * groups * packed_per_group];
        for (index, byte) in weights.iter_mut().enumerate() {
            let low = (index * 3 + 1) % 16;
            let high = (index * 5 + 7) % 16;
            *byte = low as u8 | ((high as u8) << 4);
        }
        let scales = vec![0.125, -0.25, 0.5, 0.0625, -0.125, 0.375];
        let biases = vec![-0.75, 1.25, 0.5, -1.0, 0.25, -0.5];
        let x = vec![
            -1.5, 0.25, 2.0, -0.75, 0.5, 1.25, -2.0, 0.125, 0.75, -1.0, 1.5, 0.375,
            -0.625,
        ];
        let expected = reference(&weights, &scales, &biases, rows, cols, group_size, &x);
        let mut actual = vec![0.0; rows];
        matmul_affine_i4_f32(
            &weights,
            &scales,
            &biases,
            rows,
            cols,
            group_size,
            &x,
            &mut actual,
        );
        for (row, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "row {row}: actual={actual} expected={expected}"
            );
        }
    }

    // Group sizes below 32 never enter the wide NEON block, so the sizes here
    // are chosen to cover it plus every tail length that follows it.
    #[test]
    fn affine_i4_matches_scalar_across_wide_and_tail_lengths() {
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 16
        };
        for &group_size in &[32usize, 64, 96, 128] {
            for &cols in &[
                group_size,
                group_size + 1,
                group_size + 7,
                group_size * 2 + 3,
                group_size * 3 + 17,
            ] {
                let rows = 5usize;
                let groups = cols.div_ceil(group_size);
                let packed_per_group = group_size.div_ceil(2);
                let weights: Vec<u8> = (0..rows * groups * packed_per_group)
                    .map(|_| (next() & 0xff) as u8)
                    .collect();
                let scales: Vec<f32> = (0..rows * groups)
                    .map(|_| (next() as f32 / 65_536.0) - 0.5)
                    .collect();
                let biases: Vec<f32> = (0..rows * groups)
                    .map(|_| (next() as f32 / 65_536.0) - 0.5)
                    .collect();
                let x: Vec<f32> = (0..cols).map(|_| (next() as f32 / 65_536.0) - 0.5).collect();

                let expected =
                    reference(&weights, &scales, &biases, rows, cols, group_size, &x);
                let mut actual = vec![0.0; rows];
                matmul_affine_i4_f32(
                    &weights,
                    &scales,
                    &biases,
                    rows,
                    cols,
                    group_size,
                    &x,
                    &mut actual,
                );
                for (row, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (actual - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                        "gs={group_size} cols={cols} row {row}: actual={actual} expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn affine_i4_gemm_is_bit_identical_to_per_token_gemv() {
        use super::gemm_affine_i4_f32;

        let mut seed = 0x0bad_c0deu32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 16
        };
        let rows = 7usize;
        let cols = 131usize;
        let group_size = 64usize;
        let n_tokens = 5usize;
        let groups = cols.div_ceil(group_size);
        let packed_per_group = group_size.div_ceil(2);

        let weights: Vec<u8> = (0..rows * groups * packed_per_group)
            .map(|_| (next() & 0xff) as u8)
            .collect();
        let scales: Vec<f32> = (0..rows * groups)
            .map(|_| (next() as f32 / 65_536.0) - 0.5)
            .collect();
        let biases: Vec<f32> = (0..rows * groups)
            .map(|_| (next() as f32 / 65_536.0) - 0.5)
            .collect();
        let x: Vec<f32> = (0..n_tokens * cols)
            .map(|_| (next() as f32 / 65_536.0) - 0.5)
            .collect();

        let mut expected = vec![0.0f32; n_tokens * rows];
        for t in 0..n_tokens {
            matmul_affine_i4_f32(
                &weights,
                &scales,
                &biases,
                rows,
                cols,
                group_size,
                &x[t * cols..(t + 1) * cols],
                &mut expected[t * rows..(t + 1) * rows],
            );
        }

        let mut actual = vec![0.0f32; n_tokens * rows];
        gemm_affine_i4_f32(
            &weights,
            &scales,
            &biases,
            rows,
            cols,
            group_size,
            &x,
            &mut actual,
            n_tokens,
        );

        assert_eq!(actual, expected, "GEMM must match per-token GEMV exactly");
    }

    #[test]
    fn affine_i4_w4a8_gemm_is_close_to_f32_gemv() {
        use super::gemm_affine_i4_f32;

        // cols % group_size == 0 and group_size % 32 == 0, so this shape takes
        // the SDOT path rather than the f32 fallback the test above exercises.
        let rows = 9usize;
        let cols = 128usize;
        let group_size = 64usize;
        let n_tokens = 6usize;
        let groups = cols / group_size;
        let packed_per_group = group_size / 2;

        let mut seed = 0x5eed_1234u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 16
        };
        let weights: Vec<u8> = (0..rows * groups * packed_per_group)
            .map(|_| (next() & 0xff) as u8)
            .collect();
        let scales: Vec<f32> = (0..rows * groups)
            .map(|_| ((next() as f32 / 65_536.0) - 0.5) * 0.1)
            .collect();
        let biases: Vec<f32> = (0..rows * groups)
            .map(|_| ((next() as f32 / 65_536.0) - 0.5) * 0.1)
            .collect();
        let x: Vec<f32> = (0..n_tokens * cols)
            .map(|_| (next() as f32 / 65_536.0) - 0.5)
            .collect();

        let mut expected = vec![0.0f32; n_tokens * rows];
        for t in 0..n_tokens {
            matmul_affine_i4_f32(
                &weights,
                &scales,
                &biases,
                rows,
                cols,
                group_size,
                &x[t * cols..(t + 1) * cols],
                &mut expected[t * rows..(t + 1) * rows],
            );
        }

        let mut actual = vec![0.0f32; n_tokens * rows];
        gemm_affine_i4_f32(
            &weights,
            &scales,
            &biases,
            rows,
            cols,
            group_size,
            &x,
            &mut actual,
            n_tokens,
        );

        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 0.02 * scale,
                "slot {i}: w4a8={a} f32={e} (tolerance {})",
                0.02 * scale
            );
        }
    }
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

/// Apply RoPE to a buffer using precomputed inverse frequencies (used for llama3-style scaling).
pub fn rope_non_interleaved_inplace_f32_with_freqs(
    x: &mut [f32],
    _n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    pos: usize,
    inv_freqs: &[f32],
) {
    let half = rotary_dim / 2;
    if x.len() < 2048 {
        for head in x.chunks_exact_mut(head_dim) {
            for i in 0..half {
                let angle = pos as f32 * inv_freqs[i];
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
                let angle = pos as f32 * inv_freqs[i];
                let (sin, cos) = angle.sin_cos();
                let x0 = head[i];
                let x1 = head[half + i];
                head[i] = x0 * cos - x1 * sin;
                head[half + i] = x1 * cos + x0 * sin;
            }
        });
    }
}

pub fn rope_interleaved_inplace_f32_with_freqs(
    x: &mut [f32],
    _n_heads: usize,
    head_dim: usize,
    pos: usize,
    inv_freqs: &[f32],
) {
    let half = head_dim / 2;
    x.par_chunks_exact_mut(head_dim).for_each(|head| {
        for i in 0..half {
            let angle = pos as f32 * inv_freqs[i];
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
