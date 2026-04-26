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

    let mut mean_sq = 0.0f32;
    // Use parallel reduction for large vectors if needed, but usually hidden_size is 4k-8k.
    // Scalar is fine for reduction, but let's at least ensure autovectorization.
    for &v in x {
        mean_sq += v * v;
    }
    mean_sq /= x.len() as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();

    // Parallelize the output application if len is large.
    out.par_iter_mut().zip(x.par_iter()).zip(weight.par_iter()).for_each(|((o, &xi), &wi)| {
        *o = xi * inv_rms * wi;
    });
}

pub fn matmul_f32(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    if n == 1 {
        // Matrix-vector product - parallelize across rows.
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row = &a[i * k..(i + 1) * k];
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += row[kk] * b[kk];
            }
            *o = acc;
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

pub fn matmul_f16_f32(
    a_f16: &[u16],
    m: usize,
    k: usize,
    b: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(a_f16.len(), m * k);
    debug_assert_eq!(out.len(), m);

    out.par_iter_mut().enumerate().for_each(|(i, o)| {
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
        for j in 0..k {
            let b_idx = j / 2;
            let n = if j % 2 == 0 { row[b_idx] & 0xf } else { row[b_idx] >> 4 };
            let q = (n as i8) - 8;
            let scale = f16::from_bits(rs[j / gs]).to_f32();
            dot += (q as f32) * scale * b[j];
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
