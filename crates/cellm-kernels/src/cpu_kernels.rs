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
                
                while i_inner + 8 <= k {
                    let xv0 = vld1q_f32(b.as_ptr().add(i_inner));
                    let xv1 = vld1q_f32(b.as_ptr().add(i_inner + 4));
                    
                    let w0 = f16::from_bits(row[i_inner]).to_f32();
                    let w1 = f16::from_bits(row[i_inner+1]).to_f32();
                    let w2 = f16::from_bits(row[i_inner+2]).to_f32();
                    let w3 = f16::from_bits(row[i_inner+3]).to_f32();
                    
                    let mut wf0 = vdupq_n_f32(0.0);
                    wf0 = vsetq_lane_f32(w0, wf0, 0);
                    wf0 = vsetq_lane_f32(w1, wf0, 1);
                    wf0 = vsetq_lane_f32(w2, wf0, 2);
                    wf0 = vsetq_lane_f32(w3, wf0, 3);
                    
                    let w4 = f16::from_bits(row[i_inner+4]).to_f32();
                    let w5 = f16::from_bits(row[i_inner+5]).to_f32();
                    let w6 = f16::from_bits(row[i_inner+6]).to_f32();
                    let w7 = f16::from_bits(row[i_inner+7]).to_f32();
                    
                    let mut wf1 = vdupq_n_f32(0.0);
                    wf1 = vsetq_lane_f32(w4, wf1, 0);
                    wf1 = vsetq_lane_f32(w5, wf1, 1);
                    wf1 = vsetq_lane_f32(w6, wf1, 2);
                    wf1 = vsetq_lane_f32(w7, wf1, 3);

                    sum0 = vmlaq_f32(sum0, wf0, xv0);
                    sum1 = vmlaq_f32(sum1, wf1, xv1);
                    i_inner += 8;

                }
                let res = vaddq_f32(sum0, sum1);
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
    b: &[f32],
    out: &mut [f32],
) {
    let row_stride = k.div_ceil(2);
    debug_assert_eq!(a_i4.len(), m * row_stride);
    debug_assert_eq!(a_scales_f16.len(), m);

    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let row = &a_i4[i * row_stride..(i + 1) * row_stride];
        let scale = f16::from_bits(a_scales_f16[i]).to_f32();
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += unpack_i4(row, kk) * b[kk];
        }
        *o = acc * scale;
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

pub fn rope_non_interleaved_inplace_f32(x: &mut [f32], n_heads: usize, head_dim: usize, rotary_dim: usize, pos: usize, theta: f32) {
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

pub fn rope_interleaved_inplace_f32(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
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
                    let out_v = vld1q_f32(out_h.as_ptr().add(i));
                    let val_v = vld1q_f32(vt.as_ptr().add(i));
                    vst1q_f32(out_h.as_mut_ptr().add(i), vmlaq_f32(out_v, wv, val_v));
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
