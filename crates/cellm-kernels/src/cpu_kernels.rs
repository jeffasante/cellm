use std::f32;

pub fn rms_norm_f32(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());

    let mut mean_sq = 0.0f32;
    for &v in x {
        mean_sq += v * v;
    }
    mean_sq /= x.len() as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();

    for i in 0..x.len() {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/// Row-major matmul: out[m, n] = a[m, k] @ b[k, n]
pub fn matmul_f32(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    out.fill(0.0);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let out_row = &mut out[i * n..(i + 1) * n];
        for kk in 0..k {
            let av = a_row[kk];
            let b_row = &b[kk * n..(kk + 1) * n];
            for j in 0..n {
                out_row[j] += av * b_row[j];
            }
        }
    }
}

pub fn softmax_f32_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 {
        return;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Rotary position embedding, applied in-place.
///
/// `x` is shaped [n_heads, head_dim] flattened. RoPE is applied to the first
/// `head_dim` dims for each head (pairwise rotation).
pub fn rope_inplace_f32(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

    for h in 0..n_heads {
        let base = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let dim = i / 2;
            let inv_freq = theta.powf(-(2.0 * dim as f32) / head_dim as f32);
            let angle = pos as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + i];
            let x1 = x[base + i + 1];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + i + 1] = x0 * sin + x1 * cos;
        }
    }
}

/// Rotary position embedding (non-interleaved / rotate_half style), applied in-place.
///
/// `x` is shaped [n_heads, head_dim] flattened. Pairs are formed between the
/// first and second halves of each head: `(i, i + head_dim/2)`.
pub fn rope_non_interleaved_inplace_f32(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
    let half = head_dim / 2;

    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = pos as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + i + half] = x0 * sin + x1 * cos;
        }
    }
}

/// Single-token grouped-query attention.
///
/// - `q`: [n_heads, head_dim]
/// - `k`: [seq, n_kv_heads, head_dim]
/// - `v`: [seq, n_kv_heads, head_dim]
/// - `out`: [n_heads, head_dim]
pub fn attention_single_token_f32_gqa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(k.len(), seq * n_kv_heads * head_dim);
    debug_assert_eq!(v.len(), seq * n_kv_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);

    out.fill(0.0);
    if seq == 0 {
        return;
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let group_size = (n_heads / n_kv_heads).max(1);

    let mut scores = vec![0.0f32; seq];

    for h in 0..n_heads {
        let kv_h = (h / group_size).min(n_kv_heads.saturating_sub(1));
        let qh = &q[h * head_dim..(h + 1) * head_dim];

        for t in 0..seq {
            let kt_base = (t * n_kv_heads + kv_h) * head_dim;
            let kt = &k[kt_base..kt_base + head_dim];
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += qh[i] * kt[i];
            }
            scores[t] = dot * scale;
        }

        softmax_f32_inplace(&mut scores);

        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for t in 0..seq {
            let vt_base = (t * n_kv_heads + kv_h) * head_dim;
            let vt = &v[vt_base..vt_base + head_dim];
            let w = scores[t];
            for i in 0..head_dim {
                out_h[i] += w * vt[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_f32_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_shapes() {
        // 2x3 @ 3x2 -> 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = vec![0.0f32; 4];
        matmul_f32(&a, 2, 3, &b, 2, &mut out);
        assert_eq!(out.len(), 4);
        assert!((out[0] - (1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0)).abs() < 1e-5);
    }
}
