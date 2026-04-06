#![cfg(target_os = "macos")]

use cellm_kernels::{cpu_kernels, MetalOps};
use half::f16;

fn try_metal() -> Option<MetalOps> {
    match MetalOps::create() {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("Skipping Metal parity test (Metal unavailable): {e}");
            None
        }
    }
}

fn lcg_fill(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    let mut out = vec![0.0f32; len];
    for v in &mut out {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((s >> 32) as u32) as f32 / (u32::MAX as f32);
        *v = (x * 2.0 - 1.0) * scale;
    }
    out
}

fn to_f16_bits(xs: &[f32]) -> Vec<u16> {
    xs.iter().map(|&v| f16::from_f32(v).to_bits()).collect()
}

fn assert_close(a: &[f32], b: &[f32], max_abs_tol: f32, mean_abs_tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
    }
    let mean_abs = sum_abs / (a.len() as f32);
    assert!(
        max_abs <= max_abs_tol && mean_abs <= mean_abs_tol,
        "{label}: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} (tol max={max_abs_tol:.6e} mean={mean_abs_tol:.6e})"
    );
}

#[test]
fn metal_rms_norm_f16w_matches_cpu() {
    let Some(mut metal) = try_metal() else {
        return;
    };

    let n = 4096usize;
    let x = lcg_fill(n, 0xA11CE, 0.8);
    let w = lcg_fill(n, 0xBEEF, 0.2);
    let w_f16 = to_f16_bits(&w);
    let eps = 1e-5f32;

    let mut cpu_out = vec![0.0f32; n];
    let mut metal_out = vec![0.0f32; n];

    // Gemma path adds 1.0 to norm weights.
    let w_add_one = true;
    let mut w_effective = vec![0.0f32; n];
    for i in 0..n {
        w_effective[i] = f16::from_bits(w_f16[i]).to_f32() + 1.0;
    }
    cpu_kernels::rms_norm_f32(&x, &w_effective, eps, &mut cpu_out);
    metal
        .rms_norm_f16w(&x, &w_f16, eps, w_add_one, &mut metal_out)
        .expect("metal rms_norm_f16w");

    assert_close(&cpu_out, &metal_out, 2e-3, 4e-4, "rms_norm_f16w");
}

#[test]
fn metal_rope_adj_matches_cpu() {
    let Some(mut metal) = try_metal() else {
        return;
    };

    let n_heads = 8usize;
    let head_dim = 128usize;
    let pos = 123usize;
    let theta = 10000.0f32;
    let mut cpu_x = lcg_fill(n_heads * head_dim, 0x1234_5678, 0.5);
    let mut metal_x = cpu_x.clone();

    cpu_kernels::rope_inplace_f32(&mut cpu_x, n_heads, head_dim, pos, theta);
    metal
        .rope_adj_f32(&mut metal_x, n_heads, head_dim, pos, theta)
        .expect("metal rope_adj_f32");

    assert_close(&cpu_x, &metal_x, 5e-4, 1e-4, "rope_adj_f32");
}

fn rope_inplace_rotate_half_cpu(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
) {
    let rotary_dim = head_dim;
    let half = rotary_dim / 2;
    let p = pos as f32;
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut i = 0usize;
        while i < half {
            let inv_freq = theta.powf(-2.0 * (i as f32) / (rotary_dim as f32));
            let a = p * inv_freq;
            let (s, c) = a.sin_cos();

            let x1 = x[base + i];
            let x2 = x[base + i + half];
            x[base + i] = x1 * c - x2 * s;
            x[base + i + half] = x2 * c + x1 * s;
            i += 1;
        }
    }
}

#[test]
fn metal_rope_half_matches_rotate_half_cpu() {
    let Some(mut metal) = try_metal() else {
        return;
    };

    let n_heads = 8usize;
    let head_dim = 128usize;
    let pos = 123usize;
    let theta = 10000.0f32;
    let mut cpu_x = lcg_fill(n_heads * head_dim, 0x9988_7766, 0.5);
    let mut metal_x = cpu_x.clone();

    rope_inplace_rotate_half_cpu(&mut cpu_x, n_heads, head_dim, pos, theta);
    metal
        .rope_half_f32(&mut metal_x, n_heads, head_dim, head_dim, pos, theta)
        .expect("metal rope_half_f32");

    assert_close(&cpu_x, &metal_x, 8e-4, 2e-4, "rope_half_f32");
}

#[test]
fn metal_logits_f16_matches_cpu_matvec() {
    let Some(mut metal) = try_metal() else {
        return;
    };

    let vocab = 2048usize;
    let hidden = 256usize;
    let x = lcg_fill(hidden, 0xDEAD_BEEF, 0.4);
    let embed = lcg_fill(vocab * hidden, 0xFACE_B00C, 0.4);
    let embed_f16 = to_f16_bits(&embed);

    let mut cpu_logits = vec![0.0f32; vocab];
    let mut metal_logits = vec![0.0f32; vocab];

    for r in 0..vocab {
        let row = &embed_f16[r * hidden..(r + 1) * hidden];
        let mut acc = 0.0f32;
        for c in 0..hidden {
            acc += f16::from_bits(row[c]).to_f32() * x[c];
        }
        cpu_logits[r] = acc;
    }

    metal
        .logits_f16(
            &x,
            &embed_f16,
            vocab,
            hidden,
            "test_logits_f16_parity",
            &mut metal_logits,
        )
        .expect("metal logits_f16");

    assert_close(&cpu_logits, &metal_logits, 1e-2, 2e-3, "logits_f16");
}
