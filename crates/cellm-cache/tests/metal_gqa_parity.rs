// Author: Jeffrey Asante (https://jeffasante.github.io/)
#![cfg(target_os = "macos")]

/// Metal GQA attention parity tests.
///
/// Validates that the Metal `kv_attn_single_gqa_f32` kernel produces output
/// numerically equivalent to the CPU reference (`KvCacheReadView::attention_single_token_gqa_f32`).
///
/// Two scenarios are covered:
///   1. Standard attention (no soft-cap) – Llama-style.
///   2. Soft-capped attention (`soft_cap = Some(50.0)`) – Gemma-4 style.

use cellm_cache::kvcache::{KvEncodingKind, MetalKvStorage};
use cellm_core::kv_cache::DeviceKvStorage;

// ── helpers ────────

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = ((s >> 33) as u32) as f32 / u32::MAX as f32;
        x * 2.0 - 1.0
    }
}

fn fill_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = lcg(seed);
    (0..len).map(|_| rng() * scale).collect()
}

fn assert_allclose(a: &[f32], b: &[f32], max_tol: f32, mean_tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d;
    }
    let mean_abs = sum_abs / a.len() as f32;
    assert!(
        max_abs <= max_tol && mean_abs <= mean_tol,
        "{label}: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} \
         (tol max={max_tol:.4e} mean={mean_tol:.4e})\n\
         CPU[..4]={:?}\nGPU[..4]={:?}",
        &a[..a.len().min(4)],
        &b[..b.len().min(4)],
    );
}

// CPU reference: dot-product GQA with optional tanh soft-cap
fn cpu_gqa_attention(
    k: &[f32],
    v: &[f32],
    bases: &[usize],
    q: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    soft_cap: Option<f32>,
    out: &mut [f32],
) {
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let group = (n_heads / n_kv_heads).max(1);
    let seq = bases.len();
    for h in 0..n_heads {
        let kv_h = (h / group).min(n_kv_heads - 1);
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0f32; seq];
        for (t, &base) in bases.iter().enumerate() {
            let kt = &k[base + kv_h * head_dim..base + kv_h * head_dim + head_dim];
            let dot: f32 = qh.iter().zip(kt.iter()).map(|(a, b)| a * b).sum();
            let mut s = dot * scale;
            if let Some(cap) = soft_cap {
                s = (s / cap).tanh() * cap;
            }
            scores[t] = s;
        }
        // softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            denom += *s;
        }
        if denom > 0.0 {
            for s in &mut scores {
                *s /= denom;
            }
        }
        // weighted sum
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        for (t, &base) in bases.iter().enumerate() {
            let vt = &v[base + kv_h * head_dim..base + kv_h * head_dim + head_dim];
            for i in 0..head_dim {
                oh[i] += scores[t] * vt[i];
            }
        }
    }
}

// tests

/// Shared core: builds a MetalKvStorage, writes SEQ token K/V entries,
/// runs Metal attention, compares against CPU reference.
fn run_gqa_parity(soft_cap: Option<f32>, label: &str) {
    const N_HEADS: usize = 4;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 32;
    const SEQ: usize = 8;

    let kv_dim = N_KV_HEADS * HEAD_DIM;
    let total_elems = SEQ * kv_dim;

    // Build Metal storage — skip test if Metal unavailable.
    let mut storage = match MetalKvStorage::new(total_elems, N_KV_HEADS, HEAD_DIM, KvEncodingKind::F16) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping {label}: Metal unavailable: {e}");
            return;
        }
    };

    // Generate deterministic K/V data and write it into the Metal cache.
    let mut k_flat = vec![0.0f32; total_elems];
    let mut v_flat = vec![0.0f32; total_elems];
    let bases: Vec<usize> = (0..SEQ).map(|t| t * kv_dim).collect();

    for t in 0..SEQ {
        let k_token = fill_vec(kv_dim, 0x1234_0000 + t as u64, 0.4);
        let v_token = fill_vec(kv_dim, 0xABCD_0000 + t as u64, 0.4);
        // Write to Metal cache
        storage
            .write_token_f32(bases[t], &k_token, &v_token)
            .expect("write_token_f32 failed");
        // Keep CPU-side copy
        k_flat[bases[t]..bases[t] + kv_dim].copy_from_slice(&k_token);
        v_flat[bases[t]..bases[t] + kv_dim].copy_from_slice(&v_token);
    }

    // Query vector.
    let q = fill_vec(N_HEADS * HEAD_DIM, 0x9999_BEEF, 0.3);

    // CPU reference.
    let mut cpu_out = vec![0.0f32; N_HEADS * HEAD_DIM];
    cpu_gqa_attention(
        &k_flat,
        &v_flat,
        &bases,
        &q,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        soft_cap,
        &mut cpu_out,
    );

    // Metal output.
    let mut metal_out = vec![0.0f32; N_HEADS * HEAD_DIM];
    storage
        .attention_single_token_gqa_f32(
            &bases,
            &q,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            None,          // attn_scale (default)
            soft_cap,
            &mut metal_out,
        )
        .expect("Metal GQA attention failed");

    // Verify non-trivial output (not all-zeros).
    assert!(
        cpu_out.iter().any(|&v| v.abs() > 1e-6),
        "{label}: CPU output is unexpectedly all-zeros"
    );
    assert!(
        metal_out.iter().any(|&v| v.abs() > 1e-6),
        "{label}: Metal output is unexpectedly all-zeros — kernel may not be running"
    );

    // Parity check (f16 round-trip gives ~3e-3 max error in practice).
    assert_allclose(&cpu_out, &metal_out, 5e-3, 1e-3, label);
}

#[test]
fn metal_gqa_attention_no_softcap_matches_cpu() {
    run_gqa_parity(None, "metal_gqa_no_softcap");
}

#[test]
fn metal_gqa_attention_softcap50_matches_cpu() {
    run_gqa_parity(Some(50.0), "metal_gqa_softcap50");
}
