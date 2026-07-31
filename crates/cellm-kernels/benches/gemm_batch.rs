//! Measures the batched INT8 GEMM against the per-token GEMV it replaces.
//!
//! Shapes are the real LFM2-350M projections. Run with:
//! `cargo run --release -p cellm-kernels --bench gemm_batch`

use cellm_kernels::cpu_kernels::{gemm_i8_w8a8, gemv_i8_w8a8};
use half::f16;
use std::time::Instant;

fn build(out_dim: usize, in_dim: usize, n_tokens: usize) -> (Vec<i8>, Vec<u16>, Vec<f32>) {
    let mut seed = 0x12345678u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 16) & 0xff) as i32 - 128
    };
    let w: Vec<i8> = (0..out_dim * in_dim)
        .map(|_| next().clamp(-127, 127) as i8)
        .collect();
    let scales: Vec<u16> = (0..out_dim)
        .map(|r| f16::from_f32(0.001 + (r % 7) as f32 * 0.0003).to_bits())
        .collect();
    let x: Vec<f32> = (0..n_tokens * in_dim)
        .map(|i| ((i * 37 % 211) as f32 - 105.0) / 90.0)
        .collect();
    (w, scales, x)
}

fn main() {
    // The projections that dominate LFM2-350M prefill.
    let shapes: &[(&str, usize, usize)] = &[
        ("conv.in_proj   3072x1024", 3072, 1024),
        ("conv.out_proj  1024x1024", 1024, 1024),
        ("ffn.w1         4608x1024", 4608, 1024),
        ("ffn.w2         1024x4608", 1024, 4608),
    ];
    let batches = [1usize, 8, 16, 32, 64];

    println!(
        "{:<26} {:>6} {:>10} {:>10} {:>8}",
        "shape", "batch", "gemv ms", "gemm ms", "speedup"
    );

    for &(name, out_dim, in_dim) in shapes {
        for &nt in &batches {
            let (w, scales, x) = build(out_dim, in_dim, nt);
            let mut out_v = vec![0.0f32; nt * out_dim];
            let mut out_g = vec![0.0f32; nt * out_dim];

            // Warm the caches so the first shape is not penalised.
            for t in 0..nt {
                gemv_i8_w8a8(
                    &w,
                    &scales,
                    &x[t * in_dim..(t + 1) * in_dim],
                    &mut out_v[t * out_dim..(t + 1) * out_dim],
                    out_dim,
                    in_dim,
                );
            }

            let reps = 3;
            let t0 = Instant::now();
            for _ in 0..reps {
                for t in 0..nt {
                    gemv_i8_w8a8(
                        &w,
                        &scales,
                        &x[t * in_dim..(t + 1) * in_dim],
                        &mut out_v[t * out_dim..(t + 1) * out_dim],
                        out_dim,
                        in_dim,
                    );
                }
            }
            let gemv_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

            let t1 = Instant::now();
            for _ in 0..reps {
                gemm_i8_w8a8(&w, &scales, &x, &mut out_g, nt, out_dim, in_dim);
            }
            let gemm_ms = t1.elapsed().as_secs_f64() * 1000.0 / reps as f64;

            // Guard against a fast-but-wrong kernel.
            let mismatches = out_v
                .iter()
                .zip(&out_g)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(mismatches, 0, "{name} batch {nt}: {mismatches} mismatches");

            println!(
                "{:<26} {:>6} {:>10.2} {:>10.2} {:>7.2}x",
                name,
                nt,
                gemv_ms,
                gemm_ms,
                gemv_ms / gemm_ms
            );
        }
    }
}
