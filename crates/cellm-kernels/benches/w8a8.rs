// Author: Jeffrey Asante (https://jeffasante.github.io/)
// Rough throughput probe for the INT8 GEMV kernels.
// Run with: cargo run --release -p cellm-kernels --bench w8a8
use cellm_kernels::cpu_kernels::{gemv_i8_f32, gemv_i8_w8a8};
use half::f16;
use std::time::Instant;

fn bench(out_dim: usize, in_dim: usize, iters: usize) {
    let mut seed = 0x2545F491u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (((seed >> 16) & 0xff) as i32 - 128).clamp(-127, 127) as i8
    };
    let w: Vec<i8> = (0..out_dim * in_dim).map(|_| next()).collect();
    let scales: Vec<u16> = (0..out_dim).map(|_| f16::from_f32(0.001).to_bits()).collect();
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i * 37 % 211) as f32 - 105.0) / 90.0)
        .collect();
    let mut out = vec![0.0f32; out_dim];

    let macs = (out_dim * in_dim * iters) as f64;

    gemv_i8_w8a8(&w, &scales, &x, &mut out, out_dim, in_dim);
    let t = Instant::now();
    for _ in 0..iters {
        gemv_i8_w8a8(&w, &scales, &x, &mut out, out_dim, in_dim);
    }
    let dot = t.elapsed().as_secs_f64();

    gemv_i8_f32(&w, &scales, &x, &mut out, out_dim, in_dim);
    let t = Instant::now();
    for _ in 0..iters {
        gemv_i8_f32(&w, &scales, &x, &mut out, out_dim, in_dim);
    }
    let fp = t.elapsed().as_secs_f64();

    println!(
        "{out_dim:>6} x {in_dim:<6}  sdot {:7.1} GMAC/s ({:6.1} us)   fpmac {:7.1} GMAC/s   speedup {:.2}x",
        macs / dot / 1e9,
        dot / iters as f64 * 1e6,
        macs / fp / 1e9,
        fp / dot
    );
}

fn main() {
    println!("rayon threads: {}", rayon::current_num_threads());
    bench(1024, 1024, 2000);
    bench(3072, 1024, 2000);
    bench(4608, 1024, 2000);
    bench(1024, 4608, 2000);
    bench(65536, 1024, 200);
}
