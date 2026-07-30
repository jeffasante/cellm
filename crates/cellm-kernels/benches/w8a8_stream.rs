// Author: Jeffrey Asante (https://jeffasante.github.io/)
// Does the INT8 GEMV hit the DRAM bandwidth roof?
//
// The w8a8 bench reuses one small matrix, so it runs cache-resident and reports
// optimistic numbers. Real decode streams the whole model from DRAM every token.
// Here we cycle through a working set far larger than L2 to measure the honest
// streaming rate, and compare against a plain memcpy-style read of the same data.
use cellm_kernels::cpu_kernels::gemv_i8_w8a8;
use half::f16;
use std::time::Instant;

fn main() {
    println!("rayon threads: {}", rayon::current_num_threads());

    let out_dim = 4608usize;
    let in_dim = 1024usize;
    let bytes_per_mat = out_dim * in_dim;

    // ~512 MB of distinct weights so nothing survives in cache between touches.
    let n_mats = (512 * 1024 * 1024) / bytes_per_mat;
    println!(
        "working set: {} matrices x {} KB = {} MB",
        n_mats,
        bytes_per_mat / 1024,
        n_mats * bytes_per_mat / 1024 / 1024
    );

    let mut seed = 0x2545F491u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (((seed >> 16) & 0xff) as i32 - 128).clamp(-127, 127) as i8
    };
    let mats: Vec<Vec<i8>> = (0..n_mats)
        .map(|_| (0..bytes_per_mat).map(|_| next()).collect())
        .collect();

    let scales: Vec<u16> = (0..out_dim)
        .map(|_| f16::from_f32(0.001).to_bits())
        .collect();
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i * 37 % 211) as f32 - 105.0) / 90.0)
        .collect();
    let mut out = vec![0.0f32; out_dim];

    // Pass 1: the real kernel, streaming.
    let t = Instant::now();
    for m in &mats {
        gemv_i8_w8a8(m, &scales, &x, &mut out, out_dim, in_dim);
    }
    let gemv = t.elapsed().as_secs_f64();
    let total_bytes = (n_mats * bytes_per_mat) as f64;
    println!(
        "gemv streaming : {:6.1} GB/s  ({:.0} GMAC/s)",
        total_bytes / gemv / 1e9,
        total_bytes / gemv / 1e9
    );

    // Pass 2: pure read bandwidth over the same buffers - the roof.
    let t = Instant::now();
    let mut sink = 0i64;
    for m in &mats {
        let mut acc = 0i64;
        let chunks = m.chunks_exact(64);
        for c in chunks {
            acc += c[0] as i64 + c[63] as i64;
        }
        sink += acc;
    }
    let read = t.elapsed().as_secs_f64();
    println!(
        "pure read roof : {:6.1} GB/s   (sink {})",
        total_bytes / read / 1e9,
        sink
    );

    println!(
        "\ngemv is at {:.0}% of the streaming read roof",
        (read / gemv) * 100.0
    );

    // Decode re-reads the *same* weights every token, so TLB entries and DRAM
    // pages stay warm. The pass above allocates each matrix separately and so
    // understates the achievable rate. Repeat over one resident buffer instead.
    let resident: Vec<i8> = mats[0..8].concat();
    let rows = resident.len() / in_dim;
    let rscales: Vec<u16> = (0..rows).map(|_| f16::from_f32(0.001).to_bits()).collect();
    let mut rout = vec![0.0f32; rows];
    let reps = 24;

    let t = Instant::now();
    for _ in 0..reps {
        gemv_i8_w8a8(&resident, &rscales, &x, &mut rout, rows, in_dim);
    }
    let rg = t.elapsed().as_secs_f64();
    let rbytes = (reps * resident.len()) as f64;

    let t = Instant::now();
    let mut sink2 = 0i64;
    for _ in 0..reps {
        for c in resident.chunks_exact(64) {
            sink2 += c[0] as i64 + c[63] as i64;
        }
    }
    let rr = t.elapsed().as_secs_f64();

    println!(
        "\nresident ({} MB, reread {}x):",
        resident.len() / 1024 / 1024,
        reps
    );
    println!("  gemv      : {:6.1} GB/s", rbytes / rg / 1e9);
    println!(
        "  read roof : {:6.1} GB/s   (sink {})",
        rbytes / rr / 1e9,
        sink2
    );
    println!("  gemv is at {:.0}% of roof", (rr / rg) * 100.0);
}
