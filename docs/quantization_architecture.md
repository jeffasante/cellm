# Cellm Quantization Architecture

Cellm uses **Weight-Only Quantization** to balance memory footprint and inference speed on edge devices (iOS, macOS, Android). This document outlines the technical implementation of our quantization strategy.

## 1. Supported Data Types
The core supported types are defined in `cellm-core/src/dtype.rs`. While full-precision formats are supported, the engine is optimized for 8-bit quantized weights.

```rust:crates/cellm-core/src/dtype.rs
pub enum DType {
    F32,     // 32-bit float
    F16,     // 16-bit half-precision float
    BF16,    // 16-bit bfloat
    I8,      // 8-bit integer with per-row scaling
    U8,      // 8-bit unsigned integer
    Q8_0,    // 8-bit symmetric block quantization (block size = 32)
}
```

---

## 2. Int8 with Per-Row Scaling (`i8`)
Most Cellm models utilize an `i8` format with associated **qscales**. This provides a significant 50% memory reduction over `f16` while maintaining high accuracy.

### Storage Layout
In a `.cellm` file, a quantized linear layer consists of two tensors:
1. `weights`: Raw `i8` integer bytes.
2. `weights.qscale`: One `f16` scaling factor **per row** of the weight matrix.

### On-the-Fly Dequantization
The weights are dequantized during the dot-product calculation. This avoids the need to ever expand the full model into high-precision memory (FP32/FP16).

```rust:crates/cellm-model/src/llama.rs
// Implementation of a quantized linear projection in the LlamaRunner
"i8" => {
    let w = self.tensor_i8_by_exact_name(&resolved)?;
    let scales = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;

    for j in 0..out_dim {
        let row = &w[j * in_dim..(j + 1) * in_dim];            // Raw i8 weights
        let scale = f16::from_bits(scales[j]).to_f32();       // Per-row scale
        let mut acc = 0.0f32;

        for i in 0..in_dim {
            // Apply scale during multiplication to minimize precision loss
            acc += x[i] * ((row[i] as f32) * scale);
        }
        out[j] = acc;
    }
}
```

---

## 3. Block Quantization (`Q8_0`)
For even tighter constraints, we support `Q8_0` (8-bit symmetric block quantization). This is a more complex storage format where small groups of weights (e.g., 32 values) share a single scaling factor.

- **Block Size**: 32 elements.
- **Storage**: 32 `i8` values + 1 `f32` scale per block.
- **Overhead**: 36 bytes for 32 weights (approx. 1.125 bytes per element).

---

## 4. Why Quantize for Edge Devices?

### Memory Efficiency
A typical 2B parameter VLM model takes up ~4GB in FP16. On an iPhone with 6GB-8GB of total RAM (shared with the OS and UI), this is risky. Our `i8` quantization brings this down to ~2GB, allowing the model to run comfortably on older devices.

### Reduced Disk/Flash I/O
Since `.cellm` files are **memory-mapped**, the engine only reads bytes from storage when they are needed for a calculation. Quantizing to 8-bit halves the amount of data the OS needs to swap from flash memory to the cache during every forward pass.

### Hardware Acceleration
By keeping weights in a compact format, we optimize cache locality. Our **Metal** kernels are designed to leverage these smaller types for faster throughput on Apple Silicon GPUs.
