# Converting Quantized HF Models

`cellm`’s `.cellm` format is designed so the runtime can memory-map model tensors.

Today, the converter (`tools/convert`) targets a simple baseline:

- Output dtype: **f16**
- Input dtypes supported directly: **f16**, **bf16**, **f32**

## 4-bit affine checkpoints (packed `uint32`)

Some HuggingFace checkpoints store 4-bit weights as:

- `*.weight` as **packed `uint32`** (each `u32` contains 8 x 4-bit values)
- plus side tensors:
  - `*.scales` (per-group scale)
  - `*.biases` (per-group bias / offset)

Example folder:

- `models/hf/qwen3.5-0.8b-4bit/`

### What `convert` does

If you pass `--dequant-4bit-affine`, `convert` will:

- Detect `uint32` `*.weight` tensors that have matching `*.scales` and `*.biases`
- Expand them into full **f16** matrices in the `.cellm` output
- Drop the `*.scales` / `*.biases` tensors from the output (because the weights are already dequantized)

Command:

```bash
cargo run --release --bin convert -- \
  --input  models/hf/qwen3.5-0.8b-4bit \
  --output models/qwen3.5-0.8b.cellm \
  --dtype  f16 \
  --dequant-4bit-affine
```

### Important tradeoff

Dequantizing 4-bit weights to f16 can make the `.cellm` file **much larger** (often ~4x vs int4).

Long term, the “right” solution is:

- Keep int4 weights in `.cellm` with explicit metadata
- Implement int4 matmul kernels (CPU/Metal)

But `--dequant-4bit-affine` is a practical way to get a model running while kernels and model-family adapters are still being built.

