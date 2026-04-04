---
license: gemma
base_model: google/gemma-3-1b-it
library_name: cellm
tags:
- mobile
- rust
- memory-efficient
- quantized
- gemma
- gemma-3
datasets: []
metrics:
- latency
- throughput
pipeline_tag: text-generation
---

# Gemma-3 1B IT Int8 for Cellm

This repository contains the **Gemma-3 1B Instruct** model quantized to **Int8 Symmetric Weight-Only** format, optimized for the [cellm](https://github.com/jeffasante/cellm) memory-efficient serving engine.

## Model Details
- **Architecture**: Gemma-3
- **Parameters**: 1B
- **Quantization**: Int8 Symmetric (Per-row)
- **Format**: `.cellmd` (Native cellm distribution format)
- **Size**: ~1.3 GB

## Performance Snapshots
Measured locally on April 3, 2026 (Prompt: "hi", Gen: 8, report = median / P95):

| Backend | Startup (s) | Prefill (s) | Decode (s) |
|---|---:|---:|---:|
| **CPU** | `1.02 / 1.11` | `0.56 / 2.75` | `4.12 / 4.16` |
| **Metal** | `1.06 / 1.07` | `0.80 / 0.98` | `3.62 / 4.01` |

## Usage

### Prerequisites
You need the [cellm](https://github.com/jeffasante/cellm) engine built from source.

### Run Inference
```bash
./target/release/infer \
  --model gemma-3-1b-it-int8.cellmd \
  --tokenizer tokenizer.json \
  --prompt "Write one short sentence about Rust programming." \
  --chat \
  --gen 24 \
  --backend metal
```

## License
This model is subject to the **Gemma Terms of Use**. By downloading or using these weights, you agree to the terms listed at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms).
