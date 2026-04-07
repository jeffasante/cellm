---
license: gemma
base_model: google/gemma-4-2p3b-it
library_name: cellm
tags:
- mobile
- rust
- memory-efficient
- quantized
- gemma
- gemma-4
- litert
datasets: []
metrics:
- latency
- throughput
pipeline_tag: text-generation
---

# Gemma-4 2.3B IT for Cellm

This repository contains the **Gemma-4 2.3B Instruct** model optimized for the [cellm](https://github.com/jeffasante/cellm) memory-efficient serving engine.

## Model Details
- **Architecture**: Gemma-4
- **Parameters**: 2.3B
- **Format**: `.cellm` (Native cellm distribution format)
- **Size**: ~2.4 GB

## Usage

### Prerequisites
You need the [cellm](https://github.com/jeffasante/cellm) engine built from source.

### Run Inference
```bash
./target/release/infer \
  --model gemma-4-2p3b-it-litert.cellm \
  --tokenizer tokenizer.json \
  --prompt "Write one short sentence about Rust programming." \
  --chat \
  --gen 24 \
  --backend metal
```

## License
This model is subject to the **Gemma Terms of Use**. By downloading or using these weights, you agree to the terms listed at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms).
