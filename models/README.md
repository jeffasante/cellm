---
library_name: cellm
tags:
- mobile
- rust
- memory-efficient
- quantized
---

# Cellm Models Hub

This repository contains a collection of optimized Large Language Models (LLMs) in the `.cellm` format. These models are specifically tuned for high-performance inference using the [Cellm](https://github.com/jeffasante/cellm) engine, featuring Metal-accelerated kernels and memory-mapped efficiency.

## Models

### Gemma 3 1B IT (Int8)
- **Path**: `gemma-3-1b-it-int8/gemma-3-1b-it-int8.cellmd`
- **Size**: 1.3 GB
- **Type**: Quantized Int8 (Symmetric Weight-Only)

### Gemma 4 2.3B IT (LiteRT)
- **Path**: `gemma-4-2p3b-it-litert/gemma-4-2p3b-it-litert.cellm`
- **Size**: 2.4 GB
- **Type**: LiteRT-optimized for Cellm

## Usage

You can use these models directly with the Cellm CLI or in your applications:

```bash
cellm run --model-path jeffasante/cellm-models/gemma-4-2p3b-it-litert/gemma-4-2p3b-it-litert.cellm
```

## About Cellm
Cellm is a high-performance inference engine for local LLMs, written in Rust with a focus on Metal GPU acceleration and minimal memory overhead.

## License
This model is subject to the **Gemma Terms of Use**. By downloading or using these weights, you agree to the terms listed at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms).
