---
library_name: cellm
tags:
- mobile
- rust
- memory-efficient
- quantized
---

# Cellm Models Hub

This folder contains `.cellm` model artifacts tested with the Cellm Rust CLI.

## Models

### SmolLM2 360M Instruct (INT8)
- **Path**: `models/smollm2-360m-int8-v1.cellm`
- **Size**: ~391 MB
- **Tokenizer**: `models/hf/smollm2-360m/tokenizer.json`
- **Type**: INT8 symmetric weight-only
- **Runtime note**: set `CELLM_LLAMA_ROPE_INTERLEAVED=0` when running SmolLM2

### Qwen2.5 0.5B Instruct (INT8)
- **Path**: `models/qwen2.5-0.5b-int8-v1.cellm`
- **Size**: ~472 MB
- **Tokenizer**: `models/qwen2.5-0.5b-bnb4/tokenizer.json`
- **Type**: INT8 symmetric weight-only

### Gemma-3 1B IT (INT4, smallest)
- **Path**: `models/gemma-3-1b-it-int4-v1.cellm`
- **Size**: ~478 MB
- **Tokenizer**: `models/hf/gemma-3-1b-it-full/tokenizer.json`
- **Type**: INT4 symmetric weight-only

### Gemma-3 1B IT (Mixed INT4, recommended)
- **Path**: `models/gemma-3-1b-it-mixed-int4-v1.cellm`
- **Size**: ~1.0 GB
- **Tokenizer**: `models/hf/gemma-3-1b-it-full/tokenizer.json`
- **Type**: Mixed precision (attention/embeddings higher precision, MLP mostly INT4)

### Gemma-3 1B IT (INT8, most stable)
- **Path**: `models/gemma-3-1b-it-int8-v1.cellm`
- **Size**: ~1.2 GB
- **Tokenizer**: `models/hf/gemma-3-1b-it-full/tokenizer.json`
- **Type**: INT8 symmetric weight-only

## Usage

From `.`, run:

```bash
CELLM_LLAMA_ROPE_INTERLEAVED=0 ./target/release/infer \
  --model models/smollm2-360m-int8-v1.cellm \
  --tokenizer models/hf/smollm2-360m/tokenizer.json \
  --prompt "What is sycophancy?" \
  --chat \
  --chat-format auto \
  --gen 48 \
  --temperature 0 \
  --backend metal \
  --kv-encoding f16
```

```bash
./target/release/infer \
  --model models/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/qwen2.5-0.5b-bnb4/tokenizer.json \
  --prompt "What is sycophancy?" \
  --chat \
  --gen 64 \
  --temperature 0 \
  --backend metal \
  --kv-encoding f16
```

```bash
./target/release/infer \
  --model models/gemma-3-1b-it-mixed-int4-v1.cellm \
  --tokenizer models/hf/gemma-3-1b-it-full/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat \
  --chat-format plain \
  --gen 48 \
  --temperature 0 \
  --backend metal \
  --kv-encoding f16
```

## About Cellm
Cellm is a Rust-native inference runtime focused on mobile/desktop local LLM serving with Metal acceleration and memory-mapped model loading.

## License
Please follow each upstream model license (Qwen and Gemma terms) when redistributing weights and tokenizers.
