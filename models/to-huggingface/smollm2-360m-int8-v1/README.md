---
library_name: cellm
tags:
- mobile
- rust
- memory-efficient
- quantized
- smollm2
---

# SmolLM2 360M Instruct (Cellm Int8)

This folder contains a Cellm-converted SmolLM2 360M model and tokenizer assets, ready for publishing to Hugging Face.

## Files
- `smollm2-360m-int8-v1.cellm`
- `tokenizer.json`
- `tokenizer_config.json`
- `generation_config.json`

## Model Details
- **Base model**: `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Format**: `.cellm`
- **Quantization**: INT8 symmetric weight-only
- **Size**: ~391 MB

## Usage (Cellm CLI)

```bash
cd .
CELLM_LLAMA_ROPE_INTERLEAVED=0 ./target/release/infer \
  --model models/to-huggingface/smollm2-360m-int8-v1/smollm2-360m-int8-v1.cellm \
  --tokenizer models/to-huggingface/smollm2-360m-int8-v1/tokenizer.json \
  --prompt "What is sycophancy?" \
  --chat \
  --chat-format auto \
  --gen 48 \
  --temperature 0 \
  --backend metal \
  --kv-encoding f16
```

## Notes
- SmolLM2 requires non-interleaved RoPE in the current Llama runner path. Use `CELLM_LLAMA_ROPE_INTERLEAVED=0`.
- Tokenizer and generation behavior were cross-checked against Python/Transformers.

## License
Please follow the original SmolLM2 model license/terms when redistributing weights.
