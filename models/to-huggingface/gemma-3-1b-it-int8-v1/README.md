---
library_name: cellm
tags:
- mobile
- rust
- memory-efficient
- quantized
- gemma
---

# Gemma 3 1B IT (Cellm Int8)

This folder contains a Cellm-converted Gemma 3 1B Instruct model and tokenizer assets, ready for publishing to Hugging Face.

## Files
- `gemma-3-1b-it-int8-v1.cellm`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`

## Model Details
- **Base model**: `google/gemma-3-1b-it`
- **Format**: `.cellm`
- **Quantization**: INT8 symmetric weight-only
- **Size**: ~1.2 GB

## Inference Check (cellm)

```bash
cd .
./target/release/infer \
  --model models/to-huggingface/gemma-3-1b-it-int8-v1/gemma-3-1b-it-int8-v1.cellm \
  --tokenizer models/to-huggingface/gemma-3-1b-it-int8-v1/tokenizer.json \
  --prompt "what's twitch.com?" \
  --chat \
  --chat-format plain \
  --gen 48 \
  --temperature 0 \
  --backend cpu \
  --kv-encoding f16
```

## Notes
- This INT8 variant produced coherent output in local validation.
- INT4 variant was smaller (~481 MB) but quality was significantly worse.

## License
Subject to Gemma terms and upstream license constraints.
