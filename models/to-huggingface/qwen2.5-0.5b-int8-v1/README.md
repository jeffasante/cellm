---
library_name: cellm
tags:
- mobile
- rust
- memory-efficient
- quantized
- qwen
---

# Qwen2.5 0.5B Instruct (Cellm Int8)

This folder contains a Cellm-converted Qwen2.5 0.5B model and tokenizer assets, ready for publishing to Hugging Face.

## Files
- `qwen2.5-0.5b-int8-v1.cellm`
- `tokenizer.json`
- `tokenizer_config.json`

## Model Details
- **Base model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Format**: `.cellm`
- **Quantization**: INT8 symmetric weight-only
- **Size**: ~487 MB

## Usage (Cellm CLI)

```bash
cd .
./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "what's twitch.com?" \
  --chat \
  --gen 64 \
  --temperature 0 \
  --backend cpu \
  --kv-encoding f16
```

## Notes
- This model required Qwen2.5 runtime parity fixes in `cellm` (optional q/k norm handling and projection bias support) to produce coherent output.
- Tokenizer consistency was verified against Python/Transformers for special IDs and chat-template tokenization.

## License
Please follow the original Qwen model license/terms when redistributing weights.
