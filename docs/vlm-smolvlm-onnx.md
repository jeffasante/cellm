# SmolVLM (VLM) — Rust Validation via ONNX (SmolVLM-256M-Instruct)

This repo’s long-term goal is to run **VLMs natively inside `cellm`** (same runtime as LLMs, plus Metal later).

Right now, the fastest way to validate “can we run a VLM end-to-end on this machine?” is to use the **official ONNX exports** that ship with the SmolVLM repo and drive them from Rust.

This document explains what was implemented and how to run it.

## What we have locally

- Model folder (HF-style): `models/hf/smolvlm-256m-instruct/`
  - `tokenizer.json`, `config.json`, `added_tokens.json`
  - `onnx/vision_encoder_*.onnx`
  - `onnx/embed_tokens_*.onnx`
  - `onnx/decoder_model_merged_*.onnx`
- Test images:
  - `models/test_images/rococo.jpg`
  - `models/test_images/rococo_1.jpg`

## What was implemented

Tool: `tools/vlm-onnx-infer` (binary name: `vlm-infer`)

Pipeline (mirrors the sample code in the model card):

1. **Preprocess image**
   - Decode JPEG
   - Resize (longest edge → 512) and pad to 512×512
   - Normalize to `[-1, 1]` range (mean/std 0.5)
   - Optional local tiling (`--split-image`) to add row/col image chunks for higher detail prompts
   - Produce:
     - `pixel_values`: `[1, num_images, 3, 512, 512]` `f32`
     - `pixel_attention_mask`: `[1, num_images, 512, 512]` `bool`

2. **Tokenize prompt**
   - Build a single-turn prompt that matches the tokenizer chat template (`<|im_start|>User:...<end_of_utterance>\nAssistant:`)
   - Expand each image chunk into `image_seq_len` copies of `<image>` (64), wrapped with SmolVLM image-structure tokens
   - Encode with `tokenizers` (without auto-adding extra special tokens)

3. **Vision encoder**
   - Run `vision_encoder_*.onnx` once to get `image_features` `[num_images * 64, 576]`

4. **Embed + decode loop**
   - Run `embed_tokens_*.onnx` to get `inputs_embeds`
   - Replace the embedding rows where `input_id == image_token_id` with the vision `image_features`
   - Run `decoder_model_merged_*.onnx` in a loop, feeding:
     - `inputs_embeds`, `attention_mask`, `position_ids`
     - `past_key_values.*` (updated every step from `present.*` outputs)
   - Keep `attention_mask` as full decode history and compute position ids from that mask (prevents repetitive/garbled responses)

## How to run

Build:

```bash
cargo build --release -p cellm-vlm-onnx-infer
```

Dump ONNX I/O (names + dtypes + shapes):

```bash
./target/release/vlm-infer --model-dir models/hf/smolvlm-256m-instruct --image models/test_images/rococo.jpg --dump-io
```

Run a caption prompt (fp16 variant):

```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image in detail." \
  --backend cpu \
  --split-image \
  --max-new-tokens 128 \
  --min-new-tokens 32 \
  --temperature 0.7 \
  --top-k 40 \
  --seed 1
```

Debug token-by-token generation:

```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --max-new-tokens 64 \
  --show-tokens
```

Fast mode (no local image tiling):

```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --max-new-tokens 64
```

Try the second test image:

```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo_1.jpg \
  --prompt "What do you see in this image?" \
  --max-new-tokens 128
```

Native `.cellm` decoder with ONNX vision:

```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --cellm-model models/smolvlm-256m-int8.cellm \
  --vision-backend onnx \
  --decoder-backend cellm \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --max-new-tokens 24
```

Native `.cellm` vision + decoder (experimental, slower):

```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --cellm-model models/smolvlm-256m.cellm \
  --vision-backend cellm \
  --decoder-backend cellm \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --max-new-tokens 24
```

## Notes / current limitations

- This tool is for **local validation**. It is not the long-term `cellm` runtime path.
- Output quality still depends on exact processor parity with HuggingFace `Idefics3Processor`, but current prompt/image packing and decode masking are now stable enough for meaningful local validation.
- `.cellm` conversion for multimodal repos preserves text-backbone routing via `text_config.model_type` (so wrappers like `idefics3` can map to `llama`/`qwen` runners).
- `vlm-infer` supports `--decoder-backend cellm --cellm-model <path>` for native `.cellm` decoder execution.
- `vlm-infer` supports `--vision-backend cellm` with a native Rust vision path (patch embedding + full ViT encoder + projector).
- Native `.cellm` vision now uses SIMD-optimized BLAS matmuls on macOS (`Accelerate` SGEMM) and is dramatically faster than the previous scalar path.
- ONNX Runtime is still faster for vision in current builds.
- `--backend metal` runs a Metal smoke check; in sandboxed/restricted shells Metal discovery may fail and fall back to CPU.
- The next “real” step toward native VLM inside `cellm` is to:
  - Move native vision/decode math into optimized backend kernels (SIMD + Metal)
  - Keep processor behavior aligned with HF templates across model variants



cd /Users/jeff/Desktop/cellm
./target/release/infer \
  --model models/gemma-4-E2B-it-f16.cellmd \
  --tokenizer models/gemma-4-E2B-it/tokenizer.json \
  --prompt "what is symcophancy?" \
  --chat --chat-format gemma4 \
  --gen 24 --temperature 0.7 --top-k 40 \
  --backend cpu --kv-encoding f16


cd /Users/jeff/Desktop/cellm
./target/release/infer \
  --model models/gemma-4-E2B-it-f16.cellmd \
  --tokenizer models/gemma-4-E2B-it/tokenizer.json \
  --prompt "Hello" \
  --chat --chat-format gemma4 \
  --gen 24 --temperature 0.7 --top-k 40 \
  --backend metal --kv-encoding f16


  cd /Users/jeff/Desktop/cellm
./target/release/infer \
  --model models/gemma-4-E2B-it-f16.cellmd \
  --tokenizer models/gemma-4-E2B-it/tokenizer.json \
  --prompt "what is symcophancy?" \
  --chat --chat-format gemma4 \
  --gen 24 --temperature 0.7 --top-k 40 \
  --backend metal --kv-encoding f16