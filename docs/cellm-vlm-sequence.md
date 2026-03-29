# `.cellm` VLM Sequence Status

This file tracks the sequence for moving from ONNX validation to native `.cellm` VLM execution.

## 1) VLM-aware `.cellm` schema

Implemented:
- `.cellm` header now stores:
  - `source_model_type`
  - `text_tensor_prefix`
  - `vision_tensor_prefix`
  - `projector_tensor_prefix`
  - `source_text_config`
  - `source_vision_config`
  - `source_projector_config`
- Converter fills these fields from HF `config.json` and tensor-name inspection.

## 2) SmolVLM `.safetensors -> .cellm` conversion

Implemented:
- `convert` now routes model type from `text_config.model_type` for multimodal wrappers.
- SmolVLM conversion now succeeds end-to-end with:

```bash
cargo run --release --bin convert -- \
  --input models/hf/smolvlm-256m-instruct \
  --output models/smolvlm-256m.cellm \
  --dtype f16
```

## 3) `.cellm` text-stack inference path

Implemented:
- `infer` and SDK choose text runner from `source_text_config.model_type` when present.
- Llama runner now supports multimodal text tensor naming (`model.text_model.*`) and prefixed layouts.
- `vlm-infer` now supports `--decoder-backend cellm --cellm-model <path>`:
  - vision encoder stays ONNX
  - decoder runs through native `.cellm` Llama path (including int8 quantized `.cellm`)
- `vlm-infer` now also supports `--vision-backend cellm` (experimental):
  - loads vision + projector tensors from `.cellm`
  - runs patch embedding + full ViT encoder blocks + post layernorm + connector projection in Rust
  - computes image features (`[64, hidden]`) without ONNX model execution

Current limitation:
- Native vision runtime now uses SIMD-optimized BLAS matmuls on macOS and is much faster than the earlier scalar implementation.
- ONNX Runtime remains faster; Metal-native fused kernels are still pending.

## 4) Quantization currently available

Implemented (working):
- `--quantize-int8-symmetric` in `convert` for Llama text stacks.
- Weight-only per-row symmetric int8 + f16 row scales.
- Runtime dequant support in Llama linear layers and logits source.

## 5) Next implementation steps

1. Move native vision kernels from `vlm-infer` into reusable backend/kernel crates.
2. Add optimized kernels (SIMD/Metal) for ViT attention + MLP.
3. Add additional source formats:
   - PyTorch `.bin`/`.pt`
   - Flax/JAX
   - ONNX -> `.cellm`
