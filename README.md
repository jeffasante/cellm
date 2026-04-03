# cellm — Mobile-Native LLM Serving Engine

**cellm** is a ground-up LLM serving engine for iOS and Android, written in Rust. It brings serving-engine concepts — paged KV cache management, continuous decode scheduling, multi-session concurrency, and a high-performance CLI — to phones running under 512MB RAM. 

Not a wrapper around `llama.cpp`. Not a port of `vLLM`. A new runtime designed for mobile constraints from scratch.

## Current Status: Phase 6 (Multimodal Vision)
**cellm** has evolved from an honest baseline into a multimodal-ready inference engine:

- [x] **Paged KV Cache**: Fixed-size block allocation using `BlockAllocator` & `PageTable`.
- [x] **Multi-session Scheduler**: Round-robin interleaved decoding for concurrent users.
- [x] **4-bit Affine Dequantization**: Native support for high-precision 4-bit packed weights from MLX/HF.
- [x] **Multimodal Vision**: Native ViT/SigLIP vision encoder and linear projector integration.
- [x] **Accelerated Math**: Metal (macOS/iOS) compute kernels and SIMD-optimized CPU fallbacks.
- [x] **High-Performance CLI**: Suite of tools for `.cellm` conversion, latency benchmarking, and debug inference.
- [ ] **Vulkan Support**: Cross-platform compute kernels (Active Research).
- [ ] **Android Integration**: Native Kotlin/JNI bindings and performance tuning (Coming Soon).
- [ ] **Qwen iOS Porting**: Integrate and optimize Qwen inference path for native iOS deployment.

---

## Getting Started

### Prerequisites
- Rust 1.75+ (Modern toolchain recommended)

### Build
To build the workspace:
```bash
cargo build --release
```

### Run Inference (Smoke Test)
Run unoptimized CPU inference for a `.cellm` model:
```bash
cargo run --release --bin infer -- \
  --model models/smollm2-135m.cellm \
  --tokenizer models/hf/smollm2-135m/tokenizer.json \
  --prompt "Hello, how are you?" \
  --chat \
  --gen 32
```

Notes:
- `--chat` auto-detects ChatML-style tokens (for SmolLM2 this uses `<|im_start|>` / `<|im_end|>`). Without chat formatting, many base models behave like “text completion” and may not answer directly.
- Use `--chat-format plain` to force the simpler `User:/Assistant:` style. `--chat-format auto` (default) only uses ChatML when the tokenizer advertises a chat template in `tokenizer_config.json`.
- `--max-layers` is only for debugging; using fewer layers will significantly degrade quality.
- `--backend metal` now performs a Metal kernel smoke check before inference, then falls back to current CPU math paths (full Metal forward kernels are still in progress).

Run with backend selection:
```bash
# CPU
cargo run --release --bin infer -- \
  --model models/smollm2-135m-int8.cellm \
  --tokenizer models/hf/smollm2-135m/tokenizer.json \
  --prompt "Hello" \
  --chat \
  --gen 16 \
  --backend cpu

# Metal (auto-falls back to CPU if unavailable)
cargo run --release --bin infer -- \
  --model models/smollm2-135m-int8.cellm \
  --tokenizer models/hf/smollm2-135m/tokenizer.json \
  --prompt "Hello" \
  --chat \
  --gen 16 \
  --backend metal
```

Qwen3.5 notes:
- Qwen3.5 4-bit MLX tokenizers sometimes store BPE merges as `[[a,b], ...]` instead of `["a b", ...]`. `infer` auto-normalizes this on load.
- Qwen3.5 mixes `full_attention` layers and `linear_attention` (DeltaNet / gated delta rule) layers. `infer` includes a CPU reference implementation for both.
- DeltaNet is stateful per session (separate from the paged KV cache). See `docs/qwen3_5-deltanet.md`.

Qwen3.5 quantization findings (March 30, 2026):
- Baseline text+vision `.cellm`: `models/qwen3.5-0.8b.cellm` = `1746.9 MB`
- Int8 weight-only: `models/qwen3.5-0.8b-int8.cellm` = `1706.0 MB`
- Int4 weight-only (all stacks): `models/qwen3.5-0.8b-int4.cellm` = `620.4 MB`
- Int4 weight-only + text-only tensors: `models/qwen3.5-0.8b-int4-textonly.cellm` = `378.3 MB` (< 500MB)

Create Qwen3.5 int4:
```bash
./target/release/convert \
  --input models/hf/qwen3.5-0.8b \
  --output models/qwen3.5-0.8b-int4.cellm \
  --quantize-int4-symmetric
```

Create Qwen3.5 int4 text-only:
```bash
./target/release/convert \
  --input models/hf/qwen3.5-0.8b \
  --output models/qwen3.5-0.8b-int4-textonly.cellm \
  --quantize-int4-symmetric \
  --text-only
```

Run prompt test on `qwen3.5-0.8b-int4.cellm`:
```bash
./target/release/infer \
  --model models/qwen3.5-0.8b-int4.cellm \
  --tokenizer models/hf/qwen3.5-0.8b/tokenizer.json \
  --prompt "Write one short sentence about Rust programming." \
  --chat \
  --chat-format chatml \
  --gen 24 \
  --temperature 0
```

Sample output:
```text
Rust programming serves as a concise and concise language that simplifies the creation of programs, making it easier to understand the
```

Qwen int4 Metal run note:
- `--backend metal` currently reports: `Backend: metal (smoke ok). Forward path currently uses CPU math kernels.`
- This means Metal device/kernel smoke passes, but Qwen forward execution in `infer` is still CPU-path today.

Repro command:
```bash
./target/release/infer \
  --model models/qwen3.5-0.8b-int4-textonly.cellm \
  --tokenizer models/hf/qwen3.5-0.8b/tokenizer.json \
  --prompt "Return exactly one uppercase letter: R" \
  --chat \
  --chat-format chatml \
  --gen 4 \
  --temperature 0 \
  --backend metal
```

Run prompt test on `qwen3.5-0.8b-int4-textonly.cellm`:
```bash
./target/release/infer \
  --model models/qwen3.5-0.8b-int4-textonly.cellm \
  --tokenizer models/hf/qwen3.5-0.8b/tokenizer.json \
  --prompt "Write one short sentence about Rust programming." \
  --chat \
  --chat-format chatml \
  --gen 24 \
  --temperature 0
```

Sample output:
```text
Rust programming serves as a concise and concise language that simplifies the creation of programs, making it easier to understand the
```

FunctionGemma mobile-actions (Gemma3 text) action smoke test:
```bash
./target/release/infer \
  --model models/functiongemma-270m-mobile-actions.cellm \
  --tokenizer models/hf/functiongemma-270m-mobile-actions/mobile/tokenizer.json \
  --prompt "You are a phone automation model. User request: Turn on Wi-Fi and open Settings. Return only a short action plan." \
  --chat \
  --chat-format plain \
  --gen 32 \
  --temperature 0 \
  --backend metal
```

Note:
- This confirms Gemma linear layers run on Metal (`LLM linear backend: metal (Gemma linear layers)`).
- Current output quality for this converted mobile checkpoint is still poor/repetitive; architecture support is in active bring-up.

Observed output (April 3, 2026):
```text
setPrototypeOfsetPrototypeOfsetPrototypeOf...
```

Supported Mobile Actions (reference prompts):

| Function | Example Prompt |
| --- | --- |
| Flashlight | "Turn on the flashlight" |
| Contacts | "Create a contact for John Doe with phone 555-1234" |
| Email | "Send email to john@example.com" |
| Maps | "Show Times Square on the map" |
| WiFi | "Turn off WiFi" |
| Calendar | "Create a calendar event for Team Meeting tomorrow at 2 PM" |

### Run VLM (SmolVLM-256M via ONNX, Rust validation)
SmolVLM-256M-Instruct ships official ONNX exports (`vision_encoder`, `embed_tokens`, `decoder_model_merged`). For quick local validation on macOS, use:

```bash
cargo build --release -p cellm-vlm-onnx-infer

./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image in detail." \
  --backend cpu \
  --split-image \
  --max-new-tokens 128
```

Recommended “works now” test command:

```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --split-image \
  --max-new-tokens 96 \
  --min-new-tokens 24 \
  --temperature 0.7 \
  --top-k 40 \
  --seed 1
```

Second image test:

```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo_1.jpg \
  --prompt "Describe this image." \
  --split-image \
  --max-new-tokens 96 \
  --min-new-tokens 24 \
  --temperature 0.7 \
  --top-k 40 \
  --seed 1
```

Notes:
- Use `--split-image` for best quality on detailed scenes (global + local tiles). It is slower, but improves caption relevance.
- Keep `--split-image` off for faster smoke tests.
- `vlm-infer` now computes decoder `position_ids` from the full attention history and grows `attention_mask` across decode steps, which fixes repetitive/garbled outputs from earlier builds.

Native `.cellm` decoder path (experimental):
```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --cellm-model models/smolvlm-256m-int8.cellm \
  --decoder-backend cellm \
  --onnx-variant fp16 \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --max-new-tokens 24
```
- This uses ONNX for vision encoding and native `.cellm` for decoder text-stack execution.

Native `.cellm` vision + decoder path (experimental):
```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --cellm-model models/smolvlm-256m.cellm \
  --vision-backend cellm \
  --decoder-backend cellm \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --max-new-tokens 12
```
- This bypasses ONNX model execution for both vision and decoder.
- Native vision now runs a full ViT encoder path in Rust (patch embed + 12 transformer blocks + post layernorm + connector projection).
- The native path now uses SIMD-optimized BLAS (`Accelerate` SGEMM on macOS) for linear layers and attention matmuls.
- Current limitation: ONNX Runtime is still faster for vision on the same machine.

SDK FFI VLM smoke test (same native `.cellm` vision+decoder stack used by iOS):
```bash
CELLM_VLM_TOKENIZER=models/hf/smolvlm-256m-instruct/tokenizer.json \
cargo run --release --bin vlm-smoke -- \
  --model models/smolvlm-256m-int8.cellm \
  --image models/test_images/rococo_1.jpg \
  --prompt "Describe this image."
```

Run VLM with backend selection:
```bash
# ONNX vision+decoder, CPU backend selection
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --backend cpu

# ONNX vision+decoder, Metal requested (auto-fallback if unavailable)
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --backend metal
```

See [`docs/vlm-smolvlm-onnx.md`](docs/vlm-smolvlm-onnx.md) for details, debug flags, and current limitations.
Sequence progress is tracked in [`docs/cellm-vlm-sequence.md`](docs/cellm-vlm-sequence.md).

### Run Benchmarks
Test the latency and memory throughput of the engine:
```bash
# Run with tiny test configuration
cargo run --release --bin bench -- --model tiny

# Run with SmolLM2-135M configuration
cargo run --release --bin bench -- --model smollm2-135m --seq 128 --gen 64
```

Recent run-time snapshots (measured locally on March 29, 2026; these are reference numbers, not guaranteed across machines):
| Run | Key Params | Vision Time | Prefill Time | Decode Time |
|---|---|---:|---:|---:|
| `infer` | `smollm2-135m.cellm`, 30 layers, prompt `"Hello, how are you?"`, `--chat`, `--gen 8` | N/A | `12` tokens in `2.35s` | `8` tokens in `1.62s` |
| `infer` | `smollm2-135m-int8.cellm`, prompt `"Hello, how are you?"`, `--chat`, `--gen 16` | N/A | `12` tokens in `2.38s` | `16` tokens in `3.36s` |
| `vlm-infer` ONNX fp16 | `rococo.jpg`, `--max-new-tokens 16` | `[64, 576]` in `0.99s` | N/A | `16` tokens in `1.54s` |
| `vlm-infer` ONNX quantized | `rococo.jpg`, `--onnx-variant quantized`, `--max-new-tokens 24` | `[64, 576]` in `1.44s` | N/A | `18` tokens in `0.35s` (EOS step `17`) |
| `vlm-infer` native vision + ONNX decoder | `rococo.jpg`, `--vision-backend cellm --decoder-backend onnx --max-new-tokens 16` | `[64, 576]` in `5.96s` | N/A | `16` tokens in `4.15s` |
| `vlm-infer` native vision + native decoder | `rococo.jpg`, `--vision-backend cellm --decoder-backend cellm --max-new-tokens 24` | `[64, 576]` in `5.65s` | N/A | `24` tokens in `18.39s` |

CPU vs Metal request benchmark (same prompts/settings, local run on March 29, 2026):
| Tool | Backend Arg | Host Log | Vision Time | Prefill Time | Decode Time |
|---|---|---|---:|---:|---:|
| `infer` (`smollm2-135m-int8`, `--gen 16`) | `--backend cpu` | `Backend: cpu (macos/aarch64)` | N/A | `12` tokens in `1.77s` | `16` tokens in `2.07s` |
| `infer` (`smollm2-135m-int8`, `--gen 16`) | `--backend metal` | `Backend: metal (smoke ok)` | N/A | `12` tokens in `1.75s` | `16` tokens in `2.07s` |
| `vlm-infer` (`fp16`, `rococo.jpg`, `--max-new-tokens 16`) | `--backend cpu` | `Backend: cpu (macos/aarch64)` | `[64, 576]` in `2.28s` | N/A | `16` tokens in `2.37s` |
| `vlm-infer` (`fp16`, `rococo.jpg`, `--max-new-tokens 16`) | `--backend metal` | `Backend: metal (smoke ok)` | `[64, 576]` in `1.85s` | N/A | `16` tokens in `2.56s` |

CPU vs Metal LLM benchmark snapshot (3 passes, local run on April 3, 2026; prompt=`"hi"`, `--gen 8`, report = median / P95):
| Model | Backend | Startup (s) | Prefill (s) | Decode (s) |
|---|---|---:|---:|---:|
| `smollm2-135m-int8.cellm` | `cpu` | `0.04 / 0.04` | `0.07 / 0.30` | `0.50 / 0.51` |
| `smollm2-135m-int8.cellm` | `metal` | `0.14 / 0.15` | `0.95 / 1.36` | `0.33 / 0.86` |
| `gemma-3-1b-it-int8.cellmd` | `cpu` | `1.02 / 1.11` | `0.56 / 2.75` | `4.12 / 4.16` |
| `gemma-3-1b-it-int8.cellmd` | `metal` | `1.06 / 1.07` | `0.80 / 0.98` | `3.62 / 4.01` |
| `qwen3.5-0.8b-int8.cellm` | `cpu` | `0.28 / 0.32` | `0.52 / 1.98` | `3.79 / 3.85` |
| `qwen3.5-0.8b-int8.cellm` | `metal` | `0.34 / 0.34` | `0.46 / 0.52` | `2.47 / 2.55` |

Note: in restricted/sandboxed shells, Metal device discovery can fail and trigger CPU fallback. On host macOS runs, Metal smoke succeeds.

For a dedicated benchmark page (commands + tables), see `docs/benchmarks/README.md`.

Metal troubleshooting:
```bash
# 1) Verify Metal device access
cargo run --release --bin metal-smoke

# 2) Verify infer picks Metal
./target/release/infer \
  --model models/smollm2-135m-int8.cellm \
  --tokenizer models/hf/smollm2-135m/tokenizer.json \
  --prompt "hello" \
  --gen 8 \
  --backend metal
```

### Convert Models
(Local) Convert HuggingFace safetensors to `.cellm`:
```bash
cargo run --bin convert -- \
  --input  ./models/hf/smollm2-135m \
  --output ./models/smollm2-135m.cellm \
  --dtype  f16
```

PyTorch checkpoint import (`.bin` / `.pt`) is also supported (auto-converted to temporary safetensors first):
```bash
cargo run --bin convert -- \
  --input  ./models/hf/some-model/pytorch_model.bin \
  --output ./models/some-model.cellm \
  --dtype  f16
```

Working quantization option (Llama text stacks):
```bash
cargo run --bin convert -- \
  --input  ./models/hf/smollm2-135m \
  --output ./models/smollm2-135m-int8.cellm \
  --dtype  f16 \
  --quantize-int8-symmetric
```
- This is weight-only per-row symmetric int8 for attention/MLP linear weights.
- `infer` runs these quantized weights directly (with per-row f16 scales).

Multimodal checkpoints (example: SmolVLM):
- `convert` now reads `text_config.model_type` (not just top-level `model_type`) when writing `.cellm`, so multimodal wrappers can map to the correct text backbone runner (`llama`/`qwen`) in `infer`/SDK.
- `.cellm` headers now include VLM-aware sections: `text_tensor_prefix`, `vision_tensor_prefix`, `projector_tensor_prefix`, plus source `vision_config`/projector config metadata.
- If conversion fails with `metadata incomplete`, the local `.safetensors` file is truncated and must be re-downloaded before conversion.
- Keep enough free disk space for conversion (typically at least model_size + output_size + working headroom).
- Native `.cellm` VLM execution is now available in `vlm-infer` with `--vision-backend cellm --decoder-backend cellm` (experimental CPU path).

Quantized validation status:
- Text (quantized `.cellm`): tested and working.
- `smollm2-135m-int8.cellm` runs in `infer` and generates output.
- `smolvlm-256m-int8.cellm` also runs through the text path (`infer`).
- Qwen3.5 int8 parity note: for `qwen3.5-0.8b-int8.cellm`, keep `linear_attn.*` projection weights in f16 during quantization (do not int8 those tensors) to avoid degenerate output.
- Vision (quantized): tested with ONNX VLM path (`vlm-infer --onnx-variant quantized`) and produces image-relevant captions.
- Native `.cellm` vision execution is implemented in `vlm-infer` (`--vision-backend cellm`) and tested with both ONNX and native decoders.
- Native `.cellm` vision currently prioritizes correctness over speed (CPU-only Rust math, no fused kernels yet).

Quantized sizes:
- `models/smollm2-135m.cellm`: `257M`
- `models/smollm2-135m-int8.cellm`: `156M` (~39% smaller)
- `models/smolvlm-256m.cellm`: `489M`
- `models/smolvlm-256m-int8.cellm`: `308M` (~37% smaller)

Quantized checkpoints:
- Some HF folders (e.g. 4-bit affine: `uint32` packed weights + `*.scales`/`*.biases`) require expanding weights to f16 during conversion: add `--dequant-4bit-affine`. This increases output size.

**Recommended Model**: [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main)

### Sample Models
Sample `.cellm` checkpoints are included in the repository (tracked via Git LFS) and can be used for immediate testing:
- `models/smollm2-135m-int8.cellm`
- `models/smolvlm-256m-int8.cellm`
- `models/qwen3.5-0.8b-int4-textonly.cellm`


---

## Directory Structure
- `crates/cellm-core`: Memory arena, tensor layout, and op dispatch.
- `crates/cellm-model`: Model format, configuration, and weight management.
- `crates/cellm-cache`: Paged KV cache building blocks (allocator, page table, physical KV storage).
- `crates/cellm-sdk`: High-level public API for mobile consumers.
- `tools/bench`: Benchmark harness for TTFT and tok/s metrics.
- `tools/convert`: HuggingFace to `.cellm` conversion pipeline.
- `tools/infer`: Simple Rust inference runner for debugging models and cache behavior.
- `tools/vlm-onnx-infer`: Rust runner for SmolVLM ONNX exports (VLM validation on desktop).

---

*Phase 6 provides native vision-language reasoning on device.*

See `docs/paged-kv-cache-foundation.md` for a plain-English walkthrough of the BlockAllocator/PageTable/KVCache foundation.

### Metal Smoke Test (macOS)
Verify the Metal toolchain works on Apple Silicon/macOS (compile + dispatch a tiny compute kernel):
```bash
cargo run --release --bin metal-smoke
```

### Build Swift XCFramework (iOS + macOS)
Build `bindings/swift/CellmFFI.xcframework` (staticlib + headers) so the Swift package works on both macOS (M-series dev) and iOS:
```bash
./scripts/build_xcframework.sh
```

### iOS Demo App (LLM now, VLM stub)
There is a small SwiftUI demo app scaffold under:
- `bindings/ios/CellmDemo`

It uses the C FFI from `cellm-sdk`:
- `cellm_engine_create_v3(...)` for engine + sampling + backend config (`cpu` / `metal`)
- `cellm_engine_backend_name(...)` to confirm active backend in-app
- `cellm_tokenizer_create/encode/decode(...)` for prompt tokenization in-app

See `bindings/ios/CellmDemo/README.md` for the Xcode steps.

## Design References

- **Sampling PRNG**: `cellm` uses simple, high-performance PRNGs for stochastic sampling.
  - [Linear Congruential Generator (LCG)](https://en.wikipedia.org/wiki/Linear_congruential_generator) — Background on simple PRNG architectures.
  - Xorshift — The specific 64-bit implementation used in `cellm-sdk`.

## License

Licensed under either of:

- MIT license (`LICENSE-MIT`)
- Apache License, Version 2.0 (`LICENSE-APACHE`)

at your option.
