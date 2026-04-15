# CellmDemo (iOS) — test app for cellm

This folder contains SwiftUI source files for a small iOS app that can load a `.cellm` model + `tokenizer.json` and run text generation through `cellm`’s Rust core via the C FFI.

## What works today

- LLM text generation (tokenize prompt → prefill → decode tokens)
- VLM image description through native `.cellm` path in `cellm-sdk` (vision encoder + multimodal prompt packing + text decode)
- Backend request from iOS UI (`CPU` / `Metal`) through FFI (`cellm_engine_create_v4`)
- KV encoding + TurboQuant runtime knobs are also passed through `cellm_engine_create_v4`
- Active backend reporting (`cellm_engine_backend_name`) so app confirms what was selected
- One-tap sample asset download in-app (GitHub-hosted `.cellm` + sample image + tokenizer)

Note: backend selection is strict in this build. If `Metal` is requested and unavailable, engine creation fails instead of silently falling back to CPU.

## Current limits

- Native VLM path is currently CPU math in this phase.

## How to run it in Xcode

1) Build the XCFramework used by the app target:
   ```bash
   cd /cellm
   zsh scripts/build_xcframework.sh
   ```

2) Open the generated project:
   - `/cellm/bindings/ios/CellmDemo.xcodeproj`

3) Add the Swift files from this folder into your app target:
   - `CellmFFI.swift`
   - `LLMView.swift`
   - `VLMView.swift`
   - `CellmDemoApp.swift` (optional; or copy the view code into your app’s existing `App` entry)

4) Build + Run on a real iPhone (recommended). You can either:
   - tap the in-app sample download buttons, or
   - use the document picker manually.

   Manual picker flow:
   - the model file (example: `qwen2.5-0.5b-int8-v1.cellm` or `gemma-4-E2B-it-int4-aggr-v5.cellmd`)
   - the tokenizer file (example: `tokenizer.json`)
   - backend (`Metal` recommended on iPhone/iPad with Apple GPU)

## Qwen iOS smoke test flow

- In `LLM` tab, tap **Download Qwen stable model + tokenizer (~1.6 GB)**
- Tap **Run Qwen Smoke Test**
- Default smoke prompt: `Return exactly one uppercase letter: R`
- The output panel shows generation diagnostics:
  - `prompt_tokens`
  - `generated_tokens`
  - `first_piece`
  - `prefill/decode/total` timing in ms

## Gemma iOS quick run flow

- In `LLM` tab, tap **Download Gemma 3 1B int8 model + tokenizer (~1.2 GB)**
- Prompt example:
  - `What is the capital of France?`
  - `If I buy 12 donuts and eat 5, how many donuts are left for tomorrow?`
- Choose `Metal` for acceleration, or `CPU` for deterministic CPU-only validation.

## Sample hosted assets used by the app

- `https://huggingface.co/jeffasante/cellm-models/resolve/main/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm?download=true`
- `https://huggingface.co/jeffasante/cellm-models/resolve/main/qwen2.5-0.5b-int8-v1/tokenizer.json?download=true`
- `https://huggingface.co/jeffasante/cellm-models/resolve/main/qwen2.5-0.5b-int8-v1/tokenizer_config.json?download=true`
- `https://github.com/jeffasante/cellm/blob/main/models/smollm2-135m-int8.cellm`
- `https://huggingface.co/jeffasante/cellm-models/resolve/main/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd?download=true`
- `https://huggingface.co/jeffasante/cellm-models/resolve/main/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json?download=true`
- `https://huggingface.co/jeffasante/cellm-models/resolve/main/gemma-4-E2B-it-int4-aggr-v5/tokenizer_config.json?download=true`
- `https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/tokenizer.json`
- `https://github.com/jeffasante/cellm/blob/main/models/smolvlm-256m-int8.cellm`
- `https://github.com/jeffasante/cellm/blob/main/models/test_images/rococo_1.jpg`

The app normalizes GitHub `blob` URLs to raw-download URLs before fetching.
For Gemma3 in LLM tab, Advanced Actions now support downloading model-only or tokenizer JSONs-only for phone import workflows.

## Notes

- Large model files are slow to load over the simulator and can exceed simulator storage limits. A physical device is the fastest way to validate end-to-end.
- Keep `tokenizer.json` next to the `.cellm` model when you manage files on disk; the app lets you pick both explicitly.
- Qwen and LLM backend selection are strict; no automatic CPU fallback when `Metal` is requested.
- Qwen compact int4 can still be degenerate; use the stable Qwen model when validating response quality on-device.

## Latest Metal KV optimization patch (what was changed)

To reduce long decode stalls and memory churn on iPhone when using Qwen with Metal selected, we patched the KV cache Metal path in:

- `crates/cellm-cache/src/kvcache.rs`

### What we changed

1) Added reusable scratch buffers inside `MetalKvStorage`:
- `k_f16`, `v_f16`, `k_f32`, `v_f32`, `q_f32`, `out_f32`, `bases_u32`
- these are kept and grown as needed, instead of allocating every token step.

2) Replaced tiny scalar Metal buffers with inline constants:
- switched `base/len/seq/head_dim/...` kernel args to `set_bytes(...)`
- removes many tiny per-dispatch buffer allocations.

3) Wrapped command submission in `autoreleasepool`:
- dispatch path now uses an autorelease pool around command buffer + encoder work
- helps prevent memory buildup during long generation loops on iOS.

### Why this helps

- Lower allocation pressure in decode hot path.
- Lower risk of iOS memory kill (`IDEDebugSessionErrorDomain Code 11`) during long runs.
- Better baseline for the next phase (full attention/math parity and full Metal path).

### Validation run

- `cargo check --workspace` passed.
- `xcodebuild ... CellmDemo ...` simulator build passed (`BUILD SUCCEEDED`).
