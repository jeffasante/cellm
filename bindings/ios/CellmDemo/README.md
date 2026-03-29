# CellmDemo (iOS) — test app for cellm

This folder contains SwiftUI source files for a small iOS app that can load a `.cellm` model + `tokenizer.json` and run text generation through `cellm`’s Rust core via the C FFI.

## What works today

- LLM text generation (tokenize prompt → prefill → decode tokens)
- Backend request from iOS UI (`CPU` / `Metal`) through FFI (`cellm_engine_create_v3`)
- Active backend reporting (`cellm_engine_backend_name`) so app confirms what was selected
- One-tap sample asset download in-app (GitHub-hosted `.cellm` + sample image)

Note: requesting `Metal` currently verifies Metal availability and selects the Metal backend label, but full LLM/VLM forward math kernels are still CPU in this phase.

## What is stubbed for now

- VLM (image/video) inference: the UI can be added now, but the Rust runtime currently does not implement a vision encoder / multimodal prompt packing yet.

## How to run it in Xcode

1) Build the XCFramework used by the app target:
   ```bash
   cd /Users/jeff/Desktop/cellm
   zsh scripts/build_xcframework.sh
   ```

2) Open the generated project:
   - `/Users/jeff/Desktop/cellm/bindings/ios/CellmDemo.xcodeproj`

3) Add the Swift files from this folder into your app target:
   - `CellmFFI.swift`
   - `LLMView.swift`
   - `VLMView.swift`
   - `CellmDemoApp.swift` (optional; or copy the view code into your app’s existing `App` entry)

4) Build + Run on a real iPhone (recommended). You can either:
   - tap the in-app sample download buttons, or
   - use the document picker manually.

   Manual picker flow:
   - the model file (example: `qwen3.5-0.8b.cellm`)
   - the tokenizer file (example: `tokenizer.json`)
   - backend (`Metal` recommended on iPhone/iPad with Apple GPU)

## Sample hosted assets used by the app

- `https://github.com/jeffasante/cellm/blob/main/models/smollm2-135m-int8.cellm`
- `https://github.com/jeffasante/cellm/blob/main/models/smolvlm-256m-int8.cellm`
- `https://github.com/jeffasante/cellm/blob/main/models/test_images/rococo_1.jpg`

The app normalizes GitHub `blob` URLs to raw-download URLs before fetching.

## Notes

- Large model files are slow to load over the simulator and can exceed simulator storage limits. A physical device is the fastest way to validate end-to-end.
- Keep `tokenizer.json` next to the `.cellm` model when you manage files on disk; the app lets you pick both explicitly.
