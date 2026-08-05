# cellm — Mobile-Native LLM Serving Engine

> A ground-up LLM inference engine for iOS and Android, written in Rust. Brings server-grade serving concepts — paged KV cache, continuous decode scheduling, multi-session concurrency — to phones running under 512MB RAM.

Not a wrapper around `llama.cpp`. Not a port of `vLLM`. A new runtime designed for mobile constraints from scratch.

> [!NOTE]
> This is just a research project—don't get mad at me lol!

![Inference Demo](docs/assets/inference_hq.gif)

## Quick Links

| Resource | Path |
|---|---|
| **Getting Started** | [Quick Start](#quick-start) below |
| **Architecture & Design** | [`docs/project_architecture.md`](docs/project_architecture.md) |
| **Paged KV Cache Deep Dive** | [`docs/paged-kv-cache-foundation.md`](docs/paged-kv-cache-foundation.md) |
| **Benchmarks** | [`docs/benchmarks/README.md`](docs/benchmarks/README.md) |
| **Model Conversion** | [`docs/convert-quantized-models.md`](docs/convert-quantized-models.md) |
| **VLM (Vision) Guide** | [`docs/vlm-smolvlm-onnx.md`](docs/vlm-smolvlm-onnx.md) |
| **iOS Demo App** | [`bindings/ios/CellmDemo`](bindings/ios/CellmDemo) |
| **Android Bindings** | [`bindings/kotlin`](bindings/kotlin) |
| **WASM & WebGPU** | [`docs/wasm-backend.md`](docs/wasm-backend.md) |
| **Live WASM Demo** | [cellm-wasm (](https://jeffasante.github.io/cellm/wasm/index.html) (Research Preview) |
| **Git Commit Messages** | [Local LLM commit messages](#commit-messages) below |
| **Docker (CPU tooling)** | [Docker](#docker) below |

## Quick Start

### Prerequisites
- **Rust 1.75+** (modern stable toolchain)
- **macOS / iOS** for Metal acceleration (Linux/Android builds use CPU path)
- **Git LFS** (for bundled sample models)

### 1. Build
```bash
cargo build --release
```

### 2. Run a smoke test (CPU)
```bash
cargo run --release --bin infer -- \
  --model models/smollm2-135m.cellm \
  --tokenizer models/hf/smollm2-135m/tokenizer.json \
  --prompt "Hello, how are you?" \
  --chat \
  --gen 32
```

### 3. Run with Metal (macOS/iOS)
```bash
cargo run --release --bin infer -- \
  --model models/smollm2-135m-int8.cellm \
  --tokenizer models/hf/smollm2-135m/tokenizer.json \
  --prompt "Hello" \
  --chat \
  --gen 16 \
  --backend metal
```

> **Tip:** Use `--chat` for ChatML-style formatting. Without it, many base models behave like text-completion engines and may not answer directly.

### 4. Metal verification
```bash
cargo run --release --bin metal-smoke
```

---

## Architecture Overview

```mermaid
flowchart LR
    U["User Prompt"] --> API["App/UI Request Layer"]
    API --> ORCH["CPU Orchestrator"]
    ORCH --> TOK[Tokenizer]
    TOK --> FMT["Prompt Formatter"]
    FMT --> SCH["Decode Scheduler / Batcher"]

    SCH -->|"prefill/decode jobs"| ENG["Engine Dispatcher"]
    ENG -->|backend=CPU| CPUPATH["CPU Kernels"]
    ENG -->|backend=Metal| METAL["Metal Kernels"]

    METAL --> MATMUL["QKV / MLP MatMul"]
    METAL --> ATTN["Attention + GroupKV Cache"]
    METAL --> NORM["RMSNorm / RoPE / Logits"]
    MATMUL --> SAMPLER
    ATTN --> SAMPLER
    NORM --> SAMPLER
    CPUPATH --> SAMPLER["Sampler + Stop Rules"]

    SAMPLER --> DETOK[Detokenizer]
    DETOK --> STREAM["Streaming Output"]
    STREAM --> API
    API --> U

    subgraph ModelAssets["Local Model Assets"]
      W[".cellm / .cellmd mmap Weights"]
      T["tokenizer.json + config"]
    end

    W --> ENG
    T --> TOK

    subgraph SessionState["Per-Session State"]
      KV["KV Cache (GroupKV layout)"]
      PT["Page Table / Sequence Cursor"]
      TH["Thermal + QoS Policy"]
    end

    SCH --> KV
    SCH --> PT
    ORCH --> TH
    TH --> SCH
```

---

## What Makes cellm Different?

| Feature | `llama.cpp` | `MLX` | `ExecuTorch` | **`cellm`** |
| :--- | :--- | :--- | :--- | :--- |
| **Language** | C++ | C++/Python | C++ | **Rust** |
| **KV Cache** | Contiguous | Contiguous | Contiguous | **Paged (Block-based)** |
| **Focus** | Portability | Apple Native | Model Export | **Mobile Multi-session** |
| **Scheduling** | Static Batch | Mostly Single | N/A | **Round-Robin Interleaved**|
| **Memory** | Manual/Static | Managed Buffer | Static Graph | **Dynamic Block Allocator**|

---

## Project Structure

```
cellm/
├── crates/
│   ├── cellm-core/          # Memory arena, tensor layout, op dispatch
│   ├── cellm-model/         # Model format, configuration, weight management
│   ├── cellm-cache/         # Paged KV cache: BlockAllocator, PageTable, physical storage
│   ├── cellm-kernels/       # CPU, Metal, WASM & WebGPU compute kernels
│   ├── cellm-scheduler/     # Decode scheduler & batching logic
│   ├── cellm-wasm/          # WebAssembly bindings & JavaScript API
│   └── cellm-sdk/           # Public C FFI + high-level API for mobile consumers
├── bindings/
│   ├── ios/CellmDemo/       # SwiftUI demo app (LLM + VLM stub)
│   ├── kotlin/              # Android Kotlin/JNI bindings
│   └── swift/               # Swift Package + XCFramework build scripts
├── tools/
│   ├── infer/               # CLI inference runner (debug & validation)
│   ├── vlm-onnx-infer/      # VLM runner for SmolVLM ONNX exports
│   ├── vlm-smoke/           # SDK FFI VLM smoke test
│   ├── convert/             # HF Safetensors/GGUF/PyTorch -> .cellm converter
│   ├── bench/               # Latency & throughput benchmark harness
│   └── metal-smoke/         # Minimal Metal kernel compile + dispatch test
├── docs/                    # Architecture deep-dives, benchmarks, model guides
└── models/                  # Sample .cellm checkpoints (Git LFS)
```

---

## Development Commands

### Convert a Model
Convert HuggingFace Safetensors or GGUF to `.cellm`:
```bash
cargo run --bin convert -- \
  --input  ./models/hf/smollm2-135m \
  --output ./models/smollm2-135m.cellm \
  --dtype  f16
```

Quantize during conversion:
```bash
cargo run --bin convert -- \
  --input  ./models/hf/smollm2-135m \
  --output ./models/smollm2-135m-int8.cellm \
  --dtype  f16 \
  --quantize-int8-symmetric
```

See [`docs/convert-quantized-models.md`](docs/convert-quantized-models.md) for GGUF, PyTorch, and 4-bit affine workflows.

### Run Benchmarks
```bash
# Quick smoke benchmark
cargo run --release --bin bench -- --model tiny

# Full LLM backend matrix (CPU vs Metal)
tools/bench/run_llm_backend_matrix.sh
```

Detailed benchmark reports live in [`docs/benchmarks/`](docs/benchmarks/).

### Run VLM (Vision-Language)
```bash
# ONNX vision + ONNX decoder (recommended)
cargo build --release -p cellm-vlm-onnx-infer

./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --split-image \
  --max-new-tokens 96
```

Native `.cellm` vision + decoder is experimental:
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

See [`docs/vlm-smolvlm-onnx.md`](docs/vlm-smolvlm-onnx.md) for full VLM docs.

### iOS SwiftUI Demo
Build the XCFramework:
```bash
./scripts/build_xcframework.sh
```

Then open `bindings/ios/CellmDemo` in Xcode.

### Browser / WebAssembly
Build and run the WASM engine with WebGPU acceleration:
```bash
# Build the WASM module
./scripts/build-wasm.sh --release

# Serve the demo page
python3 -m http.server 8080 --directory crates/cellm-wasm/www/
```
Then open `http://localhost:8080` and use `engine.try_init_webgpu()` to enable hardware acceleration.

---

## Docker

cellm's whole point is *phone* deployment (iOS/Android via Metal/Vulkan), so Docker can't run the actual mobile targets — there's no GPU passthrough for Apple's Metal API in a Linux container. Where Docker *does* help is the CPU-backend and tooling side of the repo: CI, headless benchmarking, model conversion pipelines, and testing without owning a Mac.

| Image | Backend | Status | Use case |
|---|---|---|---|
| `cellm:cpu` | CPU kernels | Planned | CI, headless benchmarking, model conversion, testing without a Mac |
| `cellm:vulkan` | Vulkan compute | Research | Once the Vulkan backend stabilizes (see Feature Status) |
| `cellm:metal` | Metal | **Not possible** | No Metal GPU passthrough on Linux containers — Metal only runs on real macOS/iOS hardware |

Unlike `llama.cpp`'s Docker matrix (CUDA/ROCm/Vulkan/SYCL variants for `full`/`light`/`server`), cellm doesn't need a wide backend matrix — this is a mobile-focused research project, not a server deployment target. A single CPU image covers the actual need for reproducible tooling.

Example `Dockerfile` shape:
```dockerfile
FROM rust:1.75 AS build
WORKDIR /app
COPY . .
RUN cargo build --release --bin infer --bin convert --bin bench

FROM debian:bookworm-slim AS cpu
COPY --from=build /app/target/release/infer /app/target/release/convert /app/target/release/bench /usr/local/bin/
ENTRYPOINT ["infer"]
```

Usage once built:
```bash
docker run -v /path/to/models:/models cellm:cpu \
  --model /models/smollm2-135m.cellm \
  --tokenizer /models/hf/smollm2-135m/tokenizer.json \
  --prompt "Hello" --chat --gen 32
```

> Not wired up yet — tracked as a follow-up. If you want to help, a `Dockerfile` + GitHub Actions workflow (mirroring `.github/workflows/docker.yml`-style multi-arch builds) for the `cpu` variant is the right first PR.

---

## Git Commit Messages

Use cellm's local LLM to generate git commit messages from staged changes.
All inference runs locally on CPU — no API keys, no data leaves your machine.

### Prerequisites
- A model converted to `.cellm` format (e.g., Qwen2.5 0.5B int8)
- The matching `tokenizer.json`
- `./target/release/infer` built (`cargo build --release`)

### Quick usage
```bash
# Stage your changes first
git add -A

# Generate a commit message from staged diff
./tools/git-cellm-commit.sh

# Or pipe any diff
 git diff HEAD~1 | ./tools/git-cellm-commit.sh
```

### How it works
1. `git diff --cached` captures your staged changes
2. The diff is passed as the prompt to Qwen2.5 0.5B int8
3. The model generates a commit message + summary locally
4. Output is printed to stdout — pipe it anywhere

### Configuration
Set these environment variables to use a different model:

| Env var | Default | Description |
|---|---|---|
| `CELLM_COMMIT_MODEL` | `models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm` | Path to `.cellm` model |
| `CELLM_COMMIT_TOKENIZER` | `models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json` | Path to `tokenizer.json` |
| `CELLM_COMMIT_INFER` | `./target/release/infer` | Path to the `infer` binary |

Example with the LFM model:
```bash
CELLM_COMMIT_MODEL=models/LFM2.5-230M-int4-v2.cellm \
CELLM_COMMIT_TOKENIZER=models/to-huggingface/LFM2.5-230M/tokenizer.json \
  ./tools/git-cellm-commit.sh
```

---

## Supported Models

| Model | Size | Best For | Notes |
|---|---|---|---|
| **SmolLM2** | 135M-360M | Fast smoke tests, small devices | Best LLM starter model |
| **LFM2.5** | 350M | Long-context, efficient inference | Linear attention, up to 256K context |
| **Qwen2.5 / Qwen3.0 / Qwen3.5** | 0.5B-0.8B | Multilingual, reasoning | DeltaNet layers supported (CPU ref) |
| **Gemma-3** | 1B | Quality vs size tradeoff | Metal path active, CPU-safe fallback |
| **Bonsai** | 1.7B | High-quality local chat | 1-bit quantized; see `docs/bonsai_1bit_analysis.md` |
| **Gemma-4** | 2B-4B | Larger mobile workloads | Experimental; see `docs/gemma4_*` |
| **SmolVLM** | 256M | Vision-language (ONNX) | Native `.cellm` VLM path in progress |
| **FunctionGemma** | 270M | Mobile actions / tool use | Experimental quality; see [docs/function-calling-gemma.md](docs/function-calling-gemma.md) and [HF builds](https://huggingface.co/jeffasante/cellm-models/tree/main/function-calls/functiongemma-270m-cellm) |

Recommended first download: [`SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main)

Sample checkpoints bundled in this repo (via Git LFS):
- `models/smollm2-135m-int8.cellm`
- `models/smolvlm-256m-int8.cellm`
- `models/qwen3.5-0.8b-int4-textonly.cellm`

---

## Feature Status

- [x] **Paged KV Cache** - Fixed-size block allocation with `BlockAllocator` & `PageTable`
- [x] **Multi-session Scheduler** - Round-robin interleaved decoding
- [x] **4-bit Affine Dequantization** - Native MLX/HF packed weight support
- [x] **Multimodal Vision** - Native ViT/SigLIP encoder + linear projector
- [x] **Accelerated Math** - Metal + WASM SIMD + WebGPU compute kernels
- [x] **WebAssembly Support** - Run LLMs in the browser with `wasm-bindgen`
- [x] **High-Performance CLI** - Conversion, benchmarking, debug inference
- [x] **Git Commit Messages** - Generate commit messages from local LLM via `tools/git-cellm-commit.sh`
- [ ] **Vulkan Support** - Cross-platform compute kernels (research)
- [ ] **Android Integration** - Kotlin/JNI bindings & tuning (coming soon)
- [ ] **Qwen iOS Porting** - Optimize Qwen inference for native iOS
- [ ] **Docker (CPU tooling image)** - Reproducible CI/benchmarking image for `infer`/`convert`/`bench` (see [Docker](#docker))

---

## Documentation Index

| Topic | Doc |
|---|---|
| Architecture & crate design | [`docs/project_architecture.md`](docs/project_architecture.md) |
| Paged KV cache internals | [`docs/paged-kv-cache-foundation.md`](docs/paged-kv-cache-foundation.md) |
| Scheduler & continuous batching | [`docs/phase4-continuous-batching.md`](docs/phase4-continuous-batching.md) |
| Model conversion & quantization | [`docs/convert-quantized-models.md`](docs/convert-quantized-models.md) |
| On-device function calling | [`docs/function-calling-gemma.md`](docs/function-calling-gemma.md) |
| TurboQuant KV compression | [`docs/turboquant_dataflow.md`](docs/turboquant_dataflow.md) |
| VLM / SmolVLM ONNX guide | [`docs/vlm-smolvlm-onnx.md`](docs/vlm-smolvlm-onnx.md) |
| VLM sequence tracking | [`docs/cellm-vlm-sequence.md`](docs/cellm-vlm-sequence.md) |
| Qwen3.5 / DeltaNet | [`docs/qwen3_5-deltanet.md`](docs/qwen3_5-deltanet.md) |
| Metal acceleration notes | [`docs/LFM_Metal_Acceleration.md`](docs/LFM_Metal_Acceleration.md) |
| Benchmark history & raw runs | [`docs/benchmarks/`](docs/benchmarks/) |
| Data flow diagrams | [`docs/data_flow.md`](docs/data_flow.md) |
| Format specification | [`docs/format.md`](docs/format.md) |
| Inference graph | [`docs/inference_graph.md`](docs/inference_graph.md) |
| WASM & WebGPU Backend | [`docs/wasm-backend.md`](docs/wasm-backend.md) |

---

## Troubleshooting

### Metal is not being used
```bash
# 1. Verify Metal device access
cargo run --release --bin metal-smoke

# 2. Verify infer picks Metal
./target/release/infer \
  --model models/smollm2-135m-int8.cellm \
  --tokenizer models/hf/smollm2-135m/tokenizer.json \
  --prompt "hello" --gen 8 --backend metal
```

In restricted/sandboxed shells, Metal device discovery can fail. `infer --backend metal` now **errors** instead of silently falling back to CPU.

### SmolLM2 360M needs non-interleaved RoPE
```bash
CELLM_LLAMA_ROPE_INTERLEAVED=0 ./target/release/infer ...
```

### Gemma-3 Metal quality knobs
Default keeps norm/RoPE/logits on CPU-safe path for quality parity. Opt-in Metal paths:
```bash
CELLM_GEMMA_USE_METAL_NORM=1   # enable Metal RMSNorm
CELLM_GEMMA_USE_METAL_ROPE=1   # enable Metal RoPE
CELLM_GEMMA_USE_METAL_LOGITS=1 # enable Metal final logits matvec
```

### Llama graph path (experimental speed)
```bash
CELLM_LLAMA_ENABLE_GRAPH=1 ./target/release/infer ...
```

### Model-specific env flags
| Model | Flag | Purpose |
|---|---|---|
| SmolLM2 360M | `CELLM_LLAMA_ROPE_INTERLEAVED=0` | Correct RoPE layout |
| Llama | `CELLM_LLAMA_USE_METAL_NORM=1` | Force Metal norm |
| Llama | `CELLM_LLAMA_USE_METAL_ROPE=1` | Force Metal RoPE |
| Qwen VLM | `CELLM_VLM_TOKENIZER=...` | Set tokenizer path for vlm-smoke |

For more debug flags and backend-specific notes, see the per-model docs in `docs/`.

---

## Citation

If you use cellm in your research, please cite:

```bibtex
@software{asante2026cellm,
  author       = {Asante, Jeffrey},
  title        = {cellm: A Mobile-Native LLM Serving Engine},
  year         = {2026},
  url          = {https://github.com/jeffasante/cellm},
  note         = {Rust inference engine with paged KV cache and multi-session scheduling}
}
```

---

## License

Apache License, Version 2.0 (`LICENSE-APACHE`)
