# Benchmarks

This page tracks practical runtime numbers for `cellm` CLI tools.

All numbers below are from local runs on **March 29, 2026** and are reference-only.

## How to run

### CPU/Metal LLM matrix (automated)

```bash
tools/bench/run_llm_backend_matrix.sh
```

Outputs:
- Markdown summary table: `docs/benchmarks/runs/llm_backend_matrix_<timestamp>_summary.md`
- Raw run CSV: `docs/benchmarks/runs/llm_backend_matrix_<timestamp>.csv`

Useful overrides:

```bash
PASSES=3 GEN_TOKENS=8 PROMPT_TEXT="hi" tools/bench/run_llm_backend_matrix.sh
```

```bash
# Skip rebuild if infer is already built
BUILD_INFER=0 tools/bench/run_llm_backend_matrix.sh
```

```bash
# Restrict to one backend
BACKENDS="cpu" tools/bench/run_llm_backend_matrix.sh
```

Note: in restricted/sandboxed shells, Metal may be unavailable and report `n/a` for Metal rows.

### Text benchmark (`infer`)

```bash
./target/release/infer \
  --model models/smollm2-135m-int8.cellm \
  --tokenizer models/hf/smollm2-135m/tokenizer.json \
  --prompt "Hello, how are you?" \
  --chat \
  --gen 16 \
  --backend cpu
```

### VLM benchmark (`vlm-infer`)

```bash
./target/release/vlm-infer \
  --model-dir models/hf/smolvlm-256m-instruct \
  --onnx-variant fp16 \
  --image models/test_images/rococo.jpg \
  --prompt "Describe this image." \
  --max-new-tokens 16 \
  --backend cpu
```

## Baseline runs

| Tool | Run | Vision Time | Prefill | Decode |
|---|---|---:|---:|---:|
| `infer` | `smollm2-135m.cellm`, `--chat`, `--gen 8` | N/A | `12` toks in `2.35s` | `8` toks in `1.62s` |
| `infer` | `smollm2-135m-int8.cellm`, `--chat`, `--gen 16` | N/A | `12` toks in `2.38s` | `16` toks in `3.36s` |
| `vlm-infer` ONNX fp16 | `rococo.jpg`, `--max-new-tokens 16` | `[64,576]` in `0.99s` | N/A | `16` toks in `1.54s` |
| `vlm-infer` ONNX quantized | `rococo.jpg`, `--max-new-tokens 24` | `[64,576]` in `1.44s` | N/A | `18` toks in `0.35s` |
| `vlm-infer` native vision + ONNX decoder | `--vision-backend cellm --decoder-backend onnx` | `[64,576]` in `5.96s` | N/A | `16` toks in `4.15s` |
| `vlm-infer` native vision + native decoder | `--vision-backend cellm --decoder-backend cellm` | `[64,576]` in `5.65s` | N/A | `24` toks in `18.39s` |

## CPU vs Metal request runs

| Tool | Backend Arg | Host Log | Vision Time | Prefill | Decode |
|---|---|---|---:|---:|---:|
| `infer` (`smollm2-135m-int8`, `--gen 16`) | `--backend cpu` | `Backend: cpu (macos/aarch64)` | N/A | `12` toks in `1.77s` | `16` toks in `2.07s` |
| `infer` (`smollm2-135m-int8`, `--gen 16`) | `--backend metal` | `Backend: metal (smoke ok)` | N/A | `12` toks in `1.75s` | `16` toks in `2.07s` |
| `vlm-infer` (`fp16`, `rococo.jpg`, `--max-new-tokens 16`) | `--backend cpu` | `Backend: cpu (macos/aarch64)` | `[64,576]` in `2.28s` | N/A | `16` toks in `2.37s` |
| `vlm-infer` (`fp16`, `rococo.jpg`, `--max-new-tokens 16`) | `--backend metal` | `Backend: metal (smoke ok)` | `[64,576]` in `1.85s` | N/A | `16` toks in `2.56s` |

## Notes

- Metal support is currently validated with smoke + backend selection.
- Full forward kernels are still being expanded, so CPU paths remain the main execution path for several operators.
- For VLM quality, `--split-image` helps caption relevance but increases latency.
