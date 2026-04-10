# Qwen2.5 0.5B in cellm: What Finally Worked

## Goal
Get a small `.cellm` model that generates coherent output, and understand why earlier conversions failed.

## Model Sources Used
- HF source checkpoint: `models/hf/Qwen2.5-0.5B-Instruct-full`
- Python bitsandbytes reference: `models/qwen2.5-0.5b-bnb4` (used for parity checks)

## Key Findings

1. **Tokenizer mismatch was not the root cause**
- Python cross-check showed identical tokenizer behavior between HF full and bnb4:
  - same vocab/special token IDs
  - same chat template string
  - same input token IDs for the same prompt

2. **Main runtime parity bugs were in Qwen runner**
- The runner assumed `q_norm/k_norm` always existed; Qwen2.5 may not have them.
- Qwen2.5 attention projection biases (`q/k/v`, and potentially `o`) were not being applied.

3. **Size vs quality tradeoff was real**
- Very aggressive int4 could hit/beat target size but produced poor text quality.
- A slightly larger int8 variant gave correct/coherent output.

## Code Fixes Applied

### 1) Qwen runtime: optional Q/K norm usage
File: `crates/cellm-model/src/qwen.rs`
- Changed logic to apply per-head Q/K RMSNorm only if both tensors exist:
  - `...self_attn.q_norm.weight`
  - `...self_attn.k_norm.weight`

### 2) Qwen runtime: apply projection biases when present
File: `crates/cellm-model/src/qwen.rs`
- Added helper to apply `*.bias` for projection outputs when matching bias tensor exists.
- Wired this into full-attention path for:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`

### 3) Converter: Qwen quantization selection fixes
File: `tools/convert/src/main.rs`
- Updated Qwen tensor selection patterns so Qwen2.5 naming (`model.layers.*`) is handled correctly.
- Tried several quantization profiles to balance size and quality.

## Conversion Attempts and Outcomes

### A) int4 aggressive profile
Output: `models/qwen2.5-0.5b-int4-v3.cellm`
- Size: **443MB**
- Status: Runs, but quality still weaker (more unstable outputs)

### B) int8 profile (recommended working result)
Output: `models/qwen2.5-0.5b-int8-v1.cellm`
- Size: **487MB**
- Status: **Working and coherent output**

## Working Inference Command

```bash
cd /Users/jeff/Desktop/cellm
./target/release/infer \
  --model models/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/hf/Qwen2.5-0.5B-Instruct-full/tokenizer.json \
  --prompt "what's twitch.com?" \
  --chat \
  --gen 64 \
  --temperature 0 \
  --backend cpu \
  --kv-encoding f16
```

## Why 447MB `.cellm` Was Hard
- bitsandbytes 4-bit storage is highly compact and format-specific.
- `cellm` currently stores quantized tensors in its own mmap-friendly format (weights + scales), which changes size/quality behavior.
- We reached **443MB** with int4, but quality was not as reliable as the **487MB int8** variant.

## Final Recommendation
For a reliable production baseline right now, use:
- `models/qwen2.5-0.5b-int8-v1.cellm` (487MB)

Keep `models/qwen2.5-0.5b-int4-v3.cellm` (443MB) as an experimental size-first artifact.
