#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INFER_BIN="${INFER_BIN:-./target/release/infer}"
PASSES="${PASSES:-1}"
GEN_TOKENS="${GEN_TOKENS:-32}"
PROMPT_TEXT="${PROMPT_TEXT:-The mathematical basis for deep learning is}"
OUT_DIR="${OUT_DIR:-docs/benchmarks/runs}"
BUILD_INFER="${BUILD_INFER:-0}" # Already built

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RAW_CSV="$OUT_DIR/full_model_matrix_${STAMP}.csv"
SUMMARY_MD="$OUT_DIR/full_model_matrix_${STAMP}_summary.md"

MODELS=(
  "Bonsai-1.7B_v2|models/to-huggingface/Bonsai-1.7B_v2/Bonsai-1.7B_v2.cellm|models/to-huggingface/Bonsai-1.7B_v2/tokenizer.json"
  "gemma-3-1b-it-int8-v1|models/to-huggingface/gemma-3-1b-it-int8-v1/gemma-3-1b-it-int8-v1.cellm|models/to-huggingface/gemma-3-1b-it-int8-v1/tokenizer.json"
  "lfm2.5-350m-v1|models/to-huggingface/lfm2.5-350m-v1/lfm2.5-350m-v1.cellm|models/to-huggingface/lfm2.5-350m-v1/tokenizer.json"
  "qwen2.5-0.5b-int8-v1|models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm|models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json"
  "qwen3.5-0.8b-i4-v1|models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-i4.cellm|models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json"
  "smollm2-360m-int8-v1|models/to-huggingface/smollm2-360m-int8-v1/smollm2-360m-int8-v1.cellm|models/to-huggingface/smollm2-360m-int8-v1/tokenizer.json"
)

if [[ "$BUILD_INFER" == "1" ]]; then
  cargo build --release --bin infer
fi

printf 'model,model_path,backend,pass,startup_s,prefill_s,decode_s,status,error\n' > "$RAW_CSV"

run_one() {
  local model_name="$1"
  local model_path="$2"
  local tok_path="$3"
  local backend="$4"
  local pass="$5"

  local tmp
  tmp="$(mktemp)"

  set +e
  "$INFER_BIN" \
    --model "$model_path" \
    --tokenizer "$tok_path" \
    --prompt "$PROMPT_TEXT" \
    --gen "$GEN_TOKENS" \
    --temperature 0 \
    --backend "$backend" > "$tmp" 2>&1
  local rc=$?
  set -e

  local s p d err_line
  # Using grep -E for portability
  s="$(grep -oE 'Startup: total before prefill [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/.* ([0-9.]+)s/\1/' || true)"
  p="$(grep -oE 'Prefill: [0-9]+ tokens in [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/.* in ([0-9.]+)s/\1/' || true)"
  d="$(grep -oE 'Decode: [0-9]+ tokens in [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/.* in ([0-9.]+)s/\1/' || true)"
  err_line="$(head -n 1 "$tmp" | tr ',' ';' | tr '\n' ' ' || true)"

  if [[ $rc -eq 0 && -n "$s" && -n "$p" && -n "$d" ]]; then
    printf '%s,%s,%s,%s,%s,%s,%s,ok,\n' \
      "$model_name" "$model_path" "$backend" "$pass" "$s" "$p" "$d" >> "$RAW_CSV"
  else
    printf '%s,%s,%s,%s,,,,fail,%s (rc=%s)\n' \
      "$model_name" "$model_path" "$backend" "$pass" "$err_line" "$rc" >> "$RAW_CSV"
  fi

  rm -f "$tmp"
}

for spec in "${MODELS[@]}"; do
  IFS='|' read -r model_name model_path tok_path <<< "$spec"
  for backend in cpu; do
    for pass in $(seq 1 "$PASSES"); do
      echo "running: model=$model_name backend=$backend pass=$pass/$PASSES"
      run_one "$model_name" "$model_path" "$tok_path" "$backend" "$pass"
    done
  done
done

{
  echo "# Full Model Benchmark History ($STAMP)"
  echo "Performance metrics for all research models on CPU backend."
  echo
  echo "| Model | Startup (s) | Prefill (s) | Decode (s) |"
  echo "|---|---:|---:|---:|"
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r model_name _ _ <<< "$spec"
    s="$(awk -F',' -v m="$model_name" '$1==m && $8=="ok" { print $5 }' "$RAW_CSV" | head -n1)"
    p="$(awk -F',' -v m="$model_name" '$1==m && $8=="ok" { print $6 }' "$RAW_CSV" | head -n1)"
    d="$(awk -F',' -v m="$model_name" '$1==m && $8=="ok" { print $7 }' "$RAW_CSV" | head -n1)"
    echo "| \`$model_name\` | \`${s:-fail}\` | \`${p:-fail}\` | \`${d:-fail}\` |"
  done
  echo
  echo "**Setup Details:**"
  echo "- **Prompt:** \`$PROMPT_TEXT\`"
  echo "- **Tokens:** $GEN_TOKENS"
  echo "- **Hardware:** Apple Silicon (macOS)"
} | tee "$SUMMARY_MD"

echo "Done. Summary: $SUMMARY_MD"
