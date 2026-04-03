#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INFER_BIN="${INFER_BIN:-./target/release/infer}"
PASSES="${PASSES:-3}"
GEN_TOKENS="${GEN_TOKENS:-8}"
PROMPT_TEXT="${PROMPT_TEXT:-hi}"
CHAT_FORMAT="${CHAT_FORMAT:-plain}"
BACKENDS="${BACKENDS:-cpu metal}"
OUT_DIR="${OUT_DIR:-docs/benchmarks/runs}"
BUILD_INFER="${BUILD_INFER:-1}"

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RAW_CSV="$OUT_DIR/llm_backend_matrix_${STAMP}.csv"
SUMMARY_MD="$OUT_DIR/llm_backend_matrix_${STAMP}_summary.md"

MODELS=(
  "smollm2-135m-int8.cellm|models/smollm2-135m-int8.cellm|models/hf/smollm2-135m/tokenizer.json"
  "gemma-3-1b-it-int8.cellmd|models/gemma-3-1b-it-int8.cellmd|models/hf/gemma-3-1b-it/tokenizer.json"
  "qwen3.5-0.8b-int8.cellm|models/qwen3.5-0.8b-int8.cellm|models/hf/qwen3.5-0.8b/tokenizer.json"
)

if [[ "$BUILD_INFER" == "1" ]]; then
  cargo build --release --bin infer
fi

if [[ ! -x "$INFER_BIN" ]]; then
  echo "error: infer binary not found or not executable at: $INFER_BIN" >&2
  echo "hint: set INFER_BIN, or run with BUILD_INFER=1 (default)" >&2
  exit 1
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
    --chat-format "$CHAT_FORMAT" \
    --backend "$backend" > "$tmp" 2>&1
  local rc=$?
  set -e

  local s p d err_line
  s="$(rg -o 'Startup: total before prefill [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/.* ([0-9.]+)s/\1/' || true)"
  p="$(rg -o 'Prefill: [0-9]+ tokens in [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/.* in ([0-9.]+)s/\1/' || true)"
  d="$(rg -o 'Decode: [0-9]+ tokens in [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/.* in ([0-9.]+)s/\1/' || true)"
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
  for backend in $BACKENDS; do
    for pass in $(seq 1 "$PASSES"); do
      echo "running: model=$model_name backend=$backend pass=$pass/$PASSES"
      run_one "$model_name" "$model_path" "$tok_path" "$backend" "$pass"
    done
  done
done

compute_stat_pair() {
  local model_name="$1"
  local backend="$2"
  local col="$3"

  local vals n mid1 mid2 p95_idx median p95
  vals="$(awk -F',' -v m="$model_name" -v b="$backend" -v c="$col" \
    '$1==m && $3==b && $8=="ok" { print $c }' "$RAW_CSV" | sort -n)"
  n="$(printf '%s\n' "$vals" | sed '/^$/d' | wc -l | tr -d ' ')"

  if [[ "$n" == "0" ]]; then
    printf 'n/a / n/a'
    return
  fi

  if (( n % 2 == 1 )); then
    mid1=$(( (n + 1) / 2 ))
    median="$(printf '%s\n' "$vals" | sed -n "${mid1}p")"
  else
    mid1=$(( n / 2 ))
    mid2=$(( mid1 + 1 ))
    median="$(awk -v a="$(printf '%s\n' "$vals" | sed -n "${mid1}p")" \
                  -v b="$(printf '%s\n' "$vals" | sed -n "${mid2}p")" \
                  'BEGIN { printf "%.4f", (a+b)/2.0 }')"
  fi

  p95_idx=$(( (95 * n + 99) / 100 ))
  p95="$(printf '%s\n' "$vals" | sed -n "${p95_idx}p")"

  awk -v m="$median" -v p="$p95" 'BEGIN { printf "%.2f / %.2f", m, p }'
}

{
  echo "CPU vs Metal LLM benchmark snapshot (${PASSES} passes, host run on $(date '+%B %-d, %Y'); prompt=\`\"$PROMPT_TEXT\"\`, \`--gen $GEN_TOKENS\`, report = median / P95):"
  echo "| Model | Backend | Startup (s) | Prefill (s) | Decode (s) |"
  echo "|---|---|---:|---:|---:|"
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r model_name _ _ <<< "$spec"
    for backend in $BACKENDS; do
      startup_pair="$(compute_stat_pair "$model_name" "$backend" 5)"
      prefill_pair="$(compute_stat_pair "$model_name" "$backend" 6)"
      decode_pair="$(compute_stat_pair "$model_name" "$backend" 7)"
      echo "| \`$model_name\` | \`$backend\` | \`$startup_pair\` | \`$prefill_pair\` | \`$decode_pair\` |"
    done
  done
  echo
  echo "Raw CSV: \`$RAW_CSV\`"
} | tee "$SUMMARY_MD"

echo
echo "Done."
echo "Summary markdown: $SUMMARY_MD"
echo "Raw results CSV:  $RAW_CSV"
