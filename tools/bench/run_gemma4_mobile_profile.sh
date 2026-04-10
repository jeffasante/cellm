#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INFER_BIN="${INFER_BIN:-./target/release/infer}"
MODEL_PATH="${MODEL_PATH:-models/gemma-4-E2B-it-int4-aggr-v5.cellmd}"
TOKENIZER_PATH="${TOKENIZER_PATH:-models/gemma-4-E2B-it/tokenizer.json}"
BACKENDS="${BACKENDS:-cpu metal}"
PASSES="${PASSES:-1}"
GEN_TOKENS="${GEN_TOKENS:-48}"
KV_BY_BACKEND="${KV_BY_BACKEND:-auto}" # auto => cpu:f16, metal:turboquant
BUILD_INFER="${BUILD_INFER:-1}"
OUT_DIR="${OUT_DIR:-docs/benchmarks/runs}"

PROMPTS=(
  "What is consciousness?"
  "What's twitch.com?"
  "Explain TCP vs UDP in simple terms."
)

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RAW_CSV="$OUT_DIR/gemma4_mobile_profile_${STAMP}.csv"
SUMMARY_MD="$OUT_DIR/gemma4_mobile_profile_${STAMP}_summary.md"

if [[ "$BUILD_INFER" == "1" ]]; then
  cargo build --release --bin infer >/dev/null
fi

if [[ ! -x "$INFER_BIN" ]]; then
  echo "error: infer binary not found or not executable at: $INFER_BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: model not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "error: tokenizer not found: $TOKENIZER_PATH" >&2
  exit 1
fi

model_size_bytes="$(wc -c < "$MODEL_PATH" | tr -d ' ')"
model_size_gb="$(awk -v b="$model_size_bytes" 'BEGIN { printf "%.2f", b/1024/1024/1024 }')"

printf 'backend,pass,prompt_idx,prompt,startup_s,prefill_tokens,prefill_s,prefill_tps,decode_tokens,decode_s,decode_tps,output_sanitized,status,error\n' > "$RAW_CSV"

kv_for_backend() {
  local backend="$1"
  if [[ "$KV_BY_BACKEND" != "auto" ]]; then
    printf '%s' "$KV_BY_BACKEND"
    return
  fi
  if [[ "$backend" == "metal" ]]; then
    printf 'turboquant'
  else
    printf 'f16'
  fi
}

run_one() {
  local backend="$1"
  local pass="$2"
  local prompt_idx="$3"
  local prompt="$4"

  local kv tmp rc s p_tok p_sec d_tok d_sec p_tps d_tps out_text out_line err_line prompt_csv out_csv err_csv
  kv="$(kv_for_backend "$backend")"
  tmp="$(mktemp)"

  set +e
  "$INFER_BIN" \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --prompt "$prompt" \
    --chat \
    --chat-format auto \
    --gen "$GEN_TOKENS" \
    --temperature 0 \
    --backend "$backend" \
    --kv-encoding "$kv" > "$tmp" 2>&1
  rc=$?
  set -e

  s="$(rg -o 'Startup: total before prefill [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/.* ([0-9.]+)s/\1/' || true)"
  p_tok="$(rg -o 'Prefill: [0-9]+ tokens in [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/Prefill: ([0-9]+) tokens in ([0-9.]+)s/\1/' || true)"
  p_sec="$(rg -o 'Prefill: [0-9]+ tokens in [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/Prefill: ([0-9]+) tokens in ([0-9.]+)s/\2/' || true)"
  d_tok="$(rg -o 'Decode: [0-9]+ tokens in [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/Decode: ([0-9]+) tokens in ([0-9.]+)s/\1/' || true)"
  d_sec="$(rg -o 'Decode: [0-9]+ tokens in [0-9.]+s' "$tmp" | tail -n1 | sed -E 's/Decode: ([0-9]+) tokens in ([0-9.]+)s/\2/' || true)"
  out_text="$(awk '/^---[[:space:]]*$/{f=1; next} f {print}' "$tmp" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | sed 's/^ //; s/ $//' || true)"
  out_line="$(printf '%s' "$out_text" | cut -c1-220 | tr ',' ';')"
  err_line="$(head -n 1 "$tmp" | tr ',' ';' | tr '\n' ' ' || true)"
  prompt_csv="${prompt//\"/\'}"
  out_csv="${out_line//\"/\'}"
  err_csv="${err_line//\"/\'}"

  p_tps=""
  d_tps=""
  if [[ -n "$p_tok" && -n "$p_sec" && "$p_sec" != "0" ]]; then
    p_tps="$(awk -v t="$p_tok" -v s="$p_sec" 'BEGIN { printf "%.2f", t/s }')"
  fi
  if [[ -n "$d_tok" && -n "$d_sec" && "$d_sec" != "0" ]]; then
    d_tps="$(awk -v t="$d_tok" -v s="$d_sec" 'BEGIN { printf "%.2f", t/s }')"
  fi

  if [[ $rc -eq 0 && -n "$s" && -n "$p_tok" && -n "$p_sec" && -n "$d_tok" && -n "$d_sec" ]]; then
    printf '%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,"%s",ok,\n' \
      "$backend" "$pass" "$prompt_idx" "$prompt_csv" \
      "$s" "$p_tok" "$p_sec" "$p_tps" "$d_tok" "$d_sec" "$d_tps" "$out_csv" >> "$RAW_CSV"
  else
    printf '%s,%s,%s,"%s",,,,,,,,"%s",fail,"%s (rc=%s)"\n' \
      "$backend" "$pass" "$prompt_idx" "$prompt_csv" "$out_csv" "$err_csv" "$rc" >> "$RAW_CSV"
  fi

  rm -f "$tmp"
}

for backend in $BACKENDS; do
  for pass in $(seq 1 "$PASSES"); do
    idx=0
    for prompt in "${PROMPTS[@]}"; do
      idx=$((idx + 1))
      echo "running: backend=$backend pass=$pass/$PASSES prompt=$idx/${#PROMPTS[@]}"
      run_one "$backend" "$pass" "$idx" "$prompt"
    done
  done
done

median() {
  local backend="$1"
  local col="$2"
  awk -F',' -v b="$backend" -v c="$col" '$1==b && $13=="ok" { gsub(/"/, "", $c); print $c }' "$RAW_CSV" | sort -n | awk '
    BEGIN { n=0 }
    { a[++n]=$1 }
    END {
      if (n==0) { print "n/a"; exit }
      if (n%2==1) { printf "%.2f", a[(n+1)/2]; exit }
      printf "%.2f", (a[n/2] + a[n/2+1]) / 2.0
    }'
}

pass_rate() {
  local backend="$1"
  awk -F',' -v b="$backend" '
    $1==b { t++ }
    $1==b && $13=="ok" { ok++ }
    END {
      if (t==0) { print "0/0"; exit }
      printf "%d/%d", ok, t
    }' "$RAW_CSV"
}

gate_row() {
  local backend="$1"
  local p95_prefill_target="$2"
  local p95_decode_target="$3"
  local prefill_med decode_med verdict prefill_disp decode_disp
  prefill_med="$(median "$backend" 7)"
  decode_med="$(median "$backend" 10)"
  prefill_disp="${prefill_med}s"
  decode_disp="${decode_med}s"
  if [[ "$prefill_med" == "n/a" ]]; then
    prefill_disp="n/a"
  fi
  if [[ "$decode_med" == "n/a" ]]; then
    decode_disp="n/a"
  fi
  verdict="FAIL"
  if [[ "$prefill_med" != "n/a" && "$decode_med" != "n/a" ]]; then
    if awk -v p="$prefill_med" -v d="$decode_med" -v pt="$p95_prefill_target" -v dt="$p95_decode_target" 'BEGIN{exit !((p<=pt)&&(d<=dt))}'; then
      verdict="PASS"
    fi
  fi
  echo "| \`$backend\` | <=${p95_prefill_target}s | <=${p95_decode_target}s | ${prefill_disp} | ${decode_disp} | **$verdict** |"
}

{
  echo "Gemma4 mobile production profile ($(date '+%B %-d, %Y'))"
  echo
  echo "- Model: \`$MODEL_PATH\` (${model_size_gb} GB)"
  echo "- Tokenizer: \`$TOKENIZER_PATH\`"
  echo "- Passes: \`$PASSES\`"
  echo "- Gen tokens per prompt: \`$GEN_TOKENS\`"
  echo "- Backends: \`$BACKENDS\`"
  echo
  echo "**Median performance by backend**"
  echo "| Backend | Startup (s) | Prefill (s) | Prefill tok/s | Decode (s) | Decode tok/s | Pass Rate |"
  echo "|---|---:|---:|---:|---:|---:|---:|"
  for backend in $BACKENDS; do
    echo "| \`$backend\` | $(median "$backend" 5) | $(median "$backend" 7) | $(median "$backend" 8) | $(median "$backend" 10) | $(median "$backend" 11) | $(pass_rate "$backend") |"
  done
  echo
  echo "**Broad-phone readiness gates (host proxy check)**"
  echo "| Backend | Prefill Target | Decode Target | Measured Prefill | Measured Decode | Verdict |"
  echo "|---|---:|---:|---:|---:|---|"
  for backend in $BACKENDS; do
    gate_row "$backend" 15 45
  done
  echo
  echo "Raw CSV: \`$RAW_CSV\`"
} | tee "$SUMMARY_MD"

echo
echo "Done."
echo "Summary markdown: $SUMMARY_MD"
echo "Raw results CSV:  $RAW_CSV"
