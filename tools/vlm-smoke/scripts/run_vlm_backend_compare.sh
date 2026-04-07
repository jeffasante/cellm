#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-models/smolvlm-256m-int8.cellm}"
IMAGE="${IMAGE:-models/test_images/rococo_1.jpg}"
TOKENIZER="${TOKENIZER:-models/hf/smolvlm-256m-instruct/tokenizer.json}"
PROMPT="${PROMPT:-Describe this image in one sentence.}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"

if [[ ! -f "$MODEL" ]]; then echo "missing model: $MODEL"; exit 1; fi
if [[ ! -f "$IMAGE" ]]; then echo "missing image: $IMAGE"; exit 1; fi
if [[ ! -f "$TOKENIZER" ]]; then echo "missing tokenizer: $TOKENIZER"; exit 1; fi

echo "Building vlm-smoke..."
cargo build -p vlm-smoke >/dev/null

run_backend() {
  local backend="$1"
  echo ""
  echo "=== backend=$backend ==="
  local timeout_cmd=""
  if command -v timeout >/dev/null 2>&1; then
    timeout_cmd="timeout"
  elif command -v gtimeout >/dev/null 2>&1; then
    timeout_cmd="gtimeout"
  fi

  local rc=0
  if [[ -n "$timeout_cmd" ]]; then
    CELLM_VLM_TOKENIZER="$TOKENIZER" "$timeout_cmd" "$TIMEOUT_SEC" ./target/debug/vlm-smoke --model "$MODEL" --image "$IMAGE" --prompt "$PROMPT" --backend "$backend" || rc=$?
  else
    # Portable fallback for macOS hosts without coreutils timeout.
    CELLM_VLM_TOKENIZER="$TOKENIZER" perl -e 'alarm shift; exec @ARGV' "$TIMEOUT_SEC" ./target/debug/vlm-smoke --model "$MODEL" --image "$IMAGE" --prompt "$PROMPT" --backend "$backend" || rc=$?
  fi

  if [[ $rc -ne 0 ]]; then
    if [[ $rc -eq 124 || $rc -eq 142 ]]; then
      echo "timed out after ${TIMEOUT_SEC}s"
    else
      echo "backend failed with exit $rc"
    fi
  fi
}

run_backend cpu
run_backend metal
