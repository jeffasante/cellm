#!/bin/sh
# git-cellm-commit — Generate a git commit message using cellm's local LLM
#
# Usage:
#   git-cellm-commit                    # staged changes (git diff --cached)
#   git diff HEAD~1 | git-cellm-commit  # piped diff
#
# Depends on: ./target/release/infer, a .cellm model, a tokenizer.json

set -e

MODEL="${CELLM_COMMIT_MODEL:-models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm}"
TOKENIZER="${CELLM_COMMIT_TOKENIZER:-models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json}"
INFER="${CELLM_COMMIT_INFER:-./target/release/infer}"

# Get the diff
if [ ! -t 0 ]; then
    DIFF=$(cat)
else
    DIFF=$(git diff --cached --no-color 2>/dev/null || echo "")
fi

if [ -z "$DIFF" ]; then
    echo "No diff found. Stage changes first, or pipe a diff." >&2
    exit 1
fi

# Truncate to avoid blowing the context window
DIFF=$(echo "$DIFF" | head -60)

PROMPT="Write a concise git commit message for this diff:\n$DIFF"

echo "Generating commit message..." >&2

# Run inference to a temp file, suppressing stderr
TMP=$(mktemp)
"$INFER" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --prompt "$PROMPT" \
  --gen 80 \
  --temperature 0.3 \
  --top-k 40 \
  --repeat-penalty 1.1 \
  --stop-eos \
  --backend cpu > "$TMP" 2>/dev/null || true

# Extract text after the final "---" divider
awk 'BEGIN{f=0} /^---$/ {f=1; next} f' "$TMP"
rm -f "$TMP"
