# gemma-4-E2B-it-int4-aggr-v5 (cellm)

## Files

- `gemma-4-E2B-it-int4-aggr-v5.cellmd` (~3.3GB)
- `tokenizer.json`
- `tokenizer_config.json`

## Run (CPU)

```bash
cd /cellm
./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format auto \
  --gen 48 \
  --temperature 0 \
  --backend cpu \
  --kv-encoding f16
```

## Run (Metal)

```bash
cd /cellm
./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format auto \
  --gen 48 \
  --temperature 0 \
  --backend metal \
  --kv-encoding turboquant
```
