# cellm Inference Cheatsheet

## Build

```sh
cargo build --release --bin infer
cargo build --release -p cellm-sdk -p vlm-smoke
```

---

## Qwen 2.5 0.5B int8

```sh
# CPU — short factual (f16)
./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "what's twitch.com?" \
  --chat --gen 64 --temperature 0 --backend cpu --kv-encoding f16

./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "what's twitch.com?" \
  --chat --gen 100 --backend cpu


# CPU — long philosophical (f16)
./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "How can we reconcile the need for technological progress with the preservation of human, non-digital experiences?" \
  --chat --gen 100 --temperature 0 --backend cpu --kv-encoding f16

# Metal — sycophancy (f16)
./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "what's sycophancy?" \
  --chat --gen 100 --temperature 0 --backend metal --kv-encoding f16

# Metal — longer generation (f16)
./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "how much money does a machine learning engineer make? in one paragraph" \
  --chat --gen 300 --temperature 0 --backend metal --kv-encoding f16

# Metal — turboquant KV
./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "what's sycophancy?" \
  --chat --gen 64 --temperature 0 --backend metal --kv-encoding turboquant

# CPU — turboquant KV
./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "what's sycophancy?" \
  --chat --gen 64 --temperature 0 --backend cpu --kv-encoding turboquant

# Debug position encoding
CELLM_QWEN_DEBUG_POS=0 cargo build --release -p cellm-infer && \
./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "How can we reconcile the need for technological progress with the preservation of human, non-digital experiences?" \
  --chat --gen 100 --temperature 0 --backend metal --kv-encoding f16
```


```bash
./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "what's sycophancy?" \
  --chat --gen 100 --backend metal
```

"Explain the fundamental differences between quantum computing and classical computing, including the principle
nce, and how these properties enable quantum algorithms to achieve exponential speedups for certain computational problems."

./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "Explain the fundamental differences between quantum computing and classical computing, including the principle
nce, and how these properties enable quantum algorithms to achieve exponential speedups for certain computational problems." \
  --chat --gen 100 --backend metal



```bash

./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "Create a Flappy Bird game in Python" \
  --chat --gen 3000 --backend metal


```

```bash

./target/release/infer \
  --model models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "You rewrite outreach messages. Do not add new facts.

Return only one message.

Rules:
- Under 45 words.
- Warm, simple, human.
- Do not mention AI.
- Do not invent events, meetings, help, or promises.
- Do not say “recent” unless last_contacted is known.
- Use only the base draft and contact facts.
- Keep the same intent.

Contact facts:
name: Sarah
role: Professional
workplace: Accra Tech Summit
relationship_strength: 2/5
last_contacted: never
known_context: Met at Accra Tech Summit
capabilities: Contract Review, Firebase, Flutter, Swift
channel: Messages
intent: Reconnect

Base draft:
Hi Sarah, hope you’re doing well. We met at Accra Tech Summit, and I realized I never properly followed up. I’d love to reconnect sometime this week if you have a free moment.

Rewrite the base draft." \
  --chat --gen 100 --backend metal

```
---

## Qwen 3 0.6B int4



---

## Qwen3 0.6B

### Overview

Qwen3 uses a non-standard attention architecture where `hidden_size != n_heads * head_dim`. For Qwen3-0.6B:
- hidden_size = 1024
- n_heads = 16, head_dim = 128
- attn_dim = 2048 (separate from hidden)

### Convert from HuggingFace

```sh
# Convert f16 model
cargo run --release --bin convert -- \
  --input models/Qwen3-0.6B \
  --output models/Qwen3-0.6B.cellm \
  --dtype f16

# Convert with int4 quantization
cargo run --release --bin convert -- \
  --input models/Qwen3-0.6B \
  --output models/Qwen3-0.6B-int4.cellm \
  --quantize-int4-symmetric
```

### Inference (Metal)

```sh
# Basic inference with sampling (recommended for 0.6B model)
./target/release/infer \
  --model models/to-huggingface/qwen3-0.6b-v1/qwen3-0.6b-int8.cellm \
  --tokenizer models/Qwen3-0.6B/tokenizer.json \
  --prompt "Explain consciousness:" \
  --gen 50 --temperature 0.7 --top-k 40 --backend metal

# Chat mode with system prompt
./target/release/infer \
  --model models/Qwen3-0.6B-new.cellm \
  --tokenizer models/Qwen3-0.6B/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format auto \
  --gen 50 --temperature 0.7 --top-k 40 \
  --backend metal --kv-encoding f16 --stop-eos
```

```sh
# int4 - simple prompt (works)
./target/release/infer \
  --model models/Qwen3-0.6B-fixed-int4.cellm \
  --tokenizer models/Qwen3-0.6B/tokenizer.json \
  --prompt "Explain consciousness:" \
  --gen 50 --temperature 0.7 --top-k 40 \
  --backend metal


  # int4 - simple prompt (works)
./target/release/infer \
  --model models/Qwen3-0.6B-fixed-int4.cellm \
  --tokenizer models/Qwen3-0.6B/tokenizer.json \
  --prompt "who is elon musk" \
  --gen 50 --temperature 0.7 --top-k 40 \
  --backend metal



# f16 - chat mode (works)
./target/release/infer \
  --model models/Qwen3-0.6B-new.cellm \
  --tokenizer models/Qwen3-0.6B/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format auto \
  --gen 50 --temperature 0.7 --top-k 40 \
  --backend metal --kv-encoding f16 --stop-eos


  # f16 - chat mode (works)
./target/release/infer \
  --model models/Qwen3-0.6B-new.cellm \
  --tokenizer models/to-huggingface/qwen3-0.6b-v1/tokenizer.json \
  --prompt "who is elon musk" \
  --chat --chat-format auto \
  --gen 50 --temperature 0.7 --top-k 40 \
  --backend metal --kv-encoding f16 --stop-eos
```


### Inference (CPU)

```sh
./target/release/infer \
  --model models/to-huggingface/qwen3-0.6b-v1/qwen3-0.6b-int8.cellm \
  --tokenizer models/to-huggingface/qwen3-0.6b-v1/tokenizer.json \
  --prompt "What is the capital of France?" \
  --gen 20 --backend cpu
```





### Notes

- Use `--temperature 0.7` or higher for varied outputs (greedy decoding causes repetition in small models)
- Metal backend provides full acceleration for Qwen3 architecture
- Int4 quantization reduces model size ~4x while maintaining coherence
- Requires architectural fix for non-standard attention dimensions


Model	Status
Qwen3-0.6B.cellm (f16)	Works
Qwen3-0.6B-new.cellm (f16)	Works
Qwen3-0.6B-fixed-int4.cellm (int4)	Now works!
Qwen3-0.6B-int4.cellm (old int4)	Garbage (pre-fix)


---

## SmolLM2 135M int8

```sh
./target/release/infer \
  --model models/to-huggingface/smollm2-360m-int8-v1/smollm2-360m-int8-v1.cellm \
  --tokenizer models/to-huggingface/smollm2-360m-int8-v1/tokenizer.json \
  --prompt "Hello" \
  --chat --gen 16 --backend metal
```

---

## SmolLM2 360M int8

```sh
# CPU
./target/release/infer \
  --model models/to-huggingface/smollm2-360m-int8-v1/smollm2-360m-int8-v1.cellm \
  --tokenizer models/to-huggingface/smollm2-360m-int8-v1/tokenizer.json \
  --prompt "what's sycophancy?" \
  --chat --gen 100 --temperature 0 --backend cpu --kv-encoding f16

  ./target/release/infer \
    --model models/to-huggingface/smollm2-360m-q1-v1/smollm2-360m-int8-v1.cellm \
    --tokenizer models/to-huggingface/smollm2-360m-q1-v1/tokenizer.json \
    --prompt "what's sycophancy?" \
    --chat --gen 100










    --temperature 0 --backend cpu --kv-encoding f16

# Metal
./target/release/infer \
  --model models/to-huggingface/smollm2-360m-int8-v1/smollm2-360m-int8-v1.cellm \
  --tokenizer models/to-huggingface/smollm2-360m-int8-v1/tokenizer.json \
  --prompt "what's sycophancy?" \
  --chat --gen 100 --temperature 0 --backend metal --kv-encoding f16
```

---

## Gemma 3 1B int4

```sh
# CPU
./target/release/infer \
  --model models/to-huggingface/gemma-3-1b-it-mixed-int4-v1/gemma-3-1b-it-mixed-int4-v1.cellm \
  --tokenizer models/to-huggingface/gemma-3-1b-it-int8-v1/tokenizer.json \
  --prompt "what's twitch.com?" \
  --chat --chat-format plain --gen 48 --temperature 0 --backend cpu --kv-encoding f16

# Metal
./target/release/infer \
  --model models/to-huggingface/gemma-3-1b-it-mixed-int4-v1/gemma-3-1b-it-mixed-int4-v1.cellm \
  --tokenizer models/to-huggingface/gemma-3-1b-it-int8-v1/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format plain --gen 48 --temperature 0 --backend metal --kv-encoding f16


  ./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellmd \
  --tokenizer models/to-huggingface/gemma-3-1b-it-int8-v1/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format plain --gen 48 --temperature 0 --backend metal --kv-encoding f16
```

---

## Gemma 3 1B int8

```sh
# CPU
./target/release/infer \
  --model models/to-huggingface/gemma-3-1b-it-int8-v1/gemma-3-1b-it-int8-v1.cellm \
  --tokenizer models/to-huggingface/gemma-3-1b-it-int8-v1/tokenizer.json \
  --prompt "who is elon musk and is he the richest man in the world?" \
  --chat --chat-format plain --gen 48 --temperature 0 --backend cpu --kv-encoding f16

# Metal
./target/release/infer \
  --model models/to-huggingface/gemma-3-1b-it-int8-v1/gemma-3-1b-it-int8-v1.cellm \
  --tokenizer models/to-huggingface/gemma-3-1b-it-int8-v1/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format plain --gen 80 --temperature 0 --backend metal --kv-encoding f16

# Metal — technical
./target/release/infer \
  --model models/to-huggingface/gemma-3-1b-it-int8-v1/gemma-3-1b-it-int8-v1.cellm \
  --tokenizer models/to-huggingface/gemma-3-1b-it-int8-v1/tokenizer.json \
  --prompt "what is Graph Lowering Compiler Techniques for Neural Networks in one sentence." \
  --chat --chat-format plain --gen 80 --temperature 0 --backend metal --kv-encoding f16
```

---

## Gemma 3 1B mixed-int4

```sh [ performance here is bad]
# Metal — turboquant KV
./target/release/infer \
  --model models/to-huggingface/gemma-3-1b-it-mixed-int4-v1/gemma-3-1b-it-mixed-int4-v1.cellm \
  --tokenizer models/gemma-4-E2B-it/tokenizer.json \
  --prompt $'Graph Lowering Compiler Techniques for\nNeural Networks?' \
  --chat --chat-format auto --gen 48 --temperature 0 --backend metal --kv-encoding turboquant
```

---

## Gemma 4 E2B int4-aggr-v5

### Text

```sh
# CPU
./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellm \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format auto --gen 32 --temperature 0 --backend cpu --kv-encoding f16

# Metal — f16
./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellm \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "what's sycophancy?" \
  --chat --gen 32 --temperature 0 --backend metal --kv-encoding f16



# Metal — turboquant KV
./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellm \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt $'Graph Lowering Compiler Techniques for\nNeural Networks?' \
  --chat --chat-format auto --gen 100 --temperature 0 --backend metal --kv-encoding turboquant

# CPU — who is elon musk
./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellm \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "who is elon musk?" \
  --chat --tokens 100 --backend cpu


  ./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellm \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "who is elon musk?" \
  --chat --gen 100 --backend metal


    ./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellm \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "Graph Lowering Compiler Techniques for\nNeural Networks?" \
  --chat --gen 100 --backend metal

```

### Vision

```sh
MODEL=models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellm

./target/release/vlm-direct \
  --model "$MODEL" \
  --image models/test_images/bird.jpg \
  --prompt "What is in this image?" \
  --backend cpu --tokens 16

# With feature stats debug output
CELLM_VLM_DEBUG_FEATURE_STATS=1 ./target/release/vlm-direct \
  --model "$MODEL" \
  --image models/test_images/bird.jpg \
  --prompt "What is in this image?" \
  --backend cpu --tokens 16


  MODEL=models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellm

  ./target/release/vlm-direct \
    --model "$MODEL" \
    --image models/test_images/bird.jpg \
    --prompt "What is in this image?" \
    --backend metal --tokens 16




    MODEL=models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellm
./target/release/vlm-direct \
  --model "$MODEL" \
  --image models/test_images/bird.jpg \
  --prompt "What is in this image?" \
  --backend metal \
  --tokens 16


```

### Audio

```sh
MODEL=models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellm

./target/release/audio-direct \
  --model "$MODEL" \
  --audio /tmp/test_audio.wav \
  --prompt "What instrument do you hear?" \
  --tokens 40

# With audio debug stats
CELLM_AUDIO_DEBUG=1 ./target/release/audio-direct \
  --model "$MODEL" \
  --audio /tmp/test_audio.wav \
  --prompt "Describe what you hear." \
  --tokens 80
```

---

## Bonsai 1.7B (1-bit)

```sh
# Metal
./target/release/infer \
  --model models/to-huggingface/Bonsai-1.7B_v2/Bonsai-1.7B_v2.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "what's sycophancy?" \
  --backend metal --tokens 32

# CPU
./target/release/infer \
  --model models/to-huggingface/Bonsai-1.7B_v2/Bonsai-1.7B_v2.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "The capital of France is" \
  --backend cpu --tokens 32
```


./target/release/infer \
  --model models/to-huggingface/Bonsai-1.7B_v2/Bonsai-1.7B_v2.cellm \
  --tokenizer models/to-huggingface/Bonsai-1.7B_v2/tokenizer.json \
  --prompt "what is sycophancy?" \
  --backend cpu --tokens 64


./target/release/infer \
  --model models/to-huggingface/Bonsai-1.7B_v2/Bonsai-1.7B_v2.cellm \
  --tokenizer models/to-huggingface/Bonsai-1.7B_v2/tokenizer.json \
  --prompt "what is sycophancy?" \
  --backend metal --tokens 100

./target/release/infer \
  --model models/to-huggingface/Bonsai-1.7B_v2/Bonsai-1.7B_v2.cellm \
  --tokenizer models/to-huggingface/Bonsai-1.7B_v2/tokenizer.json \
  --prompt "what is sycophancy?" \
  --backend metal --gen 200

./target/release/infer \
  --model models/to-huggingface/Bonsai-1.7B_v2/Bonsai-1.7B_v2.cellm \
  --tokenizer models/to-huggingface/Bonsai-1.7B_v2/tokenizer.json \
  --chat --chat-format auto \
  --system "I am a 1-bit model developed by PrismML..." \
  --prompt "What is your purpose?" \
  --backend metal \
  --gen 100


---

## Granite 4.0 350M f16

```sh
./target/release/infer \
  --model models/granite-4.0-350m-f16-v2.cellm \
  --tokenizer models/hf/granite-4.0-350m/tokenizer.json \
  --prompt "Write a short poem about space." \
  --backend metal --gen 100

# Sanity check
./target/release/infer \
  --model models/granite-4.0-350m-f16-v2.cellm \
  --tokenizer models/hf/granite-4.0-350m/tokenizer.json \
  --prompt "1 + 1 =" \
  --backend cpu
```

---

## SmolVLM-256M-Instruct

### Convert from HuggingFace

```sh
# Download model files
mkdir -p models/smolvlm-256m-instruct
curl -L "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/model.safetensors?download=true" \
  -o models/smolvlm-256m-instruct/model.safetensors
curl -L "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/config.json" \
  -o models/smolvlm-256m-instruct/config.json
curl -L "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/tokenizer.json?download=true" \
  -o models/smolvlm-256m-instruct/tokenizer.json

# Convert to cellm (with vision tower)
cargo run --release --bin convert -- \
  --input models/smolvlm-256m-instruct \
  --output models/to-huggingface/smolvlm-256m-instruct-f16-full/smolvlm-256m-instruct-f16-full.cellm \
  --dtype f16
```

### Vision

```sh
# CPU — image description
./target/release/vlm-direct \
  --model models/to-huggingface/smolvlm-256m-instruct-f16-full/smolvlm-256m-instruct-f16-full.cellm \
  --image models/test_images/bird.jpg \
  --prompt "What do you see?" \
  --tokens 64 --backend cpu

  ./target/release/vlm-direct \
  --model models/to-huggingface/smolvlm-256m-instruct-int8-v1/smolvlm-256m-instruct-int8-v1.cellm\
  --image models/test_images/bird.jpg \
  --prompt "What do you see?" \
  --tokens 64 --backend cpu

# Output: "A black and white owl is staring at the camera."
# Timings: patch=32.6ms, encoder=4.78s, decode=17.80s, total=23.2s

# Metal — image description
./target/release/vlm-direct \
  --model models/to-huggingface/smolvlm-256m-instruct-f16-full/smolvlm-256m-instruct-f16-full.cellm \
  --image models/test_images/bird.jpg \
  --prompt "What do you see?" \
  --tokens 64 \
  --backend metal

# Output: "A black and white owl is staring at the camera."
# Timings: patch=58.2ms, encoder=9.78s, decode=6.75s, total=17.35s
```

```bash

./target/release/vlm-direct \
  --model models/to-huggingface/smolvlm-256m-instruct-f16-full/smolvlm-256m-instruct-f16-full.cellm \
  --image models/test_images/bird.jpg \
  --prompt "What do you see?" \
  --backend metal \
  --tokens 100
```


```bash
./target/release/vlm-direct \
  --model models/to-huggingface/smolvlm-256m-instruct-f16-full/smolvlm-256m-instruct-f16-full.cellm \
  --image image_5FB898F1-0AC7-401C-AB1D-63E304A75599.png \
  --prompt "What do you see?" \
  --backend metal \
  --tokens 100
```



```bash

export CELLM_LLAMA_USE_MV=1 CELLM_LLAMA_USE_METAL_NORM=1 CELLM_LLAMA_USE_METAL_ROPE=1
export CELLM_LLAMA_ROPE_INTERLEAVED=0   # only for rotate‑half models

./target/release/vlm-direct \
  --model models/to-huggingface/smolvlm-256m-instruct-f16-full/smolvlm-256m-instruct-f16-full.cellm \
  --image models/test_images/bird.jpg \
  --prompt "What do you see?" \
  --backend metal \
  --tokens 32


```


----

The 4-bit quantized model () is approximately **356MB**, fitting well within the 500MB target. It quantizes all linear projections, embeddings, and the LM head while maintaining excellent coherence.


---

## Qwen3 0.6B

### Overview

Qwen3 uses a non-standard attention architecture where `hidden_size != n_heads * head_dim`. For Qwen3-0.6B:
- hidden_size = 1024
- n_heads = 16, head_dim = 128
- attn_dim = 2048 (separate from hidden)

### Convert from HuggingFace

```sh
# Convert f16 model
cargo run --release --bin convert -- \
  --input models/Qwen3-0.6B \
  --output models/Qwen3-0.6B.cellm \
  --dtype f16

# Convert with int4 quantization
cargo run --release --bin convert -- \
  --input models/Qwen3-0.6B \
  --output models/Qwen3-0.6B-int4.cellm \
  --quantize-int4-symmetric
```

### Inference (Metal)

```sh
# Basic inference with sampling (recommended for 0.6B model)
./target/release/infer \
  --model models/Qwen3-0.6B.cellm \
  --tokenizer models/Qwen3-0.6B/tokenizer.json \
  --prompt "Explain consciousness:" \
  --gen 50 --temperature 0.7 --top-k 40 --backend metal

# Chat mode with system prompt
./target/release/infer \
  --model models/Qwen3-0.6B-new.cellm \
  --tokenizer models/Qwen3-0.6B/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format auto \
  --gen 50 --temperature 0.7 --top-k 40 \
  --backend metal --kv-encoding f16 --stop-eos
```

### Inference (CPU)

```sh
./target/release/infer \
  --model models/to-huggingface/qwen3-0.6b-v1/qwen3-0.6b-int8.cellm \
  --tokenizer models/to-huggingface/qwen3-0.6b-v1/tokenizer.json \
  --prompt "What is the capital of France?" \
  --gen 20 --backend cpu
```

```sh
./target/release/infer \
  --model models/to-huggingface/qwen3-0.6b-v1/qwen3-0.6b-int8.cellm \
  --tokenizer models/to-huggingface/qwen3-0.6b-v1/tokenizer.json \
  --prompt "What is the capital of France?" \
  --gen 20 --backend metal
```



### Notes

- Use `--temperature 0.7` or higher for varied outputs (greedy decoding causes repetition in small models)
- Metal backend provides full acceleration for Qwen3 architecture
- Int4 quantization reduces model size ~4x while maintaining coherence
- Requires architectural fix for non-standard attention dimensions

---

## LFM2.5 350M

### Overview

LFM2.5 (Liquid Foundation Model 2.5) uses a hybrid architecture combining:
- LIV Convolution blocks for short-range dependencies
- Grouped Query Attention (GQA) for long-range dependencies
- SwiGLU feedforward networks
- RMSNorm normalization

Architecture: 16 layers alternating between conv and attention blocks.

### Convert from MLX

python3 tools/convert_lfm.py models/LFM2.5-350M-MLX-4bit models/to-huggingface/lfm2.5-350m-v1/lfm2.5-350m-v1.cellm

### Inference (CPU)
```bash
./target/release/infer \
  --model models/to-huggingface/lfm2.5-350m-v1/lfm2.5-350m-v1.cellm \
  --tokenizer models/to-huggingface/lfm2.5-350m-v1/tokenizer.json \
  --prompt "The quick brown fox" \
  --tokens 32 --backend cpu
```



```bash
./target/release/infer \
  --model models/to-huggingface/lfm2.5-350m-v1/lfm2.5-350m-v1.cellm \
  --tokenizer models/to-huggingface/lfm2.5-350m-v1/tokenizer.json \
  --prompt "The quick brown fox" \
  --gen 32 --backend cpu
  ---
  jumps over the lazy dog, but the slowest ever comes first.
```


```bash
./target/release/infer \
  --model models/to-huggingface/LFM2.5-230M/LFM2.5-230M.cellm \
  --tokenizer models/to-huggingface/LFM2.5-230M/tokenizer.json \
  --prompt "What is consciousness? in one paragraph" \
  --gen 100 --backend cpu

#
  ./target/release/infer \
  --model models/to-huggingface/LFM2.5-230M/LFM2.5-230M-int4-v2.cellm \
  --tokenizer models/to-huggingface/LFM2.5-230M/tokenizer.json \
  --prompt "What is consciousness? in one paragraph" \
  --gen 300 --backend cpu

  ---
Consciousness refers to the ability of an individual to be aware of their thoughts, feelings, and surroundings. It is a fundamental aspect of human experience that allows us to navigate our daily lives with ease.
```

```bash
./target/release/infer \
  --model models/to-huggingface/lfm2.5-350m-v1/lfm2.5-350m-v1.cellm \
  --tokenizer models/to-huggingface/lfm2.5-350m-v1/tokenizer.json \
  --prompt "What is consciousness? in one paragraph" \
  --gen 100 --backend cpu

  ---
Consciousness refers to the ability of an individual to be aware of their thoughts, feelings, and surroundings. It is a fundamental aspect of human experience that allows us to navigate our daily lives with ease.
```

```bash
./target/release/infer \
  --model models/to-huggingface/lfm2.5-350m-v1/lfm2.5-350m-v1.cellm \
  --tokenizer models/to-huggingface/lfm2.5-350m-v1/tokenizer.json \
  --prompt "What is consciousness?" \
  --gen 100 --backend metal

  Decode: 100 tokens in 1.19s
  ---
  Consciousness refers to the state of being aware of oneself and one's surroundings.
```



```bash
./target/release/infer \
  --model models/to-huggingface/lfm2.5-350m-v1/lfm2.5-350m-v1.cellm \
  --tokenizer models/to-huggingface/lfm2.5-350m-v1/tokenizer.json \
  --prompt "You rewrite outreach messages. Do not add new facts.

Return only one message.

Rules:
- Under 45 words.
- Warm, simple, human.
- Do not mention AI.
- Do not invent events, meetings, help, or promises.
- Do not say “recent” unless last_contacted is known.
- Use only the base draft and contact facts.
- Keep the same intent.

Contact facts:
name: Sarah
role: Professional
workplace: Accra Tech Summit
relationship_strength: 2/5
last_contacted: never
known_context: Met at Accra Tech Summit
capabilities: Contract Review, Firebase, Flutter, Swift
channel: Messages
intent: Reconnect

Base draft:
Hi Sarah, hope you’re doing well. We met at Accra Tech Summit, and I realized I never properly followed up. I’d love to reconnect sometime this week if you have a free moment.

Rewrite the base draft." \
  --gen 100 --backend metal


  ```

### Performance

- Model size: ~211 MB (4-bit quantized, scales kept as f32)
- CPU speed: ~15 tok/s (Apple Silicon)
- Metal: not implemented for LFM2 runner (passes --backend metal silently falls back to CPU)

## LFM2.5 350M int8 (W8A8 SDOT, CPU)

Per-row symmetric int8 weights with int8-quantized activations, using ARM
`SDOT` integer dot products. Measured on an Apple M4 (4 P-cores + 6 E-cores).

Convert from GGUF:

```bash
cargo build --release -p convert
./target/release/convert \
  --input models/LFM2.5-350M-Q4_0.gguf \
  --output models/to-huggingface/lfm2.5-350m-int8-v1/lfm2.5-350m-int8-v1.cellm \
  --quantize int8
```

LFM2 needs its chat template applied manually via `--prompt`:

```bash
./target/release/infer \
  --model models/to-huggingface/lfm2.5-350m-int8-v1/lfm2.5-350m-int8-v1.cellm \
  --tokenizer models/to-huggingface/lfm2.5-350m-int8-v1/tokenizer.json \
  --prompt '<|startoftext|><|im_start|>user
What is consciousness? Answer in one paragraph.<|im_end|>
<|im_start|>assistant
' \
  --gen 200 --backend cpu

  Prefill: 18 tokens in 0.70s
  Decode:  98 tokens in 0.87s   (~113 tok/s, stopped at EOS)
  ---
  Consciousness is the state of subjective experience, the ability to have
  thoughts, emotions, and perceptions directly within the mind; it involves
  awareness of oneself as an individual and awareness of the environment,
  including internal mental states such as sensations, feelings, and beliefs;
  this inner experience can be fragmented or unified, depending on how
  consciousness manifests in different contexts, from a fleeting moment to a
  full-day reflection, and it arises without a clear external cause or
  boundary between mind and body.
```

Steady-state decode, 300 tokens, three consecutive runs:

```bash
./target/release/infer \
  --model models/to-huggingface/lfm2.5-350m-int8-v1/lfm2.5-350m-int8-v1.cellm \
  --tokenizer models/to-huggingface/lfm2.5-350m-int8-v1/tokenizer.json \
  --prompt '<|startoftext|><|im_start|>user
Write a detailed explanation of how photosynthesis works.<|im_end|>
<|im_start|>assistant
' \
  --gen 300 --backend cpu

  Prefill: 18 tokens in 0.15s
  Decode:  300 tokens in 2.56s / 2.60s / 2.56s   (~117 tok/s)
```

### Speed

| Stage                              | Decode (350M, CPU) |
| ---------------------------------- | ------------------ |
| f16 baseline                       | ~25 tok/s          |
| int8 weights, f32 MAC              | ~88 tok/s          |
| int8 W8A8 + `SDOT` + P-core pool   | **~117-123 tok/s** |

Two changes mattered, and the second was the larger one:

1. `cpu_kernels::gemv_i8_w8a8` quantizes the activation to int8 and uses ARM
   `SDOT`. Isolated, this is ~10x the old f32-MAC kernel.
2. Thread-pool sizing. That 10x kernel only moved end-to-end speed 1.3x. A
   leaf-weighted profile showed just 23% of samples in the kernel and ~57% in
   rayon fork/join. On the M4's 4 P + 6 E cores, rayon's default 10 threads
   makes the E-cores straggle at every join, so `init_decode_thread_pool()`
   now caps the pool at `hw.perflevel0.logicalcpu`.

Note: `vdotq_s32` is still unstable on stable Rust (`stdarch_neon_dotprod`),
so the kernel emits the instruction via inline `asm!`.

### This is at the memory roof — further tuning will not help

Decode streams essentially the whole weight set per token. Exact traffic from
the model header (`scratch/roofline.py`):

```
per-token weight bytes: 355.1 MB
  mlp                    226.8 MB  (63.9%)
  embed/logits (tied)     67.2 MB  (18.9%)
  conv blocks             42.1 MB  (11.9%)
  attention               18.9 MB  ( 5.3%)

at 123 tok/s  ->  43.7 GB/s of weight traffic
```

`benches/w8a8_stream.rs` compares the kernel against a *pure read* of the same
buffers. On a warm resident working set the GEMV runs at **98-100% of the read
roof** (~45-52 GB/s) — it is purely bandwidth-bound, and 3 vs 6 threads land at
the same GB/s.

Consequences:

- **SMMLA / `i8mm` is not worth implementing here.** It doubles compute
  throughput, but compute is already free: every MAC is waiting on a weight
  byte. It would cost a converter change and a blocked weight layout for ~0%.
- A profile dominated by `__psynch_cvwait` / `swtch_pri` here is *not*
  evidence of lock overhead. Those threads are waiting on DRAM. This was
  initially misread as "1.5-2x left in thread orchestration"; the roofline
  bench disproved it.
- The only remaining levers reduce *bytes moved*, and both trade accuracy for
  speed: Q4 weights (halves the 355 MB), or avoiding a full read of all 65536
  logit rows (19% of traffic to produce one token).

### Quality

Fluent and well-structured, but the model confabulates at this size. In the
photosynthesis run above it invented the term "photophysis", mislabeled the
light-dependent stage as "the dark phase", and swapped the contents of the
light-dependent and light-independent reactions. Treat 350M output as drafting
material, not reference text.

## Qwen3.0

```bash
./target/release/infer \
  --model models/to-huggingface/qwen3-0.6b-v1/qwen3-0.6b-int8.cellm \
  --tokenizer models/to-huggingface/qwen3-0.6b-v1/tokenizer.json \
  --prompt "whats the weather like in Accra?" \
  --gen 64  --backend cpu
```



## Qwen3.5-0.8B

### Download

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir models/hf/qwen3.5-0.8b
```

### Convert

```bash
# f16
./target/release/convert \
  --input models/hf/qwen3.5-0.8b \
  --output models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-f16.cellm \
  --dtype f16

# int4
./target/release/convert \
  --input models/hf/qwen3.5-0.8b \
  --output models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-i4.cellm \
  --quantize-int4-symmetric

# 1-bit
./target/release/convert \
  --input models/hf/qwen3.5-0.8b \
  --output models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-q1.cellm \
  --quantize-int1-symmetric
```

### Run (f16)

```bash
# CPU
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-f16.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "Hello, who are you?" \
  --chat --chat-format auto \
  --gen 64 --temperature 0 --backend cpu --kv-encoding f16
```



```bash
./target/release/infer \
  --model models/to-huggingface/qwen3-0.6b-v1/qwen3-0.6b-int8.cellm \
  --tokenizer models/to-huggingface/qwen3-0.6b-v1/tokenizer.json \
  --prompt "Hello, who are you?" \
  --chat --chat-format auto \
  --gen 64 --temperature 0 --backend cpu --kv-encoding f16
```


```bash
# CPU
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-q1.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "who owns twitch?" \
  --gen 64 --backend cpu
```


```bash
# CPU
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-q1.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "Hello, who are you?" \
  --chat --chat-format auto \
  --gen 64 --temperature 0 --backend cpu --kv-encoding f16
```



```bash
# metal
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-f16.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "2 + 2?" \
  --chat --chat-format auto \
  --gen 64 --temperature 0 --backend metal --kv-encoding f16
```


### Run (int4)

```bash
# CPU
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-i4.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "Hello, who are you?" \
  --chat --chat-format auto \
  --gen 64 --temperature 0 --backend cpu --kv-encoding f16
```

```bash
# CPU
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-i4.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "Hello, who are you?" \
  --chat --chat-format auto \
  --gen 64 --backend cpu
```


```bash
# Default: thinking enabled
./target/release/infer \
--model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-i4.cellm \ --tokenizer  models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "What is 2+2?" --chat --chat-format auto --gen 100 --backend cpu

# Skip thinking
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-i4.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "What is 2+2?" --chat --chat-format auto --gen 50 --backend cpu --no-think


```


```bash
# Default: thinking enabled
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-q1.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "What is 2+2?" --chat --chat-format auto --gen 100 --backend cpu

# Skip thinking
./target/release/infer --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-q1.cellm \ --tokenizer  models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "What is 2+2?" --chat --chat-format auto --gen 50 --backend cpu --no-think


```



```bash
# METAL
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-i4.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "Hello, who are you?" \
  --chat --chat-format auto \
  --gen 64 --temperature 0 --backend metal --kv-encoding f16
```


```bash
# METAL
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-i4.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "Hello, who are you?" \
  --gen 64 --backend metal
```


```bash
# METAL
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-i4.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "2 + 2?" \
  --gen 64 --backend metal
```


### Run (1-bit)

```bash
# CPU (research only, output is incoherent at 0.8B)
./target/release/infer \
  --model models/to-huggingface/qwen3.5-0.8b-v1/qwen3.5-0.8b-q1.cellm \
  --tokenizer models/to-huggingface/qwen3.5-0.8b-v1/tokenizer.json \
  --prompt "Hello" \
  --gen 32 --temperature 0 --backend cpu --kv-encoding f16
```

### Performance

| Variant | Size | Quality |
|---------|------|---------|
| f16 | 1.6 GB | Excellent |
| int4 | 755 MB | Good |
| int2 | 836 MB | Incoherent |
| 1-bit | 200 MB | Incoherent |

- Architecture: 24 layers with hybrid linear/full attention (DeltaNet)
- CPU speed: ~7 tok/s (Apple Silicon, f16)
- int4 is the smallest quantization that preserves quality at 0.8B
- 1-bit and int2 require quantization-aware training for coherent output at this model size

---

## DeepSeek-V4 (Nanowhale-100m)

DeepSeek-V4 architecture (HC + MLA + MoE) research model.

```sh
# CPU — Nanowhale-100m (f16)
./target/release/infer \
  --model models/nanowhale-100m.cellm \
  --tokenizer models/nanowhale-100m/tokenizer.json \
  --prompt "what's sycophancy?" \
  --chat --gen 100 --temperature 0 --backend cpu --kv-encoding f16
```
---

## DeepSeek-V4 (MLA + MoE)

### Overview

DeepSeek-V4 introduces a high-efficiency architecture featuring:
- **MLA (Multi-head Latent Attention)**: Dramatically reduces KV cache size via low-rank compression.
- **MoE (Mixture of Experts)**: Utilizes a sparse "DeepSeekMoE" architecture for high capacity with low active FLOPs.
- **Multi-Token Prediction**: (Not yet utilized in current runner).

### Inference (CPU)

```sh
./target/release/infer \
  --model models/deepseek-v4-toy.cellm \
  --tokenizer models/deepseek-v4-toy/tokenizer.json \
  --prompt "The capital of France is" \
  --gen 32 --backend cpu

  ./target/release/infer \
  --model models/to-huggingface/nanowhale-100m-v1/nanowhale-100m-v1.cellm \
  --tokenizer models/to-huggingface/nanowhale-100m-v1/tokenizer.json \
  --prompt "What are 3 benefits of exercise?" \
  --gen 100 --backend cpu


  ./target/release/infer \
  --model models/to-huggingface/nanowhale-100m-v1/nanowhale-100m-v1.cellm \
  --tokenizer models/to-huggingface/nanowhale-100m-v1/tokenizer.json \
  --prompt "What are 3 benefits of exercise?" \
  --gen 100 --backend metal


  ./target/release/infer \
  --model models/to-huggingface/nanowhale-100m-v1/nanowhale-100m-v1.cellm \
  --tokenizer models/to-huggingface/nanowhale-100m-v1/tokenizer.json \
  --prompt "<｜begin of sentence｜><｜User｜>what's sycophancy?<｜Assistant｜>" \
  --gen 100 --temperature 0 --backend cpu --kv-encoding f16



  ./target/release/infer \
  --model models/to-huggingface/nanowhale-100m-v1/nanowhale-100m-v1.cellm \
  --tokenizer models/to-huggingface/nanowhale-100m-v1/tokenizer.json \
  --prompt "<｜begin of sentence｜><｜User｜>what's sycophancy?<｜Assistant｜>" \
  --gen 100 --temperature 0 --backend metal --kv-encoding f16
```

### Notes
- Current runner implementation is CPU-only.
- Supports MLA with Sinkhorn normalization and MoE expert routing.
- Optimized for large-scale efficient inference.
------

## Llama 3.2 3B Instruct Q4 (`.base` to `.cellm`)

### Overview

The BaseRT `base_q4` source uses asymmetric group-wise quantization with groups of 64 values:

```text
weight = q_unsigned * scale + bias
```

For this model, scales and biases are stored as `bf16` in row-major group order. The `.base` weights blob begins at the first 64 KiB boundary after the JSON header. Llama 3.2 also requires `llama3` RoPE scaling (`factor=32`, original context 8192, low/high frequency factors 1/4). BaseRT's fixed Q/K permutation uses interleaved (adjacent-pair) rotary layout; the converter records this in the cellm header.

### Convert

```sh
python3 tools/convert_base_to_cellm.py \
  --input models/gguf/Llama-3.2-3B-Instruct-Q4.base \
  --output models/Llama-3.2-3B-Instruct-Q4-affine-v1.cellm \
  --tokenizer models/Llama-3.2-3B-Instruct \
  --quantize affine-i4
```

The resulting model contains 646 tensors and is 2,550,044,032 bytes (2.37 GiB). Embeddings and normalization weights remain `f16`; projection matrices preserve the source unsigned affine Q4 nibbles plus per-group `f32` scales and biases. Do not use row-wise symmetric `i4` for this already-quantized source: collapsing each row's group parameters causes severe quality loss.

### Inference (CPU)

```sh
./target/release/infer \
  --model models/Llama-3.2-3B-Instruct-Q4-affine-v1.cellm \
  --tokenizer models/Llama-3.2-3B-Instruct/tokenizer.json \
  --prompt "What is the capital of France?" --chat \
  --gen 32 --temperature 0 --top-k 1 --repeat-penalty 1.0 \
  --backend cpu --kv-encoding f16
```

`--chat --chat-format auto` detects the embedded Llama 3 template, and generation stops on `<|eot_id|>`/`<|eom_id|>`. Validated CPU output: `The capital of France is Paris.`

---

## openai/privacy-filter (PII token classifier)

### Overview

An encoder-only token classifier, not a generator, so it does not run through `infer`. It has its own runner (`crates/cellm-model/src/privacy_filter.rs`) and its own binary (`pii`). Six things kept it off the existing paths: bidirectional attention, per-head attention sinks, the `head_dim**-0.25` scale applied separately to Q *and* K, interleaved YaRN RoPE, a 128-expert top-4 MoE with clamped SwiGLU, and group-32 int4 with `f16` scale/bias sidecars (`lfm.rs` hardcodes group 64 and `f32`).

640 hidden, 8 layers, 14 heads / 2 KV heads, head_dim 64, 128 experts top-4, vocab 200064, 33 BIOES labels, bidirectional sliding window of 128.

### Convert

```sh
python3 tools/convert_privacy_filter_hf.py \
  models/hf/privacy-filter \
  models/privacy-filter/privacy-filter-int4-g32-f16s.cellm \
  --quant int4 --group-size 32 --quant-embedding
```

6269 tensors, 942,793,408 bytes (899 MiB). Group 32 is the smallest recipe that leaks nothing: at group 64 five entities present in the f32 baseline go undetected. `--quant-embedding` stores `embed_tokens` as int8 (−128 MB); int4 there costs 18 missed entities, so it is not offered. Omitting `--f32-scales` keeps the sidecars in `f16`, which is where 157 MB of the earlier 1049 MB build went — measured accuracy is identical.

### Build

```sh
cargo build --release -p cellm-pii
```

### Inference (CPU)

```sh
./target/release/pii \
  models/privacy-filter/privacy-filter-int4-g32-f16s.cellm \
  --tokenizer models/hf/privacy-filter/tokenizer.json \
  --text "Contact Bob Smith at bob.smith@example.com or (555) 123-4567."
```


```text
=== Contact Bob Smith at bob.smith@example.com or (555) 123-4567.
    private_person   [   7:  17]  " Bob Smith"
    private_email    [  20:  42]  " bob.smith@example.com"
    private_phone    [  45:  60]  " (555) 123-4567"
```



#### test samples

```text
./target/release/pii \
  models/privacy-filter/privacy-filter-int4-g32-f16s.cellm \
  --tokenizer models/hf/privacy-filter/tokenizer.json \
  --text "To: John Doe (john.doe@email.com), Cc: finance-team@company.com. Subject: Cloud Server Config. Hey Team, use API key AKIAIOSFODNN7EXAMPLE and Stripe test card 4242-4242-4242-4242 (exp: 08/28, CVV: 123) for the new billing portal setup."
```

./target/release/pii \
  models/privacy-filter/privacy-filter-int4-g32-f16s.cellm \
  --tokenizer models/hf/privacy-filter/tokenizer.json \
  --text "John Smith lives at 1600 Pennsylvania Ave, Washington DC. His email is john@gmail.com"


```sh
./target/release/pii \
  models/privacy-filter/privacy-filter-int4-g32-f16s.cellm \
  --tokenizer models/hf/privacy-filter/tokenizer.json \
  --text "Dr. Ama Serwaa Mensah, born on 14 February 1989, recently moved from House No. 17B, Mango Street, East Legon, Accra, Ghana to Apartment 4C, 221B Baker Street, London NW1 6XE. Her Ghana Card number is GHA-123456789-0, passport number G3456789, employee ID ECG-ICT-004721, and tax identification number P0012345678. You can reach her at ama.mensah+work@example.co.uk, backup_email99@gmail.com, +233 24 555 0198, 024-555-0198, or on WhatsApp at +44 7700 900123. Her bank account is 0040163411018 at Republic Bank, SWIFT code HFCAGHAC, sort code 11-01-04, and card number 4111 1111 1111 1111, expiring 09/29 with CVV 317. She connected from IPv4 address 192.168.10.45, public IP 102.176.94.21, IPv6 address 2001:0db8:85a3:0000:0000:8a2e:0370:7334, and device MAC address A4:5E:60:11:22:33. Her username is ama_mensah89 and the temporary password is Summer2026!DoNotShare. The medical record ECG-2026-77891 states that she visited Ridge Hospital on 3 August 2026. Emergency contact: Kwame Osei, phone +233 20 111 2233, living near Independence Square, Accra. Please note that Apple, Republic Bank, Ridge Hospital, Washington, and ECG may be organization or location names rather than private persons. The example IP 8.8.8.8 is a public DNS server, order number ORD-2026-00081 is not a national ID, and 123456 is merely a verification example."
```

```sh
./target/release/pii \
models/privacy-filter/privacy-filter-int3-g128.cellm\
  --tokenizer models/hf/privacy-filter/tokenizer.json \
  --text "Dr. Ama Serwaa Mensah, born on 14 February 1989, recently moved from House No. 17B, Mango Street, East Legon, Accra, Ghana to Apartment 4C, 221B Baker Street, London NW1 6XE. Her Ghana Card number is GHA-123456789-0, passport number G3456789, employee ID ECG-ICT-004721, and tax identification number P0012345678. You can reach her at ama.mensah+work@example.co.uk, backup_email99@gmail.com, +233 24 555 0198, 024-555-0198, or on WhatsApp at +44 7700 900123. Her bank account is 0040163411018 at Republic Bank, SWIFT code HFCAGHAC, sort code 11-01-04, and card number 4111 1111 1111 1111, expiring 09/29 with CVV 317. She connected from IPv4 address 192.168.10.45, public IP 102.176.94.21, IPv6 address 2001:0db8:85a3:0000:0000:8a2e:0370:7334, and device MAC address A4:5E:60:11:22:33. Her username is ama_mensah89 and the temporary password is Summer2026!DoNotShare. The medical record ECG-2026-77891 states that she visited Ridge Hospital on 3 August 2026. Emergency contact: Kwame Osei, phone +233 20 111 2233, living near Independence Square, Accra. Please note that Apple, Republic Bank, Ridge Hospital, Washington, and ECG may be organization or location names rather than private persons. The example IP 8.8.8.8 is a public DNS server, order number ORD-2026-00081 is not a national ID, and 123456 is merely a verification example."
```


```sh
./target/release/pii \
models/privacy-filter/privacy-filter-int4-emb8.cellm\
  --tokenizer models/hf/privacy-filter/tokenizer.json \
  --text "Dr. Ama Serwaa Mensah, born on 14 February 1989, recently moved from House No. 17B, Mango Street, East Legon, Accra, Ghana to Apartment 4C, 221B Baker Street, London NW1 6XE. Her Ghana Card number is GHA-123456789-0, passport number G3456789, employee ID ECG-ICT-004721, and tax identification number P0012345678. You can reach her at ama.mensah+work@example.co.uk, backup_email99@gmail.com, +233 24 555 0198, 024-555-0198, or on WhatsApp at +44 7700 900123. Her bank account is 0040163411018 at Republic Bank, SWIFT code HFCAGHAC, sort code 11-01-04, and card number 4111 1111 1111 1111, expiring 09/29 with CVV 317. She connected from IPv4 address 192.168.10.45, public IP 102.176.94.21, IPv6 address 2001:0db8:85a3:0000:0000:8a2e:0370:7334, and device MAC address A4:5E:60:11:22:33. Her username is ama_mensah89 and the temporary password is Summer2026!DoNotShare. The medical record ECG-2026-77891 states that she visited Ridge Hospital on 3 August 2026. Emergency contact: Kwame Osei, phone +233 20 111 2233, living near Independence Square, Accra. Please note that Apple, Republic Bank, Ridge Hospital, Washington, and ECG may be organization or location names rather than private persons. The example IP 8.8.8.8 is a public DNS server, order number ORD-2026-00081 is not a national ID, and 123456 is merely a verification example."
```


```

./target/release/pii \
  models/privacy-filter/privacy-filter-int4-g32-f16s.cellm \
  --tokenizer models/hf/privacy-filter/tokenizer.json \
  --text 'CONFIDENTIAL INCIDENT REPORT
Employee: Nana Yaw Boateng
Preferred name: Yaw
Date of birth: 22 September 1992
Residential address: Flat 7B, Adom Heights, Boundary Road, East Legon Hills, Accra, Ghana
Previous address: P.O. Box CT 1847, Cantonments, Accra
Personal email: nana.boateng+private@samplemail.com
Work email: yaw.boateng@ecg-example.org
Primary phone: +233 (0)24 718 9032
Alternative phone: 020-555-0147
Emergency contact: Akosua Owusu, his sister, reachable at +233 50 444 8219.

Identification details:
Ghana Card: GHA-987654321-4
Passport number: G00048291
Taxpayer ID: P0098765432
Employee number: ECG/ICT/2026/00481
Health insurance membership number: NHIS-2039-8841-752
Driver licence: DVLA-GH-92-104883
Medical file number: RIDGE-MRN-2026-008741

Payroll details:
Bank: Example Republic Bank
Account holder: Nana Yaw Boateng
Account number: 0040163411099
Branch: Ridge
Sort code: 11-01-04
SWIFT/BIC: HFCAGHAC
Payment card: 4111 1111 1111 1111
Expiry date: 09/29
CVV: 317
Mobile Money wallet: +233 24 718 9032
Monthly salary: GHS 6,250.00

Medical note:
On 1 August 2026, Dr. Efua Sarpong recorded that Nana had a severe peanut allergy and prescribed TESTMED-25MG. His blood type is O+, and his next appointment is scheduled for 18 August 2026 at Ridge Hospital.

Security incident:
The employee reported receiving a password-reset message. His username is nboateng92, temporary password is Temp#Access2026!, recovery answer is BlueMango47, and one-time code is 483921. The internal service generated API key sk_test_7vN2pQ8mL4xR9aBc, bearer token eyJhbGciOiJIUzI1NiJ9.test-signature, and session ID sess_9f2d71a8c3e64b2f.

Device information:
Hostname: ECG-LAPTOP-0481
Private IPv4: 10.30.41.87
Public IPv4: 102.176.94.21
IPv6: 2001:db8:85a3::8a2e:370:7334
MAC address: A4:5E:60:11:22:33
Wi-Fi SSID: ECG-Staff
Device serial number: C02TEST9MD6T
Last login: 2026-08-04T14:32:18Z

Relevant application log:
{
  "customer_name": "Nana Yaw Boateng",
  "email": "nana.boateng+private@samplemail.com",
  "phone": "+233247189032",
  "ghana_card": "GHA-987654321-4",
  "account": "0040163411099",
  "ip": "10.30.41.87",
  "authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.test-signature",
  "password": "Temp#Access2026!",
  "callback_url": "https://example.test/reset?token=abc987XYZ",
  "message": "Customer requested assistance."
}

Database error:
postgres://nboateng:DbPass%402026@10.30.41.12:5432/hr_records
Connection failed for user nboateng from 10.30.41.87.

Support transcript:
Agent: Please confirm the last four digits of your account.
Customer: They are 1099. My full account is 0040163411099.
Agent: Do not share your password.
Customer: I already emailed Temp#Access2026! to yaw.boateng@ecg-example.org.

Control statements:
Apple released a software update on 14 July.
Republic Bank opened a new branch.
The server processed 123456 records.
Order ORD-2026-00081 was delivered successfully.
Version 10.30.41 was released internally.
The value 4111 is only a count.
Contact support@example.test for fictional testing.
Google public DNS is available at 8.8.8.8.

END OF REPORT'

```


`--text` repeats for batching. `--redact` additionally prints the spliced string:

```text
    redacted: Contact[PRIVATE_PERSON] at[PRIVATE_EMAIL] or[PRIVATE_PHONE].
```

Offsets come straight from the tokenizer, so a leading space belongs to the token and lands inside the span — a caller wanting clean text should trim.

### Parity

`--dump-logits <path>` writes, per text, `u32 seq_len`, `u32 n_labels`, then `seq*labels` little-endian `f32`, for comparison against a HuggingFace reference:

```sh
./target/release/pii ... --dump-logits /tmp/pf_rust.bin
```

100% argmax parity with HF fp32 on both a 53-token corpus (max |Δlogit| 2.96) and a 328-token corpus (max |Δlogit| 8.35). The long input is the one that matters: below 129 tokens the sliding window never engages, and an off-by-one in the mask is invisible.




### Notes

- CPU-only, and unoptimized: every layer is dequantized on each `forward()` with no caching, and the matmul is a naive loop rather than `cellm_kernels::matmul_f32`. ~6 s wall for three short texts, cold.
- Span decoding is greedy BIOES. `models/hf/privacy-filter/viterbi_calibration.json` ships transition biases for a constrained decode that is not yet implemented.
- The `tokenizer.json` stores merges as arrays, which needs `tokenizers` 0.21+; `tools/infer` and `tools/bench` still pin 0.15 and would fail to load it.
- Accuracy evidence is a 30-text / 1025-token sweep, not a broad benchmark.
- The model recognizes 8 entity types: `account_number`, `private_address`, `private_date`, `private_email`, `private_person`, `private_phone`, `private_url`, `secret`. Anything else (national ID, license plate, medical record) is `O`.
- **Documentation placeholder keys are not flagged.** `AKIAIOSFODNN7EXAMPLE` scores `O` at p≈1.000 across every token, while `AKIA4TZQ8W2LMXPVK9RJ` is caught as `secret`. Bisecting the string shows the trigger is the trailing `EXAMPLE`/`SAMPLE`/`AMPLE` token, not the `AKIA` prefix — the model learned that placeholder-looking keys are not real credentials. This is training-data behavior, verified identical in HF fp32, not a quantization or runner artifact. It means canned examples in test fixtures will under-report.
- Structured secrets split rather than span: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` yields two fragments (`EMI/K7`, `/bPxRfi`) instead of one. A redactor should merge or widen `secret` spans before splicing.
- Bare numerics are labeled by surrounding context, and the context can be wrong: a card CVV `123` came back `private_address` at p=0.47 vs `private_date` at p=0.42 — low confidence and effectively a coin flip. Card numbers land in `account_number`; there is no dedicated card/CVV class.
