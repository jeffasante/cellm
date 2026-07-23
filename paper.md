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
