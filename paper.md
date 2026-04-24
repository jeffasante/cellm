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
  --tokenizer models/Qwen3-0.6B/tokenizer.json \
  --prompt "who is elon musk" \
  --chat --chat-format auto \
  --gen 50 --temperature 0.7 --top-k 40 \
  --backend metal --kv-encoding f16 --stop-eos
```


### Inference (CPU)

```sh
./target/release/infer \
  --model models/Qwen3-0.6B.cellm \
  --tokenizer models/Qwen3-0.6B/tokenizer.json \
  --prompt "What is the capital of France?" \
  --gen 20 --backend cpu
```





### Notes

- Use `--temperature 0.7` or higher for varied outputs (greedy decoding causes repetition in small models)
- Metal backend provides full acceleration for Qwen3 architecture
- Int4 quantization reduces model size ~4x while maintaining coherence
- Requires architectural fix for non-standard attention dimensions


Model	Status
Qwen3-0.6B.cellm (f16)	✅ Works
Qwen3-0.6B-new.cellm (f16)	✅ Works
Qwen3-0.6B-fixed-int4.cellm (int4)	✅ Now works!
Qwen3-0.6B-int4.cellm (old int4)	❌ Garbage (pre-fix)


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
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.tokenizer.json \
  --prompt "what's twitch.com?" \
  --chat --chat-format plain --gen 48 --temperature 0 --backend cpu --kv-encoding f16

# Metal
./target/release/infer \
  --model models/to-huggingface/gemma-3-1b-it-mixed-int4-v1/gemma-3-1b-it-mixed-int4-v1.cellm \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format plain --gen 48 --temperature 0 --backend metal --kv-encoding f16


  ./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.cellmd \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v2/gemma-4-E2B-it-int4-aggr-v2.tokenizer.json \
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

```sh
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
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "What is consciousness?" \
  --chat --chat-format auto --gen 32 --temperature 0 --backend cpu --kv-encoding f16

# Metal — f16
./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "what's sycophancy?" \
  --chat --gen 32 --temperature 0 --backend metal --kv-encoding f16

# Metal — turboquant KV
./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt $'Graph Lowering Compiler Techniques for\nNeural Networks?' \
  --chat --chat-format auto --gen 100 --temperature 0 --backend metal --kv-encoding turboquant

# CPU — who is elon musk
./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "who is elon musk?" \
  --chat --tokens 100 --backend cpu


  ./target/release/infer \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd \
  --tokenizer models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/tokenizer.json \
  --prompt "who is elon musk?" \
  --chat --gen 100 --backend metal

```

### Vision

```sh
MODEL=models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd

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
  
  
  MODEL=models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd
  
  ./target/release/vlm-direct \
    --model "$MODEL" \
    --image models/test_images/bird.jpg \
    --prompt "What is in this image?" \
    --backend metal --tokens 16
    



    MODEL=models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd
./target/release/vlm-direct \
  --model "$MODEL" \
  --image models/test_images/bird.jpg \
  --prompt "What is in this image?" \
  --backend metal \
  --tokens 16


```

### Audio

```sh
MODEL=models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd

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
  --model models/bonsai-1.7b/Bonsai-1.7B_v2.cellm \
  --tokenizer models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json \
  --prompt "what's sycophancy?" \
  --backend metal --tokens 32

# CPU
./target/release/infer \
  --model models/bonsai-1.7b/Bonsai-1.7B_v2.cellm \
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
  --backend metal

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
  --tokens 64

# Output: "A black and white owl is staring at the camera."
# Timings: patch=642ms, encoder=166s, decode=22s, total=190s
```

./target/release/vlm-direct \
  --model models/to-huggingface/smolvlm-256m-instruct-f16-full/smolvlm-256m-instruct-f16-full.cellm \
  --image models/test_images/bird.jpg \
  --prompt "What do you see?" \
    --backend metal
  --tokens 64


```bash

./target/release/vlm-direct \
  --model models/to-huggingface/smolvlm-256m-instruct-f16-full/smolvlm-256m-instruct-f16-full.cellm \
  --image models/test_images/bird.jpg \
  --prompt "What do you see?" \
  --backend metal \
  --gen 100
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
  --model models/Qwen3-0.6B.cellm \
  --tokenizer models/Qwen3-0.6B/tokenizer.json \
  --prompt "What is the capital of France?" \
  --gen 20 --backend cpu
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

python3 tools/convert_lfm.py models/LFM2.5-350M-MLX-4bit models/LFM2.5-350M.cellm

### Inference (CPU)

./target/release/infer \
  --model models/LFM2.5-350M.cellm \
  --tokenizer models/LFM2.5-350M-MLX-4bit/tokenizer.json \
  --prompt "The quick brown fox" \
  --tokens 32 --backend cpu

### Performance

- Model size: ~190 MB (4-bit quantized)
- CPU speed: ~1.5 tok/s (Apple Silicon)
- Metal backend: not yet implemented

## Qwen3.5-0.8B

### Download

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir models/hf/qwen3.5-0.8b
```

### Convert

```bash
./target/release/convert \
  --input models/hf/qwen3.5-0.8b \
  --output models/qwen3.5-0.8b.cellm \
  --dtype f16
```

### Run

```bash
./target/release/infer \
  --model models/qwen3.5-0.8b.cellm \
  --tokenizer models/hf/qwen3.5-0.8b/tokenizer.json \
  --prompt "Hello, how are you?" \
  --chat \
  --gen 64 \
  --backend cpu \
  --kv-encoding f16
```

### Performance

- Model size: ~1.6 GB (f16)
- CPU speed: ~7 tok/s (Apple Silicon)
- Architecture: 24 layers with hybrid linear/full attention (DeltaNet)

