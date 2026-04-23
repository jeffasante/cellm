# Gemma 4 E2B INT4 Aggressive v2

**File:** `gemma-4-E2B-it-int4-aggr-v2.cellmd`  
**Size:** 2.4 GB  
**Source:** `google/gemma-4-E2B-it`

## What This Model Includes

This is a fully multimodal Gemma 4 E2B model with all modalities preserved:
- **Text** - 35 layers, 1536 hidden size
- **Vision** - 16-layer ViT encoder
- **Audio** - 12-layer USM Conformer encoder
- **Video** - Uses vision encoder

## Quantization

All 2D weight tensors are quantized to INT4 symmetric:
- Text stack (embeddings, attention, MLP projections)
- Vision tower weights
- Audio tower weights
- Multimodal embedders

Kept in F16:
- Normalization layers (norm, q_norm, k_norm)
- Position embeddings
- Per-layer projection norms

## Audio Dimension Alignment

The E2B model has correct audio→text dimension alignment:
```
audio_tower.output_proj.weight     [1536, 1024]  # audio hidden → intermediate
embed_audio.embedding_projection   [1536, 1536]  # intermediate → text hidden
```

Text hidden size = 1536, matching the audio output projection.

## How It Was Created

1. Downloaded `google/gemma-4-E2B-it` from HuggingFace (10.2 GB safetensors)

2. Modified `tools/convert/src/main.rs` to quantize vision/audio tensors:
```rust
// Added vision and audio stack detection
let is_vision_stack = name.starts_with("model.vision_tower.")
    || name.contains("embed_vision.");
let is_audio_stack = name.starts_with("model.audio_tower.")
    || name.contains("embed_audio.");
return is_text_stack || is_vision_stack || is_audio_stack;
```

3. Converted with aggressive INT4:
```bash
cargo run --release --bin convert -- \
  --input models/gemma-4-E2B-it \
  --output models/gemma-4-E2B-it-int4-aggr-v2.cellmd \
  --gemma4-aggressive-int4 \
  --quantize-int4-symmetric
```

4. Copied tokenizer:
```bash
cp models/gemma-4-E2B-it/tokenizer.json models/gemma-4-E2B-it-int4-aggr-v2.tokenizer.json
```

## Usage

```bash
# Audio inference (requires 16kHz audio)
cargo run --release --bin audio-direct -- \
  --model models/gemma-4-E2B-it-int4-aggr-v2.cellmd \
  --audio audio.wav \
  --prompt "Transcribe this audio:" \
  --tokens 100

# Vision inference
cargo run --release --bin vlm-direct -- \
  --model models/gemma-4-E2B-it-int4-aggr-v2.cellmd \
  --image image.jpg \
  --prompt "Describe this image:"
```

## Size Breakdown

| Component | Original (BF16) | Quantized (INT4) |
|-----------|-----------------|------------------|
| Text      | 9.5 GB          | ~2.2 GB          |
| Vision    | 0.34 GB         | ~0.1 GB          |
| Audio     | 0.62 GB         | ~0.1 GB          |
| **Total** | **10.2 GB**     | **2.4 GB**       |

Compression ratio: ~4.3x
