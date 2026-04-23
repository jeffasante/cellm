# Audio Inference — Gemma 4 E2B

How to run audio-conditioned generation with the Gemma 4 E2B model using the `audio-direct` test binary.

---

## Prerequisites

- Rust toolchain installed (`cargo`)
- Model file: `models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd`
- A **16 kHz mono WAV file** (PCM-16). The encoder rejects any other sample rate.

---

## 1. Build

```bash
cd /Users/jeff/Desktop/cellm

cargo build --release -p cellm-sdk -p vlm-smoke
```

The binary lands at `./target/release/audio-direct`.

---

## 2. Prepare a test audio file

The encoder requires **16 kHz, mono, PCM-16 WAV**. If you have a different file, resample it first (e.g. with `ffmpeg` or `sox`).

Download a ready-to-use sample:

```bash
# Cantina Band, ~2 s, 16 kHz mono
curl -sL "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand60.wav" -o /tmp/test_audio.wav
file /tmp/test_audio.wav   # should say: RIFF, Wave audio, Microsoft PCM, 16 bit, mono 11025 Hz
```

> If the sample rate shown is not 16000 Hz, resample before running:
> ```bash
> ffmpeg -i /tmp/test_audio.wav -ar 16000 -ac 1 /tmp/test_audio_16k.wav
> ```

---

## 3. Basic inference

```bash
cd /Users/jeff/Desktop/cellm

./target/release/audio-direct \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd \
  --audio /tmp/test_audio.wav \
  --prompt "Describe what you hear in this audio." \
  --tokens 80
```

Expected output format:
```
Running audio VLM with max_new_tokens=80
Timings: mel=11ms encoder=15940ms decode=20290ms total=38240ms
Output:
<generated text here>
```

---

## 4. Example prompts

Ask different questions about the same audio clip:

```bash
MODEL=models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd
AUDIO=/tmp/test_audio.wav

# What music is this?
./target/release/audio-direct \
  --model "$MODEL" --audio "$AUDIO" \
  --prompt "What music do you hear?" \
  --tokens 60

# Identify the instrument
./target/release/audio-direct \
  --model "$MODEL" --audio "$AUDIO" \
  --prompt "What instrument do you hear?" \
  --tokens 40

# General description
./target/release/audio-direct \
  --model "$MODEL" --audio "$AUDIO" \
  --prompt "Describe what you hear in this audio." \
  --tokens 80

# Yes/no question
./target/release/audio-direct \
  --model "$MODEL" --audio "$AUDIO" \
  --prompt "Is there a piano in this recording?" \
  --tokens 20
```

---

## 5. CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to `.cellmd` model file |
| `--audio` | *(required)* | Path to `.wav` audio file (must be 16 kHz mono PCM-16) |
| `--prompt` | `"Describe what you hear in this audio."` | User question about the audio |
| `--tokens` | `80` | Maximum new tokens to generate |
| `--backend` | `cpu` | Compute backend: `cpu` or `metal` (Apple Silicon GPU) |

---

## 6. Debug / diagnostic environment variables

These are read at runtime — no rebuild required:

| Variable | Effect |
|----------|--------|
| `CELLM_AUDIO_DEBUG=1` | Print token IDs, prompt structure, and layer-by-layer activation statistics |
| `CELLM_VLM_DEBUG_FEATURE_STATS=1` | Print audio feature stats (min/max/mean/std) after the encoder |
| `CELLM_GEN_DEBUG=1` | Print each generated token ID as it is sampled |
| `CELLM_VLM_ZERO_IMAGE_FEATURES=1` | Zero out audio embeddings — use to confirm audio is actually conditioning output |
| `CELLM_DUMP_AUDIO_TENSORS=1` | List all audio/embed tensor names and shapes from the model header |
| `CELLM_VLM_GEMMA4_BIDIR_IMAGE=false` | Disable bidirectional prefill for the audio block (may improve speed slightly) |

### Example: verify the audio is being used

Running with zeroed features should produce generic, non-audio-aware output. Running normally should produce audio-conditioned output. If they produce the same text, the audio pipeline is broken.

```bash
MODEL=models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd
AUDIO=/tmp/test_audio.wav

# With real audio features
./target/release/audio-direct \
  --model "$MODEL" --audio "$AUDIO" \
  --prompt "Describe what you hear." --tokens 40

# With zeroed audio (should produce different/generic output)
CELLM_VLM_ZERO_IMAGE_FEATURES=1 ./target/release/audio-direct \
  --model "$MODEL" --audio "$AUDIO" \
  --prompt "Describe what you hear." --tokens 40
```

### Example: check activation statistics

```bash
CELLM_AUDIO_DEBUG=1 ./target/release/audio-direct \
  --model models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd \
  --audio /tmp/test_audio.wav \
  --prompt "x" --tokens 1 \
  2>&1 | grep AUDIO_STATS
```

---

## 7. Audio format requirements

| Property | Required value |
|----------|---------------|
| Container | WAV (RIFF) |
| Encoding | PCM 16-bit signed |
| Sample rate | 16000 Hz exactly |
| Channels | Mono (1 channel) |

The mel spectrogram pipeline uses:
- FFT length: 512
- Frame length: 320 samples (20 ms)
- Hop length: 160 samples (10 ms)
- Mel bins: 128
- Frequency range: 0–8000 Hz

---

## 8. Architecture notes

The audio pipeline stages are:

1. **Mel spectrogram** — WAV → log-mel features `[T, 128]`
2. **Subsampling** — two stride-2 Conv2d layers reduce time by 4×
3. **Conformer encoder** — 12 layers (FFN1 → Attention → LightConv1d → FFN2 → LayerNorm), hidden size 1024
4. **Output projection** — 1024 → 1536 (text hidden size)
5. **Embedder** — RMSNorm (no scale) + Linear 1536→1536
6. **Text decoder** — Gemma 4 E2B with audio soft tokens inserted at `<|audio|>` positions

The encoder runs ~15–16 s on CPU for a 2-second clip (no Metal acceleration yet for the conformer).
