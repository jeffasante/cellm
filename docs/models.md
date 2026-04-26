# cellm Model Repository

All research models are hosted on the Hugging Face [jeffasante/cellm-models](https://huggingface.co/jeffasante/cellm-models) repository. These models are optimized for the `.cellm` format and can be used with the `infer` tool.

## Available Models

| Model | Variant | Size | Hugging Face Link |
| :--- | :--- | :--- | :--- |
| **Qwen 3.5 0.8B** | f16, int4, q1 | 1.6GB (f16) | [View on HF](https://huggingface.co/jeffasante/cellm-models/tree/main/qwen3.5-0.8b-v1) |
| **Qwen 3.0 0.6B** | f16, int8, int4 | 1.2GB (f16) | [View on HF](https://huggingface.co/jeffasante/cellm-models/tree/main/qwen3-0.6b-v1) |
| **LFM 2.5 350M** | 4-bit | 190MB | [View on HF](https://huggingface.co/jeffasante/cellm-models/tree/main/lfm2.5-350m-v1) |
| **Gemma 4 E2B** | int4 | 2.1GB | [View on HF](https://huggingface.co/jeffasante/cellm-models/tree/main/gemma-4-E2B-it-int4-aggr-v5) |
| **Gemma 3 1B** | int8, mixed-i4 | 1.1GB | [View on HF](https://huggingface.co/jeffasante/cellm-models/tree/main/gemma-3-1b-it-int8-v1) |
| **SmolVLM 256M** | f16 | 513MB | [View on HF](https://huggingface.co/jeffasante/cellm-models/tree/main/smolvlm-256m-instruct-f16-full) |
| **SmolLM 2 360M** | int8, q1 | 360MB | [View on HF](https://huggingface.co/jeffasante/cellm-models/tree/main/smollm2-360m-int8-v1) |
| **Bonsai 1.7B** | v2 | 1.7GB | [View on HF](https://huggingface.co/jeffasante/cellm-models/tree/main/Bonsai-1.7B_v2) |
| **Qwen 2.5 0.5B** | int8 | 500MB | [View on HF](https://huggingface.co/jeffasante/cellm-models/tree/main/qwen2.5-0.5b-int8-v1) |

## Usage

To run a model from the repository:

1. Download the `.cellm` model file and the `tokenizer.json`.
2. Execute inference using the `infer` tool:

```bash
./target/release/infer \
  --model path/to/model.cellm \
  --tokenizer path/to/tokenizer.json \
  --prompt "Hello, cellm!" \
  --backend cpu
```

> [!NOTE]
> For mobile deployment, ensure the model size fits within your device's memory constraints. int4 and int8 variants are recommended for devices with less than 8GB RAM.
