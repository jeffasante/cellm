#!/usr/bin/env python3
"""
Convert LiquidAI/LFM2.5-Embedding-350M (Lfm2BidirectionalModel) to .cellm format.

Differences from the causal LFM2 converter (convert_lfm_hf.py):
  * Checkpoint tensor names have no ``model.`` prefix; the runner expects one.
  * ``config.intermediate_size`` is a *pre-adjustment* value
    (``block_auto_adjust_ff_dim: true``), so the real FF dim must be read off
    the ``feed_forward.w1.weight`` shape instead.
  * ``conv.conv.weight`` ships as ``[dim, 1, k]``; the runner indexes it as a
    flat ``[dim, k]``.
  * The header is tagged ``bidirectional: true`` so the runner selects the
    non-causal encoder path.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_lfm_hf import write_cellm  # noqa: E402


def convert(input_dir: Path, output_path: Path):
    with open(input_dir / "config.json") as f:
        config = json.load(f)

    arches = config.get("architectures") or []
    if "Lfm2BidirectionalModel" not in arches:
        raise ValueError(
            f"expected Lfm2BidirectionalModel in architectures, got {arches}"
        )

    safetensors_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {input_dir}")

    tensors_bytes = {}
    tensors_shape = {}
    for st_file in safetensors_files:
        print(f"Loading {st_file.name}...")
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if tensor.dtype in (torch.bfloat16, torch.float32):
                    tensor = tensor.to(torch.float16)
                elif tensor.dtype != torch.float16:
                    raise ValueError(f"Unsupported dtype {tensor.dtype} for {name}")

                # Depthwise conv ships as [dim, 1, k]; the runner reads [dim, k].
                if name.endswith("conv.conv.weight") and tensor.dim() == 3:
                    tensor = tensor.squeeze(1)

                out_name = name if name.startswith("model.") else f"model.{name}"
                tensors_bytes[out_name] = tensor.contiguous().numpy().tobytes()
                tensors_shape[out_name] = list(tensor.shape)

    total_bytes = sum(len(b) for b in tensors_bytes.values())
    print(f"Loaded {len(tensors_bytes)} tensors, {total_bytes / 1024 / 1024:.1f} MB in f16")

    # block_auto_adjust_ff_dim rewrites intermediate_size at build time, so the
    # config value is not what the checkpoint actually holds. Read the shape.
    w1_shape = tensors_shape.get("model.layers.0.feed_forward.w1.weight")
    if w1_shape is None:
        raise ValueError("missing model.layers.0.feed_forward.w1.weight")
    intermediate_size = w1_shape[0]
    if intermediate_size != config.get("intermediate_size"):
        print(
            f"intermediate_size: using {intermediate_size} from w1 shape "
            f"(config said {config.get('intermediate_size')})"
        )

    k_shape = tensors_shape.get("model.layers.2.self_attn.k_proj.weight")
    num_kv_heads = config.get("num_key_value_heads", 8)
    head_dim = config.get("head_dim") or (k_shape[0] // num_kv_heads)

    rope_theta = config.get("rope_parameters", {}).get(
        "rope_theta", config.get("rope_theta", 1000000.0)
    )

    header = {
        "model_type": "lfm2",
        "source_model_type": config.get("model_type", "lfm2"),
        "source_safetensors_format": "pt",
        "text_tensor_prefix": "model",
        "vocab_size": config.get("vocab_size", 65536),
        "hidden_dim": config.get("hidden_size", 1024),
        "intermediate_size": intermediate_size,
        "num_layers": config.get("num_hidden_layers", 16),
        "num_heads": config.get("num_attention_heads", 16),
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "rms_norm_eps": config.get("norm_eps", 1e-5),
        "rope_theta": rope_theta,
        "bos_token_id": config.get("bos_token_id", 1),
        "eos_token_id": config.get("eos_token_id", 7),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "tie_word_embeddings": config.get("tie_embedding", True),
        "source_torch_dtype": "bfloat16",
        "source_architectures": arches,
        "source_text_config": {
            **config,
            "intermediate_size": intermediate_size,
            # Consumed by LfmRunner to select the non-causal encoder path.
            "bidirectional": True,
            "pooling_mode": "cls",
            "max_seq_length": 512,
            "query_prompt": "query: ",
            "document_prompt": "document: ",
        },
    }

    print(f"Writing to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_cellm(output_path, header, tensors_bytes, tensors_shape)
    print(f"Done! Output size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: convert_lfm_embedding_hf.py <input_dir> <output.cellm>")
        sys.exit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
