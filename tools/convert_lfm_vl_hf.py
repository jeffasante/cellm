#!/usr/bin/env python3
"""
Convert an LFM2.5-VL HuggingFace model (bfloat16 safetensors) to .cellm format.

LFM2-VL = SigLIP2 NaFlex vision tower + 2-layer MLP projector + LFM2 text model.

Tensor name remapping:
  model.language_model.*      -> model.*        (matches cellm-model/src/lfm.rs)
  model.vision_tower.*        -> kept as-is
  model.multi_modal_projector.* -> kept as-is
"""

import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

LM_SRC_PREFIX = "model.language_model."
LM_DST_PREFIX = "model."
VISION_PREFIX = "model.vision_tower.vision_model."
PROJECTOR_PREFIX = "model.multi_modal_projector."


def write_cellm(output_path: Path, header: dict, tensors_bytes: dict, tensors_shape: dict,
                tensors_dtype: dict):
    """Write a .cellm file: magic + version + header_len + JSON header + 64B-aligned tensors."""
    names = sorted(tensors_bytes.keys())

    tensor_index = [
        {
            "name": name,
            "offset_bytes": 0,
            "nbytes": len(tensors_bytes[name]),
            "shape": tensors_shape[name],
            "dtype": tensors_dtype.get(name, "f16"),
        }
        for name in names
    ]

    # Two passes: header length depends on offsets, offsets depend on header length.
    # The offsets are fixed-width enough after one pass that a second settles it.
    for _ in range(2):
        header["tensors"] = tensor_index
        header_json = json.dumps(header).encode("utf-8")
        data_start = (5 + 1 + 4 + len(header_json) + 63) & ~63
        offset = data_start
        for item in tensor_index:
            offset = (offset + 63) & ~63
            item["offset_bytes"] = offset
            offset += item["nbytes"]

    header["tensors"] = tensor_index
    header_json = json.dumps(header).encode("utf-8")
    data_start = (5 + 1 + 4 + len(header_json) + 63) & ~63

    with open(output_path, "wb") as f:
        f.write(b"CELLM")
        f.write(struct.pack("<B", 1))
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)

        pos = 5 + 1 + 4 + len(header_json)
        if pos < data_start:
            f.write(b"\x00" * (data_start - pos))

        for item in tensor_index:
            pos = f.tell()
            if pos < item["offset_bytes"]:
                f.write(b"\x00" * (item["offset_bytes"] - pos))
            elif pos > item["offset_bytes"]:
                raise RuntimeError(
                    f"offset overrun for {item['name']}: at {pos}, expected {item['offset_bytes']}"
                )
            f.write(tensors_bytes[item["name"]])


def remap_name(name: str) -> str:
    if name.startswith(LM_SRC_PREFIX):
        return LM_DST_PREFIX + name[len(LM_SRC_PREFIX):]
    return name


def quantize_i8_per_row(tensor: torch.Tensor):
    """Per-row symmetric int8, matching what lfm.rs and vlm.rs expect.

    Returns (int8 weights, f16 per-row scales) where `w[i][j] * scale[i]`
    reconstructs the original value.
    """
    w = tensor.to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = (amax / 127.0).clamp(min=1e-12)
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(1).to(torch.float16)


INT4_GROUP_SIZE = 64


def quantize_i4_grouped(tensor: torch.Tensor, group_size: int = INT4_GROUP_SIZE):
    """Group-wise symmetric int4, two weights per byte, biased by +8.

    Element `2i` goes in the low nibble and `2i+1` in the high nibble, which is
    the layout `gemv_i4_w4a8` unpacks with a mask and a shift. Values are stored
    as `q + 8` so the nibble range 0..15 covers `-7..=7`; the kernel removes that
    bias algebraically.

    A single scale per row leaves 15 levels to span a whole 1024-wide row, which
    produced visibly broken text. One scale per 64 weights keeps the step size
    local and costs 2 bytes per 32 bytes of weights.

    Returns (packed bytes, f16 scales shaped [rows, groups_per_row]).
    """
    w = tensor.to(torch.float32)
    rows, cols = w.shape
    groups = cols // group_size
    g = w.view(rows, groups, group_size)

    amax = g.abs().amax(dim=2, keepdim=True)
    scale = (amax / 7.0).clamp(min=1e-12)
    q = torch.round(g / scale).clamp(-7, 7).to(torch.int8).view(rows, cols)

    nib = (q + 8).to(torch.uint8)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return packed.contiguous(), scale.view(rows, groups).to(torch.float16)


def int4_shape_ok(shape: list, group_size: int = INT4_GROUP_SIZE) -> bool:
    """The kernel needs whole groups of whole 16-byte SIMD loads."""
    return len(shape) == 2 and shape[1] % group_size == 0


def int4_eligible(name: str) -> bool:
    """Text-side weights only.

    `tensor_to_f32` in vlm.rs decodes f16/f32/bf16/i8 and nothing else, so the
    vision tower and projector would fail outright on 4-bit weights. They are
    also the accuracy-critical part of the pipeline, where a 15-level per-row
    grid compounds across 988 patch tokens.
    """
    return not name.startswith(VISION_PREFIX) and not name.startswith(PROJECTOR_PREFIX)


def should_quantize(name: str, shape: list, quantize_vision: bool) -> bool:
    """Pick the 2D projection weights that dominate file size.

    Norms, biases, and the depthwise conv kernel are tiny and precision-critical,
    so they stay f16. The vision tower is only ~180 MB of 856 MB and it runs a
    full bidirectional encoder where int8 error compounds across 988 tokens, so
    it is opt-in.
    """
    if len(shape) != 2 or not name.endswith(".weight"):
        return False
    if "norm" in name or "layernorm" in name:
        return False
    # Rows must be quantizable per-row and big enough for the scale to pay off.
    if min(shape) < 64:
        return False

    if name.startswith(VISION_PREFIX):
        return quantize_vision and (".self_attn." in name or ".mlp." in name)
    if name.startswith(PROJECTOR_PREFIX):
        return quantize_vision
    # Text model: all projections plus the tied embedding table.
    if name == "model.embed_tokens.weight":
        return True
    return name.startswith("model.layers.") and (
        ".self_attn." in name or ".feed_forward." in name or ".conv.in_proj." in name
        or ".conv.out_proj." in name
    )


def convert(input_dir: Path, output_path: Path, quantize: bool = False,
            quantize_vision: bool = False, quantize_int4: bool = False):
    with open(input_dir / "config.json") as f:
        config = json.load(f)

    text_config = config.get("text_config", {})
    vision_config = config.get("vision_config", {})

    print(f"Model type: {config.get('model_type')}")
    print(f"Text hidden: {text_config.get('hidden_size')}, layers: {text_config.get('num_hidden_layers')}")
    print(f"Vision hidden: {vision_config.get('hidden_size')}, layers: {vision_config.get('num_hidden_layers')}")

    safetensors_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {input_dir}")

    tensors_bytes = {}
    tensors_shape = {}
    tensors_dtype = {}
    counts = {"lm": 0, "vision": 0, "projector": 0, "other": 0}
    num_i8 = 0
    num_i4 = 0

    for st_file in safetensors_files:
        print(f"Loading {st_file.name}...")
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if tensor.dtype not in (torch.bfloat16, torch.float32, torch.float16):
                    raise ValueError(f"Unsupported dtype {tensor.dtype} for {name}")

                out_name = remap_name(name)
                if name.startswith(LM_SRC_PREFIX):
                    counts["lm"] += 1
                elif name.startswith("model.vision_tower."):
                    counts["vision"] += 1
                elif name.startswith(PROJECTOR_PREFIX):
                    counts["projector"] += 1
                else:
                    counts["other"] += 1
                    print(f"  [unclassified] {name}: {list(tensor.shape)}")

                shape = list(tensor.shape)
                if quantize and should_quantize(out_name, shape, quantize_vision):
                    use_i4 = (quantize_int4 and int4_eligible(out_name)
                              and int4_shape_ok(shape))
                    if use_i4:
                        q, scale = quantize_i4_grouped(tensor)
                        num_i4 += 1
                    else:
                        q, scale = quantize_i8_per_row(tensor)
                        num_i8 += 1
                    tensors_bytes[out_name] = q.contiguous().numpy().tobytes()
                    tensors_shape[out_name] = shape
                    tensors_dtype[out_name] = "i4" if use_i4 else "i8"
                    # Runtime looks up the sidecar as "{weight_name}.qscale".
                    scale_name = f"{out_name}.qscale"
                    tensors_bytes[scale_name] = scale.contiguous().numpy().tobytes()
                    tensors_shape[scale_name] = list(scale.shape)
                    tensors_dtype[scale_name] = "f16"
                else:
                    tensors_bytes[out_name] = tensor.to(torch.float16).contiguous().numpy().tobytes()
                    tensors_shape[out_name] = shape
                    tensors_dtype[out_name] = "f16"

    total_bytes = sum(len(b) for b in tensors_bytes.values())
    print(
        f"Loaded {len(tensors_bytes)} tensors "
        f"(lm={counts['lm']} vision={counts['vision']} projector={counts['projector']} "
        f"other={counts['other']}), {total_bytes / 1024 / 1024:.1f} MB "
        f"({num_i4} int4, {num_i8} int8, rest f16)"
    )

    # --- Sanity checks on the tensors we depend on downstream ---
    required = [
        f"{VISION_PREFIX}embeddings.patch_embedding.weight",
        f"{VISION_PREFIX}embeddings.position_embedding.weight",
        f"{VISION_PREFIX}post_layernorm.weight",
        f"{PROJECTOR_PREFIX}linear_1.weight",
        f"{PROJECTOR_PREFIX}linear_2.weight",
        "model.embed_tokens.weight",
        "model.embedding_norm.weight",
    ]
    missing = [n for n in required if n not in tensors_shape]
    if missing:
        raise ValueError(f"Missing required tensors after remap: {missing}")

    # --- Derive dims from the actual weights, not the config ---
    # block_auto_adjust_ff_dim means config intermediate_size is not the real FF dim.
    intermediate_size = None
    for name, shape in tensors_shape.items():
        if name.endswith("feed_forward.w1.weight"):
            intermediate_size = shape[0]
            break
    if intermediate_size is None:
        intermediate_size = text_config.get("intermediate_size", 6656)
    print(f"Inferred intermediate_size={intermediate_size} (config says {text_config.get('intermediate_size')})")

    head_dim = text_config.get("head_dim")
    if head_dim is None:
        for name, shape in tensors_shape.items():
            if name.endswith("self_attn.k_proj.weight"):
                head_dim = shape[0] // text_config.get("num_key_value_heads", 8)
                print(f"Inferred head_dim={head_dim} from {name}")
                break

    rope_theta = text_config.get("rope_parameters", {}).get("rope_theta", 1000000.0)

    header = {
        "model_type": "lfm2",
        "source_model_type": config.get("model_type", "lfm2_vl"),
        "source_safetensors_format": "pt",
        "text_tensor_prefix": "model",
        "vision_tensor_prefix": VISION_PREFIX,
        "projector_tensor_prefix": PROJECTOR_PREFIX,
        "vocab_size": text_config.get("vocab_size", 65536),
        "hidden_dim": text_config.get("hidden_size", 1024),
        "intermediate_size": intermediate_size,
        "num_layers": text_config.get("num_hidden_layers", 16),
        "num_heads": text_config.get("num_attention_heads", 16),
        "num_kv_heads": text_config.get("num_key_value_heads", 8),
        "head_dim": head_dim,
        "rms_norm_eps": text_config.get("norm_eps", 1e-5),
        "rope_theta": rope_theta,
        "bos_token_id": text_config.get("bos_token_id"),
        "eos_token_id": text_config.get("eos_token_id", 7),
        "max_position_embeddings": text_config.get("max_position_embeddings"),
        "tie_word_embeddings": text_config.get("tie_word_embeddings", True),
        "source_torch_dtype": "bfloat16",
        "quantization": (
            ("int4_symmetric_per_row_text" if quantize_int4 else "int8_symmetric_per_row")
            if quantize
            else None
        ),
        "source_architectures": config.get("architectures"),
        "source_text_config": text_config,  # carries layer_types + conv_L_cache
        "source_vision_config": vision_config,
        "source_projector_config": {
            "projector_hidden_size": config.get("projector_hidden_size", 2048),
            "projector_hidden_act": config.get("projector_hidden_act", "gelu"),
            "projector_bias": config.get("projector_bias", True),
            "projector_use_layernorm": config.get("projector_use_layernorm", False),
            "downsample_factor": config.get("downsample_factor", 2),
            "encoder_patch_size": config.get("encoder_patch_size", 16),
            "image_token_id": config.get("image_token_id", 396),
            "tile_size": config.get("tile_size", 512),
            "max_image_tokens": config.get("max_image_tokens", 256),
            "min_image_tokens": config.get("min_image_tokens", 64),
        },
    }

    print(f"Writing to {output_path}...")
    write_cellm(output_path, header, tensors_bytes, tensors_shape, tensors_dtype)
    print(f"Done! Output size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert LFM2.5-VL HF weights to .cellm")
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--quantize-int8", action="store_true",
                    help="per-row symmetric int8 for the text model's 2D weights")
    ap.add_argument("--quantize-vision", action="store_true",
                    help="also quantize the vision tower and projector (more size, more drift)")
    ap.add_argument("--quantize-int4", action="store_true",
                    help="per-row symmetric int4 for the text model; vision stays int8/f16")
    args = ap.parse_args()

    if not args.input_dir.exists():
        ap.error(f"Input directory {args.input_dir} does not exist")

    convert(args.input_dir, args.output,
            quantize=args.quantize_int8 or args.quantize_vision or args.quantize_int4,
            quantize_vision=args.quantize_vision,
            quantize_int4=args.quantize_int4)
