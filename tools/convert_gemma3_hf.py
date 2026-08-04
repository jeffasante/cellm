#!/usr/bin/env python3
"""Convert a Gemma3 text HuggingFace checkpoint (bf16 safetensors) to .cellm.

Written for google/functiongemma-270m-it but applies to any `gemma3_text`
model, including gemma-3-1b-it.

    python3 tools/convert_gemma3_hf.py \
        models/hf/functiongemma-270m-it \
        models/functiongemma-270m-it-f16.cellm
"""

import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

ALIGN = 64


def write_cellm(output_path: Path, header: dict, tensors_bytes: dict,
                tensors_shape: dict, tensors_dtype: dict):
    names = sorted(tensors_bytes)

    def layout(start: int):
        offsets, cur = {}, start
        for name in names:
            cur = (cur + ALIGN - 1) & ~(ALIGN - 1)
            offsets[name] = cur
            cur += len(tensors_bytes[name])
        return offsets

    index = [{"name": n, "offset_bytes": 0, "nbytes": len(tensors_bytes[n]),
              "shape": tensors_shape[n], "dtype": tensors_dtype[n]} for n in names]
    header["tensors"] = index

    # Offsets are absolute, so the header length feeds back into them; two
    # passes converge because the json length is fixed once the digits are.
    for _ in range(3):
        header_len = len(json.dumps(header).encode())
        data_start = (5 + 1 + 4 + header_len + ALIGN - 1) & ~(ALIGN - 1)
        offsets = layout(data_start)
        for item in index:
            item["offset_bytes"] = offsets[item["name"]]

    header_json = json.dumps(header).encode()
    header_len = len(header_json)
    data_start = (5 + 1 + 4 + header_len + ALIGN - 1) & ~(ALIGN - 1)

    with open(output_path, "wb") as f:
        f.write(b"CELLM")
        f.write(struct.pack("<B", 1))
        f.write(struct.pack("<I", header_len))
        f.write(header_json)
        f.write(b"\x00" * (data_start - f.tell()))
        for name in names:
            pad = ((f.tell() + ALIGN - 1) & ~(ALIGN - 1)) - f.tell()
            f.write(b"\x00" * pad)
            assert f.tell() == offsets[name], f"{name} offset drift"
            f.write(tensors_bytes[name])


def quantize_int8_rows(t: torch.Tensor):
    """Per-row symmetric int8. Returns (i8 bytes, f16 scale bytes)."""
    w = t.to(torch.float32)
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q.numpy().tobytes(), scale.squeeze(1).to(torch.float16).numpy().tobytes()


def quantize_int4_rows(t: torch.Tensor, group_size: int = 0):
    """Symmetric int4, two weights per byte, low nibble first.

    Runtime reads this as `(nibble as i8 - 8)`, so the stored nibble is the
    signed level biased by +8. `group_size=0` means one scale per row; a
    group size that divides the row emits `cols/group_size` scales per row.
    """
    w = t.to(torch.float32)
    rows, cols = w.shape
    g = cols if group_size in (0, None) or cols % group_size else group_size
    wg = w.reshape(rows, cols // g, g)
    scale = wg.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (torch.round(wg / scale).clamp(-7, 7).to(torch.int32) + 8).reshape(rows, cols)

    if cols % 2:
        q = torch.cat([q, torch.full((rows, 1), 8, dtype=q.dtype)], dim=1)
    q = q.reshape(rows, -1, 2)
    packed = (q[:, :, 0] | (q[:, :, 1] << 4)).to(torch.uint8)
    return packed.numpy().tobytes(), scale.reshape(rows, -1).to(torch.float16).numpy().tobytes()


# Runtime codebook for i2; fixed levels, one f16 scale per row.
I2_LEVELS = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32)


def quantize_int2_rows(t: torch.Tensor, group_size: int = 0, refine_iters: int = 12):
    """Int2 against the fixed {-1.5,-0.5,0.5,1.5} codebook, 4 weights per byte.

    Only the scale is free, so it is fitted by alternating nearest-level
    assignment with a least-squares refit rather than taken as amax/1.5,
    which would waste most of the range on groups with outliers.
    """
    w = t.to(torch.float32)
    rows, cols = w.shape
    levels = I2_LEVELS.to(w.dtype)
    g = cols if group_size in (0, None) or cols % group_size else group_size
    wg = w.reshape(rows, cols // g, g)

    scale = wg.abs().amax(dim=-1, keepdim=True) / 1.5
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    for _ in range(refine_iters):
        idx = torch.argmin((wg.unsqueeze(-1) - scale.unsqueeze(-1) * levels).abs(), dim=-1)
        lv = levels[idx]
        denom = (lv * lv).sum(dim=-1, keepdim=True)
        new_scale = torch.where(denom == 0, scale, (wg * lv).sum(dim=-1, keepdim=True) / denom)
        new_scale = torch.where(new_scale <= 0, scale, new_scale)
        if torch.allclose(new_scale, scale, rtol=1e-5, atol=0):
            scale = new_scale
            break
        scale = new_scale
    idx = torch.argmin((wg.unsqueeze(-1) - scale.unsqueeze(-1) * levels).abs(), dim=-1)
    idx = idx.reshape(rows, cols)

    pad = (-cols) % 4
    if pad:
        idx = torch.cat([idx, torch.full((rows, pad), 2, dtype=idx.dtype)], dim=1)
    q = idx.reshape(rows, -1, 4).to(torch.int32)
    packed = (q[:, :, 0] | (q[:, :, 1] << 2) | (q[:, :, 2] << 4) | (q[:, :, 3] << 6)).to(torch.uint8)
    return packed.numpy().tobytes(), scale.reshape(rows, -1).to(torch.float16).numpy().tobytes()


QUANTIZERS = {
    "int8": (quantize_int8_rows, "i8"),
    "int4": (quantize_int4_rows, "i4"),
    "int2": (quantize_int2_rows, "i2"),
}


def convert(input_dir: Path, output_path: Path, quant: str, quant_embed, group_size: int = 0):
    config = json.loads((input_dir / "config.json").read_text())
    if not config.get("model_type", "").startswith("gemma3"):
        raise SystemExit(f"expected a gemma3 model, got {config.get('model_type')!r}")

    files = sorted(input_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors in {input_dir}")

    tensors_bytes, tensors_shape, tensors_dtype = {}, {}, {}
    n_quant = 0
    for st in files:
        with safe_open(str(st), framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                # Norms stay f16 (tiny, and scale-sensitive). The embedding
                # table is 63% of this model, so quantizing it is where the
                # savings actually are.
                is_embed = "embed_tokens" in name
                mode = quant_embed if is_embed else quant
                quantizable = mode in QUANTIZERS and t.ndim == 2 and "norm" not in name
                if quantizable:
                    fn, dt = QUANTIZERS[mode]
                    qb, sb = fn(t) if mode == "int8" else fn(t, group_size)
                    tensors_bytes[name] = qb
                    tensors_shape[name] = list(t.shape)
                    tensors_dtype[name] = dt
                    n_scales = len(sb) // 2
                    tensors_bytes[name + ".qscale"] = sb
                    tensors_shape[name + ".qscale"] = (
                        [t.shape[0]] if n_scales == t.shape[0]
                        else [t.shape[0], n_scales // t.shape[0]]
                    )
                    tensors_dtype[name + ".qscale"] = "f16"
                    n_quant += 1
                else:
                    tensors_bytes[name] = t.to(torch.float16).numpy().tobytes()
                    tensors_shape[name] = list(t.shape)
                    tensors_dtype[name] = "f16"

    total = sum(len(b) for b in tensors_bytes.values())
    print(f"tensors={len(tensors_bytes)} quantized={n_quant} payload={total / 1e9:.3f} GB")

    eos = config.get("eos_token_id")
    if isinstance(eos, list):
        eos = eos[0]

    header = {
        "model_type": "gemma3",
        "source_model_type": config.get("model_type"),
        "source_safetensors_format": "pt",
        "text_tensor_prefix": "model",
        "vocab_size": config["vocab_size"],
        "hidden_dim": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_layers": config["num_hidden_layers"],
        "num_heads": config["num_attention_heads"],
        "num_kv_heads": config["num_key_value_heads"],
        "head_dim": config.get("head_dim"),
        "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
        "rope_theta": config.get("rope_theta", 1_000_000.0),
        "bos_token_id": config.get("bos_token_id"),
        "eos_token_id": eos,
        "max_position_embeddings": config.get("max_position_embeddings"),
        "tie_word_embeddings": config.get("tie_word_embeddings", True),
        "source_torch_dtype": config.get("torch_dtype"),
        "source_architectures": config.get("architectures"),
        # Carries layer_types, sliding_window and rope_local_base_freq, which
        # the runner needs to place the full-attention layers correctly.
        "source_text_config": config,
    }

    write_cellm(output_path, header, tensors_bytes, tensors_shape, tensors_dtype)
    print(f"wrote {output_path} ({output_path.stat().st_size / 1e9:.3f} GB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--quant", choices=["none", "int8", "int4", "int2"], default="none")
    # Bare --quant-embed keeps its old meaning (int8) so earlier commands still work.
    ap.add_argument("--quant-embed", nargs="?", const="int8", default="none",
                   choices=["none", "int8", "int4", "int2"],
                   help="recipe for the embedding table (63%% of this model)")
    ap.add_argument("--group-size", type=int, default=0,
                   help="scales per group along the input dim; 0 = one scale per row")
    a = ap.parse_args()
    convert(a.input_dir, a.output, a.quant, a.quant_embed, a.group_size)
