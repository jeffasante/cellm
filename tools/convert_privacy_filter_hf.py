#!/usr/bin/env python3
"""Convert openai/privacy-filter (HF safetensors) to .cellm.

The MoE expert stacks dominate this checkpoint (~1.26B of ~1.4B params), so the
`--quant` mode is applied to experts only; attention/router/embedding/score stay
f16 because quantizing them costs little size but a lot of accuracy.

Usage:
    python tools/convert_privacy_filter_hf.py models/hf/privacy-filter out.cellm
    python tools/convert_privacy_filter_hf.py <in> <out> --quant int8
    python tools/convert_privacy_filter_hf.py <in> <out> --quant int4
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

EXPECTED_ARCH = "OpenAIPrivacyFilterForTokenClassification"


def write_cellm(output_path, header, tensors, shapes, dtypes):
    """Write a .cellm file.

    Offsets are absolute, so data_start depends on the header length, which in
    turn depends on the offsets. Iterate to a fixed point instead of assuming
    one recalculation converges -- otherwise the header grows after offsets are
    assigned and the first tensor ends up before data_start.
    """
    names = sorted(tensors)
    index = [
        {
            "name": n,
            "offset_bytes": 0,
            "nbytes": len(tensors[n]),
            "shape": shapes[n],
            "dtype": dtypes[n],
        }
        for n in names
    ]
    header["tensors"] = index

    data_start = 0
    for _ in range(64):
        header_len = len(json.dumps(header).encode())
        new_start = (5 + 1 + 4 + header_len + 63) & ~63
        offset = new_start
        for item in index:
            offset = (offset + 63) & ~63
            item["offset_bytes"] = offset
            offset += item["nbytes"]
        if new_start == data_start:
            break
        data_start = new_start
    else:
        raise SystemExit("header length did not converge")

    header_json = json.dumps(header).encode()
    assert (5 + 1 + 4 + len(header_json) + 63) & ~63 == data_start

    with open(output_path, "wb") as f:
        f.write(b"CELLM")
        f.write(struct.pack("<B", 1))
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        f.write(b"\x00" * (data_start - f.tell()))
        for item in index:
            f.write(b"\x00" * (item["offset_bytes"] - f.tell()))
            f.write(tensors[item["name"]])


def quantize_int8_rows(w: np.ndarray):
    """Per-row symmetric int8. w is [rows, cols] f32."""
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax == 0.0, 1.0, amax / 127.0).astype(np.float32)
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale.astype(np.float16)


def quantize_int4_groups(w: np.ndarray, group_size: int = 64, half_params=True, bits: int = 4):
    """MLX-style affine quant, `32 // bits` values per u32. w is [rows, cols] f32.

    f16 scale/bias sidecars are the default: at group 32 they would otherwise be
    ~29% of the file, and f16 was measured to cost no additional missed spans.
    Quantization uses the rounded scale so the error is not compounded at load.
    """
    # Values never straddle a word; for bits=3 that wastes the top 2 bits.
    per_word = 32 // bits
    qmax = (1 << bits) - 1
    rows, cols = w.shape
    if cols % group_size != 0:
        raise ValueError(f"cols {cols} not divisible by group_size {group_size}")
    if cols % per_word:
        raise ValueError(f"cols {cols} not divisible by {per_word} values/word")
    g = w.reshape(rows, cols // group_size, group_size)
    g_min = g.min(axis=2)
    g_max = g.max(axis=2)
    span = g_max - g_min
    scales = np.where(span <= 0, 0.0, span / qmax).astype(np.float32)
    biases = g_min.astype(np.float32)
    if half_params:
        scales = scales.astype(np.float16).astype(np.float32)
        biases = biases.astype(np.float16).astype(np.float32)
    safe = np.where(scales == 0, 1.0, scales)
    q = np.rint((g - biases[:, :, None]) / safe[:, :, None])
    q = q.clip(0, qmax).astype(np.uint32).reshape(rows, cols)
    packed = np.zeros((rows, cols // per_word), dtype=np.uint32)
    for i in range(per_word):
        packed |= (q[:, i::per_word] & qmax) << (bits * i)
    out_dt = np.float16 if half_params else np.float32
    return packed, scales.astype(out_dt), biases.astype(out_dt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument(
        "--quant", choices=["none", "int8", "int4", "int3", "int2"], default="none"
    )
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument(
        "--quant-embedding",
        action="store_true",
        help="also store embed_tokens as int8 (saves ~128 MB; int4 here is "
        "measurably lossy, so only int8 is offered)",
    )
    ap.add_argument(
        "--f32-scales",
        action="store_true",
        help="store int4 scale/bias sidecars as f32 (+157 MB, no measured gain)",
    )
    args = ap.parse_args()

    # 0 means "not group-packed" (none/int8), which the header reports as null.
    quant_bits = {"int4": 4, "int3": 3, "int2": 2}.get(args.quant, 0)

    cfg = json.loads((args.input_dir / "config.json").read_text())
    if EXPECTED_ARCH not in cfg.get("architectures", []):
        raise SystemExit(f"expected {EXPECTED_ARCH}, got {cfg.get('architectures')}")

    st_path = args.input_dir / "model.safetensors"
    tensors: dict[str, bytes] = {}
    shapes: dict[str, list] = {}
    dtypes: dict[str, str] = {}

    def put(name, arr, dtype):
        tensors[name] = arr.tobytes()
        shapes[name] = list(arr.shape)
        dtypes[name] = dtype

    n_experts = cfg["num_local_experts"]

    with safe_open(st_path, framework="pt") as f:
        for name in f.keys():
            arr = f.get_tensor(name).float().numpy()

            is_expert = ".experts." in name and name.endswith(
                ("gate_up_proj", "down_proj")
            )

            if name.endswith("self_attn.sinks"):
                # HF marks sinks keep-in-fp32; they feed the softmax denominator.
                put(name, arr.astype(np.float32), "f32")
            elif name.endswith("embed_tokens.weight") and args.quant_embedding:
                q, s = quantize_int8_rows(arr)
                put(name, q, "i8")
                put(name.replace(".weight", ".scales"), s, "f16")
            elif is_expert and args.quant != "none":
                # [E, K, N] -> flatten each expert to 2D and quantize along K.
                # Store transposed so rows are output channels (matvec-friendly).
                for e in range(n_experts):
                    mat = arr[e].T.copy()  # [N, K]
                    ename = f"{name}.{e}"
                    if args.quant == "int8":
                        q, s = quantize_int8_rows(mat)
                        put(ename, q, "i8")
                        put(f"{ename}.scales", s, "f16")
                    else:
                        p, s, b = quantize_int4_groups(
                            mat, args.group_size, not args.f32_scales, quant_bits
                        )
                        sdt = "f32" if args.f32_scales else "f16"
                        put(ename, p, "u32")
                        put(f"{ename}.scales", s, sdt)
                        put(f"{ename}.biases", b, sdt)
            elif is_expert:
                for e in range(n_experts):
                    put(f"{name}.{e}", arr[e].T.copy().astype(np.float16), "f16")
            else:
                put(name, arr.astype(np.float16), "f16")

    rope = cfg["rope_parameters"]
    header = {
        "model_type": "openai_privacy_filter",
        "source_model_type": cfg["model_type"],
        "source_architectures": cfg["architectures"],
        "vocab_size": cfg["vocab_size"],
        "hidden_dim": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "num_layers": cfg["num_hidden_layers"],
        "num_heads": cfg["num_attention_heads"],
        "num_kv_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": rope["rope_theta"],
        # `_apply_rotary_emb` splits with x[..., ::2] / x[..., 1::2] and
        # re-interleaves via stack(-1).flatten(-2), so pairs are adjacent.
        "rope_interleaved": True,
        "eos_token_id": cfg["eos_token_id"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "rope_scaling_type": rope["rope_type"],
        "rope_scaling_factor": rope["factor"],
        "rope_scaling_original_max_position_embeddings": rope[
            "original_max_position_embeddings"
        ],
        "tie_word_embeddings": cfg["tie_word_embeddings"],
        "source_torch_dtype": cfg["dtype"],
        "n_routed_experts": cfg["num_local_experts"],
        "num_experts_per_tok": cfg["num_experts_per_tok"],
        "moe_intermediate_size": cfg["intermediate_size"],
        "source_text_config": {
            "sliding_window": cfg["sliding_window"],
            "attention_bias": cfg["attention_bias"],
            "id2label": cfg["id2label"],
            "num_labels": len(cfg["id2label"]),
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "beta_fast": rope["beta_fast"],
            "beta_slow": rope["beta_slow"],
            "quant": args.quant,
            "quant_bits": quant_bits,
            "quant_group_size": args.group_size if quant_bits else None,
            "quant_embedding": "int8" if args.quant_embedding else None,
            # The reader must dispatch on this: lfm.rs assumes f32 sidecars.
            "quant_scale_dtype": "f32" if args.f32_scales else "f16",
        },
        "_shapes": shapes,
        "_dtypes": dtypes,
    }

    write_cellm(args.output, header, tensors, shapes, dtypes)

    total = sum(len(v) for v in tensors.values())
    print(f"quant={args.quant}  tensors={len(tensors)}  payload={total/1e9:.3f} GB")
    print(f"wrote {args.output} ({args.output.stat().st_size/1e9:.3f} GB)")


if __name__ == "__main__":
    main()
