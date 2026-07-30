#!/usr/bin/env python3
"""
Quantize an LFM2 bidirectional embedding .cellm model.

Two modes, both matching formats the Rust `LfmRunner` already decodes:

  int8  per-row symmetric  -> `<name>` (i8) + `<name>.qscale` (f16)
  int4  MLX-style affine   -> `<name>` (u32) + `<name>.scales`/`.biases` (f32)

Embedding-model specific choices (differ from the generative quantizer):

  * `model.embed_tokens.weight` is quantized. It is 67% of the file and the
    encoder only ever gathers rows from it, so error stays local.
  * `conv.conv.weight` (the depthwise kernel, 3 floats per channel) stays f16.
    It is ~50 KB total and per-row symmetric scaling over 3 elements is
    numerically poor.
  * All `*norm*` gains stay f16.

Usage:
  quantize_lfm_embedding.py <input.cellm> <output.cellm> [--mode int8|int4]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_lfm_cellm import read_cellm, write_cellm, quantize_weight_2d  # noqa: E402


def quantize_int8_symmetric(weight_f16: np.ndarray):
    """Per-row symmetric int8. Returns (q_i8 [out,in], scales_f16 [out])."""
    w = weight_f16.astype(np.float32)
    max_abs = np.abs(w).max(axis=1)
    scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    q = np.round(w / scales[:, None]).clip(-127, 127).astype(np.int8)
    # Round-trip the scale through f16 so the runner dequantizes with the exact
    # value we quantized against.
    return q, scales.astype(np.float16)


def should_quantize(name: str, shape: list) -> bool:
    if len(shape) != 2 or not name.endswith(".weight"):
        return False
    if "norm" in name:
        return False
    if name.endswith("conv.conv.weight"):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mode", choices=["int8", "int4"], default="int8")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Reading {input_path}...")
    header, tensors = read_cellm(input_path)
    print(f"Found {len(tensors)} tensors, model_type={header.get('model_type')}")

    new_tensors = {}
    header["_shapes"] = {}
    header["_dtypes"] = {}

    n_quant = 0
    n_keep = 0
    src_bytes = 0
    dst_bytes = 0

    for name in sorted(tensors.keys()):
        info = tensors[name]
        shape = info["shape"]

        if not should_quantize(name, shape):
            new_tensors[name] = info["data"]
            header["_shapes"][name] = shape
            header["_dtypes"][name] = info["dtype"]
            n_keep += 1
            continue

        if info["dtype"] != "f16":
            raise SystemExit(f"{name}: expected f16 source, got {info['dtype']}")

        arr = np.frombuffer(info["data"], dtype=np.float16).reshape(shape)
        before = len(info["data"])

        if args.mode == "int8":
            q, scales = quantize_int8_symmetric(arr)
            new_tensors[name] = q.tobytes()
            new_tensors[f"{name}.qscale"] = scales.tobytes()
            header["_shapes"][name] = list(shape)
            header["_dtypes"][name] = "i8"
            header["_shapes"][f"{name}.qscale"] = [shape[0]]
            header["_dtypes"][f"{name}.qscale"] = "f16"
            after = len(new_tensors[name]) + len(new_tensors[f"{name}.qscale"])
        else:
            packed, scales, biases = quantize_weight_2d(arr, group_size=64)
            base = name[: -len(".weight")]
            new_tensors[name] = packed.tobytes()
            new_tensors[f"{base}.scales"] = scales.tobytes()
            new_tensors[f"{base}.biases"] = biases.tobytes()
            header["_shapes"][name] = list(packed.shape)
            header["_dtypes"][name] = "u32"
            header["_shapes"][f"{base}.scales"] = list(scales.shape)
            header["_dtypes"][f"{base}.scales"] = "f32"
            header["_shapes"][f"{base}.biases"] = list(biases.shape)
            header["_dtypes"][f"{base}.biases"] = "f32"
            after = sum(
                len(new_tensors[k])
                for k in (name, f"{base}.scales", f"{base}.biases")
            )

        src_bytes += before
        dst_bytes += after
        n_quant += 1
        print(f"  Q {name:58s} {str(shape):18s} {after / before * 100:5.1f}%")

    print(f"\nQuantized {n_quant} tensors to {args.mode}, kept {n_keep} as-is")
    print(
        f"quantized portion: {src_bytes / 1e6:.1f} MB -> {dst_bytes / 1e6:.1f} MB "
        f"({dst_bytes / src_bytes * 100:.1f}%)"
    )

    print(f"\nWriting {output_path}...")
    write_cellm(output_path, header, new_tensors)
    out_size = output_path.stat().st_size
    in_size = input_path.stat().st_size
    print(
        f"Done! {in_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB "
        f"({out_size / in_size * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
