#!/usr/bin/env python3
"""
Quantize LFM2.5 .cellm model weights to int4 (affine, group_size=64)
using the MLX-style format that the LFM runner already supports.

Produces: u32 packed weights + f32 scales + f32 biases
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def quantize_weight_2d(weight_f16: np.ndarray, group_size: int = 64):
    """Quantize a 2D f16 weight [out_dim, in_dim] to int4 MLX format.

    Returns (packed_u32 [out,in/8], scales_f32 [out,groups], biases_f32 [out,groups]).
    """
    w = weight_f16.astype(np.float32)
    out_dim, in_dim = w.shape
    if in_dim % group_size != 0 or in_dim % 8 != 0:
        raise ValueError(f"in_dim {in_dim} must be a multiple of {group_size} and 8")
    n_groups = in_dim // group_size

    g = w.reshape(out_dim, n_groups, group_size)
    g_min = g.min(axis=2)
    g_max = g.max(axis=2)
    span = g_max - g_min
    # A constant group has no range to encode; store the value in the bias and
    # a zero scale so dequant reproduces it exactly.
    degenerate = span == 0
    scales = np.where(degenerate, 0.0, span / 15.0).astype(np.float32)
    biases = g_min.astype(np.float32)

    safe_scale = np.where(degenerate, 1.0, scales)
    q = np.rint((g - biases[:, :, None]) / safe_scale[:, :, None])
    q = np.clip(q, 0, 15).astype(np.uint8)
    q[degenerate] = 0
    q = q.reshape(out_dim, in_dim)

    # Pack 8 nibbles per uint32, little-endian nibble order.
    nib = q.reshape(out_dim, in_dim // 8, 8).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4)
    packed = np.bitwise_or.reduce(nib << shifts, axis=2).astype(np.uint32)

    return packed, scales, biases


def read_cellm(path: Path):
    """Read a .cellm file, return (header, tensor_name -> (offset, nbytes, shape, dtype))."""
    with open(path, "rb") as f:
        magic = f.read(5)
        assert magic == b"CELLM", f"Bad magic: {magic}"
        ver = struct.unpack("<B", f.read(1))[0]
        hdr_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(hdr_len).decode("utf-8"))
        data_start = (5 + 1 + 4 + hdr_len + 63) & ~63

        tensors = {}
        for t in header["tensors"]:
            f.seek(t["offset_bytes"])
            data = f.read(t["nbytes"])
            tensors[t["name"]] = {
                "data": data,
                "shape": t["shape"],
                "dtype": t["dtype"],
                "nbytes": t["nbytes"],
            }
    return header, tensors


def write_cellm(output_path: Path, header: dict, tensors: dict):
    """Write a .cellm file from header and {name: bytes_data} dict.

    Offsets and the header length are mutually dependent: writing larger offsets
    into the index makes the JSON longer, which pushes `data_start` further out,
    which makes the offsets larger again. Solve it as a fixed point and verify,
    rather than assuming a single correction pass converges.
    """
    names = sorted(tensors.keys())

    # Clean up internal fields before serializing.
    shapes = header.pop("_shapes")
    dtypes = header.pop("_dtypes")

    def layout(hdr_len: int):
        data_start = (5 + 1 + 4 + hdr_len + 63) & ~63
        offsets = {}
        cur = data_start
        for name in names:
            cur = (cur + 63) & ~63
            offsets[name] = cur
            cur += len(tensors[name])
        return data_start, offsets

    hdr_len = 0
    hdr_json = b""
    data_start = 0
    for _ in range(16):
        data_start, offsets = layout(hdr_len)
        header["tensors"] = [
            {
                "name": name,
                "offset_bytes": offsets[name],
                "nbytes": len(tensors[name]),
                "shape": list(shapes[name]),
                "dtype": dtypes[name],
            }
            for name in names
        ]
        hdr_json = json.dumps(header).encode("utf-8")
        if len(hdr_json) == hdr_len:
            break
        hdr_len = len(hdr_json)
    else:
        raise RuntimeError("cellm header layout did not converge")

    with open(output_path, "wb") as f:
        f.write(b"CELLM")
        f.write(struct.pack("<B", 1))
        f.write(struct.pack("<I", hdr_len))
        f.write(hdr_json)
        pos = 5 + 1 + 4 + hdr_len
        if pos < data_start:
            f.write(b"\x00" * (data_start - pos))
        for name in names:
            pos = f.tell()
            aligned = (pos + 63) & ~63
            if pos < aligned:
                f.write(b"\x00" * (aligned - pos))
            # A tensor written anywhere other than its indexed offset silently
            # corrupts the model, so assert instead of trusting the arithmetic.
            assert f.tell() == offsets[name], (
                f"{name}: wrote at {f.tell()}, index says {offsets[name]}"
            )
            f.write(tensors[name])

    return output_path


def should_quantize(name: str, shape: list) -> bool:
    """Decide if a tensor should be quantized to int4."""
    # Keep norms, conv kernels, layernorms in f16 for accuracy
    if name.endswith(".weight") and len(shape) == 2:
        if "norm" in name or "layernorm" in name:
            return False
        if name == "model.embed_tokens.weight":
            # Quantize embeddings too — they're 57% of the file
            return True
        return True
    return False


def group_size_for(name: str, shape: list) -> int:
    """LFM runner hardcodes group_size=64 in linear_i4_out_in."""
    return 64


def main():
    if len(sys.argv) != 3:
        print("Usage: quantize_lfm_cellm.py <input.cellm> <output.cellm>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    print(f"Reading {input_path}...")
    header, tensors = read_cellm(input_path)
    print(f"Found {len(tensors)} tensors in .cellm file")
    print(f"Model type: {header.get('model_type')}")

    # Track new tensors
    new_tensors = {}
    shapes = dict(header.get("_shapes", {}))
    dtypes = dict(header.get("_dtypes", {}))

    # Store shapes/dtypes for write_cellm
    header["_shapes"] = {}
    header["_dtypes"] = {}

    quantized_count = 0
    f16_count = 0
    total_f16_bytes = 0
    total_quant_bytes = 0

    for name in sorted(tensors.keys()):
        info = tensors[name]
        shape = info["shape"]

        if should_quantize(name, shape):
            # Read f16 data
            f16_arr = np.frombuffer(info["data"], dtype=np.float16).reshape(shape)

            # Quantize to int4 MLX format with per-tensor group size
            gs = group_size_for(name, shape)
            packed, scales, biases = quantize_weight_2d(f16_arr, group_size=gs)
            out_dim, packed_in = packed.shape
            n_groups = scales.shape[1]

            # Store quantized weight as u32 bytes
            weight_name = name
            scales_name = name.replace(".weight", ".scales")
            biases_name = name.replace(".weight", ".biases")

            new_tensors[weight_name] = packed.tobytes()
            new_tensors[scales_name] = scales.tobytes()
            new_tensors[biases_name] = biases.tobytes()

            header["_shapes"][weight_name] = [out_dim, packed_in]
            header["_dtypes"][weight_name] = "u32"
            header["_shapes"][scales_name] = [out_dim, n_groups]
            header["_dtypes"][scales_name] = "f32"
            header["_shapes"][biases_name] = [out_dim, n_groups]
            header["_dtypes"][biases_name] = "f32"

            f16_bytes = len(info["data"])
            quant_bytes = (
                len(packed.tobytes()) + len(scales.tobytes()) + len(biases.tobytes())
            )
            total_f16_bytes += f16_bytes
            total_quant_bytes += quant_bytes
            quantized_count += 1
            ratio = quant_bytes / f16_bytes * 100 if f16_bytes > 0 else 0
            print(f"  Q {name:55s}  gs={gs:>3d}  {list(shape)}  {ratio:5.1f}%")
        else:
            # Keep as f16
            new_tensors[name] = info["data"]
            header["_shapes"][name] = shape
            header["_dtypes"][name] = info["dtype"]
            f16_count += 1

    print(f"\nQuantized {quantized_count} weight tensors to int4")
    print(f"Kept {f16_count} tensors in f16 (norms, embeddings, conv kernels)")
    f16_mb = total_f16_bytes / 1024 / 1024
    quant_mb = total_quant_bytes / 1024 / 1024
    print(
        f"f16 portion: {f16_mb:.1f} MB -> int4 portion: {quant_mb:.1f} MB ({quant_mb / f16_mb * 100:.1f}%)"
    )

    # Update header
    print(f"\nWriting {output_path}...")
    write_cellm(output_path, header, new_tensors)

    out_size = output_path.stat().st_size
    in_size = input_path.stat().st_size
    print(
        f"Done! {in_size / 1024 / 1024:.1f} MB -> {out_size / 1024 / 1024:.1f} MB ({out_size / in_size * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
