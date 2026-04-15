#!/usr/bin/env python3

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

NP_DTYPES = {
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT64: np.int64,
}


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass
class TensorEntry:
    name: str
    shape: list[int]
    nbytes: int
    writer: object
    offset_bytes: int = 0


class OnnxExternalStore:
    def __init__(self) -> None:
        self._mmaps: dict[str, np.memmap] = {}

    def mmap(self, path: Path) -> np.memmap:
        key = str(path)
        if key not in self._mmaps:
            self._mmaps[key] = np.memmap(path, mode="r", dtype=np.uint8)
        return self._mmaps[key]


def external_info(tensor: onnx.TensorProto, model_dir: Path) -> tuple[Path, int, int]:
    location = None
    offset = 0
    length = None
    for item in tensor.external_data:
        if item.key == "location":
            location = item.value
        elif item.key == "offset":
            offset = int(item.value)
        elif item.key == "length":
            length = int(item.value)
    if location is None or length is None:
        raise ValueError(f"missing external data info for {tensor.name}")
    return model_dir / location, offset, length


def load_numpy_tensor(
    tensor: onnx.TensorProto,
    model_dir: Path,
    store: OnnxExternalStore,
    row_slice: slice | None = None,
) -> np.ndarray:
    if tensor.data_location == TensorProto.EXTERNAL:
        path, offset, length = external_info(tensor, model_dir)
        data = store.mmap(path)[offset : offset + length]
        dtype = NP_DTYPES[tensor.data_type]
        arr = np.frombuffer(data, dtype=dtype).reshape(tuple(tensor.dims))
        if row_slice is not None:
            arr = arr[row_slice]
        return np.asarray(arr)
    arr = numpy_helper.to_array(tensor)
    if row_slice is not None:
        arr = arr[row_slice]
    return np.asarray(arr)


def dequant_q4_rows(
    quant: np.ndarray,
    scales: np.ndarray,
    zp: np.ndarray,
) -> np.ndarray:
    if quant.ndim == 2:
        groups = scales.shape[1]
        quant = quant.reshape(quant.shape[0], groups, quant.shape[1] // groups)
    low = quant & 0x0F
    high = quant >> 4
    packed = np.empty((quant.shape[0], quant.shape[1], quant.shape[2] * 2), dtype=np.uint8)
    packed[:, :, 0::2] = low
    packed[:, :, 1::2] = high

    zp_low = zp & 0x0F
    zp_high = zp >> 4
    zp_full = np.empty((zp.shape[0], zp.shape[1] * 2), dtype=np.uint8)
    zp_full[:, 0::2] = zp_low
    zp_full[:, 1::2] = zp_high

    out = (packed.astype(np.float32) - zp_full[:, :, None].astype(np.float32)) * scales[:, :, None]
    return out.reshape(out.shape[0], out.shape[1] * out.shape[2]).astype(np.float16)


def write_q4_tensor(
    fh,
    quant_tensor: onnx.TensorProto,
    scales_tensor: onnx.TensorProto,
    zp_tensor: onnx.TensorProto,
    model_dir: Path,
    store: OnnxExternalStore,
    row_start: int = 0,
    row_end: int | None = None,
    chunk_rows: int = 128,
) -> None:
    n_rows = quant_tensor.dims[0]
    if row_end is None:
        row_end = n_rows
    for start in range(row_start, row_end, chunk_rows):
        stop = min(start + chunk_rows, row_end)
        row_slice = slice(start, stop)
        quant = load_numpy_tensor(quant_tensor, model_dir, store, row_slice)
        scales = load_numpy_tensor(scales_tensor, model_dir, store, row_slice)
        zp = load_numpy_tensor(zp_tensor, model_dir, store, row_slice)
        out = dequant_q4_rows(quant, scales, zp)
        fh.write(out.tobytes())


def write_fp_tensor(
    fh,
    tensor: onnx.TensorProto,
    model_dir: Path,
    store: OnnxExternalStore,
) -> None:
    arr = load_numpy_tensor(tensor, model_dir, store)
    if arr.dtype != np.float16:
        arr = arr.astype(np.float16)
    fh.write(arr.tobytes())


def build_entries(embed_model, decoder_model, config: dict) -> list[TensorEntry]:
    embed = {t.name: t for t in embed_model.graph.initializer}
    dec = {t.name: t for t in decoder_model.graph.initializer}
    entries: list[TensorEntry] = []
    text_cfg = config["text_config"]
    layer_types = text_cfg["layer_types"]
    shared_kv_layers = text_cfg.get("num_kv_shared_layers", 0)
    shared_start = text_cfg["num_hidden_layers"] - shared_kv_layers

    def add_fp(src_name: str, dst_name: str, shape: list[int] | None = None) -> None:
        tensor = dec[src_name]
        if shape is None:
            shape = list(tensor.dims)
        entries.append(
            TensorEntry(
                name=dst_name,
                shape=shape,
                nbytes=int(np.prod(shape)) * 2,
                writer=("fp", tensor),
            )
        )

    def add_q4(
        quant_name: str,
        scales_name: str,
        zp_name: str,
        dst_name: str,
        shape: list[int],
        row_start: int = 0,
        row_end: int | None = None,
    ) -> None:
        entries.append(
            TensorEntry(
                name=dst_name,
                shape=shape,
                nbytes=int(np.prod(shape)) * 2,
                writer=("q4", dec.get(quant_name, embed.get(quant_name)), dec.get(scales_name, embed.get(scales_name)), dec.get(zp_name, embed.get(zp_name)), row_start, row_end),
            )
        )

    add_q4(
        "model_embed_tokens_weight_quant",
        "model_embed_tokens_weight_scales",
        "model_embed_tokens_weight_zp",
        "model.language_model.embed_tokens.weight",
        [262144, 1536],
    )
    add_q4(
        "model_embed_tokens_per_layer_weight_quant",
        "model_embed_tokens_per_layer_weight_scales",
        "model_embed_tokens_per_layer_weight_zp",
        "model.language_model.embed_tokens_per_layer.weight",
        [262144, 8960],
    )
    add_q4(
        "model_per_layer_projection_MatMul_weight_quant",
        "model_per_layer_projection_MatMul_weight_scales",
        "model_per_layer_projection_MatMul_weight_zp",
        "model.language_model.per_layer_model_projection.weight",
        [8960, 1536],
    )
    add_fp(
        "model.per_layer_projection_norm.weight",
        "model.language_model.per_layer_projection_norm.weight",
    )
    add_fp(
        "model.layers.35.final_norm_layernorm.weight",
        "model.language_model.norm.weight",
        [1536],
    )

    for layer in range(35):
        base = f"model.language_model.layers.{layer}"
        add_fp(
            f"model.layers.{layer}.input_layernorm.weight",
            f"{base}.input_layernorm.weight",
        )
        add_fp(f"model.layers.{layer}.layer_scalar", f"{base}.layer_scalar", [1])
        add_fp(
            f"model.layers.{layer}.post_attention_layernorm.weight",
            f"{base}.post_attention_layernorm.weight",
        )
        add_fp(
            f"model.layers.{layer}.pre_feedforward_layernorm.weight",
            f"{base}.pre_feedforward_layernorm.weight",
        )
        add_fp(
            f"model.layers.{layer}.post_feedforward_layernorm.weight",
            f"{base}.post_feedforward_layernorm.weight",
        )
        add_fp(
            f"model.layers.{layer}.post_per_layer_input_norm.weight",
            f"{base}.post_per_layer_input_norm.weight",
        )

        q_norm = dec.get(f"model.layers.{layer}.attn.q_norm.layernorm.weight")
        if q_norm is not None:
            add_fp(q_norm.name, f"{base}.self_attn.q_norm.weight")
        k_norm = dec.get(f"model.layers.{layer}.attn.k_norm.layernorm.weight")
        if k_norm is None and layer >= shared_start:
            donor = shared_start - 2 if layer_types[layer] == "sliding_attention" else shared_start - 1
            k_norm = dec.get(f"model.layers.{donor}.attn.k_norm.layernorm.weight")
        if k_norm is not None:
            add_fp(k_norm.name, f"{base}.self_attn.k_norm.weight")

        for proj in ["q", "k", "v", "o"]:
            quant = dec.get(f"model_layers_{layer}_attn_{proj}_proj_MatMul_weight_quant")
            if quant is None and proj in {"k", "v"} and layer >= shared_start:
                donor = shared_start - 2 if layer_types[layer] == "sliding_attention" else shared_start - 1
                quant = dec.get(f"model_layers_{donor}_attn_{proj}_proj_MatMul_weight_quant")
            if quant is None:
                continue
            src_layer = layer
            if f"model_layers_{layer}_attn_{proj}_proj_MatMul_weight_scales" not in dec and proj in {"k", "v"} and layer >= shared_start:
                src_layer = shared_start - 2 if layer_types[layer] == "sliding_attention" else shared_start - 1
            scales = dec[f"model_layers_{src_layer}_attn_{proj}_proj_MatMul_weight_scales"]
            zp = dec[f"model_layers_{src_layer}_attn_{proj}_proj_MatMul_weight_zp"]
            shape = [quant.dims[0], quant.dims[1] * 32]
            add_q4(
                quant.name,
                scales.name,
                zp.name,
                f"{base}.self_attn.{proj}_proj.weight",
                shape,
            )

        gate_up = dec.get(f"model_layers_{layer}_mlp_gate_up_proj_MatMul_weight_quant")
        if gate_up is not None:
            scales = dec[f"model_layers_{layer}_mlp_gate_up_proj_MatMul_weight_scales"]
            zp = dec[f"model_layers_{layer}_mlp_gate_up_proj_MatMul_weight_zp"]
            hidden = gate_up.dims[1] * 32
            rows = gate_up.dims[0]
            half = rows // 2
            add_q4(
                gate_up.name,
                scales.name,
                zp.name,
                f"{base}.mlp.gate_proj.weight",
                [half, hidden],
                0,
                half,
            )
            add_q4(
                gate_up.name,
                scales.name,
                zp.name,
                f"{base}.mlp.up_proj.weight",
                [rows - half, hidden],
                half,
                rows,
            )

        down = dec.get(f"model_layers_{layer}_mlp_down_proj_MatMul_weight_quant")
        if down is not None:
            add_q4(
                down.name,
                f"model_layers_{layer}_mlp_down_proj_MatMul_weight_scales",
                f"model_layers_{layer}_mlp_down_proj_MatMul_weight_zp",
                f"{base}.mlp.down_proj.weight",
                [down.dims[0], down.dims[1] * 32],
            )

        per_in = dec.get(f"model_layers_{layer}_per_layer_per_layer_input_gate_MatMul_weight_quant")
        if per_in is not None:
            add_q4(
                per_in.name,
                f"model_layers_{layer}_per_layer_per_layer_input_gate_MatMul_weight_scales",
                f"model_layers_{layer}_per_layer_per_layer_input_gate_MatMul_weight_zp",
                f"{base}.per_layer_input_gate.weight",
                [per_in.dims[0], per_in.dims[1] * 32],
            )
        per_proj = dec.get(f"model_layers_{layer}_per_layer_per_layer_projection_MatMul_weight_quant")
        if per_proj is not None:
            add_q4(
                per_proj.name,
                f"model_layers_{layer}_per_layer_per_layer_projection_MatMul_weight_scales",
                f"model_layers_{layer}_per_layer_per_layer_projection_MatMul_weight_zp",
                f"{base}.per_layer_projection.weight",
                [per_proj.dims[0], per_proj.dims[1] * 32],
            )

    return entries


def build_header(config: dict, entries: list[TensorEntry]) -> dict:
    text_cfg = config["text_config"]
    return {
        "model_type": text_cfg.get("model_type", "gemma4_text"),
        "source_model_type": config.get("model_type"),
        "source_safetensors_format": None,
        "text_tensor_prefix": "model.",
        "vision_tensor_prefix": None,
        "projector_tensor_prefix": None,
        "vocab_size": text_cfg["vocab_size"],
        "hidden_dim": text_cfg["hidden_size"],
        "intermediate_size": text_cfg["intermediate_size"],
        "num_layers": text_cfg["num_hidden_layers"],
        "num_heads": text_cfg["num_attention_heads"],
        "num_kv_heads": text_cfg["num_key_value_heads"],
        "rms_norm_eps": text_cfg["rms_norm_eps"],
        "rope_theta": text_cfg["rope_parameters"]["full_attention"]["rope_theta"],
        "bos_token_id": text_cfg.get("bos_token_id"),
        "eos_token_id": text_cfg.get("eos_token_id"),
        "max_position_embeddings": text_cfg.get("max_position_embeddings"),
        "tie_word_embeddings": text_cfg.get("tie_word_embeddings", True),
        "source_torch_dtype": text_cfg.get("dtype", config.get("dtype")),
        "source_architectures": config.get("architectures"),
        "source_quantization": None,
        "source_quantization_config": None,
        "source_text_config": text_cfg,
        "source_vision_config": None,
        "source_projector_config": None,
        "tensors": [
            {
                "name": entry.name,
                "offset_bytes": entry.offset_bytes,
                "nbytes": entry.nbytes,
                "shape": entry.shape,
                "dtype": "f16",
            }
            for entry in entries
        ],
    }


def compute_offsets(config: dict, entries: list[TensorEntry]) -> dict:
    last_len = None
    header = None
    for _ in range(5):
        header = build_header(config, entries)
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        if len(header_bytes) == last_len:
            break
        last_len = len(header_bytes)
        cursor = align_up(10 + len(header_bytes), 64)
        for entry in entries:
            cursor = align_up(cursor, 64)
            entry.offset_bytes = cursor
            cursor += entry.nbytes
    if header is None:
        raise RuntimeError("failed to build header")
    return header


def write_cellmd(output_path: Path, header: dict, entries: list[TensorEntry], model_dir: Path, embed_model, decoder_model) -> None:
    store = OnnxExternalStore()
    embed = {t.name: t for t in embed_model.graph.initializer}
    dec = {t.name: t for t in decoder_model.graph.initializer}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        fh.write(b"CELLM")
        fh.write(bytes([1]))
        fh.write(struct.pack("<I", len(header_bytes)))
        fh.write(header_bytes)
        pos = 10 + len(header_bytes)
        aligned = align_up(pos, 64)
        if aligned > pos:
            fh.write(b"\x00" * (aligned - pos))
            pos = aligned

        for entry in entries:
            if pos < entry.offset_bytes:
                fh.write(b"\x00" * (entry.offset_bytes - pos))
                pos = entry.offset_bytes
            mode = entry.writer[0]
            if mode == "fp":
                tensor = entry.writer[1]
                write_fp_tensor(fh, tensor, model_dir, store)
            elif mode == "q4":
                _, quant, scales, zp, row_start, row_end = entry.writer
                write_q4_tensor(fh, quant, scales, zp, model_dir, store, row_start, row_end)
            else:
                raise ValueError(f"unsupported writer mode {mode}")
            pos += entry.nbytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, help="Gemma 4 ONNX root dir")
    parser.add_argument("--output", required=True, help="Output .cellmd path")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    onnx_dir = input_root / "onnx"
    embed_path = onnx_dir / "embed_tokens_q4.onnx"
    decoder_path = onnx_dir / "decoder_model_merged_q4.onnx"
    config_path = input_root / "config.json"

    embed_model = onnx.load(embed_path, load_external_data=False)
    decoder_model = onnx.load(decoder_path, load_external_data=False)
    config = json.loads(config_path.read_text())

    entries = build_entries(embed_model, decoder_model, config)
    header = compute_offsets(config, entries)
    write_cellmd(Path(args.output), header, entries, onnx_dir, embed_model, decoder_model)
    print(f"wrote {args.output} with {len(entries)} tensors")


if __name__ == "__main__":
    main()
