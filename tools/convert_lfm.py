#!/usr/bin/env python3
"""
Convert LFM2.5 MLX model to cellm format.
Handles pre-quantized 4-bit weights with .scales/.biases suffixes.
"""

import json
import struct
import sys
from pathlib import Path
from safetensors import safe_open
import numpy as np


def write_cellm(output_path: Path, header: dict, tensors: dict, tensors_np: dict):
    """Write a cellm file with the given header and tensors."""
    # Serialize header
    header_json = json.dumps(header).encode('utf-8')
    header_len = len(header_json)
    
    # Calculate data start position (aligned to 64 bytes)
    data_start = (5 + 1 + 4 + header_len + 63) & ~63
    
    # Calculate tensor offsets
    tensor_offsets = {}
    current_offset = data_start
    for name in sorted(tensors.keys()):
        tensor_data = tensors[name]
        # Align to 64 bytes
        current_offset = (current_offset + 63) & ~63
        tensor_offsets[name] = current_offset
        current_offset += len(tensor_data)
    
    # Build tensor index for header
    tensor_index = []
    for name in sorted(tensors.keys()):
        tensor_data = tensors[name]
        tensor_np = tensors_np[name]  # Get numpy array for shape
        tensor_index.append({
            "name": name,
            "offset_bytes": tensor_offsets[name],
            "nbytes": len(tensor_data),
            "shape": list(tensor_np.shape),
            "dtype": "u32" if tensor_np.dtype == np.uint32 else "f16" if tensor_np.dtype == np.float16 else "f32"
        })
    
    header["tensors"] = tensor_index
    header_json = json.dumps(header).encode('utf-8')
    header_len = len(header_json)
    
    # Re-calculate with correct header length
    data_start = (5 + 1 + 4 + header_len + 63) & ~63
    tensor_offsets = {}
    current_offset = data_start
    for name in sorted(tensors.keys()):
        tensor_data = tensors[name]
        current_offset = (current_offset + 63) & ~63
        tensor_offsets[name] = current_offset
        current_offset += len(tensor_data)
    
    # Update tensor index with correct offsets
    for item in tensor_index:
        item["offset_bytes"] = tensor_offsets[item["name"]]
    
    header["tensors"] = tensor_index
    header_json = json.dumps(header).encode('utf-8')
    header_len = len(header_json)
    
    # Write file
    with open(output_path, 'wb') as f:
        # Magic + version + header length
        f.write(b'CELLM')
        f.write(struct.pack('<B', 1))  # version
        f.write(struct.pack('<I', header_len))
        f.write(header_json)
        
        # Pad to data_start
        current_pos = 5 + 1 + 4 + header_len
        if current_pos < data_start:
            f.write(b'\x00' * (data_start - current_pos))
        
        # Write tensors
        for name in sorted(tensors.keys()):
            tensor_data = tensors[name]
            # Align if needed
            pos = f.tell()
            aligned_pos = (pos + 63) & ~63
            if pos < aligned_pos:
                f.write(b'\x00' * (aligned_pos - pos))
            f.write(tensor_data)


def convert_lfm_model(input_dir: Path, output_path: Path):
    """Convert LFM2.5 MLX model to cellm format."""
    
    # Load config
    with open(input_dir / "config.json") as f:
        config = json.load(f)
    
    print(f"Converting LFM model from {input_dir}")
    print(f"Model type: {config.get('model_type')}")
    print(f"Hidden size: {config.get('hidden_size')}")
    print(f"Layers: {config.get('num_hidden_layers')}")
    
    # Find safetensors files
    safetensors_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {input_dir}")
    
    print(f"Found {len(safetensors_files)} safetensors file(s)")
    
    # Load all tensors
    tensors = {}
    tensors_np = {}
    
    for st_file in safetensors_files:
        print(f"Loading {st_file.name}...")
        with safe_open(str(st_file), framework="np", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                tensors_np[name] = tensor
    
    # Convert tensors to bytes, keeping uint32 weights as-is for runner to dequantize
    # Keep .scales and .biases as f32 so the runner can read them directly
    print("Converting tensors...")
    for name in list(tensors_np.keys()):
        tensor = tensors_np[name]
        if tensor.dtype == np.uint32:
            # Keep as uint32 - runner will dequantize
            tensors[name] = tensor.tobytes()
        elif tensor.dtype == np.float32:
            if name.endswith(".scales") or name.endswith(".biases"):
                # Keep quantization parameters as f32 for accuracy
                tensors[name] = tensor.tobytes()
            else:
                # Convert norms and other f32 weights to f16
                f16_tensor = tensor.astype(np.float16)
                tensors[name] = f16_tensor.tobytes()
                tensors_np[name] = f16_tensor
        elif tensor.dtype == np.float16:
            tensors[name] = tensor.tobytes()
        else:
            print(f"Warning: unknown dtype {tensor.dtype} for {name}, converting to f16")
            f16_tensor = tensor.astype(np.float16)
            tensors[name] = f16_tensor.tobytes()
            tensors_np[name] = f16_tensor
    
    print(f"Loaded {len(tensors)} tensors")
    
    # Infer intermediate_size from actual weights (config value may be wrong)
    # Check w1 weight shape: [intermediate_size, hidden_size/8] for 4-bit quantized
    intermediate_size = config.get("intermediate_size", 6656)
    if "model.layers.0.feed_forward.w1.weight" in tensors_np:
        w1_shape = tensors_np["model.layers.0.feed_forward.w1.weight"].shape
        inferred_intermediate = w1_shape[0]
        if inferred_intermediate != intermediate_size:
            print(f"Warning: config intermediate_size={intermediate_size} but inferred from weights={inferred_intermediate}")
            print(f"Using inferred value: {inferred_intermediate}")
            intermediate_size = inferred_intermediate
    
    # Build header
    # For LFM, we need to store the full config in source_text_config
    header = {
        "model_type": "lfm2",
        "source_model_type": config.get("model_type"),
        "vocab_size": config.get("vocab_size", 65536),
        "hidden_dim": config.get("hidden_size", 1024),
        "intermediate_size": intermediate_size,
        "num_layers": config.get("num_hidden_layers", 16),
        "num_heads": config.get("num_attention_heads", 16),
        "num_kv_heads": config.get("num_key_value_heads", 8),
        "head_dim": config.get("head_dim"),  # May be None
        "rms_norm_eps": config.get("rms_norm_eps") or 1e-5,
        "rope_theta": config.get("rope_theta", 1000000.0),
        "eos_token_id": config.get("eos_token_id", 7),
        "bos_token_id": config.get("bos_token_id"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "source_torch_dtype": config.get("torch_dtype"),
        "source_text_config": config,  # Store full config here
    }
    
    # Store numpy arrays for shape info (needed for writing)
    # Don't modify tensors dict, keep np arrays separate
    tensors_np_for_write = dict(tensors_np)  # Copy for later use
    
    # Write output
    print(f"Writing to {output_path}...")
    write_cellm(output_path, header, tensors, tensors_np_for_write)
    
    # Print stats
    output_size = output_path.stat().st_size
    print(f"Done! Output size: {output_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: convert_lfm.py <input_dir> <output.cellm>")
        print("Example: convert_lfm.py models/LFM2.5-350M-MLX-4bit models/LFM2.5-350M.cellm")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    convert_lfm_model(input_dir, output_path)
