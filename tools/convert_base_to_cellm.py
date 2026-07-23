#!/usr/bin/env python3
"""Convert a .base format model file to .cellm format for use with the cellm inference engine.

Usage:
    python3 tools/convert_base_to_cellm.py \
        --input models/gguf/Llama-3.2-3B-Instruct-Q4.base \
        --output models/Llama-3.2-3B-Instruct-Q4-affine-v1.cellm \
        --tokenizer models/Llama-3.2-3B-Instruct/
"""

import argparse
import json
import os
import struct
import sys
import time
import numpy as np


def align_up(pos: int, alignment: int) -> int:
    return (pos + alignment - 1) // alignment * alignment


# BASE format blob alignment: weights blob starts at a 64 KiB boundary.
BASE_BLOB_ALIGNMENT = 64 * 1024  # 65536


def bf16_bits_to_f32_np(u16_arr: np.ndarray) -> np.ndarray:
    """Convert an array of bf16 bit patterns (stored as uint16) to f32.

    bf16 is the upper 16 bits of a float32, so the conversion is simply
    shifting left by 16 and reinterpreting as f32.
    """
    u32_arr = u16_arr.astype(np.uint32) << np.uint32(16)
    return u32_arr.view(np.float32).copy()


def quantize_i4_row_np(values: np.ndarray) -> (np.ndarray, np.float16):
    """Quantize a row of f32 values to cellm i4 format using numpy.

    Returns (packed_uint8, scale_f16_bits).
    """
    max_abs = np.max(np.abs(values))
    scale = max_abs / 7.0 if max_abs > 0.0 else 1.0

    qi = np.round(values / scale).clip(-7, 7).astype(np.int8) + 8
    # Pack: even indices in low nibble, odd in high nibble
    n = len(qi)
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    packed[:n//2] = qi[0:n:2].astype(np.uint8) | (qi[1:n:2].astype(np.uint8) << 4)
    if n % 2 == 1:
        packed[-1] = qi[-1].astype(np.uint8)

    # Clamp scale to f16 range to avoid overflow
    scale = max(-65504.0, min(65504.0, float(scale)))
    return packed, np.float16(scale)


def dequantize_base_q4_tensor_np(
    data: bytes,
    offset: int,
    scale_offset: int,
    scale_length: int,
    bias_offset: int,
    bias_length: int,
    rows: int,
    cols: int,
    group_size: int,
    scale_dtype: str = 'bf16',
) -> np.ndarray:
    """Dequantize entire base_q4 tensor to f32 values using numpy.

    Per the baserT .base format spec (FORMAT.md + CANONICAL_QUANT_SPEC.md):
      Data: row-major, [row0_g0_data, row0_g1_data, ..., row0_gN, row1_g0, ...]
            group_size/2 bytes per group (32 for group_size=64)
      Scales/biases: stored in separate contiguous regions, in row-major
            group order: scale[r*n_groups + g] for row r, group g
      Dequant formula (asymmetric): value = q_unsigned * scale + bias
            where q_unsigned is the raw nibble value in [0, 15]
      scale_dtype: 'bf16' (default) or 'f16'
    """
    n_groups = (cols + group_size - 1) // group_size
    group_nbytes = group_size // 2  # 32 bytes per group for gs=64

    total_elements = rows * cols
    result = np.empty(total_elements, dtype=np.float32)

    # Read packed data as numpy array
    data_bytes_view = np.frombuffer(
        data, dtype=np.uint8, offset=offset,
        count=rows * n_groups * group_nbytes,
    )

    # Read scales and biases as uint16 arrays, then convert to f32.
    # Scale/bias layout is row-major: index = row * n_groups + group.
    scale_u16 = np.frombuffer(
        data, dtype=np.uint16, offset=offset + scale_offset,
        count=rows * n_groups,
    ).copy()
    bias_u16 = np.frombuffer(
        data, dtype=np.uint16, offset=offset + bias_offset,
        count=rows * n_groups,
    ).copy()

    if scale_dtype == 'bf16':
        scales_f32 = bf16_bits_to_f32_np(scale_u16).reshape(rows, n_groups)
        biases_f32 = bf16_bits_to_f32_np(bias_u16).reshape(rows, n_groups)
    else:  # f16
        scales_f32 = scale_u16.view(np.float16).astype(np.float32).reshape(rows, n_groups)
        biases_f32 = bias_u16.view(np.float16).astype(np.float32).reshape(rows, n_groups)

    # Replace NaN/Inf scales with 0 (sentinel for zero-variance groups)
    bad = ~np.isfinite(scales_f32)
    scales_f32[bad] = 0.0
    bad = ~np.isfinite(biases_f32)
    biases_f32[bad] = 0.0

    # Vectorized dequant: process all groups for each row.
    for r in range(rows):
        for g in range(n_groups):
            g_eff = min(group_size, cols - g * group_size)
            d_start = r * n_groups * group_nbytes + g * group_nbytes
            packed = data_bytes_view[d_start:d_start + group_nbytes]

            # Unpack 4-bit values as UNSIGNED [0, 15]
            q = np.empty(group_size, dtype=np.float32)
            q[0::2] = (packed[:group_size // 2] & 0x0F).astype(np.float32)
            q[1::2] = ((packed[:group_size // 2] >> 4) & 0x0F).astype(np.float32)

            scale = scales_f32[r, g]
            bias = biases_f32[r, g]
            out_start = r * cols + g * group_size
            out_end = out_start + g_eff
            result[out_start:out_end] = q[:g_eff] * scale + bias

    return result


def dequantize_f16_tensor_np(data: bytes, offset: int, nbytes: int) -> np.ndarray:
    """Read f16 tensor and return as f32 numpy array."""
    f16_data = np.frombuffer(data, dtype=np.float16, offset=offset, count=nbytes // 2)
    return f16_data.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert .base model to .cellm format")
    parser.add_argument("--input", required=True, help="Path to .base file")
    parser.add_argument("--output", required=True, help="Output .cellm file path")
    parser.add_argument("--tokenizer", default=None, help="Directory to write tokenizer.json to")
    parser.add_argument("--quantize", default="affine-i4", choices=["f16", "i4", "affine-i4"],
                        help="Output weight format (default: affine-i4 preserves BASE Q4 groups)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    print(f"Reading {input_path}...")
    t0 = time.time()
    with open(input_path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

    # Parse header
    magic = bytes(file_bytes[0:4].tolist())
    if magic != b'BASE':
        print(f"ERROR: Not a BASE file (magic={magic})")
        sys.exit(1)

    version = struct.unpack('<I', file_bytes[4:8].tobytes())[0]
    meta_len = struct.unpack('<Q', file_bytes[8:16].tobytes())[0]
    meta = json.loads(bytes(file_bytes[16:16 + meta_len].tolist()))
    # The .base format aligns the weights blob to a 64 KiB boundary
    # (see FORMAT.md: BLOB_ALIGNMENT = 64 * 1024).  Tensor offsets in
    # the header are relative to this aligned blob start, NOT to
    # 16 + meta_len.
    data_start = align_up(16 + meta_len, BASE_BLOB_ALIGNMENT)

    print(f"BASE version: {version}, Arch: {meta['arch']}, Quant: {meta['quant_scheme']}")
    print(f"Parse time: {time.time() - t0:.1f}s")

    config = meta.get('config', {})
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    num_key_value_heads = config.get('num_key_value_heads', num_attention_heads)
    head_dim = config.get('head_dim', hidden_size // num_attention_heads)
    rms_norm_eps = config['rms_norm_eps']
    rope_theta = config['rope_theta']
    bos_token_id = config['bos_token_id']
    eos_token_id = config['eos_token_id']
    max_position_embeddings = config.get('max_position_embeddings', 8192)
    tie_word_embeddings = config.get('tie_word_embeddings', False)

    print(f"Model: Llama {num_hidden_layers} layers, {hidden_size} hidden, {intermediate_size} ff")

    # Extract tokenizer
    tokenizer_data = meta.get('tokenizer', {})
    if args.tokenizer:
        os.makedirs(args.tokenizer, exist_ok=True)
        tk_json = tokenizer_data.get('tokenizer.json')
        if tk_json:
            with open(os.path.join(args.tokenizer, 'tokenizer.json'), 'w') as f:
                json.dump(tk_json, f)
            print(f"Wrote tokenizer.json")

        tk_config = {}
        if 'tokenizer.chat_template' in tokenizer_data:
            tk_config['chat_template'] = tokenizer_data['tokenizer.chat_template']
        if tk_config:
            with open(os.path.join(args.tokenizer, 'tokenizer_config.json'), 'w') as f:
                json.dump(tk_config, f, indent=2)

    # ---- Map tensors ----
    tensors = meta['tensors']
    tensor_by_name = {t['name']: t for t in tensors}

    def map_tensor_name(base_name):
        if base_name == 'embed_tokens.weight':
            return 'model.embed_tokens.weight'
        if base_name == 'final_norm.weight':
            return 'model.norm.weight'
        if base_name.startswith('layers.'):
            rest = base_name[len('layers.'):]
            dot_idx = rest.index('.')
            layer = rest[:dot_idx]
            suffix = rest[dot_idx + 1:]
            suffix_map = {
                'input_norm.weight': 'input_layernorm.weight',
                'post_attn_norm.weight': 'post_attention_layernorm.weight',
            }
            suffix = suffix_map.get(suffix, suffix)
            return f'model.layers.{layer}.{suffix}'
        return None

    def should_quantize_i4(name):
        if 'norm' in name.lower():
            return False
        if name in ('model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'):
            return False
        return True

    # Build output tensors
    out_tensors = []
    total_input_bytes = 0
    total_output_bytes = 0
    output_is_i4 = (args.quantize == 'i4')
    output_is_affine_i4 = (args.quantize == 'affine-i4')

    base_file_bytes_ref = file_bytes
    raw_data_start = data_start

    for t in tensors:
        base_name = t['name']
        cellm_name = map_tensor_name(base_name)
        if cellm_name is None:
            continue

        dtype = t['dtype']
        shape = t['shape']
        offset = raw_data_start + t['offset']

        t1 = time.time()

        if dtype == 'f16':
            nbytes = t['length']
            if output_is_i4 and should_quantize_i4(cellm_name) and len(shape) == 2:
                rows, cols = shape
                f32_vals = dequantize_f16_tensor_np(base_file_bytes_ref, offset, nbytes)

                i4_packed = bytearray()
                i4_scales = bytearray()
                for r in range(rows):
                    packed, scale_val = quantize_i4_row_np(f32_vals[r*cols:(r+1)*cols])
                    i4_packed.extend(packed.tobytes())
                    i4_scales += struct.pack('<e', scale_val)

                out_tensors.append({'name': cellm_name, 'dtype': 'i4', 'shape': shape, 'data': bytes(i4_packed)})
                out_tensors.append({'name': f'{cellm_name}.qscale', 'dtype': 'f16', 'shape': [shape[0]], 'data': bytes(i4_scales)})
                elapsed = time.time() - t1
                print(f"  {base_name:<50} -> {cellm_name:<50} i4 (from f16) [{elapsed:.1f}s]")
            else:
                out_tensors.append({'name': cellm_name, 'dtype': 'f16', 'shape': shape,
                                    'data': bytes(base_file_bytes_ref[offset:offset + nbytes].tolist())})
                print(f"  {base_name:<50} -> {cellm_name:<50} f16")

        elif dtype == 'base_q4':
            group_size = t.get('group_size', 64)
            scale_dtype = t.get('scale_dtype', 'bf16')
            rows, cols = shape

            if output_is_affine_i4:
                # Preserve BASE's unsigned affine Q4 exactly. A row-wise symmetric
                # requantization loses the per-64-value bias and scale and severely
                # degrades already-quantized models.
                n_groups = (cols + group_size - 1) // group_size
                packed_nbytes = rows * n_groups * (group_size // 2)
                packed = bytes(base_file_bytes_ref[offset:offset + packed_nbytes])
                scale_u16 = np.frombuffer(
                    base_file_bytes_ref, dtype=np.uint16,
                    offset=offset + t['scale_offset'], count=rows * n_groups,
                ).copy()
                bias_u16 = np.frombuffer(
                    base_file_bytes_ref, dtype=np.uint16,
                    offset=offset + t['bias_offset'], count=rows * n_groups,
                ).copy()
                if scale_dtype == 'bf16':
                    scales = bf16_bits_to_f32_np(scale_u16)
                    biases = bf16_bits_to_f32_np(bias_u16)
                else:
                    scales = scale_u16.view(np.float16).astype(np.float32)
                    biases = bias_u16.view(np.float16).astype(np.float32)
                base = cellm_name.removesuffix('.weight')
                out_tensors.append({'name': cellm_name, 'dtype': 'u32', 'shape': shape, 'data': packed})
                out_tensors.append({'name': f'{base}.scales', 'dtype': 'f32',
                                    'shape': [rows, n_groups], 'data': scales.tobytes()})
                out_tensors.append({'name': f'{base}.biases', 'dtype': 'f32',
                                    'shape': [rows, n_groups], 'data': biases.tobytes()})
                print(f"  {base_name:<50} -> {cellm_name:<50} affine-i4 gs={group_size}")
            else:
                t2 = time.time()
                f32_vals = dequantize_base_q4_tensor_np(
                    base_file_bytes_ref, offset,
                    t['scale_offset'], t['scale_length'],
                    t['bias_offset'], t['bias_length'],
                    rows, cols, group_size,
                    scale_dtype,
                )
                dequant_time = time.time() - t2

                if not output_is_i4:
                    # Write as f16
                    f16_data = np.float16(f32_vals).tobytes()
                    out_tensors.append({'name': cellm_name, 'dtype': 'f16', 'shape': shape, 'data': f16_data})
                    print(f"  {base_name:<50} -> {cellm_name:<50} f16 (deq {dequant_time:.1f}s)")
                else:
                    t3 = time.time()
                    i4_packed = bytearray()
                    i4_scales = bytearray()
                    for r in range(rows):
                        packed, scale_val = quantize_i4_row_np(f32_vals[r*cols:(r+1)*cols])
                        i4_packed.extend(packed.tobytes())
                        i4_scales += struct.pack('<e', scale_val)
                    quant_time = time.time() - t3

                    out_tensors.append({'name': cellm_name, 'dtype': 'i4', 'shape': shape, 'data': bytes(i4_packed)})
                    out_tensors.append({'name': f'{cellm_name}.qscale', 'dtype': 'f16', 'shape': [shape[0]], 'data': bytes(i4_scales)})
                    total_time = time.time() - t1
                    print(f"  {base_name:<50} -> {cellm_name:<50} i4 (deq {dequant_time:.1f}s + quant {quant_time:.1f}s = {total_time:.1f}s)")
        else:
            print(f"  WARNING: Skipping {base_name} unsupported dtype={dtype}")

    # Sort tensors by name
    out_tensors.sort(key=lambda t: t['name'])

    # ---- Build CellmHeader ----
    header = {
        'model_type': 'llama',
        'source_model_type': 'llama',
        'source_safetensors_format': None,
        'text_tensor_prefix': 'model.',
        'vision_tensor_prefix': None,
        'projector_tensor_prefix': None,
        'vocab_size': vocab_size,
        'hidden_dim': hidden_size,
        'intermediate_size': intermediate_size,
        'num_layers': num_hidden_layers,
        'num_heads': num_attention_heads,
        'num_kv_heads': num_key_value_heads,
        'head_dim': head_dim,
        'rms_norm_eps': rms_norm_eps,
        'rope_theta': rope_theta,
        # BaseRT's fixed Q/K permutation targets adjacent-pair rotary kernels.
        'rope_interleaved': True,
         'rope_scaling_type': config.get('rope_scaling_type'),
         'rope_scaling_factor': config.get('rope_scaling_factor'),
         'rope_scaling_original_max_position_embeddings': config.get('rope_scaling_original_max_position_embeddings'),
         'rope_scaling_low_freq_factor': config.get('rope_scaling_low_freq_factor'),
         'rope_scaling_high_freq_factor': config.get('rope_scaling_high_freq_factor'),
         'bos_token_id': bos_token_id,
        'eos_token_id': eos_token_id,
        'max_position_embeddings': max_position_embeddings,
        'tie_word_embeddings': tie_word_embeddings,
        'source_torch_dtype': 'f16',
        'source_architectures': ['LlamaForCausalLM'],
        'source_quantization': {
            'source_format': 'base',
            'quant_scheme': f'base_{meta["quant_scheme"]}',
        },
        'source_quantization_config': None,
        'source_text_config': None,
        'source_vision_config': None,
        'source_projector_config': None,
        'tensors': [],
    }

    planned = [{'name': t['name'], 'shape': t['shape'], 'dtype': t['dtype'], 'nbytes': len(t['data'])}
               for t in out_tensors]

    # Iteratively solve for header length + tensor offsets.
    # The header JSON length depends on the offset values (large numbers take
    # more characters), and the offsets depend on the header length (which
    # determines where the data region starts). Iterate until stable.
    last_header_len = None
    for iteration in range(10):
        header['tensors'] = [{'name': p['name'], 'offset_bytes': p.get('offset', 0),
                              'nbytes': p['nbytes'],
                              'shape': p['shape'], 'dtype': p['dtype']} for p in planned]
        hdr_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
        if last_header_len == len(hdr_bytes):
            break
        last_header_len = len(hdr_bytes)
        data_start_offset = align_up(5 + 1 + 4 + len(hdr_bytes), 64)
        cursor = data_start_offset
        for p in planned:
            cursor = align_up(cursor, 64)
            p['offset'] = cursor
            cursor += p['nbytes']

    header['tensors'] = [{'name': p['name'], 'offset_bytes': p['offset'], 'nbytes': p['nbytes'],
                           'shape': p['shape'], 'dtype': p['dtype']} for p in planned]
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')

    # ---- Write cellm file ----
    print(f"\nWriting {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(b'CELLM')
        f.write(struct.pack('B', 1))
        f.write(struct.pack('<I', len(header_bytes)))
        f.write(header_bytes)

        pos = 5 + 1 + 4 + len(header_bytes)
        aligned = align_up(pos, 64)
        if aligned > pos:
            f.write(b'\x00' * (aligned - pos))
            pos = aligned

        for i, p in enumerate(planned):
            if pos < p['offset']:
                f.write(b'\x00' * (p['offset'] - pos))
                pos = p['offset']
            f.write(out_tensors[i]['data'])
            pos += p['nbytes']

    fsize = os.path.getsize(output_path)
    print(f"\nDone! Wrote {len(planned)} tensors to {output_path}")
    print(f"File size: {fsize:,} bytes ({fsize / 1024 / 1024:.1f} MB)")
    print(f"Total time: {time.time() - t0:.1f}s")
    if args.tokenizer:
        print(f"Tokenizer in: {args.tokenizer}/tokenizer.json")


if __name__ == '__main__':
    main()
