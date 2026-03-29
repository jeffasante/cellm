# `.cellm` Binary Format Specification (v1)

This format is designed to be simple to write, fast to memory-map, and friendly for mobile runtimes.

## Goals

- Load weights with `mmap` (no copy into RAM).
- Keep a small self-describing header (JSON).
- Store tensors in a single flat file with predictable alignment.

## File Layout

```
| "CELLM" (5 bytes) |
| version (u8)      |  = 1
| header_len (u32)  |  little-endian byte length of JSON header
| header (bytes)    |  UTF-8 JSON
| padding           |  0..63 bytes to align to 64
| tensor data       |  concatenated tensors, each 64-byte aligned
```

Alignment rules:
- The first tensor starts at the next 64-byte boundary after the header.
- Each tensor starts at a 64-byte boundary.

## Header JSON

The header is a single JSON object with:

- Model config fields (vocab size, layers, heads, etc.).
- `tensors[]`: a tensor index describing where each tensor lives in the file.

Each `tensors[]` entry contains:

- `name`: string tensor name (HuggingFace weight name)
- `dtype`: `"f16"` (current)
- `shape`: array of dimensions (usize)
- `offset_bytes`: absolute byte offset from the beginning of the file
- `nbytes`: number of bytes for the tensor payload

## Current Implementation Notes

- The converter always emits `f16` tensor data, converting from `f32` or `bf16` when needed.
- Tokenizer assets (`tokenizer.json`, merges/vocab) are not embedded yet; keep them alongside the `.cellm` file.

## Converter

Example (SmolLM2-135M local HF folder):

```bash
cargo run --bin convert -- \
  --input  /Users/jeff/Desktop/cellm/models/hf/smollm2-135m \
  --output /Users/jeff/Desktop/cellm/models/smollm2-135m.cellm \
  --dtype  f16
```

