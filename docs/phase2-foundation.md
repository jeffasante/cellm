# Phase 2 Foundation (Paged KV Cache)

The system is now a stable foundation for Phase 2: Paged KV Cache implementation.

## What I checked

- Built and ran the full Rust workspace tests to confirm everything compiles and the existing Phase 1 pieces still work.
- Focused on correctness and simple, testable building blocks (not GPU kernels yet).

## What I implemented (in plain English)

Phase 2 needs a way to store “the model’s memory” (KV cache) without growing RAM forever as conversations get longer. The key idea is to store the KV cache in fixed-size chunks (“blocks”) and keep a small lookup table that says which block holds which token.

To make that possible, I added two core pieces:

### 1) `BlockAllocator` (hands out block ids)

File: `crates/cellm-cache/src/allocator.rs`

- Think of this as a librarian that tracks which shelves are free.
- You ask for a free shelf (`alloc()`), it gives you an id.
- When you are done with a shelf, you return it (`free(id)`).
- It detects common mistakes like trying to free the same id twice.

### 2) `PageTable` (maps token positions to blocks)

File: `crates/cellm-cache/src/pagetable.rs`

- Each session has its own `PageTable`.
- As tokens are added, the page table allocates a new block only when the current one fills up.
- Given a token position (like token 0, 1, 2, …), it can tell you:
  - which block id holds that token (`block_for_token(pos)`)
  - the offset within that block (`offset_in_block(pos)`)
- When a session ends, `free_all()` returns all its blocks to the allocator.

### 3) `CacheError` (clear failures)

File: `crates/cellm-cache/src/error.rs`

- When blocks run out, you get a clear “out of blocks” error instead of silent corruption.
- Bad inputs (like asking for token 999 when you only have 10 tokens) return a clear error.

## How to run the checks

From the repo root:

```bash
cargo test -p cellm-cache
```

## How Phase 2 will use this next

- The scheduler (or engine) owns a single `BlockAllocator` for the whole device.
- Each active session owns a `PageTable`.
- During decoding, every time a new token is generated, the session calls `append_token()`.
- The model forward pass reads/writes KV using `block_for_token()` to find the right block and slot.

## Update: Physical KV cache + forward integration

After validating `BlockAllocator` and `PageTable`, the next step is to connect “block ids” to real memory.

### Physical cache storage: `KVCache`

File: `crates/cellm-cache/src/kvcache.rs`

- `KVCache` owns the actual `k` and `v` buffers (now stored as `f16` to match real KV-cache memory goals).
- It also owns the `BlockAllocator`, so the cache is a single “thing” that can both allocate ids and store bytes.
- The memory layout is a big slab broken into:
  - block id
  - layer
  - token offset inside the block
  - kv head and head dimension

### Backend extension: paged KV read/write

File: `crates/cellm-core/src/backend.rs`

- Added native `kv_write_token_f16(...)` / `kv_read_token_f16(...)` for long-term GPU backends (Metal/Vulkan) to avoid extra conversions and copies.
- Kept `kv_write_token_f32(...)` / `kv_read_token_f32(...)` as convenience APIs for CPU math and tests; these convert `f32` ↔ `f16` at the cache boundary.
- Later (Metal/Vulkan), a backend can override these to do device-side copies.

### Forward pass integration (Phase 2 behavior)

File: `crates/cellm-model/src/lib.rs`

- `forward(...)` now takes:
  - `&mut PageTable` (for the current session)
  - `&mut KVCache` (physical storage)
  - `&dyn Backend` (so backends can control KV IO later)
- For every incoming `(token, position)`:
  - it ensures the page table covers that position (allocates blocks as needed)
  - it writes deterministic per-layer K/V values into the correct `(block_id, offset)` slot
- The logits are still dummy until real attention kernels land; the goal here is correctness of paged storage and retrieval.

### Multi-turn correctness test

File: `crates/cellm-model/src/lib.rs`

- Added a unit test that:
  - pre-fills a “turn 1” prompt
  - appends more tokens for “turn 2”
  - reads back every token’s K/V across all layers and checks it matches the expected values

## Update: Minimal attention uses paged KV

File: `crates/cellm-model/src/lib.rs`

- `forward(...)` now also reads historical K/V through the `PageTable` and `KVCache` to run a small attention step (f32, naive loops).
- This proves “paged memory” is not just stored, but can be traversed and consumed to produce a next-token decision.

## CPU kernels (slice-based)

File: `crates/cellm-kernels/src/cpu_kernels.rs`

- Added small, testable CPU implementations for: RMSNorm, matmul, RoPE, softmax, and single-token GQA attention.
- The “tiny” forward path uses these kernels to compute Q/K/V and attention, then writes K/V into the paged cache.

## Engine wiring (early)

File: `crates/cellm-sdk/src/lib.rs`

- The engine now owns a single `KVCache` shared across sessions.
- Each session owns a `PageTable` and a `next_pos` cursor.
- `submit_tokens(...)` shows how prompts/turns extend the cache without losing old K/V.
- Added a minimal round-robin scheduler for decode steps so multiple sessions can advance one token at a time.
