# Paged KV Cache Foundation (Phase 1) — Plain-English Notes

This document explains, in simple terms, what we built to support a paged KV cache and how we verified it works.

## What problem are we solving?

During text generation, a transformer needs to remember “attention history” for every token you have already processed. That history is the **KV cache** (Keys and Values).

On a phone, the KV cache can become the biggest memory cost. We cannot assume we can store it as one huge, contiguous tensor forever. We need a way to:

- Allocate KV memory in **fixed-size pages/blocks**
- Keep multiple conversations (sessions) alive at once
- Grow and free cache memory incrementally

That is what “paged KV cache” means here.

## The basic idea

We split KV cache storage into equally-sized **blocks**:

- A block holds KV for `tokens_per_block` token positions.
- Each session has a “map” from token positions (0, 1, 2, …) to the block ids that contain their KV.

So instead of “KV for token 137 is at index 137 in one big tensor”, we do:

- “Token 137 lives in block X”
- “Inside block X it is offset Y”

## What we implemented

### 1) `BlockAllocator` (ids only)

File: `crates/cellm-cache/src/allocator.rs`

This is a small allocator that manages **block ids**:

- It starts with `0..total_blocks-1` in a free list.
- `alloc()` gives you a free block id.
- `free(id)` returns it to the free list.
- `alloc_n(n)` is **atomic**: if it cannot allocate all `n`, it allocates none and returns an error.

Important point: this allocator does **not** own the KV bytes. It only hands out ids.

### 2) `PageTable` (per-session mapping)

File: `crates/cellm-cache/src/pagetable.rs`

The `PageTable` is the session’s “address book”:

- `append_token()` grows the session by one token position.
  - If we crossed a block boundary, it asks the allocator for a new block id.
- `block_for_token(pos)` tells you which block holds token `pos`.
- `offset_in_block(pos)` tells you where inside the block that token lives.
- `free_all()` returns all blocks to the allocator (when a session ends).

### 3) `KVCache` (physical KV storage)

File: `crates/cellm-cache/src/kvcache.rs`

`KVCache` ties everything together:

- Owns a `BlockAllocator`
- Owns the actual KV buffers (`k` and `v`) in a single physical slab
- Exposes typed views (`KvCacheView` / `KvCacheReadView`) using the shared layout (`KvCacheLayout`)

So:

- `PageTable` decides *which* block + offset a token should use
- `KVCache` provides the bytes for *all* blocks

## How we validated it

We added unit tests in the same modules:

- `BlockAllocator` tests:
  - allocate/free roundtrip
  - exhaustion behavior
  - double-free and invalid-id errors
  - `alloc_n` atomicity
- `PageTable` tests:
  - blocks allocate only when needed
  - token → (block, offset) mapping correctness
  - freeing returns all blocks
  - out-of-blocks errors surface correctly

To run these tests:

```bash
cargo test -p cellm-cache
```

## Why this means we are ready for Phase 2

With the allocator and page table validated, we have stable “plumbing” for:

- Growing a session token-by-token without reallocating giant tensors
- Knowing exactly where each token’s KV should live
- Releasing memory cleanly when sessions end

So the system is now a stable foundation for Phase 2: integrating paged KV cache writes/reads into the model forward pass (real attention using the page table and cache storage).

