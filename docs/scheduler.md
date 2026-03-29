# cellm Scheduler Design

The scheduler manages KV-cache pages and handles incoming request streams.

## KV Block Allocator
- Page size: 16 tokens
- Policy: LRU with pre-allocation

## Thermal Policy
Dynamically adjusts batch size based on device temperature to prevent throttling.
