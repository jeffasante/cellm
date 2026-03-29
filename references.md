# cellm — Design References

This file contains links and documentation for the algorithms and papers that influenced the design of **cellm**.

## Pseudorandom Number Generation (PRNG)

`cellm` uses simple, fast PRNGs for sampling during LLM decoding.

- **[Linear Congruential Generator (LCG)](https://en.wikipedia.org/wiki/Linear_congruential_generator)** — A classic iterative algorithm for generating a sequence of pseudorandom numbers. While modern engines often use more robust generators, LCG provides a simple reference for the "seed-and-iterate" pattern.
- **Xorshift** — The actual algorithm used in `cellm-sdk` (XorShift64) for multi-session sampling. It is significantly faster and has better statistical properties for most inference use cases.

## Core Architecture

- **[PagedAttention: Memory-Efficient LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)** (vLLM) — The inspiration for `cellm`'s `BlockAllocator` and `PageTable` memory models.
- **[Llama 2 / Llama 3](https://ai.meta.com/llama/)** — The primary architecture family supported by the native runner.
- **[SigLIP: Sigmoid Loss for Language-Image Pre-training](https://arxiv.org/abs/2303.15343)** — The vision encoder backbone referenced by SmolVLM.
