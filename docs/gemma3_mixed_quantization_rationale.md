# Why We Made Gemma-3 Mixed Quantization

## Context
We tested multiple quantization strategies for `gemma-3-1b-it` in `cellm` and found a clear size-vs-quality tradeoff:

- `int8` (`gemma-3-1b-it-int8-v1.cellm`): good quality, larger size (~1.2 GB)
- `int4` (`gemma-3-1b-it-int4-v1.cellm`): small size (~481 MB), poor output quality

The pure `int4` variant was too aggressive for stable generation quality in practical prompts.

## Why Mixed Was Introduced
The mixed approach was created to keep quality-critical parts in higher precision while still reducing model size:

- Keep embeddings / attention path higher precision (f16)
- Quantize MLP-heavy weights to int4

This gives a better middle ground:

- `mixed-int4` (`gemma-3-1b-it-mixed-int4-v1.cellm`): ~1.0 GB
- Quality notably better than full int4 and acceptable for normal text prompts

## Core Idea
Not all tensors contribute equally to generation stability.

- Aggressively quantizing everything can introduce compounding errors in token prediction.
- MLP quantization gives substantial size savings.
- Preserving sensitive paths helps maintain coherent output.

## Decision Flow

```mermaid
flowchart TD
    A[Start: Gemma-3 1B HF model] --> B{Quantization strategy}
    B --> C[All INT8]
    B --> D[All INT4]
    B --> E[Mixed: MLP INT4 + sensitive paths F16]

    C --> C1[Size: ~1.2 GB]
    C --> C2[Quality: Good]

    D --> D1[Size: ~481 MB]
    D --> D2[Quality: Poor / unstable]

    E --> E1[Size: ~1.0 GB]
    E --> E2[Quality: Better than full INT4]

    C2 --> F[Production-safe baseline]
    E2 --> G[Balanced alternative]
    D2 --> H[Experimental only]
```

## Model Pipeline (Mixed)

```mermaid
flowchart LR
    A[HF Gemma-3 1B checkpoint] --> B[convert tool]
    B --> C[Tensor selection rules]
    C --> D[MLP weights -> INT4]
    C --> E[Attention/embedding critical path -> F16]
    D --> F[.cellm writer]
    E --> F
    F --> G[gemma-3-1b-it-mixed-int4-v1.cellm]
    G --> H[infer runtime]
    H --> I[Coherent text output]
```

## Practical Recommendation

- Use `int8` for highest reliability.
- Use `mixed-int4` when you need smaller size with reasonable quality.
- Avoid full `int4` for user-facing quality-sensitive generation.

## Related Artifacts
- `models/gemma-3-1b-it-int8-v1.cellm`
- `models/gemma-3-1b-it-int4-v1.cellm`
- `models/gemma-3-1b-it-mixed-int4-v1.cellm`
