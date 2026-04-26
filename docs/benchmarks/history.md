# Benchmark Run History

Below is a record of automated profiling runs and backend comparisons performed during development. Each run includes a detailed breakdown of prefill/decode latency, memory usage, and numerical parity checks.

## Gemma 4 Mobile Profiles
Detailed performance profiling for Gemma 4 models on mobile-class hardware.

*   [Profile: 2026-04-09 23:09:04](runs/gemma4_mobile_profile_20260409_230904_summary.md)
*   [Profile: 2026-04-09 23:06:17](runs/gemma4_mobile_profile_20260409_230617_summary.md)
*   [Profile: 2026-04-09 23:05:13](runs/gemma4_mobile_profile_20260409_230513_summary.md)
*   [Profile: 2026-04-09 23:03:32](runs/gemma4_mobile_profile_20260409_230332_summary.md)

## Llama 3.2 GGUF Comparisons
Cross-validation of cellm's execution engine against reference GGUF implementations.

*   [Comparison: 2026-04-03 20:28:45](runs/llama32_gguf_compare_20260403_202845_summary.md)
*   [Comparison: 2026-04-03 20:26:56](runs/llama32_gguf_compare_20260403_202656_summary.md)

## How to read these reports
- **Prefill**: The time taken to process the input prompt.
- **Decode**: The tokens-per-second (TPS) rate during generation.
- **Backend**: Indicates whether the run was executed on CPU or Apple Silicon (Metal).
