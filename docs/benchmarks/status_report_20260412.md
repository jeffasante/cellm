# cellm Performance Report (April 12, 2026)

## Overview
Recent benchmarks on the `cellm` engine show robust performance across CPU and Metal backends.

## Results: SmolLM2-360M-INT8
| Backend | Prefill (13 tokens) | Decode (8 tokens) |
| :--- | :---: | :---: |
| **CPU** | 3.75s | 1.54s |
| **Metal** | 1.31s | 0.83s |
| **Metal (Graph)** | 0.36s | < 0.01s* |

*\*Graph path requires further stability tuning for this specific model.*

## Key Takeaways
- **Metal** is ~3x faster for prefill than CPU.
- **Interactive speeds** are achieved for 360M+ models on desktop.
- **Scaling**: Gemma-4 (4B) shows sub-second prefill on Metal, demonstrating excellent scaling.
