Llama-3.2 1B GGUF-derived benchmark (1 passes, host run on April 3, 2026; prompt=`"hello"`, `--gen 2`, report = median / P95):
| Model | Backend | Startup (s) | Prefill (s) | Decode (s) |
|---|---|---:|---:|---:|
| `llama32-1b-bf16-from-gguf.cellm` | `metal` | `0.55 / 0.55` | `25.38 / 25.38` | `8.61 / 8.61` |
| `llama32-1b-bf16-q4k-from-gguf.cellm` | `metal` | `0.52 / 0.52` | `25.32 / 25.32` | `18.61 / 18.61` |
| `llama32-1b-bf16-q6k-from-gguf.cellm` | `metal` | `0.52 / 0.52` | `22.09 / 22.09` | `19.22 / 19.22` |

Raw CSV: `docs/benchmarks/runs/llama32_gguf_compare_20260403_202845.csv`
