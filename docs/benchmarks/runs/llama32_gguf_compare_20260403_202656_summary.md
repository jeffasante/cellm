Llama-3.2 1B GGUF-derived benchmark (1 passes, host run on April 3, 2026; prompt=`"hello"`, `--gen 2`, report = median / P95):
| Model | Backend | Startup (s) | Prefill (s) | Decode (s) |
|---|---|---:|---:|---:|
| `llama32-1b-bf16-from-gguf.cellm` | `cpu` | `0.47 / 0.47` | `13.25 / 13.25` | `1.62 / 1.62` |
| `llama32-1b-bf16-q4k-from-gguf.cellm` | `cpu` | `0.53 / 0.53` | `15.18 / 15.18` | `1.59 / 1.59` |
| `llama32-1b-bf16-q6k-from-gguf.cellm` | `cpu` | `0.50 / 0.50` | `13.91 / 13.91` | `1.63 / 1.63` |

Raw CSV: `docs/benchmarks/runs/llama32_gguf_compare_20260403_202656.csv`
