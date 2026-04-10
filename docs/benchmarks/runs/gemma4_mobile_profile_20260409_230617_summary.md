Gemma4 mobile production profile (April 9, 2026)

- Model: `models/gemma-4-E2B-it-int4-aggr-v5.cellmd` (3.34 GB)
- Tokenizer: `models/gemma-4-E2B-it/tokenizer.json`
- Passes: `1`
- Gen tokens per prompt: `8`
- Backends: `cpu`

**Median performance by backend**
| Backend | Startup (s) | Prefill (s) | Prefill tok/s | Decode (s) | Decode tok/s | Pass Rate |
|---|---:|---:|---:|---:|---:|---:|
| `cpu` | 2.01 | 5.36 | 2.24 | 2.92 | 2.74 | 3/3 |

**Broad-phone readiness gates (host proxy check)**
| Backend | Prefill Target | Decode Target | Measured Prefill | Measured Decode | Verdict |
|---|---:|---:|---:|---:|---|
| `cpu` | <=15s | <=45s | 5.36s | 2.92s | **PASS** |

Raw CSV: `docs/benchmarks/runs/gemma4_mobile_profile_20260409_230617.csv`
