Gemma4 mobile production profile (April 9, 2026)

- Model: `models/gemma-4-E2B-it-int4-aggr-v5.cellmd` (3.34 GB)
- Tokenizer: `models/gemma-4-E2B-it/tokenizer.json`
- Passes: `1`
- Gen tokens per prompt: `16`
- Backends: `cpu`

**Median performance by backend**
| Backend | Startup (s) | Prefill (s) | Prefill tok/s | Decode (s) | Decode tok/s | Pass Rate |
|---|---:|---:|---:|---:|---:|---:|
| `cpu` | 1.83 | 5.76 | 2.78 | 5.45 | 2.94 | 3/3 |

**Broad-phone readiness gates (host proxy check)**
| Backend | Prefill Target | Decode Target | Measured Prefill | Measured Decode | Verdict |
|---|---:|---:|---:|---:|---|
| `cpu` | <=15s | <=45s | 5.76s | 5.45s | **PASS** |

Raw CSV: `docs/benchmarks/runs/gemma4_mobile_profile_20260409_230513.csv`
