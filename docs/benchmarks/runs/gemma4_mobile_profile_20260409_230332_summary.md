Gemma4 mobile production profile (April 9, 2026)

- Model: `models/gemma-4-E2B-it-int4-aggr-v5.cellmd` (3.34 GB)
- Tokenizer: `models/gemma-4-E2B-it/tokenizer.json`
- Passes: `1`
- Gen tokens per prompt: `32`
- Backends: `cpu metal`

**Median performance by backend**
| Backend | Startup (s) | Prefill (s) | Prefill tok/s | Decode (s) | Decode tok/s | Pass Rate |
|---|---:|---:|---:|---:|---:|---:|
| `cpu` | 1.85 | 5.59 | 2.68 | 11.47 | 2.79 | 3/3 |
| `metal` | n/a | n/a | n/a | n/a | n/a | 0/3 |

**Broad-phone readiness gates (host proxy check)**
| Backend | Prefill Target | Decode Target | Measured Prefill | Measured Decode | Verdict |
|---|---:|---:|---:|---:|---|
| `cpu` | <=15s | <=45s | 5.59s | 11.47s | **PASS** |
| `metal` | <=15s | <=45s | n/as | n/as | **FAIL** |

Raw CSV: `docs/benchmarks/runs/gemma4_mobile_profile_20260409_230332.csv`
