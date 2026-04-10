Gemma4 mobile production profile (April 9, 2026)

- Model: `models/gemma-4-E2B-it-int4-aggr-v5.cellmd` (3.34 GB)
- Tokenizer: `models/gemma-4-E2B-it/tokenizer.json`
- Passes: `1`
- Gen tokens per prompt: `16`
- Backends: `metal`

**Median performance by backend**
| Backend | Startup (s) | Prefill (s) | Prefill tok/s | Decode (s) | Decode tok/s | Pass Rate |
|---|---:|---:|---:|---:|---:|---:|
| `metal` | 2.10 | 10.38 | 1.23 | 6.94 | 2.31 | 3/3 |

**Broad-phone readiness gates (host proxy check)**
| Backend | Prefill Target | Decode Target | Measured Prefill | Measured Decode | Verdict |
|---|---:|---:|---:|---:|---|
| `metal` | <=15s | <=45s | 10.38s | 6.94s | **PASS** |

Raw CSV: `docs/benchmarks/runs/gemma4_mobile_profile_20260409_230904.csv`
