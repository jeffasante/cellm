# Function Calling On-Device: FunctionGemma-270M in cellm

**4 August 2026**

A phone assistant that turns "turn off wifi and check the weather in London in
fahrenheit" into two structured API calls does not need a datacenter. The model
that does it here is 270M parameters and runs on CPU.

This is the story of converting `functiongemma-270m` to `.cellm`, making prefill
3× faster, and finding out — the hard way — that a five-prompt test set will
happily tell you a broken model is perfect.

Models: [`jeffasante/cellm-models`](https://huggingface.co/jeffasante/cellm-models/tree/main/function-calls/functiongemma-270m-cellm)

## What the model does

You declare the functions your app exposes. The model emits calls against them.

```
Turn off wifi and check the weather in London in fahrenheit

  call:set_wifi{enabled:false}
  call:get_weather{city:London,unit:f}
```

It chains multiple calls in one turn, fills in enum-typed arguments (`c` vs `f`),
and — usually — declines when nothing fits:

```
Who was the first president of Ghana?

  I apologize, but I cannot assist with historical inquiries about past
  presidents of countries. My current capabilities are focused on managing
  Wi-Fi and weather, setting alarms, sending messages, and ...
```

Architecture is `gemma3_text`: 18 layers, hidden 640, 4 query heads over 1 KV
head, head_dim 256, vocab 262144.

## Where the weight actually is

The first surprise. For a 270M-parameter model, the parameter count is not where
you think it is:

| Component | Size (f16) | Share |
| --- | --- | --- |
| `embed_tokens` | 320.0 MB | 62.6% |
| MLP | 135.0 MB | 26.4% |
| Attention | 56.2 MB | 11.0% |
| **Total** | **511.4 MB** | |

A 262144-token vocabulary at hidden 640 is 168M parameters — more than half the
model — sitting in one table. And because Gemma ties `lm_head` to
`embed_tokens`, that same 320 MB is also the output projection.

That single fact drives everything below: both the biggest performance win and
the entire quantization strategy.

## Prefill was doing 400× more work than it needed to

The first conversion ran, produced correct output, and took 26.5 seconds to
answer. Decode was fine. Prefill was pathological.

The prompt is ~400 tokens — the function declarations dominate it. For each of
those 400 positions, the runtime was computing the full logit vector:

```rust
let k = top_k.max(1).min(vocab);
```

That `.max(1)` is the bug. A caller asking for zero logits gets one anyway, and
"one logit" still means running the 640 × 262144 `lm_head` matmul to find it.
Across 400 prefill positions that is 400 passes over the largest tensor in the
model, 399 of which are discarded — only the last position's distribution is
ever sampled.

The fix is to let zero mean zero, and return before touching `lm_head`:

```rust
if top_k == 0 {
    return Ok(Vec::new());
}
let vocab = cfg.vocab_size;
let k = top_k.max(1).min(vocab);
```

and to ask for nothing at every position but the last:

```rust
Runner::Gemma(r) => {
    // Only the last prompt token's logits are sampled, so ask for
    // top_k=0 elsewhere and skip the lm_head matmul per position.
    let want = if i + 1 == prompt_tokens.len() { args.top_k } else { 0 };
    r.step_topk(tok, i, &mut page_table, &mut kv_cache, want)?
}
```

**26.51 s → 8.17 s**, with bit-identical output. The KV cache is still populated
for every position; only the discarded projection goes away.

This has a consequence that matters later: once `lm_head` is out of the prefill
loop, the embedding table is barely touched during prefill. Shrinking it stops
buying speed.

## The embedding tolerates 4 bits; the weights do not

Naive approach: quantize everything to int4, get a 4× reduction, ship it.

It produces fluent, confident, completely wrong output:

```
Turn on wifi
  I am sorry, but I cannot assist with this request.
```

No garbage tokens, no crash — the model just stops being able to do its job. It
scored 0/5.

So instead of quantizing everything at once, quantize one thing at a time:

| Variant | Size | Score |
| --- | --- | --- |
| int4 g32 weights, f16 embedding | 392 MB | **0/5** — fluent but refuses everything |
| f16 weights, int4 g32 embedding | 295 MB | **5/5** |
| f16 weights, int2 g32 embedding | 253 MB | **0/5** — `<start_function_call><escape><escape>...` |
| int8 weights, int4 g32 embedding | 195 MB | **5/5** |

The asymmetry is the finding. The embedding survives 4-bit fine. The linear
weights do not survive it at all.

That makes sense in hindsight: an embedding lookup is a *single* read whose
error enters the residual stream once, while a linear weight participates in a
640-element dot product where errors accumulate, then feeds 18 layers of
compounding. Same bit-width, very different blast radius.

Hence the shipped recipe: **int8 everywhere, int4 only on the embedding**.

## A bug that made the error metric lie

Group-wise quantization stores one scale per 32 weights instead of one per row,
which lowers reconstruction error. Enabling it made the output *worse* — full
`<unused43><unused43><unused43>` garbage — while the measured error went
**down**, 0.31 → 0.12.

That contradiction is the whole diagnosis. An error metric improving while
output degrades cannot be a quality tradeoff; something is reading the weights
wrong. The `lm_head` CPU path had:

```rust
let scale = f16::from_bits(scales[vid]).to_f32();
```

One scale per row `vid`. With grouped scales there are 20 scales per row, so
this reads scale #0 of row 0 for everything after the first row — arbitrary
numbers applied to arbitrary weights. The fix threads the group stride through:

```rust
let spr = (scales.len() / vocab).max(1);
let rs = &scales[vid * spr..(vid + 1) * spr];
dot = dot_i4_grouped_row(row, x_final.as_slice(), rs, hidden / spr);
```

with the group index computed per element:

```rust
let scale = f16::from_bits(scales[(xi / gs).min(scales.len() - 1)]).to_f32();
```

Both the parallel and sequential branches had the same hardcoded lookup. Fixing
one and testing would have shown no change on machines that took the other path.

## Why sub-100 MB is out of reach

The goal was under 100 MB. It is not achievable with this vocabulary, and the
reason is worth stating precisely because it is *not* a tuning failure.

Getting to 100 MB means the 168M-parameter embedding at roughly 2 bits. But int2
here is codebook-limited, not scale-limited. With a fixed codebook of
`{-1.5, -0.5, 0.5, 1.5}`, only the scale is free, and tightening the grouping
barely moves relative error on `embed_tokens`:

| Grouping | Relative error |
| --- | --- |
| per-row | 0.471 |
| g128 | 0.462 |
| g64 | 0.451 |
| g32 | 0.431 |

Going from one scale per row to one per 32 weights — a 20× increase in scale
storage — buys 8%. The error is in the four available levels, not in how they're
scaled. No amount of grouping fixes that; it needs a learned codebook.

The int4 quantizer, meanwhile, is already near-optimal. Comparing the cheap
`amax/7` scale against an exhaustively searched one:

| Tensor | `amax/7` | best searched |
| --- | --- | --- |
| `mlp.up_proj` g32 | 0.1008 | 0.0967 |
| `q_proj` g32 | 0.1049 | 0.1008 |

4% available from perfect scale selection. The remaining ~10% is inherent to
4 bits. The packing was verified correct by round-tripping through the runtime's
own unpack semantics in numpy — i4 relative error 0.12303, i2 0.35297, both
matching theory, ruling out a packing bug.

## The benchmark that overturned the recommendation

The 5-prompt results above said the 186 MB build was perfect: 5/5, byte-identical
to f16. That conclusion was wrong, and it was wrong for a boring reason — the
first five prompts are the easy ones. Single calls, unambiguous verbs.

Sixteen prompts, greedy decoding, CPU, measured two ways:

- **vs HF ref** — exact token match against the original `transformers` model.
  End-to-end correctness.
- **vs f16** — exact match against our own f16 `.cellm` build. Isolates
  quantization damage from behaviour the base model already had.

| Build | Size | vs HF ref | vs f16 | Avg prefill |
| --- | --- | --- | --- | --- |
| f16 | 511 MB | 12/16 | 16/16 | 4.19 s |
| int8 | 416 MB | 11/16 | 12/16 | 2.95 s |
| **int8e** | **257 MB** | **11/16** | **11/16** | **2.90 s** |
| int8-e4g32 | 186 MB | 9/16 | 9/16 | 2.94 s |

Two things fall out.

**The 186 MB build has real regressions.** It drops one of two required calls on
compound requests:

```
Turn on wifi and set brightness to 50
  f16:        call:set_wifi{enabled:true} + call:set_brightness{level:50}
  int8-e4g32: call:set_wifi{enabled:true}                    <- brightness lost
```

Single calls still work. It's the second call that goes. So the recommendation
changed to **int8e at 257 MB** — 71 MB larger, two more prompts right, and no
speed penalty.

**Below int8, shrinking buys nothing.** 257 MB and 186 MB prefill at 2.90 s and
2.94 s — indistinguishable. This is the prefill fix coming back around: with
`lm_head` skipped at 399 of 400 positions, the embedding is the one thing those
two builds differ in and the one thing prefill no longer reads. Smaller file,
same clock.

## What the f16 model gets wrong

The f16 baseline scores 12/16 against the reference, so four failures are
inherited from the base model and have nothing to do with quantization. Reading
"11/16" as "five quantization regressions" would be wrong by four.

```
Wake me up at 6:30 tomorrow
  call:set_alarm{time:14:30}          <- wrong time, every build including f16

Text Ama that I'm running late
  call:send_message{..., recipient:person@example.com}   <- invents a recipient

Tell me a joke about cats
  call:play_music{query:cat}          <- reaches for a tool it shouldn't
```

The 6:30 → 14:30 case is the useful one to sit with. Token-level parity with the
reference implementation is a *conversion* correctness check. It says the runtime
reproduces the model. It says nothing about whether the model is right, and here
it is confidently, reproducibly wrong about what time to set an alarm.

## Running it

```bash
infer \
  --model functiongemma-270m-int8e.cellm \
  --tokenizer tokenizer.json \
  --prompt "$PROMPT" \
  --gen 64 --temperature 0 \
  --stop-tokens 1,50,106
```

`--stop-tokens` is required. The converter writes `eos_token_id: 106` from the
config, but the model actually stops on 50; without the explicit list, generation
runs past the closing `<end_function_call>` and keeps inventing calls.

The prompt must use the Gemma chat template with declarations wrapped in
`<start_function_declaration>` / `<end_function_declaration>` before the
`<start_of_turn>user` turn.

**CPU only for grouped builds.** The Metal i4 path still passes `hidden` as the
group size, so `int8-e4g32` produces wrong results on GPU. `int8e` and `f16` are
unaffected and run on both.

## Takeaways

- **In small models with large vocabularies, the embedding is the model.** 63% of
  the bytes here. Both the prefill win and the quantization strategy came from
  that one table.
- **Quantize one component at a time.** The all-int4 model failed, and the reason
  — weights, not the embedding — is invisible unless the variables are separated.
- **An error metric moving opposite to output quality means a bug, not a
  tradeoff.** That contradiction found the grouped-scale bug faster than any
  amount of staring at output would have.
- **Easy prompts confirm what you hoped.** 5/5 became 9/16 on a set that included
  compound requests. Test sets that never fail aren't measuring anything.
- **Token parity is not correctness.** Every build reproduces `6:30 → 14:30`
  faithfully. Perfect fidelity to a wrong answer is still a wrong answer.
