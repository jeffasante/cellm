# Qwen3.5 “Linear Attention” (DeltaNet) support in cellm (layman notes)

Qwen3.5 models can mix two different layer types:

- **full_attention**: “normal” transformer attention with a KV cache
- **linear_attention**: a recurrent layer often called **DeltaNet / Gated Delta Rule**

This repo’s `.cellm` format can store the weights for both kinds of layers. The work in the codebase makes the runtime able to *step token-by-token* through both.

## What we implemented (high level)

For Qwen3.5 `linear_attention` layers we implemented the same conceptual pipeline used by the reference PyTorch code:

1) **Project the token embedding** into a few internal vectors:
   - Q/K/V (the parts that drive the recurrent “attention-like” update)
   - `z` (a gating vector used at the output)
   - `a` and `b` (small per-head values that control the decay and update strength)

2) **Run a small causal 1D convolution** over the Q/K/V channels.
   - This is a lightweight “local mixing” step.
   - It needs a small rolling buffer per layer to remember the last few inputs.

3) **Run the “gated delta rule” recurrent update**.
   - Instead of storing every past token’s K/V (like normal attention), this layer keeps a **compact per-head state matrix**.
   - Each new token updates the state and produces the layer’s output for that token.

4) **Apply a gated RMSNorm** using the `z` gate.

5) **Project back to hidden size** and add the residual, then run the normal MLP part of the transformer block.

## What state we cache per session

Unlike full attention (which uses the paged KV cache), DeltaNet needs two small “memories” per layer:

- **Convolution state**: remembers the last `kernel_size` inputs for the depthwise conv.
- **Recurrent state**: the compact matrix that summarizes all past tokens for the delta-rule update.

In `cellm`, this state is tracked per session id (the same id used by `PageTable`).

## Where the code lives

- Qwen runner: `/Users/cellm/crates/cellm-model/src/qwen.rs`

## Why this matters

If you skip the `linear_attention` layers, Qwen3.5 quality collapses because most layers are not “normal attention” layers.
Getting DeltaNet working is what turns “it runs” into “it behaves like a real Qwen checkpoint”.

