// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! Bidirectional (encoder) forward pass for LFM2 embedding models.
//!
//! `LiquidAI/LFM2.5-Embedding-350M` ships as `Lfm2BidirectionalModel`, which is
//! the causal LFM2 backbone with two behavioural changes:
//!
//! 1. **Attention is non-causal.** Every token attends to every other token.
//! 2. **The short conv is centered, not causal.** `F.conv1d(..., padding=k//2)`
//!    so with `conv_L_cache = 3` each position sees `[t-1, t, t+1]`.
//!
//! Neither can be expressed through [`LfmRunner::step_inner`], which processes a
//! single position at a time against a rolling conv state and a KV cache — the
//! causality there is *structural*, not mask-driven. So this module implements a
//! parallel whole-sequence forward pass that materialises `[seq, hidden]`
//! activations, reusing the same weight-loading helpers and tensor names.
//!
//! Pooling is CLS (position 0, which the tokenizer fills with
//! `<|startoftext|>`), followed by L2 normalisation, matching the
//! `1_Pooling/config.json` shipped with the model.

use half::f16;
use rayon::prelude::*;

use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::{
    gemv_i8_w8a8, matmul_f16_f32, matmul_f32, rms_norm_f32,
};

use crate::lfm::LfmRunner;

/// Prompt prefix the model was trained to see in front of a search query.
pub const QUERY_PREFIX: &str = "query: ";
/// Prompt prefix the model was trained to see in front of an indexed passage.
pub const DOCUMENT_PREFIX: &str = "document: ";

/// A projection matrix hoisted out of the mmap so it can be shared across the
/// rayon workers that process each sequence position.
///
/// The encoder streams weights layer-by-layer rather than going through
/// `LfmRunner::linear_f16_out_in`, because that helper takes `&mut self` (for
/// the int4 LRU cache) and so cannot be called from inside a parallel loop.
enum Weight {
    F16(Vec<u16>),
    /// Per-row symmetric int8 with f16 scales.
    I8(Vec<i8>, Vec<u16>),
    /// Int4 dequantized up-front to dense f32.
    F32(Vec<f32>),
}

impl Weight {
    /// `out = W @ input`, with `W` of shape `[out_dim, in_dim]`.
    fn matvec(&self, input: &[f32], out_dim: usize, in_dim: usize, out: &mut [f32]) {
        match self {
            Weight::F16(w) => matmul_f16_f32(w, out_dim, in_dim, input, out),
            Weight::I8(w, s) => gemv_i8_w8a8(w, s, input, out, out_dim, in_dim),
            Weight::F32(w) => matmul_f32(w, out_dim, in_dim, input, 1, out),
        }
    }
}

impl LfmRunner {
    /// Load a projection matrix in whatever dtype it was stored as.
    fn enc_weight(
        &self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Weight, CoreError> {
        match self.tensor_dtype(name).as_deref() {
            Some("u32") => Ok(Weight::F32(self.weight_i4_dequant(name, out_dim, in_dim)?)),
            Some("i8") => {
                let (w, s) = self.weight_i8_owned(name)?;
                Ok(Weight::I8(w, s))
            }
            _ => Ok(Weight::F16(self.weight_f16_owned(name)?)),
        }
    }
}

impl LfmRunner {
    /// True when the checkpoint is an `Lfm2BidirectionalModel` embedding model.
    pub fn is_bidirectional(&self) -> bool {
        self.file()
            .header
            .source_text_config
            .as_ref()
            .and_then(|c| c.get("bidirectional"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || self
                .file()
                .header
                .source_architectures
                .as_ref()
                .map(|a| a.iter().any(|s| s == "Lfm2BidirectionalModel"))
                .unwrap_or(false)
    }

    /// Maximum sequence length the embedding model was trained for.
    pub fn max_encode_len(&self) -> usize {
        self.file()
            .header
            .source_text_config
            .as_ref()
            .and_then(|c| c.get("max_seq_length"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(512)
    }

    /// Run the bidirectional encoder over `tokens` and return the CLS-pooled,
    /// L2-normalised sentence embedding of length `hidden_size`.
    ///
    /// Tokens beyond [`max_encode_len`] are truncated, matching the reference
    /// SentenceTransformer configuration.
    pub fn embed_sequence(&mut self, tokens: &[u32]) -> Result<Vec<f32>, CoreError> {
        let states = self.encode_sequence(tokens)?;
        let hidden = self.hidden_size();
        // CLS pooling: position 0 holds <|startoftext|>.
        let mut vec = states[0..hidden].to_vec();
        l2_normalize(&mut vec);
        Ok(vec)
    }

    /// Run the bidirectional encoder and return all `[seq, hidden]` hidden
    /// states, row-major. Exposed for pooling strategies other than CLS.
    pub fn encode_sequence(&mut self, tokens: &[u32]) -> Result<Vec<f32>, CoreError> {
        if tokens.is_empty() {
            return Err(CoreError::Backend("lfm encode: empty token sequence".into()));
        }
        let max_len = self.max_encode_len();
        let tokens = if tokens.len() > max_len {
            &tokens[..max_len]
        } else {
            tokens
        };

        let cfg = self.config().clone();
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let attn_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = cfg.intermediate_size;
        let seq = tokens.len();
        let ks = self.conv_kernel_size();
        let eps = cfg.rms_norm_eps;

        // [seq, hidden] residual stream.
        let mut x = vec![0.0f32; seq * hidden];
        for (t, &tok) in tokens.iter().enumerate() {
            self.embed_token_hidden(tok, &mut x[t * hidden..(t + 1) * hidden])?;
        }

        debug_nonfinite(&x, usize::MAX, "embed");

        let mut x_norm = vec![0.0f32; seq * hidden];
        let mut block_out = vec![0.0f32; seq * hidden];

        for layer in 0..self.max_layers() {
            let layer_type = self.layer_type(layer).to_string();

            let norm_w = self.norm_weight_f32(&format!(
                "model.layers.{layer}.operator_norm.weight"
            ))?;
            rms_norm_rows(&x, &norm_w, eps, hidden, &mut x_norm);
            debug_nonfinite(&x_norm, layer, "opnorm");

            match layer_type.as_str() {
                "conv" => {
                    self.encode_conv_block(layer, seq, hidden, ks, &x_norm, &mut block_out)?
                }
                "full_attention" | "attention" => self.encode_attention_block(
                    layer, seq, hidden, n_heads, n_kv_heads, head_dim, attn_dim, kv_dim,
                    eps, cfg.rope_theta, &x_norm, &mut block_out,
                )?,
                other => {
                    return Err(CoreError::Backend(format!(
                        "lfm encode: unknown layer type '{other}' at layer {layer}"
                    )));
                }
            }

            for i in 0..seq * hidden {
                x[i] += block_out[i];
            }
            debug_nonfinite(&x, layer, layer_type.as_str());

            // SwiGLU feed-forward.
            let ffn_w = self.norm_weight_f32(&format!("model.layers.{layer}.ffn_norm.weight"))?;
            rms_norm_rows(&x, &ffn_w, eps, hidden, &mut x_norm);

            let w1 = self.enc_weight(
                &format!("model.layers.{layer}.feed_forward.w1.weight"),
                inter,
                hidden,
            )?;
            let w3 = self.enc_weight(
                &format!("model.layers.{layer}.feed_forward.w3.weight"),
                inter,
                hidden,
            )?;
            let w2 = self.enc_weight(
                &format!("model.layers.{layer}.feed_forward.w2.weight"),
                hidden,
                inter,
            )?;

            block_out
                .par_chunks_mut(hidden)
                .zip(x_norm.par_chunks(hidden))
                .for_each(|(out_row, in_row)| {
                    let mut gate = vec![0.0f32; inter];
                    let mut up = vec![0.0f32; inter];
                    w1.matvec(in_row, inter, hidden, &mut gate);
                    w3.matvec(in_row, inter, hidden, &mut up);
                    for i in 0..inter {
                        let g = gate[i];
                        gate[i] = g * (1.0 / (1.0 + (-g).exp())) * up[i];
                    }
                    w2.matvec(&gate, hidden, inter, out_row);
                });

            for i in 0..seq * hidden {
                x[i] += block_out[i];
            }
        }

        let final_w = self.norm_weight_f32("model.embedding_norm.weight")?;
        let mut out = vec![0.0f32; seq * hidden];
        rms_norm_rows(&x, &final_w, eps, hidden, &mut out);
        Ok(out)
    }

    /// Centered depthwise short-conv block over the whole sequence.
    ///
    /// Mirrors `_noncausal_shortconv_forward`: `in_proj` → split into `(B, C, x)`
    /// → `Bx = B * x` → depthwise `conv1d(padding = k / 2)` → `y = C * conv`
    /// → `out_proj`.
    fn encode_conv_block(
        &mut self,
        layer: usize,
        seq: usize,
        hidden: usize,
        ks: usize,
        x_norm: &[f32],
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        let in_proj = self.enc_weight(
            &format!("model.layers.{layer}.conv.in_proj.weight"),
            hidden * 3,
            hidden,
        )?;
        let out_proj = self.enc_weight(
            &format!("model.layers.{layer}.conv.out_proj.weight"),
            hidden,
            hidden,
        )?;
        // The depthwise kernel always stays f16 — it is tiny and quantizing it
        // costs accuracy for no size win.
        let conv_w = self.weight_f16_owned(&format!(
            "model.layers.{layer}.conv.conv.weight"
        ))?;
        let conv_f32: Vec<f32> = conv_w.iter().map(|&w| f16::from_bits(w).to_f32()).collect();

        // [seq, 3 * hidden]
        let mut bcx = vec![0.0f32; seq * hidden * 3];
        bcx.par_chunks_mut(hidden * 3)
            .zip(x_norm.par_chunks(hidden))
            .for_each(|(out_row, in_row)| {
                in_proj.matvec(in_row, hidden * 3, hidden, out_row);
            });

        // Bx laid out channel-major ([hidden, seq]) so the conv reads contiguously.
        let mut bx = vec![0.0f32; hidden * seq];
        for t in 0..seq {
            let row = &bcx[t * hidden * 3..(t + 1) * hidden * 3];
            for i in 0..hidden {
                bx[i * seq + t] = row[i] * row[2 * hidden + i];
            }
        }

        // Centered convolution: output t sums input positions t + k - pad.
        let pad = ks / 2;
        let mut conv_out = vec![0.0f32; hidden * seq];
        conv_out
            .par_chunks_mut(seq)
            .enumerate()
            .for_each(|(ch, out_ch)| {
                let in_ch = &bx[ch * seq..(ch + 1) * seq];
                let kernel = &conv_f32[ch * ks..(ch + 1) * ks];
                for t in 0..seq {
                    let mut acc = 0.0f32;
                    for k in 0..ks {
                        let src = t as isize + k as isize - pad as isize;
                        if src >= 0 && (src as usize) < seq {
                            acc += in_ch[src as usize] * kernel[k];
                        }
                    }
                    out_ch[t] = acc;
                }
            });

        // y = C * conv_out, then out_proj.
        let mut y = vec![0.0f32; seq * hidden];
        for t in 0..seq {
            let row = &bcx[t * hidden * 3..(t + 1) * hidden * 3];
            for i in 0..hidden {
                y[t * hidden + i] = row[hidden + i] * conv_out[i * seq + t];
            }
        }

        out.par_chunks_mut(hidden)
            .zip(y.par_chunks(hidden))
            .for_each(|(out_row, in_row)| {
                out_proj.matvec(in_row, hidden, hidden, out_row);
            });

        Ok(())
    }

    /// Full bidirectional GQA over the whole sequence (no KV cache, no mask).
    #[allow(clippy::too_many_arguments)]
    fn encode_attention_block(
        &mut self,
        layer: usize,
        seq: usize,
        hidden: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        attn_dim: usize,
        kv_dim: usize,
        eps: f32,
        rope_theta: f32,
        x_norm: &[f32],
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        let q_w = self.enc_weight(
            &format!("model.layers.{layer}.self_attn.q_proj.weight"),
            attn_dim,
            hidden,
        )?;
        let k_w = self.enc_weight(
            &format!("model.layers.{layer}.self_attn.k_proj.weight"),
            kv_dim,
            hidden,
        )?;
        let v_w = self.enc_weight(
            &format!("model.layers.{layer}.self_attn.v_proj.weight"),
            kv_dim,
            hidden,
        )?;
        let o_w = self.enc_weight(
            &format!("model.layers.{layer}.self_attn.out_proj.weight"),
            hidden,
            attn_dim,
        )?;

        let q_norm = self
            .norm_weight_f32(&format!("model.layers.{layer}.self_attn.q_layernorm.weight"))
            .ok();
        let k_norm = self
            .norm_weight_f32(&format!("model.layers.{layer}.self_attn.k_layernorm.weight"))
            .ok();

        let mut q = vec![0.0f32; seq * attn_dim];
        let mut k = vec![0.0f32; seq * kv_dim];
        let mut v = vec![0.0f32; seq * kv_dim];

        q.par_chunks_mut(attn_dim)
            .zip(k.par_chunks_mut(kv_dim))
            .zip(v.par_chunks_mut(kv_dim))
            .zip(x_norm.par_chunks(hidden))
            .enumerate()
            .for_each(|(pos, (((q_row, k_row), v_row), in_row))| {
                q_w.matvec(in_row, attn_dim, hidden, q_row);
                k_w.matvec(in_row, kv_dim, hidden, k_row);
                v_w.matvec(in_row, kv_dim, hidden, v_row);

                // Per-head QK RMSNorm (LFM2 specific).
                if let Some(w) = &q_norm {
                    normalize_heads(q_row, w, eps, n_heads, head_dim);
                }
                if let Some(w) = &k_norm {
                    normalize_heads(k_row, w, eps, n_kv_heads, head_dim);
                }

                cellm_kernels::cpu_kernels::rope_non_interleaved_inplace_f32(
                    q_row, n_heads, head_dim, head_dim, pos, rope_theta,
                );
                cellm_kernels::cpu_kernels::rope_non_interleaved_inplace_f32(
                    k_row, n_kv_heads, head_dim, head_dim, pos, rope_theta,
                );
            });

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let heads_per_kv = n_heads / n_kv_heads.max(1);
        let mut attn_out = vec![0.0f32; seq * attn_dim];

        attn_out
            .par_chunks_mut(attn_dim)
            .enumerate()
            .for_each(|(t, out_row)| {
                let mut scores = vec![0.0f32; seq];
                for h in 0..n_heads {
                    let kv_h = h / heads_per_kv;
                    let q_head = &q[t * attn_dim + h * head_dim..t * attn_dim + (h + 1) * head_dim];

                    // Bidirectional: every position attends to every position.
                    let mut max_score = f32::NEG_INFINITY;
                    for s in 0..seq {
                        let k_head = &k[s * kv_dim + kv_h * head_dim
                            ..s * kv_dim + (kv_h + 1) * head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_head[d] * k_head[d];
                        }
                        let sc = dot * scale;
                        scores[s] = sc;
                        if sc > max_score {
                            max_score = sc;
                        }
                    }

                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max_score).exp();
                        sum += *s;
                    }
                    let inv = 1.0f32 / sum.max(1e-20);

                    let dst = &mut out_row[h * head_dim..(h + 1) * head_dim];
                    dst.fill(0.0);
                    for s in 0..seq {
                        let w = scores[s] * inv;
                        let v_head = &v[s * kv_dim + kv_h * head_dim
                            ..s * kv_dim + (kv_h + 1) * head_dim];
                        for d in 0..head_dim {
                            dst[d] += w * v_head[d];
                        }
                    }
                }
            });

        out.par_chunks_mut(hidden)
            .zip(attn_out.par_chunks(attn_dim))
            .for_each(|(out_row, in_row)| {
                o_w.matvec(in_row, hidden, attn_dim, out_row);
            });

        Ok(())
    }
}

fn normalize_heads(x: &mut [f32], weight: &[f32], eps: f32, n_heads: usize, head_dim: usize) {
    let mut tmp = vec![0.0f32; head_dim];
    for h in 0..n_heads {
        let s = h * head_dim;
        rms_norm_f32(&x[s..s + head_dim], weight, eps, &mut tmp);
        x[s..s + head_dim].copy_from_slice(&tmp);
    }
}

fn debug_nonfinite(x: &[f32], layer: usize, kind: &str) {
    if std::env::var_os("CELLM_ENCODER_DEBUG").is_none() {
        return;
    }
    let bad = x.iter().filter(|v| !v.is_finite()).count();
    let absmax = x.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("  layer {layer:2} {kind:14} nonfinite={bad:6} absmax={absmax:.4}");
}

fn rms_norm_rows(x: &[f32], weight: &[f32], eps: f32, hidden: usize, out: &mut [f32]) {
    out.par_chunks_mut(hidden)
        .zip(x.par_chunks(hidden))
        .for_each(|(out_row, in_row)| {
            rms_norm_f32(in_row, weight, eps, out_row);
        });
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}
