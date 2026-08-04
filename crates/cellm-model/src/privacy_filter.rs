// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! Token-classification forward pass for `openai/privacy-filter`.
//!
//! An 8-layer bidirectional MoE encoder that tags PII spans. It differs from
//! every other runner in this crate in ways that rule out reusing them:
//!
//! 1. **Encoder, not decoder.** Attention is bidirectional over a symmetric
//!    sliding window (`|q - kv| <= sliding_window`), so there is no KV cache
//!    and no single-token step; the whole sequence is materialised at once.
//! 2. **Attention sinks.** A per-head learned logit is appended to each row
//!    before the softmax and dropped after, letting a head attend to nothing.
//! 3. **Scale on Q and K separately.** `head_dim^-0.25` is applied to both
//!    tensors rather than once to their product.
//! 4. **Interleaved YaRN RoPE** with half-width cos/sin.
//! 5. **128-expert top-4 MoE** with a clamped SwiGLU (`(up + 1) * glu`).
//! 6. **Group-32 int4 weights with f16 scales/biases.** [`crate::lfm`]'s
//!    dequantizer hardcodes group 64 and f32 sidecars, so it cannot be used.
//!
//! The output is per-token logits over 33 BIOES labels, decoded into
//! character spans by [`PrivacyFilterRunner::spans`].

use std::collections::HashMap;
use std::path::Path;

use half::f16;
use rayon::prelude::*;

use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::rms_norm_f32;

use crate::cellm_file::CellmFile;

/// A dequantized `[out_dim, in_dim]` projection.
struct Matrix {
    data: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl Matrix {
    /// `out[t] = x[t] @ W^T + bias`, over `rows` rows of `x`.
    fn matmul(&self, x: &[f32], rows: usize, bias: Option<&[f32]>, out: &mut [f32]) {
        let (k, n) = (self.in_dim, self.out_dim);
        out.par_chunks_exact_mut(n)
            .enumerate()
            .take(rows)
            .for_each(|(t, o)| {
                let xr = &x[t * k..(t + 1) * k];
                for j in 0..n {
                    let w = &self.data[j * k..(j + 1) * k];
                    let mut acc = bias.map_or(0.0, |b| b[j]);
                    for i in 0..k {
                        acc += xr[i] * w[i];
                    }
                    o[j] = acc;
                }
            });
    }
}

pub struct PrivacyFilterConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub group_size: usize,
    pub quant_bits: usize,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub rope_factor: f32,
    pub rope_orig_max: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub rope_truncate: bool,
}

pub struct PrivacyFilterRunner {
    file: CellmFile,
    cfg: PrivacyFilterConfig,
    inv_freq: Vec<f32>,
    attn_scaling: f32,
    id2label: Vec<String>,
}

fn cfg_f32(tc: &serde_json::Value, key: &str, default: f32) -> f32 {
    tc.get(key).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(default)
}

fn cfg_usize(tc: &serde_json::Value, key: &str, default: usize) -> usize {
    tc.get(key).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(default)
}

impl PrivacyFilterRunner {
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let file = CellmFile::load(path)?;
        let h = &file.header;
        let tc = h
            .source_text_config
            .clone()
            .unwrap_or(serde_json::Value::Object(Default::default()));

        // A wrong group size or bit width silently corrupts every expert, so
        // never guess: older int4 files predate `quant_bits` and imply 4.
        let quant = tc.get("quant").and_then(|v| v.as_str()).unwrap_or("none");
        let packed_quant = matches!(quant, "int4" | "int3" | "int2");
        let group_size = if packed_quant { cfg_usize(&tc, "quant_group_size", 0) } else { 0 };
        let quant_bits = if packed_quant {
            cfg_usize(&tc, "quant_bits", if quant == "int4" { 4 } else { 0 })
        } else {
            0
        };
        if packed_quant && (group_size == 0 || quant_bits == 0) {
            return Err(CoreError::Backend(format!(
                "privacy-filter: {quant} file has no quant_group_size/quant_bits in header"
            )));
        }

        let cfg = PrivacyFilterConfig {
            hidden: h.hidden_dim,
            n_layers: h.num_layers,
            n_heads: h.num_heads,
            n_kv_heads: h.num_kv_heads,
            head_dim: h.head_dim.unwrap_or(h.hidden_dim / h.num_heads),
            inter: h.moe_intermediate_size.unwrap_or(h.intermediate_size),
            n_experts: h.n_routed_experts.unwrap_or(0),
            top_k: h.num_experts_per_tok.unwrap_or(4),
            eps: h.rms_norm_eps,
            rope_theta: h.rope_theta,
            sliding_window: cfg_usize(&tc, "sliding_window", 128),
            group_size: if group_size == 0 { 64 } else { group_size },
            quant_bits: if quant_bits == 0 { 4 } else { quant_bits },
            swiglu_alpha: cfg_f32(&tc, "swiglu_alpha", 1.702),
            swiglu_limit: cfg_f32(&tc, "swiglu_limit", 7.0),
            rope_factor: h.rope_scaling_factor.unwrap_or(1.0),
            rope_orig_max: h
                .rope_scaling_original_max_position_embeddings
                .unwrap_or(4096),
            beta_fast: cfg_f32(&tc, "beta_fast", 32.0),
            beta_slow: cfg_f32(&tc, "beta_slow", 1.0),
            rope_truncate: tc
                .get("rope_scaling")
                .and_then(|r| r.get("truncate"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        };

        let n_labels = cfg_usize(&tc, "num_labels", 33);
        let mut id2label = vec![String::new(); n_labels];
        if let Some(map) = tc.get("id2label").and_then(|v| v.as_object()) {
            for (k, v) in map {
                if let (Ok(i), Some(s)) = (k.parse::<usize>(), v.as_str()) {
                    if i < n_labels {
                        id2label[i] = s.to_string();
                    }
                }
            }
        }

        let (inv_freq, attn_scaling) = yarn_inv_freq(&cfg);
        Ok(Self { file, cfg, inv_freq, attn_scaling, id2label })
    }

    pub fn config(&self) -> &PrivacyFilterConfig {
        &self.cfg
    }

    pub fn id2label(&self) -> &[String] {
        &self.id2label
    }

    fn f32_tensor(&self, name: &str) -> Result<Vec<f32>, CoreError> {
        let idx = self
            .file
            .tensor_index(name)
            .ok_or_else(|| CoreError::Backend(format!("privacy-filter: missing {name}")))?;
        let dtype = idx.dtype.clone();
        let bytes = self.file.tensor_bytes(name)?;
        Ok(match dtype.as_str() {
            "f32" => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
            "f16" => bytemuck::cast_slice::<u8, u16>(bytes)
                .iter()
                .map(|&b| f16::from_bits(b).to_f32())
                .collect(),
            other => {
                return Err(CoreError::Backend(format!(
                    "privacy-filter: {name} has unexpected dtype {other}"
                )))
            }
        })
    }

    /// Dequantize a stored `[out_dim, in_dim]` matrix, whatever its dtype.
    fn matrix(&self, name: &str, out_dim: usize, in_dim: usize) -> Result<Matrix, CoreError> {
        let dtype = self
            .file
            .tensor_index(name)
            .map(|t| t.dtype.clone())
            .ok_or_else(|| CoreError::Backend(format!("privacy-filter: missing {name}")))?;
        let data = match dtype.as_str() {
            "u32" => self.dequant_int4(name, out_dim, in_dim)?,
            "i8" => {
                let q = self.file.tensor_bytes(name)?;
                let scales = self.f32_tensor(&format!("{name}.scales"))?;
                let q: &[i8] = bytemuck::cast_slice(q);
                let mut out = vec![0.0f32; out_dim * in_dim];
                out.par_chunks_exact_mut(in_dim).enumerate().for_each(|(r, row)| {
                    let s = scales[r];
                    for c in 0..in_dim {
                        row[c] = q[r * in_dim + c] as f32 * s;
                    }
                });
                out
            }
            _ => self.f32_tensor(name)?,
        };
        Ok(Matrix { data, out_dim, in_dim })
    }

    /// MLX-style affine quant: `32 / bits` values per u32, per-group scale/bias.
    ///
    /// Unlike [`crate::lfm`] this reads the group size from the header and
    /// accepts f16 sidecars, which is what makes the 943 MB build possible.
    /// At 3 bits the top 2 bits of each word are unused so values stay aligned.
    fn dequant_int4(&self, name: &str, out_dim: usize, in_dim: usize) -> Result<Vec<f32>, CoreError> {
        let packed: &[u32] = bytemuck::cast_slice(self.file.tensor_bytes(name)?);
        let scales = self.f32_tensor(&format!("{name}.scales"))?;
        let biases = self.f32_tensor(&format!("{name}.biases"))?;

        let group_size = self.cfg.group_size;
        let bits = self.cfg.quant_bits;
        let per_word = 32 / bits;
        let mask = (1u32 << bits) - 1;
        let groups_per_row = in_dim.div_ceil(group_size);
        let packed_in = in_dim / per_word;

        let mut out = vec![0.0f32; out_dim * in_dim];
        out.par_chunks_exact_mut(in_dim).enumerate().for_each(|(r, row)| {
            let base = r * packed_in;
            for jp in 0..packed_in {
                let word = packed[base + jp];
                for k in 0..per_word {
                    let col = jp * per_word + k;
                    let g = col / group_size;
                    let scale = scales[r * groups_per_row + g];
                    let bias = biases[r * groups_per_row + g];
                    let q = ((word >> (k * bits)) & mask) as f32;
                    row[col] = q * scale + bias;
                }
            }
        });
        Ok(out)
    }

    fn embed(&self, ids: &[u32]) -> Result<Vec<f32>, CoreError> {
        let hidden = self.cfg.hidden;
        let name = "model.embed_tokens.weight";
        let dtype = self.file.tensor_index(name).map(|t| t.dtype.clone()).unwrap_or_default();
        let mut out = vec![0.0f32; ids.len() * hidden];

        if dtype == "i8" {
            let q: &[i8] = bytemuck::cast_slice(self.file.tensor_bytes(name)?);
            let scales = self.f32_tensor("model.embed_tokens.scales")?;
            for (t, &id) in ids.iter().enumerate() {
                let r = id as usize;
                let s = scales[r];
                for c in 0..hidden {
                    out[t * hidden + c] = q[r * hidden + c] as f32 * s;
                }
            }
        } else {
            let w = self.f32_tensor(name)?;
            for (t, &id) in ids.iter().enumerate() {
                let r = id as usize;
                out[t * hidden..(t + 1) * hidden]
                    .copy_from_slice(&w[r * hidden..(r + 1) * hidden]);
            }
        }
        Ok(out)
    }

    /// Run the encoder and return `[seq, num_labels]` logits, row-major.
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>, CoreError> {
        if ids.is_empty() {
            return Err(CoreError::Backend("privacy-filter: empty token sequence".into()));
        }
        let c = &self.cfg;
        let t_len = ids.len();
        let hidden = c.hidden;
        let half = c.head_dim / 2;

        let mut x = self.embed(ids)?;

        // Half-width cos/sin: the model pairs adjacent dims, so there are
        // head_dim/2 angles per position, not head_dim.
        let mut cos = vec![0.0f32; t_len * half];
        let mut sin = vec![0.0f32; t_len * half];
        for p in 0..t_len {
            for i in 0..half {
                let a = p as f32 * self.inv_freq[i];
                cos[p * half + i] = a.cos() * self.attn_scaling;
                sin[p * half + i] = a.sin() * self.attn_scaling;
            }
        }

        let mut h = vec![0.0f32; t_len * hidden];
        for li in 0..c.n_layers {
            let p = format!("model.layers.{li}");
            let w = self.f32_tensor(&format!("{p}.input_layernorm.weight"))?;
            for t in 0..t_len {
                rms_norm_f32(
                    &x[t * hidden..(t + 1) * hidden],
                    &w,
                    c.eps,
                    &mut h[t * hidden..(t + 1) * hidden],
                );
            }
            let attn = self.attention(&h, t_len, li, &cos, &sin)?;
            for i in 0..x.len() {
                x[i] += attn[i];
            }

            let w = self.f32_tensor(&format!("{p}.post_attention_layernorm.weight"))?;
            for t in 0..t_len {
                rms_norm_f32(
                    &x[t * hidden..(t + 1) * hidden],
                    &w,
                    c.eps,
                    &mut h[t * hidden..(t + 1) * hidden],
                );
            }
            let moe = self.moe(&h, t_len, li)?;
            for i in 0..x.len() {
                x[i] += moe[i];
            }
        }

        let w = self.f32_tensor("model.norm.weight")?;
        for t in 0..t_len {
            let mut tmp = vec![0.0f32; hidden];
            rms_norm_f32(&x[t * hidden..(t + 1) * hidden], &w, c.eps, &mut tmp);
            x[t * hidden..(t + 1) * hidden].copy_from_slice(&tmp);
        }

        let n_labels = self.id2label.len();
        let score = self.matrix("score.weight", n_labels, hidden)?;
        let bias = self.f32_tensor("score.bias")?;
        let mut logits = vec![0.0f32; t_len * n_labels];
        score.matmul(&x, t_len, Some(&bias), &mut logits);
        Ok(logits)
    }

    fn attention(
        &self,
        h: &[f32],
        t_len: usize,
        li: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Vec<f32>, CoreError> {
        let c = &self.cfg;
        let p = format!("model.layers.{li}.self_attn");
        let hd = c.head_dim;
        let q_dim = c.n_heads * hd;
        let kv_dim = c.n_kv_heads * hd;
        let scaling = (hd as f32).powf(-0.25);

        let mut proj = |kind: &str, dim: usize| -> Result<Vec<f32>, CoreError> {
            let w = self.matrix(&format!("{p}.{kind}_proj.weight"), dim, c.hidden)?;
            let b = self.f32_tensor(&format!("{p}.{kind}_proj.bias"))?;
            let mut out = vec![0.0f32; t_len * dim];
            w.matmul(h, t_len, Some(&b), &mut out);
            Ok(out)
        };
        let mut q = proj("q", q_dim)?;
        let mut k = proj("k", kv_dim)?;
        let v = proj("v", kv_dim)?;

        apply_rope(&mut q, t_len, c.n_heads, hd, cos, sin);
        apply_rope(&mut k, t_len, c.n_kv_heads, hd, cos, sin);
        // Scale is applied to q and k separately, not once to their product.
        for val in q.iter_mut() {
            *val *= scaling;
        }
        for val in k.iter_mut() {
            *val *= scaling;
        }

        let sinks = self.f32_tensor(&format!("{p}.sinks"))?;
        let reps = c.n_heads / c.n_kv_heads;
        let window = c.sliding_window;

        let mut ctx = vec![0.0f32; t_len * q_dim];
        ctx.par_chunks_exact_mut(q_dim).enumerate().for_each(|(tq, out_row)| {
            let mut scores = vec![0.0f32; t_len];
            for hi in 0..c.n_heads {
                let kv = hi / reps;
                let qv = &q[tq * q_dim + hi * hd..tq * q_dim + (hi + 1) * hd];

                // Bidirectional band; rows outside it never enter the softmax.
                let lo = tq.saturating_sub(window);
                let hi_t = (tq + window + 1).min(t_len);

                let mut max = sinks[hi];
                for tk in lo..hi_t {
                    let kvv = &k[tk * kv_dim + kv * hd..tk * kv_dim + (kv + 1) * hd];
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += qv[d] * kvv[d];
                    }
                    scores[tk] = dot;
                    if dot > max {
                        max = dot;
                    }
                }

                // The sink participates in the max and the sum, then is
                // dropped, so rows can sum to less than 1.
                let mut denom = (sinks[hi] - max).exp();
                for tk in lo..hi_t {
                    let e = (scores[tk] - max).exp();
                    scores[tk] = e;
                    denom += e;
                }

                let o = &mut out_row[hi * hd..(hi + 1) * hd];
                o.fill(0.0);
                for tk in lo..hi_t {
                    let wgt = scores[tk] / denom;
                    let vv = &v[tk * kv_dim + kv * hd..tk * kv_dim + (kv + 1) * hd];
                    for d in 0..hd {
                        o[d] += wgt * vv[d];
                    }
                }
            }
        });

        let w = self.matrix(&format!("{p}.o_proj.weight"), c.hidden, q_dim)?;
        let b = self.f32_tensor(&format!("{p}.o_proj.bias"))?;
        let mut out = vec![0.0f32; t_len * c.hidden];
        w.matmul(&ctx, t_len, Some(&b), &mut out);
        Ok(out)
    }

    fn moe(&self, h: &[f32], t_len: usize, li: usize) -> Result<Vec<f32>, CoreError> {
        let c = &self.cfg;
        let p = format!("model.layers.{li}.mlp");
        let hidden = c.hidden;
        let inter = c.inter;

        let rw = self.matrix(&format!("{p}.router.weight"), c.n_experts, hidden)?;
        let rb = self.f32_tensor(&format!("{p}.router.bias"))?;
        let mut router = vec![0.0f32; t_len * c.n_experts];
        rw.matmul(h, t_len, Some(&rb), &mut router);

        // Per token: top-k experts, softmax over their logits, then /top_k.
        let mut routes: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for t in 0..t_len {
            let row = &router[t * c.n_experts..(t + 1) * c.n_experts];
            let mut idx: Vec<usize> = (0..c.n_experts).collect();
            idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            idx.truncate(c.top_k);

            let max = idx.iter().map(|&e| row[e]).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = idx.iter().map(|&e| (row[e] - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (slot, &e) in idx.iter().enumerate() {
                let w = exps[slot] / sum / c.top_k as f32;
                routes.entry(e).or_default().push((t, w));
            }
        }

        let gu_bias = self.f32_tensor(&format!("{p}.experts.gate_up_proj_bias"))?;
        let dn_bias = self.f32_tensor(&format!("{p}.experts.down_proj_bias"))?;

        let mut experts: Vec<(usize, Vec<(usize, f32)>)> = routes.into_iter().collect();
        experts.sort_unstable_by_key(|(e, _)| *e);

        let mut out = vec![0.0f32; t_len * hidden];
        for (e, toks) in experts {
            let w_gu = self.matrix(&format!("{p}.experts.gate_up_proj.{e}"), 2 * inter, hidden)?;
            let w_dn = self.matrix(&format!("{p}.experts.down_proj.{e}"), hidden, inter)?;

            let n = toks.len();
            let mut xs = vec![0.0f32; n * hidden];
            for (i, &(t, _)) in toks.iter().enumerate() {
                xs[i * hidden..(i + 1) * hidden]
                    .copy_from_slice(&h[t * hidden..(t + 1) * hidden]);
            }

            let gb = &gu_bias[e * 2 * inter..(e + 1) * 2 * inter];
            let mut gu = vec![0.0f32; n * 2 * inter];
            w_gu.matmul(&xs, n, Some(gb), &mut gu);

            let mut act = vec![0.0f32; n * inter];
            for i in 0..n {
                let row = &gu[i * 2 * inter..(i + 1) * 2 * inter];
                for j in 0..inter {
                    let gate = row[j].min(c.swiglu_limit);
                    let up = row[inter + j].clamp(-c.swiglu_limit, c.swiglu_limit);
                    let glu = gate / (1.0 + (-c.swiglu_alpha * gate).exp());
                    act[i * inter + j] = (up + 1.0) * glu;
                }
            }

            let db = &dn_bias[e * hidden..(e + 1) * hidden];
            let mut y = vec![0.0f32; n * hidden];
            w_dn.matmul(&act, n, Some(db), &mut y);

            for (i, &(t, wgt)) in toks.iter().enumerate() {
                for c_i in 0..hidden {
                    out[t * hidden + c_i] += y[i * hidden + c_i] * wgt;
                }
            }
        }

        // The router divides by top_k and the MLP multiplies it back; both
        // are kept so the arithmetic matches the reference.
        for val in out.iter_mut() {
            *val *= c.top_k as f32;
        }
        Ok(out)
    }

    /// Collapse per-token BIOES argmax labels into character spans.
    ///
    /// `offsets` are `(start, end)` byte offsets into `text`, one per token.
    pub fn spans(
        &self,
        logits: &[f32],
        offsets: &[(usize, usize)],
    ) -> Vec<(String, usize, usize)> {
        let n = self.id2label.len();
        let mut spans = Vec::new();
        let mut cur: Option<(String, usize, usize)> = None;

        for (t, &(a, b)) in offsets.iter().enumerate() {
            let row = &logits[t * n..(t + 1) * n];
            let best = row
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let label = self.id2label[best].as_str();

            if label == "O" || a == b {
                if let Some(s) = cur.take() {
                    spans.push(s);
                }
                continue;
            }
            let (tag, ent) = label.split_once('-').unwrap_or(("S", label));
            match cur.as_mut() {
                // Continue only within the same entity type.
                Some((e, _, end)) if (tag == "I" || tag == "E") && e == ent => {
                    *end = b;
                }
                _ => {
                    if let Some(s) = cur.take() {
                        spans.push(s);
                    }
                    cur = Some((ent.to_string(), a, b));
                }
            }
            if tag == "E" || tag == "S" {
                if let Some(s) = cur.take() {
                    spans.push(s);
                }
            }
        }
        if let Some(s) = cur.take() {
            spans.push(s);
        }
        spans
    }
}

/// Rotate in-place with the interleaved layout: adjacent dims are paired.
fn apply_rope(x: &mut [f32], t_len: usize, n_heads: usize, hd: usize, cos: &[f32], sin: &[f32]) {
    let half = hd / 2;
    let row = n_heads * hd;
    for t in 0..t_len {
        for h in 0..n_heads {
            let base = t * row + h * hd;
            for i in 0..half {
                let (c, s) = (cos[t * half + i], sin[t * half + i]);
                let a = x[base + 2 * i];
                let b = x[base + 2 * i + 1];
                x[base + 2 * i] = a * c - b * s;
                x[base + 2 * i + 1] = b * c + a * s;
            }
        }
    }
}

fn find_correction_dim(rot: f32, dim: usize, base: f32, orig_max: usize) -> f32 {
    (dim as f32 * (orig_max as f32 / (rot * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * base.ln())
}

/// YaRN inverse frequencies plus the attention temperature.
fn yarn_inv_freq(c: &PrivacyFilterConfig) -> (Vec<f32>, f32) {
    let dim = c.head_dim;
    let half = dim / 2;
    let factor = c.rope_factor;

    let mut inv_freq = vec![0.0f32; half];
    let mut low = find_correction_dim(c.beta_fast, dim, c.rope_theta, c.rope_orig_max);
    let mut high = find_correction_dim(c.beta_slow, dim, c.rope_theta, c.rope_orig_max);
    if c.rope_truncate {
        low = low.floor();
        high = high.ceil();
    }
    low = low.max(0.0);
    high = high.min(dim as f32 - 1.0);
    if (high - low).abs() < f32::EPSILON {
        high += 0.001;
    }

    for i in 0..half {
        let pos_freq = c.rope_theta.powf((2 * i) as f32 / dim as f32);
        let extrapolation = 1.0 / pos_freq;
        let interpolation = 1.0 / (factor * pos_freq);
        let ramp = (((i as f32) - low) / (high - low)).clamp(0.0, 1.0);
        let extrap_factor = 1.0 - ramp;
        inv_freq[i] = interpolation * (1.0 - extrap_factor) + extrapolation * extrap_factor;
    }

    let attention_scaling = if factor > 1.0 {
        0.1 * factor.ln() + 1.0
    } else {
        1.0
    };
    (inv_freq, attention_scaling)
}
