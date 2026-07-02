// Author: Jeffrey Asante (https://jeffasante.github.io/)
// Ported to cellm from https://github.com/cmpatino/nanowhale-100m/blob/main/modeling_deepseek_v4.py
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::{softmax_f32_inplace};
use std::path::Path;
use crate::{CellmFile, ModelConfig};

pub struct DeepSeekV4Runner {
    file: CellmFile,
    cfg: ModelConfig,
    // V4 specifics
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    o_groups: usize,
    o_lora_rank: usize,
    pub eos_token_id: Option<u32>,
    max_layers: usize,
    // Weight cache for f16 -> f32 conversion
    weight_cache: std::collections::HashMap<String, Vec<f32>>,
}

impl DeepSeekV4Runner {
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let file = CellmFile::load(path).map_err(|e| CoreError::Backend(e.to_string()))?;
        let h = file.header.clone();

        // Extract V4 specifics from header
        let hc_mult = h.hc_mult.unwrap_or(4);
        let hc_sinkhorn_iters = h.hc_sinkhorn_iters.unwrap_or(2);

        let cfg = ModelConfig {
            vocab_size: h.vocab_size,
            hidden_size: h.hidden_dim,
            num_hidden_layers: h.num_layers,
            num_attention_heads: h.num_heads,
            num_key_value_heads: h.num_kv_heads,
            intermediate_size: h.intermediate_size,
            rms_norm_eps: h.rms_norm_eps,
            rope_theta: h.rope_theta,
            head_dim: h.head_dim.unwrap_or(0),
            attention_softcap: 0.0,
            hc_mult: Some(hc_mult),
            hc_sinkhorn_iters: Some(hc_sinkhorn_iters),
            o_groups: h.o_groups,
            o_lora_rank: h.o_lora_rank,
            q_lora_rank: h.q_lora_rank,
            qk_rope_head_dim: h.qk_rope_head_dim,
            n_routed_experts: h.n_routed_experts,
            num_experts_per_tok: h.num_experts_per_tok,
            moe_intermediate_size: h.moe_intermediate_size,
            hc_eps: h.hc_eps,
            ..ModelConfig::default()
        };

        Ok(Self {
            file,
            cfg,
            hc_mult,
            hc_sinkhorn_iters,
            o_groups: h.o_groups.unwrap_or(2),
            o_lora_rank: h.o_lora_rank.unwrap_or(80),
            eos_token_id: h.eos_token_id,
            max_layers: h.num_layers,
            weight_cache: std::collections::HashMap::new(),
        })
    }

    pub fn from_file(file: CellmFile) -> Result<Self, CoreError> {
        let h = file.header.clone();

        // Extract V4 specifics from header
        let hc_mult = h.hc_mult.unwrap_or(4);
        let hc_sinkhorn_iters = h.hc_sinkhorn_iters.unwrap_or(2);

        let cfg = ModelConfig {
            vocab_size: h.vocab_size,
            hidden_size: h.hidden_dim,
            num_hidden_layers: h.num_layers,
            num_attention_heads: h.num_heads,
            num_key_value_heads: h.num_kv_heads,
            intermediate_size: h.intermediate_size,
            rms_norm_eps: h.rms_norm_eps,
            rope_theta: h.rope_theta,
            head_dim: h.head_dim.unwrap_or(0),
            attention_softcap: 0.0,
            hc_mult: Some(hc_mult),
            hc_sinkhorn_iters: Some(hc_sinkhorn_iters),
            o_groups: h.o_groups,
            o_lora_rank: h.o_lora_rank,
            q_lora_rank: h.q_lora_rank,
            qk_rope_head_dim: h.qk_rope_head_dim,
            n_routed_experts: h.n_routed_experts,
            num_experts_per_tok: h.num_experts_per_tok,
            moe_intermediate_size: h.moe_intermediate_size,
            hc_eps: h.hc_eps,
            ..ModelConfig::default()
        };

        Ok(Self {
            file,
            cfg,
            hc_mult,
            hc_sinkhorn_iters,
            o_groups: h.o_groups.unwrap_or(2),
            o_lora_rank: h.o_lora_rank.unwrap_or(80),
            eos_token_id: h.eos_token_id,
            max_layers: h.num_layers,
            weight_cache: std::collections::HashMap::new(),
        })
    }

    pub fn step_topk(
        &mut self,
        token: u32,
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>, CoreError> {
        let logits = self.forward(token, pos, page_table, kv_cache)?;

        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);

        Ok(indexed)
    }

    pub fn prefill_topk(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>, CoreError> {
        let n = tokens.len();
        if n <= 1 {
            return Ok(Vec::new());
        }
        let mut results = Vec::with_capacity(n - 1);
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            let logits = self.forward(tok, pos, page_table, kv_cache)?;
            if i < n - 1 {
                let mut indexed: Vec<(u32, f32)> = logits
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as u32, v))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(top_k);
                results.push(indexed);
            }
        }
        Ok(results)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn max_layers(&self) -> usize {
        self.max_layers
    }

    pub fn set_max_layers(&mut self, n: usize) {
        self.max_layers = n.min(self.cfg.num_hidden_layers);
    }

    /// Enable Metal acceleration. Currently disabled for DeepSeekV4 — this model
    /// is too small (hidden=320) for GPU dispatch latency to be worthwhile.
    pub fn enable_metal_full_backend(&mut self) -> bool {
        false
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    pub fn forward(
        &mut self,
        token: u32,
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
    ) -> Result<Vec<f32>, CoreError> {
        let hidden = self.cfg.hidden_size;
        let hc_mult = self.hc_mult;

        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                CoreError::Backend(format!("deepseek_v4: page_table append_token failed: {e}"))
            })?;
        }

        // 1. Embedding -> expand to hc_mult copies
        let mut x = vec![0.0f32; hidden * hc_mult];
        self.embed_token_hc(token, &mut x)?;

        for layer_idx in 0..self.max_layers {
            // 2. Attention with HC
            self.layer_attn_hc(layer_idx, &mut x, pos, page_table, kv_cache)?;

            // 3. FFN with HC
            self.layer_ffn_hc(layer_idx, &mut x)?;
        }

        // 4. Contract HC -> logits
        let mut x_final = vec![0.0f32; hidden];
        self.hc_head(&x, &mut x_final)?;

        // 5. Final norm + Logits
        let mut x_norm = vec![0.0f32; hidden];
        self.rmsnorm("model.norm.weight", &x_final, &mut x_norm)?;

        self.logits(&x_norm)
    }

    fn embed_token_hc(&mut self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        let hc_mult = self.hc_mult;
        let mut emb = vec![0.0f32; hidden];

        self.embed_token(token, &mut emb)?;

        for i in 0..hc_mult {
            out[i * hidden..(i + 1) * hidden].copy_from_slice(&emb);
        }
        Ok(())
    }

    fn layer_attn_hc(
        &mut self,
        layer_idx: usize,
        x: &mut [f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
    ) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        let hc_mult = self.hc_mult;

        let mut post = vec![0.0f32; hc_mult];
        let mut comb = vec![0.0f32; hc_mult * hc_mult];
        let mut x_reduced = vec![0.0f32; hidden];
        self.hc_pre(layer_idx, true, x, &mut x_reduced, &mut post, &mut comb)?;

        let mut x_norm = vec![0.0f32; hidden];
        let attn_norm_name = format!("model.layers.{}.attn_norm.weight", layer_idx);
        self.rmsnorm(&attn_norm_name, &x_reduced, &mut x_norm)?;

        let mut attn_out = vec![0.0f32; hidden];
        self.mla_attention(layer_idx, &x_norm, pos, page_table, kv_cache, &mut attn_out)?;

        let x_residual = x.to_vec();
        self.hc_post(&attn_out, &x_residual, &post, &comb, x)?;
        Ok(())
    }

    fn layer_ffn_hc(
        &mut self,
        layer_idx: usize,
        x: &mut [f32],
    ) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        let hc_mult = self.hc_mult;

        let mut post = vec![0.0f32; hc_mult];
        let mut comb = vec![0.0f32; hc_mult * hc_mult];
        let mut x_reduced = vec![0.0f32; hidden];
        self.hc_pre(layer_idx, false, x, &mut x_reduced, &mut post, &mut comb)?;

        let mut x_norm = vec![0.0f32; hidden];
        let ffn_norm_name = format!("model.layers.{}.ffn_norm.weight", layer_idx);
        self.rmsnorm(&ffn_norm_name, &x_reduced, &mut x_norm)?;

        let mut ffn_out = vec![0.0f32; hidden];
        self.moe_layer(layer_idx, &x_norm, &mut ffn_out)?;

        let x_residual = x.to_vec();
        self.hc_post(&ffn_out, &x_residual, &post, &comb, x)?;
        Ok(())
    }

    // --- HC Implementation ---

    fn hc_pre(
        &mut self,
        layer_idx: usize,
        is_attn: bool,
        x: &[f32], // [hc_mult, hidden]
        out: &mut [f32], // [hidden]
        post_out: &mut [f32],
        comb_out: &mut [f32],
    ) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        let hc_mult = self.hc_mult;
        let mix_hc = (2 + hc_mult) * hc_mult;

        let sublayer = if is_attn { "attn" } else { "ffn" };
        let mut mixes = vec![0.0f32; mix_hc];
        let fn_name = format!("model.layers.{}.hc_{}_fn", layer_idx, sublayer);
        let base_name = format!("model.layers.{}.hc_{}_base", layer_idx, sublayer);
        let scale_name = format!("model.layers.{}.hc_{}_scale", layer_idx, sublayer);

        let eps_norm = self.cfg.rms_norm_eps;
        let mut mean_sq = 0.0f32;
        for &v in x { mean_sq += v * v; }
        mean_sq /= x.len() as f32;
        let rsqrt = 1.0 / (mean_sq + eps_norm).sqrt();

        self.linear(&fn_name, x, &mut mixes)?;
        for v in mixes.iter_mut() { *v *= rsqrt; }

        let scales = self.tensor_f32(&scale_name)?.to_vec();
        let bases = self.tensor_f32(&base_name)?.to_vec();

        let (pre, post, comb) = self.hc_split_sinkhorn(&mixes, &scales, &bases)?;

        out.fill(0.0);
        for i in 0..hc_mult {
            let p = pre[i];
            let x_i = &x[i * hidden..(i + 1) * hidden];
            for j in 0..hidden {
                out[j] += p * x_i[j];
            }
        }

        post_out.copy_from_slice(&post);
        comb_out.copy_from_slice(&comb);
        Ok(())
    }

    fn hc_post(
        &self,
        y: &[f32], // [hidden]
        residual: &[f32], // [hc_mult, hidden]
        post: &[f32], // [hc_mult]
        comb: &[f32], // [hc_mult, hc_mult]
        out: &mut [f32], // [hc_mult, hidden]
    ) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        let hc_mult = self.hc_mult;

        for i in 0..hc_mult {
            let p = post[i];
            let out_i = &mut out[i * hidden..(i + 1) * hidden];

            for j in 0..hidden {
                out_i[j] = p * y[j];
            }

            for j in 0..hc_mult {
                let c = comb[i * hc_mult + j];
                let res_j = &residual[j * hidden..(j + 1) * hidden];
                for k in 0..hidden {
                    out_i[k] += c * res_j[k];
                }
            }
        }
        Ok(())
    }

    fn hc_split_sinkhorn(
        &self,
        mixes: &[f32],
        scales: &[f32],
        bases: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), CoreError> {
        let hc_mult = self.hc_mult;
        let eps = self.cfg.hc_eps.unwrap_or(1e-6);
        let sinkhorn_iters = self.cfg.hc_sinkhorn_iters.unwrap_or(1);

        let mut pre = vec![0.0f32; hc_mult];
        for i in 0..hc_mult {
            pre[i] = 1.0 / (1.0 + (-(mixes[i] * scales[0] + bases[i])).exp()) + eps;
        }

        let mut post = vec![0.0f32; hc_mult];
        for i in 0..hc_mult {
            post[i] = 2.0 / (1.0 + (-(mixes[hc_mult + i] * scales[1] + bases[hc_mult + i])).exp());
        }

        let offset = 2 * hc_mult;
        let mut comb = vec![0.0f32; hc_mult * hc_mult];
        for i in 0..(hc_mult * hc_mult) {
            comb[i] = mixes[offset + i] * scales[2] + bases[offset + i];
        }

        // Sinkhorn iterations
        // 1. Initial Row norm (Softmax) + eps
        for i in 0..hc_mult {
            let row = &mut comb[i * hc_mult..(i + 1) * hc_mult];
            softmax_f32_inplace(row);
            for v in row.iter_mut() { *v += eps; }
        }

        // 2. Initial Col norm
        for j in 0..hc_mult {
            let mut sum = 0.0f32;
            for i in 0..hc_mult { sum += comb[i * hc_mult + j]; }
            let inv_sum = 1.0 / (sum + eps);
            for i in 0..hc_mult { comb[i * hc_mult + j] *= inv_sum; }
        }

        // 3. Further iterations
        for _ in 0..sinkhorn_iters.saturating_sub(1) {
            // Row norm
            for i in 0..hc_mult {
                let mut sum = 0.0f32;
                for j in 0..hc_mult { sum += comb[i * hc_mult + j]; }
                let inv_sum = 1.0 / (sum + eps);
                for j in 0..hc_mult { comb[i * hc_mult + j] *= inv_sum; }
            }
            // Col norm
            for j in 0..hc_mult {
                let mut sum = 0.0f32;
                for i in 0..hc_mult { sum += comb[i * hc_mult + j]; }
                let inv_sum = 1.0 / (sum + eps);
                for i in 0..hc_mult { comb[i * hc_mult + j] *= inv_sum; }
            }
        }

        Ok((pre, post, comb))
    }

    fn hc_head(&mut self, x: &[f32], out: &mut [f32]) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        let hc_mult = self.hc_mult;

        let fn_name = "model.hc_head_fn";
        let base_name = "model.hc_head_base";
        let scale_name = "model.hc_head_scale";

        let eps_norm = self.cfg.rms_norm_eps;
        let mut mean_sq = 0.0f32;
        for &v in x { mean_sq += v * v; }
        mean_sq /= x.len() as f32;
        let rsqrt = 1.0 / (mean_sq + eps_norm).sqrt();

        let mut mixes = vec![0.0f32; hc_mult];
        self.linear(fn_name, x, &mut mixes)?;
        for v in mixes.iter_mut() { *v *= rsqrt; }

        let eps_hc = self.cfg.hc_eps.unwrap_or(1e-6);
        let scale = self.tensor_f32(scale_name)?[0];
        let bases = self.tensor_f32(base_name)?;

        out.fill(0.0);
        for i in 0..hc_mult {
            let p = 1.0 / (1.0 + (-(mixes[i] * scale + bases[i])).exp()) + eps_hc;
            let x_i = &x[i * hidden..(i + 1) * hidden];
            for j in 0..hidden {
                out[j] += p * x_i[j];
            }
        }
        Ok(())
    }

    // --- MLA Attention ---

    fn mla_attention(
        &mut self,
        layer_idx: usize,
        x: &[f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        let n_heads = self.cfg.num_attention_heads;
        let head_dim = self.cfg.head_dim;
        let q_lora_rank = self.cfg.q_lora_rank.unwrap_or(160);
        let qk_rope_head_dim = self.cfg.qk_rope_head_dim.unwrap_or(32);

        let mut q_latent_a = vec![0.0f32; q_lora_rank];
        let wq_a_name = format!("model.layers.{}.attn.wq_a.weight", layer_idx);
        self.linear(&wq_a_name, x, &mut q_latent_a)?;

        let mut q_latent = vec![0.0f32; q_lora_rank];
        let q_norm_name = format!("model.layers.{}.attn.q_norm.weight", layer_idx);
        self.rmsnorm(&q_norm_name, &q_latent_a, &mut q_latent)?;

        let mut q = vec![0.0f32; n_heads * head_dim];
        let wq_b_name = format!("model.layers.{}.attn.wq_b.weight", layer_idx);
        self.linear(&wq_b_name, &q_latent, &mut q)?;

        // Reference line 189: RMSNorm on q per-head
        for h in 0..n_heads {
            let q_h = &mut q[h * head_dim..(h + 1) * head_dim];
            let mut sum_sq = 0.0f32;
            for v in q_h.iter() { sum_sq += v * v; }
            let rsqrt = 1.0 / ((sum_sq / head_dim as f32) + self.cfg.rms_norm_eps).sqrt();
            for v in q_h.iter_mut() { *v *= rsqrt; }
        }

        let mut kv = vec![0.0f32; head_dim];
        let wkv_name = format!("model.layers.{}.attn.wkv.weight", layer_idx);
        let kv_norm_name = format!("model.layers.{}.attn.kv_norm.weight", layer_idx);
        self.linear(&wkv_name, x, &mut kv)?;
        self.rmsnorm(&kv_norm_name, &kv.clone(), &mut kv)?; // Apply RMSNorm to KV

        // Apply RoPE to q and k using Python convention: R(-θ) instead of standard R(θ)
        // Python apply_rotary_emb: [x1*cos + x2*sin, -x1*sin + x2*cos]
        // which is rotation by -θ: [[cos, sin], [-sin, cos]]
        fn rope_python_convention(x: &mut [f32], rotary_dim: usize, pos: usize, theta: f32) {
            let half = rotary_dim / 2;
            for i in 0..half {
                let inv_freq = theta.powf(-(2.0 * i as f32) / rotary_dim as f32);
                let angle = pos as f32 * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = x[i];
                let x1 = x[half + i];
                // Python forward: R(-θ) = [[cos, sin], [-sin, cos]]
                x[i] = x0 * cos + x1 * sin;
                x[half + i] = -x0 * sin + x1 * cos;
            }
        }
        for h in 0..n_heads {
            let q_h_rope = &mut q[(h + 1) * head_dim - qk_rope_head_dim..(h + 1) * head_dim];
            rope_python_convention(q_h_rope, qk_rope_head_dim, pos, self.cfg.rope_theta);
        }
        let k_rope = &mut kv[head_dim - qk_rope_head_dim..head_dim];
        rope_python_convention(k_rope, qk_rope_head_dim, pos, self.cfg.rope_theta);

        let block_id = page_table.block_for_token(pos).map_err(|e| CoreError::Backend(e.to_string()))?;
        let token_off = page_table.offset_in_block(pos).map_err(|e| CoreError::Backend(e.to_string()))?;
        {
            let mut cv = kv_cache.view_mut();
            cv.write_token(block_id, layer_idx, token_off, &kv, &kv)?;
        }

        let mut attn_out_raw = vec![0.0f32; n_heads * head_dim];
        let seq_len = pos + 1;
        let cr = kv_cache.view();

        for h in 0..n_heads {
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let out_h = &mut attn_out_raw[h * head_dim..(h + 1) * head_dim];

            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let b_t = page_table.block_for_token(t).map_err(|e| CoreError::Backend(e.to_string()))?;
                let o_t = page_table.offset_in_block(t).map_err(|e| CoreError::Backend(e.to_string()))?;

                let mut k_t = vec![0.0f32; head_dim];
                let mut v_t_dummy = vec![0.0f32; head_dim];
                cr.read_token(b_t, layer_idx, o_t, &mut k_t, &mut v_t_dummy)?;

                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += q_h[i] * k_t[i];
                }
                // Scale by 1/sqrt(head_dim)
                scores[t] = dot / (head_dim as f32).sqrt();
            }

            softmax_f32_inplace(&mut scores);

            out_h.fill(0.0);
            for t in 0..seq_len {
                let b_t = page_table.block_for_token(t).map_err(|e| CoreError::Backend(e.to_string()))?;
                let o_t = page_table.offset_in_block(t).map_err(|e| CoreError::Backend(e.to_string()))?;

                let mut k_t_dummy = vec![0.0f32; head_dim];
                let mut v_t = vec![0.0f32; head_dim];
                cr.read_token(b_t, layer_idx, o_t, &mut k_t_dummy, &mut v_t)?;

                let w = scores[t];
                for i in 0..head_dim {
                    out_h[i] += w * v_t[i];
                }
            }

            // Reference line 228: De-rotate RoPE on output (inverse of Python's R(-θ) = apply R(θ))
            // Python de-rotation: [o1*cos - o2*sin, o1*sin + o2*cos]
            let out_h_rope = &mut out_h[head_dim - qk_rope_head_dim..head_dim];
            let half = qk_rope_head_dim / 2;
            for i in 0..half {
                let inv_freq = self.cfg.rope_theta.powf(-(2.0 * i as f32) / qk_rope_head_dim as f32);
                let angle = pos as f32 * inv_freq;
                let (sin, cos) = angle.sin_cos();

                let x0 = out_h_rope[i];
                let x1 = out_h_rope[half + i];
                // Python inverse: apply R(θ) = [[cos, -sin], [sin, cos]]
                out_h_rope[i] = x0 * cos - x1 * sin;
                out_h_rope[half + i] = x0 * sin + x1 * cos;
            }
        }

        // 8. Grouped output projection (wo_a -> wo_b)
        let o_groups = self.o_groups;
        let o_lora_rank = self.o_lora_rank;
        let group_head_dim = (n_heads * head_dim) / o_groups;

        let mut o_latent = vec![0.0f32; o_groups * o_lora_rank];
        let wo_a_name = format!("model.layers.{}.attn.wo_a.weight", layer_idx);
        let wo_a_w = self.tensor_f32(&wo_a_name)?;

        for g in 0..o_groups {
            let group_in = &attn_out_raw[g * group_head_dim..(g + 1) * group_head_dim];
            let group_out = &mut o_latent[g * o_lora_rank..(g + 1) * o_lora_rank];
            let group_weight = &wo_a_w[g * o_lora_rank * group_head_dim..(g + 1) * o_lora_rank * group_head_dim];

            for i in 0..o_lora_rank {
                let mut sum = 0.0f32;
                let row = &group_weight[i * group_head_dim..(i + 1) * group_head_dim];
                for j in 0..group_head_dim {
                    sum += group_in[j] * row[j];
                }
                group_out[i] = sum;
            }
        }

        let wo_b_name = format!("model.layers.{}.attn.wo_b.weight", layer_idx);
        self.linear(&wo_b_name, &o_latent, out)?;

        Ok(())
    }

    fn moe_layer(
        &mut self,
        layer_idx: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        let hidden = self.cfg.hidden_size;
        let n_experts = self.cfg.n_routed_experts.unwrap_or(4);
        let k = self.cfg.num_experts_per_tok.unwrap_or(2);

        let mut logits = vec![0.0f32; n_experts];
        let gate_name = format!("model.layers.{}.ffn.gate.weight", layer_idx);
        let gate_bias_name = format!("model.layers.{}.ffn.gate.bias", layer_idx);
        self.linear(&gate_name, x, &mut logits)?;

        // Compute scores: sqrt(softplus(logits)) - without bias
        let mut scores = vec![0.0f32; n_experts];
        for i in 0..n_experts {
            let sp = (1.0 + logits[i].exp()).ln();
            scores[i] = sp.sqrt();
        }

        // Python: add bias ONLY for top-k selection, NOT for weighting
        let gate_bias = self.tensor_f32(&gate_bias_name)?;
        let mut biased_scores = scores.clone();
        for i in 0..n_experts {
            biased_scores[i] += gate_bias[i];
        }

        let mut expert_indices: Vec<usize> = (0..n_experts).collect();
        expert_indices.sort_by(|&a, &b| biased_scores[b].partial_cmp(&biased_scores[a]).unwrap());

        // Reference line 312: Normalize expert weights
        let mut sum_w = 0.0f32;
        for i in 0..k { sum_w += scores[expert_indices[i]]; }
        let inv_sum_w = 1.0 / (sum_w + 1e-20);

        // Reference line 314: Apply routed_scaling_factor (1.5 for Nanowhale)
        let route_scale = 1.5f32;

        out.fill(0.0);

        for i in 0..k {
            let expert_idx = expert_indices[i];
            let score = scores[expert_idx] * inv_sum_w * route_scale;

            let mut expert_out = vec![0.0f32; hidden];
            self.expert_forward(layer_idx, expert_idx, x, &mut expert_out)?;

            for j in 0..hidden {
                out[j] += score * expert_out[j];
            }
        }

        let mut shared_out = vec![0.0f32; hidden];
        self.shared_expert_forward(layer_idx, x, &mut shared_out)?;
        for j in 0..hidden {
            out[j] += shared_out[j];
        }

        Ok(())
    }

    fn expert_forward(
        &mut self,
        layer_idx: usize,
        expert_idx: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        let inter = self.cfg.moe_intermediate_size.unwrap_or(640);
        let hidden = self.cfg.hidden_size;

        let w1_name = format!("model.layers.{}.ffn.experts.{}.w1.weight", layer_idx, expert_idx);
        let w2_name = format!("model.layers.{}.ffn.experts.{}.w2.weight", layer_idx, expert_idx);
        let w3_name = format!("model.layers.{}.ffn.experts.{}.w3.weight", layer_idx, expert_idx);

        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        self.linear_dims(&w1_name, x, inter, hidden, &mut gate)?;
        self.linear_dims(&w3_name, x, inter, hidden, &mut up)?;

        for i in 0..inter {
            let g = gate[i];
            let s = 1.0 / (1.0 + (-g).exp());
            gate[i] = (g * s) * up[i];
        }

        self.linear_dims(&w2_name, &gate, hidden, inter, out)?;
        Ok(())
    }

    fn shared_expert_forward(
        &mut self,
        layer_idx: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        let inter = self.cfg.moe_intermediate_size.unwrap_or(640);
        let hidden = self.cfg.hidden_size;

        let w1_name = format!("model.layers.{}.ffn.shared_expert.w1.weight", layer_idx);
        let w2_name = format!("model.layers.{}.ffn.shared_expert.w2.weight", layer_idx);
        let w3_name = format!("model.layers.{}.ffn.shared_expert.w3.weight", layer_idx);

        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        self.linear_dims(&w1_name, x, inter, hidden, &mut gate)?;
        self.linear_dims(&w3_name, x, inter, hidden, &mut up)?;

        for i in 0..inter {
            let g = gate[i];
            let s = 1.0 / (1.0 + (-g).exp());
            gate[i] = (g * s) * up[i];
        }

        self.linear_dims(&w2_name, &gate, hidden, inter, out)?;
        Ok(())
    }

    fn linear_dims(&mut self, name: &str, x: &[f32], out_dim: usize, in_dim: usize, out: &mut [f32]) -> Result<(), CoreError> {
        let weight = self.tensor_f32(name)?;
        cellm_kernels::cpu_kernels::matmul_f32(weight, out_dim, in_dim, x, 1, out);
        Ok(())
    }

    fn embed_token(&mut self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let hidden = out.len();
        let vocab = self.cfg.vocab_size;
        let name = "model.embed_tokens.weight";
        let t = (token as usize) % vocab;
        let embed = self.tensor_f32(name)?;
        out.copy_from_slice(&embed[t * hidden..(t + 1) * hidden]);
        Ok(())
    }

    fn rmsnorm(&mut self, name: &str, x: &[f32], out: &mut [f32]) -> Result<(), CoreError> {
        let eps = self.cfg.rms_norm_eps;
        let weight = self.tensor_f32(name)?;
        cellm_kernels::cpu_kernels::rms_norm_f32(x, weight, eps, out);
        Ok(())
    }

    fn linear(&mut self, name: &str, x: &[f32], out: &mut [f32]) -> Result<(), CoreError> {
        let meta = self.file.tensor_index(name).ok_or_else(|| CoreError::Backend(format!("unknown tensor {name}")))?;
        let out_dim = meta.shape[0];
        let in_dim = meta.shape[1];
        let weight = self.tensor_f32(name)?;
        cellm_kernels::cpu_kernels::matmul_f32(weight, out_dim, in_dim, x, 1, out);
        Ok(())
    }

    fn logits(&mut self, x: &[f32]) -> Result<Vec<f32>, CoreError> {
        let vocab = self.cfg.vocab_size;
        let hidden = self.cfg.hidden_size;
        let mut buf = vec![0.0f32; vocab];
        let weight = self.tensor_f32("lm_head.weight")?;
        cellm_kernels::cpu_kernels::matmul_f32(weight, vocab, hidden, x, 1, &mut buf);
        Ok(buf)
    }

    fn tensor_f32(&mut self, name: &str) -> Result<&[f32], CoreError> {
        if self.weight_cache.contains_key(name) {
            return Ok(self.weight_cache.get(name).unwrap());
        }

        let meta = self.file.tensor_index(name).ok_or_else(|| CoreError::Backend(format!("unknown tensor {name}")))?;
        let bytes = self.file.tensor_bytes(name).map_err(|e| CoreError::Backend(e.to_string()))?;

        let f32_vec = match meta.dtype.as_str() {
            "f32" => {
                let mut v = vec![0.0f32; bytes.len() / 4];
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, bytes.len());
                }
                v
            }
            "f16" | "bf16" => {
                let n = bytes.len() / 2;
                let mut v = vec![0.0f32; n];
                let src: &[u16] = bytemuck::cast_slice(bytes);
                for i in 0..n {
                    v[i] = half::f16::from_bits(src[i]).to_f32();
                }
                v
            }
            other => return Err(CoreError::Backend(format!("unsupported dtype {other} for tensor {name}"))),
        };

        self.weight_cache.insert(name.to_string(), f32_vec);
        Ok(self.weight_cache.get(name).unwrap())
    }
}
