use std::path::Path;

use bytemuck::cast_slice;
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::{rms_norm_f32, rope_inplace_f32};
use half::f16;

use crate::{CellmFile, ModelConfig};

pub struct LlamaRunner {
    file: CellmFile,
    cfg: ModelConfig,
    max_layers: usize,
    eos_token_id: Option<u32>,
    tensor_prefix: String,
}

impl LlamaRunner {
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let file = CellmFile::load(path)?;
        let h = file.header.clone();

        let cfg = ModelConfig {
            vocab_size: h.vocab_size,
            hidden_size: h.hidden_dim,
            num_hidden_layers: h.num_layers,
            num_attention_heads: h.num_heads,
            num_key_value_heads: h.num_kv_heads,
            intermediate_size: h.intermediate_size,
            rms_norm_eps: h.rms_norm_eps,
            rope_theta: h.rope_theta,
        };

        let tensor_prefix = detect_llama_prefix(&file)?;

        Ok(Self {
            file,
            cfg: cfg.clone(),
            max_layers: cfg.num_hidden_layers,
            eos_token_id: h.eos_token_id,
            tensor_prefix,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn set_max_layers(&mut self, n: usize) {
        self.max_layers = n.min(self.cfg.num_hidden_layers).max(1);
    }

    pub fn max_layers(&self) -> usize {
        self.max_layers
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    pub fn embed_token_hidden(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        self.embed_token(token, out)
    }

    pub fn step_topk(
        &mut self,
        token: u32,
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>, CoreError> {
        let mut x = vec![0.0f32; self.cfg.hidden_size];
        self.embed_token(token, &mut x)?;
        self.step_topk_from_hidden(&x, pos, page_table, kv_cache, top_k)
    }

    pub fn step_topk_from_hidden(
        &mut self,
        x0: &[f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>, CoreError> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden / n_heads;
        if head_dim * n_heads != hidden {
            return Err(CoreError::Backend(
                "llama: hidden_size must be divisible by num_attention_heads".into(),
            ));
        }
        let kv_dim = n_kv_heads * head_dim;

        // Ensure pagetable covers this token position.
        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                CoreError::Backend(format!("llama step: page_table append_token failed: {e}"))
            })?;
        } else if pos > page_table.token_count() {
            return Err(CoreError::Backend(format!(
                "llama step: non-contiguous pos {pos} (token_count={})",
                page_table.token_count()
            )));
        }

        let block_id = page_table.block_for_token(pos).map_err(|e| {
            CoreError::Backend(format!("llama step: page_table block_for_token failed: {e}"))
        })?;
        let token_off = page_table.offset_in_block(pos).map_err(|e| {
            CoreError::Backend(format!("llama step: page_table offset_in_block failed: {e}"))
        })?;

        if x0.len() != hidden {
            return Err(CoreError::Backend(format!(
                "llama step_from_hidden: hidden len mismatch {} != {}",
                x0.len(),
                hidden
            )));
        }
        let mut x = x0.to_vec();

        // Per-layer scratch.
        let mut attn_norm_w = vec![0.0f32; hidden];
        let mut x_norm = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; hidden];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; hidden];
        let mut attn_proj = vec![0.0f32; hidden];

        let mut post_norm_w = vec![0.0f32; hidden];
        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; cfg.intermediate_size];
        let mut up = vec![0.0f32; cfg.intermediate_size];
        let mut down = vec![0.0f32; hidden];

        let mut gather_bases: Vec<usize> = Vec::new();

        for layer in 0..self.max_layers {
            // Attention input norm.
            self.rmsnorm_weight(
                &format!("model.layers.{layer}.input_layernorm.weight"),
                &mut attn_norm_w,
            )?;
            rms_norm_f32(&x, &attn_norm_w, cfg.rms_norm_eps, &mut x_norm);

            // QKV projections (HF weights are [out, in]).
            self.linear_f16_out_in(
                &x_norm,
                &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                hidden,
                hidden,
                &mut q,
            )?;
            self.linear_f16_out_in(
                &x_norm,
                &format!("model.layers.{layer}.self_attn.k_proj.weight"),
                kv_dim,
                hidden,
                &mut k,
            )?;
            self.linear_f16_out_in(
                &x_norm,
                &format!("model.layers.{layer}.self_attn.v_proj.weight"),
                kv_dim,
                hidden,
                &mut v,
            )?;

            rope_inplace_f32(&mut q, n_heads, head_dim, pos, cfg.rope_theta);
            rope_inplace_f32(&mut k, n_kv_heads, head_dim, pos, cfg.rope_theta);

            // Write new token K/V into paged cache.
            {
                let mut cv = kv_cache.view_mut();
                cv.write_token(block_id, layer, token_off, &k, &v)?;
            }

            // Gather historical K/V and run attention for this token.
            let seq = page_table.token_count();
            let cr = kv_cache.view();
            gather_bases.clear();
            gather_bases.reserve(seq);
            for tpos in 0..seq {
                let b = page_table.block_for_token(tpos).map_err(|e| {
                    CoreError::Backend(format!("llama: block_for_token failed: {e}"))
                })?;
                let o = page_table.offset_in_block(tpos).map_err(|e| {
                    CoreError::Backend(format!("llama: offset_in_block failed: {e}"))
                })?;
                gather_bases.push(cr.layout.token_base_elem(b, layer, o)?);
            }
            cr.attention_single_token_gqa_from_bases(
                &gather_bases,
                &q,
                n_heads,
                n_kv_heads,
                head_dim,
                &mut attn_out,
            )?;

            // o_proj: hidden <- hidden
            self.linear_f16_out_in(
                &attn_out,
                &format!("model.layers.{layer}.self_attn.o_proj.weight"),
                hidden,
                hidden,
                &mut attn_proj,
            )?;

            for i in 0..hidden {
                x[i] += attn_proj[i];
            }

            // Post-attn norm.
            self.rmsnorm_weight(
                &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                &mut post_norm_w,
            )?;
            rms_norm_f32(&x, &post_norm_w, cfg.rms_norm_eps, &mut mlp_in);

            // MLP: gate_proj + up_proj -> silu(gate)*up -> down_proj
            self.linear_f16_out_in(
                &mlp_in,
                &format!("model.layers.{layer}.mlp.gate_proj.weight"),
                cfg.intermediate_size,
                hidden,
                &mut gate,
            )?;
            self.linear_f16_out_in(
                &mlp_in,
                &format!("model.layers.{layer}.mlp.up_proj.weight"),
                cfg.intermediate_size,
                hidden,
                &mut up,
            )?;

            // silu(gate) in-place: x * sigmoid(x)
            for g in gate.iter_mut() {
                let s = 1.0 / (1.0 + (-*g).exp());
                *g = *g * s;
            }
            for i in 0..gate.len() {
                gate[i] *= up[i];
            }

            self.linear_f16_out_in(
                &gate,
                &format!("model.layers.{layer}.mlp.down_proj.weight"),
                hidden,
                cfg.intermediate_size,
                &mut down,
            )?;
            for i in 0..hidden {
                x[i] += down[i];
            }
        }

        // Final norm.
        let mut norm_w = vec![0.0f32; hidden];
        self.rmsnorm_weight("model.norm.weight", &mut norm_w)?;
        let mut x_final = vec![0.0f32; hidden];
        rms_norm_f32(&x, &norm_w, cfg.rms_norm_eps, &mut x_final);

        // Logits via tied embeddings: logits[v] = dot(x_final, embed[v])
        let vocab = cfg.vocab_size;
        let k = top_k.max(1).min(vocab);
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut min_idx = 0usize;
        let mut min_val = f32::INFINITY;

        let lm_head_name = self.resolve_name("lm_head.weight");
        let maybe_lm_head = lm_head_name
            .as_ref()
            .ok()
            .and_then(|n| self.file.tensor_index(n).map(|_| n.clone()));
        let lm_src_name = maybe_lm_head
            .as_deref()
            .unwrap_or("model.embed_tokens.weight");
        let lm_src_resolved = self.resolve_name(lm_src_name)?;
        let lm_meta = self
            .tensor_meta_by_exact_name(&lm_src_resolved)
            .ok_or_else(|| CoreError::Backend(format!("unknown tensor {}", lm_src_resolved)))?;
        let lm_src_f16 = if lm_meta.dtype == "f16" {
            Some(self.tensor_f16_by_exact_name(&lm_src_resolved)?)
        } else {
            None
        };
        let lm_src_i8 = if lm_meta.dtype == "i8" {
            Some(self.tensor_i8_by_exact_name(&lm_src_resolved)?)
        } else {
            None
        };
        let lm_src_scales = if lm_meta.dtype == "i8" {
            Some(self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?)
        } else {
            None
        };

        for vid in 0..vocab {
            let mut dot = 0.0f32;
            if let Some(wf16) = lm_src_f16 {
                let row = &wf16[(vid * hidden)..(vid + 1) * hidden];
                for i in 0..hidden {
                    dot += x_final[i] * f16::from_bits(row[i]).to_f32();
                }
            } else if let (Some(wi8), Some(scales)) = (lm_src_i8, lm_src_scales) {
                let row = &wi8[(vid * hidden)..(vid + 1) * hidden];
                let scale = f16::from_bits(scales[vid]).to_f32();
                for i in 0..hidden {
                    dot += x_final[i] * ((row[i] as f32) * scale);
                }
            } else {
                return Err(CoreError::Backend(format!(
                    "unsupported lm source dtype {} for {}",
                    lm_meta.dtype, lm_src_resolved
                )));
            }

            if top.len() < k {
                top.push((vid as u32, dot));
                if dot < min_val {
                    min_val = dot;
                    min_idx = top.len() - 1;
                }
            } else if dot > min_val {
                top[min_idx] = (vid as u32, dot);
                // Recompute min (k is small).
                min_val = top[0].1;
                min_idx = 0;
                for (i, &(_, s)) in top.iter().enumerate().skip(1) {
                    if s < min_val {
                        min_val = s;
                        min_idx = i;
                    }
                }
            }
        }

        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(top)
    }

    fn tensor_f16(&self, name: &str) -> Result<&[u16], CoreError> {
        let resolved = self.resolve_name(name)?;
        self.tensor_f16_by_exact_name(&resolved)
    }

    fn tensor_f16_by_exact_name(&self, resolved: &str) -> Result<&[u16], CoreError> {
        let bytes = self.file.tensor_bytes(resolved)?;
        if bytes.len() % 2 != 0 {
            return Err(CoreError::Backend(format!("tensor {resolved} nbytes not even")));
        }
        Ok(cast_slice(bytes))
    }

    fn tensor_i8_by_exact_name(&self, resolved: &str) -> Result<&[i8], CoreError> {
        let bytes = self.file.tensor_bytes(resolved)?;
        Ok(cast_slice(bytes))
    }

    fn tensor_meta_by_exact_name(&self, resolved: &str) -> Option<&crate::CellmTensorIndex> {
        self.file.tensor_index(resolved)
    }

    fn embed_token(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let hidden = out.len();
        let embed = self.tensor_f16("model.embed_tokens.weight")?;
        let vocab = self.cfg.vocab_size;
        let t = (token as usize) % vocab;
        let row = &embed[t * hidden..(t + 1) * hidden];
        for i in 0..hidden {
            out[i] = f16::from_bits(row[i]).to_f32();
        }
        Ok(())
    }

    fn rmsnorm_weight(&self, name: &str, out: &mut [f32]) -> Result<(), CoreError> {
        let w = self.tensor_f16(name)?;
        if w.len() != out.len() {
            return Err(CoreError::Backend(format!(
                "rmsnorm weight {name} len mismatch: {} vs {}",
                w.len(),
                out.len()
            )));
        }
        for i in 0..out.len() {
            out[i] = f16::from_bits(w[i]).to_f32();
        }
        Ok(())
    }

    fn linear_f16_out_in(
        &self,
        x: &[f32],
        weight_name: &str,
        out_dim: usize,
        in_dim: usize,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        if x.len() != in_dim || out.len() != out_dim {
            return Err(CoreError::Backend(format!(
                "linear dims mismatch for {weight_name}: x={} out={} expected in={in_dim} out={out_dim}",
                x.len(),
                out.len()
            )));
        }

        let resolved = self.resolve_name(weight_name)?;
        let meta = self
            .tensor_meta_by_exact_name(&resolved)
            .ok_or_else(|| CoreError::Backend(format!("unknown tensor {resolved}")))?;
        let shape = meta.shape.clone();
        if shape.len() != 2 {
            return Err(CoreError::Backend(format!(
                "weight {weight_name} expected 2D, got {:?}",
                shape
            )));
        }
        // HF linear weight: [out, in]
        if shape[0] != out_dim || shape[1] != in_dim {
            return Err(CoreError::Backend(format!(
                "weight {weight_name} shape mismatch: {:?} expected [{out_dim},{in_dim}]",
                shape
            )));
        }
        match meta.dtype.as_str() {
            "f16" => {
                let w = self.tensor_f16_by_exact_name(&resolved)?;
                if w.len() != out_dim * in_dim {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} len mismatch: {} expected {}",
                        w.len(),
                        out_dim * in_dim
                    )));
                }
                for j in 0..out_dim {
                    let row = &w[j * in_dim..(j + 1) * in_dim];
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += x[i] * f16::from_bits(row[i]).to_f32();
                    }
                    out[j] = acc;
                }
            }
            "i8" => {
                let w = self.tensor_i8_by_exact_name(&resolved)?;
                if w.len() != out_dim * in_dim {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} len mismatch: {} expected {}",
                        w.len(),
                        out_dim * in_dim
                    )));
                }
                let scales = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                if scales.len() != out_dim {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} qscale len mismatch: {} expected {}",
                        scales.len(),
                        out_dim
                    )));
                }
                for j in 0..out_dim {
                    let row = &w[j * in_dim..(j + 1) * in_dim];
                    let scale = f16::from_bits(scales[j]).to_f32();
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += x[i] * ((row[i] as f32) * scale);
                    }
                    out[j] = acc;
                }
            }
            other => {
                return Err(CoreError::Backend(format!(
                    "unsupported weight dtype for {weight_name}: {other}"
                )));
            }
        }
        Ok(())
    }

    fn resolve_name(&self, name: &str) -> Result<String, CoreError> {
        if self.file.tensor_index(name).is_some() {
            return Ok(name.to_string());
        }
        if !self.tensor_prefix.is_empty() {
            let prefixed = format!("{}{}", self.tensor_prefix, name);
            if self.file.tensor_index(&prefixed).is_some() {
                return Ok(prefixed);
            }
        }
        if let Some(suffix) = name.strip_prefix("model.") {
            let text_model = format!("model.text_model.{suffix}");
            if self.file.tensor_index(&text_model).is_some() {
                return Ok(text_model);
            }
        }
        Err(CoreError::Backend(format!("unknown tensor {name}")))
    }
}

fn detect_llama_prefix(file: &CellmFile) -> Result<String, CoreError> {
    for prefix in ["", "language_model."] {
        let embed = format!("{prefix}model.embed_tokens.weight");
        let norm = format!("{prefix}model.norm.weight");
        if file.tensor_index(&embed).is_some() && file.tensor_index(&norm).is_some() {
            return Ok(prefix.to_string());
        }
    }
    if file
        .tensor_index("model.text_model.embed_tokens.weight")
        .is_some()
        && file.tensor_index("model.text_model.norm.weight").is_some()
    {
        return Ok(String::new());
    }
    Err(CoreError::Backend(
        "missing required llama tensors: model.embed_tokens.weight/model.norm.weight".into(),
    ))
}
