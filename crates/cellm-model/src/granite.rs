// Author: Jeffrey Asante (https://jeffasante.github.io/)
use std::path::Path;

use bytemuck::cast_slice;
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::{rms_norm_f32, rope_non_interleaved_inplace_f32};
use cellm_kernels::metal::MetalMatmul;
use cellm_kernels::{MetalKernels, MetalOps};
use half::f16;
// use serde_json::Value; // Removed unused import

use crate::{CellmFile, ModelConfig};

pub struct GraniteRunner {
    file: CellmFile,
    cfg: ModelConfig,
    max_layers: usize,
    eos_token_id: Option<u32>,
    tensor_prefix: String,
    
    // MuP multipliers from config
    embedding_multiplier: f32,
    residual_multiplier: f32,
    attention_multiplier: f32,
    logits_scaling: f32,

    linear_backend: GraniteLinearBackend,
    metal_ops: Option<MetalOps>,
    metal_strict: bool,
}

enum GraniteLinearBackend {
    Cpu,
    Metal { ctx: MetalMatmul },
}

impl GraniteRunner {
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let file = CellmFile::load(path)?;
        let h = file.header.clone();

        let cfg = ModelConfig {
            vocab_size: h.vocab_size,
            hidden_size: h.hidden_dim,
            num_hidden_layers: h.num_layers,
            num_attention_heads: h.num_heads,
            num_key_value_heads: h.num_kv_heads,
            head_dim: h.head_dim.unwrap_or(h.hidden_dim / h.num_heads),
            intermediate_size: h.intermediate_size,
            rms_norm_eps: h.rms_norm_eps,
            rope_theta: h.rope_theta,
        };

        // Extract MuP multipliers from source_text_config if available
        let mut embedding_multiplier = 1.0f32;
        let mut residual_multiplier = 1.0f32;
        let mut attention_multiplier = 1.0f32; // Standard will be 1/sqrt(head_dim) if not set
        let mut logits_scaling = 1.0f32;

        if let Some(src_cfg) = &h.source_text_config {
            if let Some(val) = src_cfg.get("embedding_multiplier") {
                embedding_multiplier = val.as_f64().unwrap_or(1.0) as f32;
            }
            if let Some(val) = src_cfg.get("residual_multiplier") {
                residual_multiplier = val.as_f64().unwrap_or(1.0) as f32;
            }
            if let Some(val) = src_cfg.get("attention_multiplier") {
                attention_multiplier = val.as_f64().unwrap_or(1.0) as f32;
            } else {
                // Default to 1/sqrt(head_dim) for Granite IF not set and head_dim available
                let head_dim = cfg.hidden_size / cfg.num_attention_heads;
                attention_multiplier = 1.0f32 / (head_dim as f32).sqrt();
            }
            if let Some(val) = src_cfg.get("logits_scaling") {
                logits_scaling = val.as_f64().unwrap_or(1.0) as f32;
            }
        } else {
            // Check top level header just in case
            // (The header currently doesn't store these directly, they are in source_text_config)
        }

        let tensor_prefix = detect_granite_prefix(&file)?;

        Ok(Self {
            file,
            cfg: cfg.clone(),
            max_layers: cfg.num_hidden_layers,
            eos_token_id: h.eos_token_id,
            tensor_prefix,
            embedding_multiplier,
            residual_multiplier,
            attention_multiplier,
            logits_scaling,
            linear_backend: GraniteLinearBackend::Cpu,
            metal_ops: None,
            metal_strict: false,
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

    pub fn enable_metal_linear_backend(&mut self) -> bool {
        match MetalKernels::create_matmul() {
            Ok(ctx) => {
                self.linear_backend = GraniteLinearBackend::Metal { ctx };
                self.metal_strict = false;
                true
            }
            Err(e) => {
                eprintln!("granite: failed to enable metal linear backend: {e}");
                self.linear_backend = GraniteLinearBackend::Cpu;
                self.metal_strict = false;
                false
            }
        }
    }

    pub fn enable_metal_full_backend(&mut self) -> bool {
        let mk_res = MetalKernels::create_matmul();
        let mo_res = MetalOps::create();
        match (mk_res, mo_res) {
            (Ok(ctx), Ok(mo)) => {
                self.linear_backend = GraniteLinearBackend::Metal { ctx };
                self.metal_ops = Some(mo);
                self.metal_strict = true;
                true
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("granite: failed to enable full metal backend: {e}");
                self.linear_backend = GraniteLinearBackend::Cpu;
                self.metal_ops = None;
                self.metal_strict = false;
                false
            }
        }
    }

    pub fn embed_token_hidden(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        self.embed_token(token, out)?;
        // Apply embedding multiplier
        for x in out.iter_mut() {
            *x *= self.embedding_multiplier;
        }
        Ok(())
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
        self.embed_token_hidden(token, &mut x)?;
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
        let logits = self.step_inner(x0, pos, page_table, kv_cache, true)?;
        self.topk_from_logits(&logits, top_k)
    }

    pub fn prefill(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
    ) -> Result<(), CoreError> {
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            if pos == page_table.token_count() {
                page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                    CoreError::Backend(format!("granite prefill: page_table append_token failed: {e}"))
                })?;
            }
            let mut x = vec![0.0f32; self.cfg.hidden_size];
            self.embed_token_hidden(tok, &mut x)?;
            self.step_inner(&x, pos, page_table, kv_cache, false)?;
        }
        Ok(())
    }

    fn step_inner(
        &mut self,
        x0: &[f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        return_logits: bool,
    ) -> Result<Vec<f32>, CoreError> {
        let cfg = self.cfg.clone();
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Ensure pagetable covers this token position.
        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                CoreError::Backend(format!("granite step: page_table append_token failed: {e}"))
            })?;
        }

        let block_id = page_table.block_for_token(pos).map_err(|e| {
            CoreError::Backend(format!("granite step: page_table block_for_token failed: {e}"))
        })?;
        let token_off = page_table.offset_in_block(pos).map_err(|e| {
            CoreError::Backend(format!("granite step: page_table offset_in_block failed: {e}"))
        })?;

        let mut x = x0.to_vec();

        // Per-layer scratch.
        let mut x_norm = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; hidden];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; hidden];
        let mut attn_proj = vec![0.0f32; hidden];

        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate_up = vec![0.0f32; cfg.intermediate_size * 2];
        let mut down = vec![0.0f32; hidden];

        let mut gather_bases: Vec<usize> = Vec::new();

        for layer in 0..self.max_layers {
            // Attention input norm.
            self.rmsnorm_scaled(
                &format!("model.layers.{layer}.input_layernorm.weight"),
                &x,
                cfg.rms_norm_eps,
                &mut x_norm,
            )?;

            // QKV projections
            let q_name = format!("model.layers.{layer}.self_attn.q_proj.weight");
            let k_name = format!("model.layers.{layer}.self_attn.k_proj.weight");
            let v_name = format!("model.layers.{layer}.self_attn.v_proj.weight");
            
            self.linear_f16_out_in(&x_norm, &q_name, hidden, hidden, &mut q)?;
            self.linear_f16_out_in(&x_norm, &k_name, kv_dim, hidden, &mut k)?;
            self.linear_f16_out_in(&x_norm, &v_name, kv_dim, hidden, &mut v)?;

            // RoPE
            if let Some(ref mut ops) = self.metal_ops {
                 ops.rope_half_f32(&mut q, n_heads, head_dim, head_dim, pos, cfg.rope_theta)
                     .map_err(|e| CoreError::Backend(e.to_string()))?;
                 ops.rope_half_f32(&mut k, n_kv_heads, head_dim, head_dim, pos, cfg.rope_theta)
                     .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                 rope_non_interleaved_inplace_f32(&mut q, n_heads, head_dim, head_dim, pos, cfg.rope_theta);
                 rope_non_interleaved_inplace_f32(&mut k, n_kv_heads, head_dim, head_dim, pos, cfg.rope_theta);
            }

            // Write new token K/V into paged cache.
            {
                let mut cv = kv_cache.view_mut();
                cv.write_token(block_id, layer, token_off, &k, &v)?;
            }

            // Gather and Attn
            let seq = page_table.token_count();
            let cr = kv_cache.view();
            gather_bases.clear();
            for tpos in 0..seq {
                let b = page_table.block_for_token(tpos).map_err(|e| {
                    CoreError::Backend(format!("granite: block_for_token failed: {e}"))
                })?;
                let o = page_table.offset_in_block(tpos).map_err(|e| {
                    CoreError::Backend(format!("granite: offset_in_block failed: {e}"))
                })?;
                gather_bases.push(cr.layout.token_base_elem(b, layer, o)?);
            }
            
            // Apply attention_multiplier
            // Granite uses attention_multiplier instead of 1/sqrt(head_dim)
            let scale = self.attention_multiplier;
            cr.attention_single_token_gqa_from_bases(
                &gather_bases,
                &q,
                n_heads,
                n_kv_heads,
                head_dim, 
                Some(scale),
                None, // soft_cap
                &mut attn_out,
            )?;

            // o_proj
            self.linear_f16_out_in(
                &attn_out,
                &format!("model.layers.{layer}.self_attn.o_proj.weight"),
                hidden,
                hidden,
                &mut attn_proj,
            )?;

            // Residual add with multiplier
            for i in 0..hidden {
                x[i] += attn_proj[i] * self.residual_multiplier;
            }

            // Post-attn norm.
            self.rmsnorm_scaled(
                &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                &x,
                cfg.rms_norm_eps,
                &mut mlp_in,
            )?;

            // Fused MLP: shared_mlp.input_linear contains [gate, up]
            self.linear_f16_out_in(
                &mlp_in,
                &format!("model.layers.{layer}.shared_mlp.input_linear.weight"),
                cfg.intermediate_size * 2,
                hidden,
                &mut gate_up,
            )?;

            // silu(gate) * up
            let (gate, up) = gate_up.split_at_mut(cfg.intermediate_size);
            for i in 0..cfg.intermediate_size {
                let g = gate[i];
                let s = 1.0 / (1.0 + (-g).exp());
                gate[i] = (g * s) * up[i];
            }

            // down_proj (shared_mlp.output_linear)
            self.linear_f16_out_in(
                gate,
                &format!("model.layers.{layer}.shared_mlp.output_linear.weight"),
                hidden,
                cfg.intermediate_size,
                &mut down,
            )?;

            // Residual add with multiplier
            for i in 0..hidden {
                x[i] += down[i] * self.residual_multiplier;
            }
        }

        // Final norm.
        let mut x_final = vec![0.0f32; hidden];
        self.rmsnorm_scaled("model.norm.weight", &x, cfg.rms_norm_eps, &mut x_final)?;

        if !return_logits {
            return Ok(vec![]);
        }

        // Logits
        let vocab = cfg.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        
        let lm_head_name = self.resolve_name("lm_head.weight")?;
        let lm_src_resolved = self.resolve_name(&lm_head_name)?;
        
        self.linear_f16_out_in(&x_final, &lm_src_resolved, vocab, hidden, &mut logits)?;
        
        // Apply logits_scaling
        if self.logits_scaling != 1.0 {
            for l in logits.iter_mut() {
                *l /= self.logits_scaling;
            }
        }
        
        Ok(logits)
    }

    fn tensor_f16_by_exact_name(&self, resolved: &str) -> Result<&[u16], CoreError> {
        let bytes = self.file.tensor_bytes(resolved)?;
        if bytes.len() % 2 != 0 {
            return Err(CoreError::Backend(format!("tensor {resolved} nbytes not even")));
        }
        Ok(cast_slice(bytes))
    }

    fn resolve_name(&self, name: &str) -> Result<String, CoreError> {
        let prefix = &self.tensor_prefix;
        let candidates = [
            format!("{prefix}{name}"),
            format!("{name}"),
        ];
        for c in candidates {
            if self.file.tensor_index(&c).is_some() {
                return Ok(c);
            }
        }
        // Fallback for embeddings/lm_head if prefix didn't work
        if name == "lm_head.weight" || name == "model.embed_tokens.weight" {
             if self.file.tensor_index("model.embed_tokens.weight").is_some() { return Ok("model.embed_tokens.weight".to_string()); }
        }
        
        Err(CoreError::Backend(format!("granite: unknown tensor {name}")))
    }

    fn embed_token(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let hidden = out.len();
        let name = self.resolve_name("model.embed_tokens.weight")?;
        let embed = self.tensor_f16_by_exact_name(&name)?;
        let vocab = self.cfg.vocab_size;
        let t = (token as usize) % vocab;
        let row = &embed[t * hidden..(t + 1) * hidden];
        for i in 0..hidden {
            out[i] = f16::from_bits(row[i]).to_f32();
        }
        Ok(())
    }

    fn rmsnorm_scaled(&self, name: &str, x: &[f32], eps: f32, out: &mut [f32]) -> Result<(), CoreError> {
        let resolved = self.resolve_name(name)?;
        let w_u16 = self.tensor_f16_by_exact_name(&resolved)?;
        let mut w_f32 = vec![0.0f32; x.len()];
        for i in 0..x.len() {
            w_f32[i] = f16::from_bits(w_u16[i]).to_f32();
        }
        rms_norm_f32(x, &w_f32, eps, out);
        Ok(())
    }

    fn linear_f16_out_in(
        &mut self,
        x: &[f32],
        weight_name: &str,
        out_dim: usize,
        in_dim: usize,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        let resolved = self.resolve_name(weight_name)?;
        
        if let Some(ref mut ops) = self.metal_ops {
            let bytes = self.file.tensor_bytes(&resolved)?;
            let w: &[u16] = cast_slice(bytes);
            ops.logits_f16(x, w, out_dim, in_dim, &resolved, out)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
            return Ok(());
        }

        let bytes = self.file.tensor_bytes(&resolved)?;
        let w: &[u16] = cast_slice(bytes);

        if let GraniteLinearBackend::Metal { ctx: _ } = &self.linear_backend {
             // ...
        }

        // CPU fallback
        cellm_kernels::cpu_kernels::matmul_f16_f32(w, out_dim, in_dim, x, out);
        Ok(())
    }

    fn topk_from_logits(&self, logits: &[f32], top_k: usize) -> Result<Vec<(u32, f32)>, CoreError> {
        let _vocab = logits.len();
        let mut indexed: Vec<(u32, f32)> = logits.iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(indexed.into_iter().take(top_k).collect())
    }
}

fn detect_granite_prefix(file: &CellmFile) -> Result<String, CoreError> {
    if file.tensor_index("model.layers.0.input_layernorm.weight").is_some() {
        return Ok("".to_string());
    }
    // Add other prefixes if found in wild Granite models
    Ok("".to_string())
}
