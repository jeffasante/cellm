use std::path::Path;

use bytemuck::cast_slice;
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::rms_norm_f32;
use cellm_kernels::MetalOps;
use cellm_kernels::metal::MetalMatmul;
use cellm_kernels::MetalKernels;
use half::f16;
use rayon::prelude::*;

use crate::{CellmFile, ModelConfig};
use serde_json::Value;

const GEMMA_METAL_LINEAR_MAX_ELEMS: usize = 262_144;

pub struct GemmaRunner {
    file: CellmFile,
    cfg: ModelConfig,
    max_layers: usize,
    eos_token_id: Option<u32>,
    tensor_prefix: String,
    layer_attn: Vec<GemmaLayerAttnSpec>,
    max_q_dim: usize,
    max_kv_dim: usize,
    max_ffn_dim: usize,
    rmsnorm_weight_is_offset: bool,
    is_gemma3_text: bool,
    is_gemma4_text: bool,
    sliding_window: usize,
    sliding_window_pattern: usize,
    gemma4_sliding_mask: Vec<bool>,
    gemma4_shared_kv_layers: usize,
    rope_theta_sliding: f32,
    final_logit_softcapping: Option<f32>,
    per_layer_input: Option<GemmaPerLayerInputSpec>,
    linear_backend: GemmaLinearBackend,
    metal_strict: bool,
    /// Present when a Metal backend is active; drives rms_norm / rope / logits on GPU.
    metal_ops: Option<MetalOps>,
    use_metal_norm: bool,
    use_metal_rope: bool,
    use_metal_logits: bool,
    gemma4_disable_q_prescale: bool,
    gemma4_disable_per_layer_input: bool,
    gemma4_disable_layer_output_scale: bool,
    gemma4_disable_embed_scale: bool,
    gemma4_disable_per_token_embed_scale: bool,
}

enum GemmaLinearBackend {
    Cpu,
    Metal { ctx: MetalMatmul },
}

#[derive(Clone, Debug)]
struct GemmaLayerAttnSpec {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
}

#[derive(Clone, Debug)]
struct GemmaPerLayerInputSpec {
    token_embd_name: String,
    model_proj_name: String,
    proj_norm_name: String,
    per_total_dim: usize,
    per_layer_dim: usize,
}

impl GemmaRunner {
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

        let tensor_prefix = detect_gemma_prefix(&file)?;
        let (layer_attn, max_q_dim, max_kv_dim, max_ffn_dim) = infer_gemma_layer_attn_specs(
            &file,
            &tensor_prefix,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        )?;
        let source_model_type = match &h.source_text_config {
            Some(Value::Object(obj)) => match obj.get("model_type") {
                Some(Value::String(mt)) if !mt.is_empty() => Some(mt.as_str()),
                _ => None,
            },
            _ => None,
        };
        let is_gemma3_text = h.model_type.starts_with("gemma3")
            || source_model_type.is_some_and(|mt| mt.starts_with("gemma3"));
        let is_gemma4_text = h.model_type.starts_with("gemma4")
            || source_model_type.is_some_and(|mt| mt.starts_with("gemma4"));
        let mut rmsnorm_weight_is_offset = !is_gemma4_text;
        if std::env::var("CELLM_GEMMA_FORCE_RMS_OFFSET_ONE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            rmsnorm_weight_is_offset = true;
        }
        // Gemma3 uses mixed attention: sliding layers use local RoPE base 10k, while
        // periodic full-attention layers use the global RoPE theta from config.
        let (sliding_window, sliding_window_pattern, gemma4_sliding_mask, gemma4_shared_kv_layers, rope_theta_sliding) =
            if is_gemma3_text {
                (512usize, 6usize, Vec::new(), 0usize, 10_000.0f32)
            } else if is_gemma4_text {
                // Gemma4 GGUF metadata defaults:
                //   sliding_window=512, sliding_window_pattern=[T,T,T,T,F,...], rope.freq_base_swa=10000.
                // Prefer explicit layer_types metadata from source config, then fallback
                // to inferring sliding layers from smaller q_dim.
                let max_q = max_q_dim.max(1);
                let mask = infer_gemma4_sliding_mask_from_cfg(
                    h.source_text_config.as_ref(),
                    cfg.num_hidden_layers,
                )
                .unwrap_or_else(|| layer_attn.iter().map(|s| s.q_dim < max_q).collect::<Vec<_>>());
                // Shared-KV layout varies by checkpoint/export. Prefer explicit metadata,
                // then infer from missing trailing K/V projection tensors; otherwise disable.
                let shared_kv = infer_gemma4_shared_kv_layers(
                    &file,
                    &tensor_prefix,
                    cfg.num_hidden_layers,
                    h.source_text_config.as_ref(),
                );
                let rope_theta_sliding = source_text_cfg_rope_theta(
                    h.source_text_config.as_ref(),
                    &["rope_parameters", "sliding_attention", "rope_theta"],
                )
                .unwrap_or(10_000.0);
                (512usize, usize::MAX, mask, shared_kv, rope_theta_sliding)
            } else {
                (usize::MAX, usize::MAX, Vec::new(), 0usize, cfg.rope_theta)
            };
        let final_logit_softcapping = if is_gemma4_text { Some(30.0f32) } else { None };
        let per_layer_input = infer_gemma_per_layer_input_spec(
            &file,
            &tensor_prefix,
            cfg.vocab_size,
            cfg.hidden_size,
            cfg.num_hidden_layers,
        )?;

        Ok(Self {
            file,
            cfg: cfg.clone(),
            max_layers: cfg.num_hidden_layers,
            eos_token_id: h.eos_token_id,
            tensor_prefix,
            layer_attn,
            max_q_dim,
            max_kv_dim,
            max_ffn_dim,
            rmsnorm_weight_is_offset,
            is_gemma3_text,
            is_gemma4_text,
            sliding_window,
            sliding_window_pattern,
            gemma4_sliding_mask,
            gemma4_shared_kv_layers,
            rope_theta_sliding,
            final_logit_softcapping,
            per_layer_input,
            linear_backend: GemmaLinearBackend::Cpu,
            metal_strict: false,
            metal_ops: None,
            use_metal_norm: std::env::var("CELLM_GEMMA_USE_METAL_NORM")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            use_metal_rope: std::env::var("CELLM_GEMMA_USE_METAL_ROPE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            use_metal_logits: std::env::var("CELLM_GEMMA_USE_METAL_LOGITS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            gemma4_disable_q_prescale: std::env::var("CELLM_GEMMA4_DISABLE_Q_PRESCALE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            gemma4_disable_per_layer_input: std::env::var("CELLM_GEMMA4_DISABLE_PER_LAYER_INPUT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            gemma4_disable_layer_output_scale: std::env::var("CELLM_GEMMA4_DISABLE_LAYER_OUTPUT_SCALE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            gemma4_disable_embed_scale: std::env::var("CELLM_GEMMA4_DISABLE_EMBED_SCALE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            gemma4_disable_per_token_embed_scale: std::env::var("CELLM_GEMMA4_DISABLE_PER_TOKEN_EMBED_SCALE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
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
    
    pub fn is_gemma3_text(&self) -> bool {
        self.is_gemma3_text
    }

    pub fn enable_metal_linear_backend(&mut self) -> bool {
        match (MetalKernels::create_matmul(), MetalOps::create()) {
            (Ok(ctx), Ok(ops)) => {
                self.linear_backend = GemmaLinearBackend::Metal { ctx };
                self.metal_ops = Some(ops);
                self.metal_strict = false;
                true
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("gemma: failed to enable metal linear backend: {e}");
                self.linear_backend = GemmaLinearBackend::Cpu;
                self.metal_ops = None;
                self.metal_strict = false;
                false
            }
        }
    }

    pub fn enable_metal_full_backend(&mut self) -> bool {
        match (MetalKernels::create_matmul(), MetalOps::create()) {
            (Ok(ctx), Ok(ops)) => {
                self.linear_backend = GemmaLinearBackend::Metal { ctx };
                self.metal_ops = Some(ops);
                self.metal_strict = true;
                // Full-metal mode must enable all eligible Gemma ops on GPU.
                self.use_metal_norm = true;
                self.use_metal_rope = true;
                self.use_metal_logits = true;
                true
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("gemma: failed to enable full metal backend: {e}");
                self.linear_backend = GemmaLinearBackend::Cpu;
                self.metal_ops = None;
                self.metal_strict = false;
                false
            }
        }
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
        let per_layer_input = self.prepare_gemma4_per_layer_inputs(Some(token), &x)?;
        self.step_topk_from_hidden_inner(&x, per_layer_input.as_deref(), pos, page_table, kv_cache, top_k)
    }

    pub fn step_topk_from_hidden(
        &mut self,
        x0: &[f32],
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>, CoreError> {
        let per_layer_input = self.prepare_gemma4_per_layer_inputs(None, x0)?;
        self.step_topk_from_hidden_inner(x0, per_layer_input.as_deref(), pos, page_table, kv_cache, top_k)
    }

    fn step_topk_from_hidden_inner(
        &mut self,
        x0: &[f32],
        per_layer_input: Option<&[f32]>,
        pos: usize,
        page_table: &mut PageTable,
        kv_cache: &mut KVCache,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>, CoreError> {
        let cfg = self.cfg.clone();
        let hidden = cfg.hidden_size;
        if self.max_q_dim == 0 || self.max_kv_dim == 0 {
            return Err(CoreError::Backend(
                "gemma: invalid attention head geometry".into(),
            ));
        }

        // Ensure pagetable covers this token position.
        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut()).map_err(|e| {
                CoreError::Backend(format!("gemma step: page_table append_token failed: {e}"))
            })?;
        } else if pos > page_table.token_count() {
            return Err(CoreError::Backend(format!(
                "gemma step: non-contiguous pos {pos} (token_count={})",
                page_table.token_count()
            )));
        }

        let block_id = page_table.block_for_token(pos).map_err(|e| {
            CoreError::Backend(format!("gemma step: page_table block_for_token failed: {e}"))
        })?;
        let token_off = page_table.offset_in_block(pos).map_err(|e| {
            CoreError::Backend(format!("gemma step: page_table offset_in_block failed: {e}"))
        })?;

        if x0.len() != hidden {
            return Err(CoreError::Backend(format!(
                "gemma step_from_hidden: hidden len mismatch {} != {}",
                x0.len(),
                hidden
            )));
        }
        let mut x = x0.to_vec();

        // Per-layer scratch.
        let mut attn_norm_w = vec![0.0f32; hidden];
        let mut x_norm = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; self.max_q_dim];
        let mut k = vec![0.0f32; self.max_kv_dim];
        let mut v = vec![0.0f32; self.max_kv_dim];
        let mut attn_out = vec![0.0f32; self.max_q_dim];
        let mut attn_proj = vec![0.0f32; hidden];

        let mut q_norm_w = Vec::<f32>::new();
        let mut k_norm_w = Vec::<f32>::new();
        let mut post_attn_norm_w = vec![0.0f32; hidden];
        let mut pre_ffn_norm_w = vec![0.0f32; hidden];
        let mut post_ffn_norm_w = vec![0.0f32; hidden];
        let mut mlp_in = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; self.max_ffn_dim];
        let mut up = vec![0.0f32; self.max_ffn_dim];
        let mut ffn_out = vec![0.0f32; self.max_ffn_dim];
        let mut down = vec![0.0f32; hidden];
        let mut per_gate: Vec<f32> = Vec::new();
        let mut per_proj = vec![0.0f32; hidden];
        let mut per_post_norm_w = vec![0.0f32; hidden];

        let mut gather_bases: Vec<usize> = Vec::new();
        let layout_kv_dim = kv_cache.view().layout.kv_dim();
        if layout_kv_dim < self.max_kv_dim {
            return Err(CoreError::Backend(format!(
                "gemma: kv cache layout too small (layout_kv_dim={} < model_max_kv_dim={})",
                layout_kv_dim, self.max_kv_dim
            )));
        }
        let mut k_store = vec![0.0f32; layout_kv_dim];
        let mut v_store = vec![0.0f32; layout_kv_dim];

        for layer in 0..self.max_layers {
            let (is_kv_shared_layer, kv_shared_source_layer) = if self.is_gemma4_text
                && self.gemma4_shared_kv_layers > 0
                && self.gemma4_shared_kv_layers < self.max_layers
            {
                let first_kv_shared = self.max_layers - self.gemma4_shared_kv_layers;
                if layer >= first_kv_shared {
                    let is_sliding = self.gemma4_sliding_mask.get(layer).copied().unwrap_or(false);
                    let src = (0..first_kv_shared)
                        .rev()
                        .find(|&i| self.gemma4_sliding_mask.get(i).copied().unwrap_or(false) == is_sliding)
                        .ok_or_else(|| {
                            CoreError::Backend(format!(
                                "gemma: failed to resolve shared-kv source layer for layer {} (is_sliding={})",
                                layer, is_sliding
                            ))
                        })?;
                    (true, src)
                } else {
                    (false, layer)
                }
            } else {
                (false, layer)
            };
            let layer_spec = self
                .layer_attn
                .get(layer)
                .ok_or_else(|| CoreError::Backend(format!("gemma: missing layer attn spec for layer {layer}")))?;
            let n_heads = layer_spec.n_heads;
            let n_kv_heads = layer_spec.n_kv_heads;
            let head_dim = layer_spec.head_dim;
            let kv_head_dim = layer_spec.kv_head_dim;
            let q_dim = layer_spec.q_dim;
            let kv_dim = layer_spec.kv_dim;
            let ffn_dim = layer_spec.ffn_dim;
            let q_slice = &mut q[..q_dim];
            let k_slice = &mut k[..kv_dim];
            let v_slice = &mut v[..kv_dim];
            let attn_out_slice = &mut attn_out[..q_dim];
            let gate_slice = &mut gate[..ffn_dim];
            let up_slice = &mut up[..ffn_dim];
            let ffn_out_slice = &mut ffn_out[..ffn_dim];

            let is_full_attention_layer = if self.is_gemma3_text {
                // Gemma3 pattern: 5 sliding layers, then 1 full-attention layer.
                self.sliding_window_pattern != 0 && (layer + 1) % self.sliding_window_pattern == 0
            } else if self.is_gemma4_text {
                // Gemma4: mask=true means sliding-window layer, false means full attention.
                !self.gemma4_sliding_mask.get(layer).copied().unwrap_or(false)
            } else {
                true
            };
            let layer_rope_theta = if (self.is_gemma3_text || self.is_gemma4_text)
                && !is_full_attention_layer
            {
                self.rope_theta_sliding
            } else {
                cfg.rope_theta
            };

            // Attention input norm 
            let use_metal_norm = self.metal_ops.is_some() && self.use_metal_norm;
            if use_metal_norm {
                let w = self
                    .tensor_f16(&format!("model.layers.{layer}.input_layernorm.weight"))?
                    .to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                self.metal_ops
                    .as_mut()
                    .unwrap()
                    .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, "gemma.norm", &mut x_norm)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.input_layernorm.weight"),
                    &mut attn_norm_w,
                )?;
                rms_norm_f32(&x, &attn_norm_w, cfg.rms_norm_eps, &mut x_norm);
            }

            // Gemma uses rotate_half-style RoPE; keep RoPE on CPU for Gemma3 to
            // preserve correctness until Metal path matches this convention.
            let use_metal_rope = self.metal_ops.is_some() && self.use_metal_rope;

            // QKV projections (HF weights are [out, in]).
            let q_name = format!("model.layers.{layer}.self_attn.q_proj.weight");
            let k_name = format!("model.layers.{layer}.self_attn.k_proj.weight");
            let v_name = format!("model.layers.{layer}.self_attn.v_proj.weight");
            if !is_kv_shared_layer {
                let fused_qkv = self.linear_qkv_f16_out_in(
                    &x_norm,
                    &q_name,
                    q_dim,
                    &k_name,
                    kv_dim,
                    &v_name,
                    kv_dim,
                    hidden,
                    q_slice,
                    k_slice,
                    v_slice,
                )?;
                if !fused_qkv {
                    self.linear_f16_out_in(&x_norm, &q_name, q_dim, hidden, q_slice)?;
                    self.linear_f16_out_in(&x_norm, &k_name, kv_dim, hidden, k_slice)?;
                    self.linear_f16_out_in(&x_norm, &v_name, kv_dim, hidden, v_slice)?;
                }
            } else {
                self.linear_f16_out_in(&x_norm, &q_name, q_dim, hidden, q_slice)?;
            }


            // ── Gemma per-head Q/K RMSNorm before RoPE ────────────────────────
            if use_metal_norm {
                let qw = self.tensor_f16(
                    &format!("model.layers.{layer}.self_attn.q_norm.weight"))?.to_vec();
                let kw = if !is_kv_shared_layer {
                    Some(
                        self.tensor_f16(
                            &format!("model.layers.{layer}.self_attn.k_norm.weight"),
                        )?
                        .to_vec(),
                    )
                } else {
                    None
                };
                let add_one = self.rmsnorm_weight_is_offset;
                let ops = self.metal_ops.as_ref().unwrap();
                for hidx in 0..n_heads {
                    let seg = &mut q_slice[hidx * head_dim..(hidx + 1) * head_dim];
                    let inp = seg.to_vec();
                    ops.rms_norm_f16w(&inp, &qw, cfg.rms_norm_eps, add_one, "gemma.norm", seg)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                if !is_kv_shared_layer {
                    let kw = kw.as_ref().unwrap();
                    for hidx in 0..n_kv_heads {
                        let seg = &mut k_slice[hidx * kv_head_dim..(hidx + 1) * kv_head_dim];
                        let inp = seg.to_vec();
                        ops.rms_norm_f16w(&inp, &kw, cfg.rms_norm_eps, add_one, "gemma.norm", seg)
                            .map_err(|e| CoreError::Backend(e.to_string()))?;
                    }
                }
            } else {
                q_norm_w.resize(head_dim, 0.0);
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.self_attn.q_norm.weight"),
                    &mut q_norm_w,
                )?;
                for hidx in 0..n_heads {
                    let start = hidx * head_dim;
                    let end = start + head_dim;
                    self.rms_norm_inplace_segment(&mut q_slice[start..end], &q_norm_w, cfg.rms_norm_eps);
                }
                if !is_kv_shared_layer {
                    k_norm_w.resize(kv_head_dim, 0.0);
                    self.rmsnorm_weight(
                        &format!("model.layers.{layer}.self_attn.k_norm.weight"),
                        &mut k_norm_w,
                    )?;
                    for hidx in 0..n_kv_heads {
                        let start = hidx * kv_head_dim;
                        let end = start + kv_head_dim;
                        self.rms_norm_inplace_segment(&mut k_slice[start..end], &k_norm_w, cfg.rms_norm_eps);
                    }
                }
            }

            // V norm uses Gemma4 RMSNorm without scale.
            if !is_kv_shared_layer && self.is_gemma4_text {
                for hidx in 0..n_kv_heads {
                    let start = hidx * kv_head_dim;
                    let end = start + kv_head_dim;
                    self.rms_norm_inplace_no_weight_segment(&mut v_slice[start..end], cfg.rms_norm_eps);
                }
            }

            // ── RoPE ──────────────
            if use_metal_rope {
                let ops = self.metal_ops.as_ref().unwrap();
                ops.rope_half_f32(q_slice, n_heads, head_dim, head_dim, pos, layer_rope_theta)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                if !is_kv_shared_layer {
                    ops.rope_half_f32(
                        k_slice,
                        n_kv_heads,
                        kv_head_dim,
                        kv_head_dim,
                        pos,
                        layer_rope_theta,
                    )
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
            } else {
                rope_inplace_rotate_half_f32(q_slice, n_heads, head_dim, pos, layer_rope_theta);
                if !is_kv_shared_layer {
                    rope_inplace_rotate_half_f32(k_slice, n_kv_heads, kv_head_dim, pos, layer_rope_theta);
                }
            }

            // Write new token K/V into paged cache.
            if !is_kv_shared_layer {
                if kv_dim == layout_kv_dim {
                    k_store.copy_from_slice(k_slice);
                    v_store.copy_from_slice(v_slice);
                } else {
                    k_store.fill(0.0);
                    v_store.fill(0.0);
                    k_store[..kv_dim].copy_from_slice(k_slice);
                    v_store[..kv_dim].copy_from_slice(v_slice);
                }
                let mut cv = kv_cache.view_mut();
                cv.write_token(block_id, layer, token_off, &k_store, &v_store)?;
            }

            // Gather historical K/V and run attention for this token.
            let seq = page_table.token_count();
            let cr = kv_cache.view();
            gather_bases.clear();
            let start_tpos = if (self.is_gemma3_text || self.is_gemma4_text) && !is_full_attention_layer {
                seq.saturating_sub(self.sliding_window)
            } else {
                0
            };
            let gather_len = seq.saturating_sub(start_tpos);
            gather_bases.reserve(gather_len);
            for tpos in start_tpos..seq {
                let b = page_table.block_for_token(tpos).map_err(|e| {
                    CoreError::Backend(format!("gemma: block_for_token failed: {e}"))
                })?;
                let o = page_table.offset_in_block(tpos).map_err(|e| {
                    CoreError::Backend(format!("gemma: offset_in_block failed: {e}"))
                })?;
                gather_bases.push(cr.layout.token_base_elem(b, kv_shared_source_layer, o)?);
            }
            // Gemma4 attention uses scaling=1.0 in HF; KV kernel applies 1/sqrt(head_dim),
            // so pre-scale queries by sqrt(head_dim) to match Gemma4 behavior.
            let q_attn_scaled: Option<Vec<f32>> = if self.is_gemma4_text && !self.gemma4_disable_q_prescale {
                let s = (head_dim as f32).sqrt();
                Some(q_slice.iter().map(|&v| v * s).collect())
            } else {
                None
            };
            let q_for_attn: &[f32] = q_attn_scaled.as_deref().unwrap_or(q_slice);
            cr.attention_single_token_gqa_from_bases(
                &gather_bases,
                q_for_attn,
                n_heads,
                n_kv_heads,
                head_dim, None,
                attn_out_slice,
            )?;

            // o_proj: hidden <- hidden
            self.linear_f16_out_in(
                attn_out_slice,
                &format!("model.layers.{layer}.self_attn.o_proj.weight"),
                hidden,
                q_dim,
                &mut attn_proj,
            )?;

            let x_residual = x.clone();

            // ── Post-attention norm on attn branch, then residual add
            if use_metal_norm {
                let w = self.tensor_f16(
                    &format!("model.layers.{layer}.post_attention_layernorm.weight"))?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                let mut x_out = vec![0.0f32; hidden];
                self.metal_ops.as_ref().unwrap()
                    .rms_norm_f16w(&attn_proj, &w, cfg.rms_norm_eps, add_one, "gemma.norm", &mut x_out)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                for i in 0..hidden { x[i] = x_out[i] + x_residual[i]; }
            } else {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                    &mut post_attn_norm_w,
                )?;
                rms_norm_f32(&attn_proj, &post_attn_norm_w, cfg.rms_norm_eps, &mut x_norm);
                for i in 0..hidden { x[i] = x_norm[i] + x_residual[i]; }
            }

            // ── Pre-FFN norm ───────
            if use_metal_norm {
                let w = self.tensor_f16(
                    &format!("model.layers.{layer}.pre_feedforward_layernorm.weight"))?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                let ck = format!("gemma.layer.{layer}.attn_norm");
                self.metal_ops.as_ref().unwrap()
                    .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, &ck, &mut mlp_in)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
            } else {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.pre_feedforward_layernorm.weight"),
                    &mut pre_ffn_norm_w,
                )?;
                rms_norm_f32(&x, &pre_ffn_norm_w, cfg.rms_norm_eps, &mut mlp_in);
            }

            self.linear_f16_out_in(
                &mlp_in,
                &format!("model.layers.{layer}.mlp.gate_proj.weight"),
                ffn_dim,
                hidden,
                gate_slice,
            )?;
            self.linear_f16_out_in(
                &mlp_in,
                &format!("model.layers.{layer}.mlp.up_proj.weight"),
                ffn_dim,
                hidden,
                up_slice,
            )?;
            for i in 0..ffn_dim {
                ffn_out_slice[i] = gelu_tanh_f32(gate_slice[i]) * up_slice[i];
            }
            self.linear_f16_out_in(
                ffn_out_slice,
                &format!("model.layers.{layer}.mlp.down_proj.weight"),
                hidden,
                ffn_dim,
                &mut down,
            )?;
            let x_residual = x.clone();

            // ── Post-FFN norm on MLP branch, then residual add ──────
            if use_metal_norm {
                let wffn = self.tensor_f16(
                    &format!("model.layers.{layer}.post_feedforward_layernorm.weight"))?.to_vec();
                let add_one = self.rmsnorm_weight_is_offset;
                let mut x_out = vec![0.0f32; hidden];
                let ck = format!("gemma.layer.{layer}.ffn_norm");
                self.metal_ops.as_ref().unwrap()
                    .rms_norm_f16w(&down, &wffn, cfg.rms_norm_eps, add_one, &ck, &mut x_out)
                    .map_err(|e| CoreError::Backend(e.to_string()))?;
                for i in 0..hidden { x[i] = x_out[i] + x_residual[i]; }
            } else {
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.post_feedforward_layernorm.weight"),
                    &mut post_ffn_norm_w,
                )?;
                rms_norm_f32(&down, &post_ffn_norm_w, cfg.rms_norm_eps, &mut x_norm);
                for i in 0..hidden { x[i] = x_norm[i] + x_residual[i]; }
            }

            if !self.gemma4_disable_per_layer_input {
            if let (Some(spec), Some(per_input_all)) = (&self.per_layer_input, per_layer_input) {
                let per_layer_dim = spec.per_layer_dim;
                let l0 = layer * per_layer_dim;
                let l1 = l0 + per_layer_dim;
                if l1 > per_input_all.len() {
                    return Err(CoreError::Backend(format!(
                        "gemma: per-layer input buffer too small for layer {} (need {}, have {})",
                        layer,
                        l1,
                        per_input_all.len()
                    )));
                }
                let per_input = &per_input_all[l0..l1];
                let residual = x.clone();
                per_gate.resize(per_layer_dim, 0.0);
                self.linear_f16_out_in(
                    &x,
                    &format!("model.layers.{layer}.per_layer_input_gate.weight"),
                    per_layer_dim,
                    hidden,
                    &mut per_gate,
                )?;
                for i in 0..per_layer_dim {
                    per_gate[i] = gelu_tanh_f32(per_gate[i]) * per_input[i];
                }
                self.linear_f16_out_in(
                    &per_gate,
                    &format!("model.layers.{layer}.per_layer_projection.weight"),
                    hidden,
                    per_layer_dim,
                    &mut per_proj,
                )?;
                self.rmsnorm_weight(
                    &format!("model.layers.{layer}.post_per_layer_input_norm.weight"),
                    &mut per_post_norm_w,
                )?;
                rms_norm_f32(&per_proj, &per_post_norm_w, cfg.rms_norm_eps, &mut x_norm);
                for i in 0..hidden {
                    x[i] = residual[i] + x_norm[i];
                }
            }
            }

            if self.is_gemma4_text && !self.gemma4_disable_layer_output_scale {
                let scale_candidates = [
                    format!("model.layers.{layer}.layer_output_scale.weight"),
                    format!("model.layers.{layer}.layer_scalar"),
                ];
                for cand in scale_candidates {
                    if let Ok(scale_name) = self.resolve_name(&cand) {
                        let scale = self.tensor_f16_by_exact_name(&scale_name)?;
                        if scale.len() == 1 {
                            let s = f16::from_bits(scale[0]).to_f32();
                            for i in 0..hidden {
                                x[i] *= s;
                            }
                        } else if scale.len() == hidden {
                            for i in 0..hidden {
                                let s = f16::from_bits(scale[i]).to_f32();
                                x[i] *= s;
                            }
                        } else {
                            return Err(CoreError::Backend(format!(
                                "gemma: invalid layer scale shape at layer {} from {} (len={} expected 1 or {})",
                                layer,
                                scale_name,
                                scale.len(),
                                hidden
                            )));
                        }
                        break;
                    }
                }
            }

        }

        // Final norm.
        let use_metal_norm = self.metal_ops.is_some() && self.use_metal_norm;
        let mut x_final = vec![0.0f32; hidden];
        if use_metal_norm {
            let w = self.tensor_f16("model.norm.weight")?.to_vec();
            let add_one = self.rmsnorm_weight_is_offset;
            self.metal_ops.as_ref().unwrap()
                .rms_norm_f16w(&x, &w, cfg.rms_norm_eps, add_one, "gemma.final_norm", &mut x_final)
                .map_err(|e| CoreError::Backend(e.to_string()))?;
        } else {
            let mut norm_w = vec![0.0f32; hidden];
            self.rmsnorm_weight("model.norm.weight", &mut norm_w)?;
            rms_norm_f32(&x, &norm_w, cfg.rms_norm_eps, &mut x_final);
        }

        // Logits via tied embeddings / lm_head.
        let vocab = cfg.vocab_size;
        let k = top_k.max(1).min(vocab);

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
        let lm_dtype = lm_meta.dtype.clone();
        let use_metal_logits =
            self.metal_ops.is_some() && self.use_metal_logits && lm_dtype != "i4" && lm_dtype != "i2";

        // Compute all vocab logits on GPU when Metal is active, otherwise CPU.
        let all_logits: Vec<f32>;
        if use_metal_logits {
            let mut buf = vec![0.0f32; vocab];
            match lm_dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16_by_exact_name(&lm_src_resolved)?;
                    self.metal_ops.as_ref().unwrap()
                        .logits_f16(&x_final, w, vocab, hidden, &lm_src_resolved, &mut buf)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                "i8" => {
                    let w = self.tensor_i8_by_exact_name(&lm_src_resolved)?;
                    let s = self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?;
                    self.metal_ops.as_ref().unwrap()
                        .logits_i8(&x_final, w, s, vocab, hidden, &lm_src_resolved, &mut buf)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                }
                other => return Err(CoreError::Backend(format!(
                    "unsupported lm dtype {other} for {lm_src_resolved}"))),
            }
            all_logits = buf;
        } else {
            // CPU path.
            let lm_src_f16 = if lm_dtype == "f16" {
                Some(self.tensor_f16_by_exact_name(&lm_src_resolved)?)
            } else { None };
            let lm_src_i8 = if lm_dtype == "i8" {
                Some(self.tensor_i8_by_exact_name(&lm_src_resolved)?)
            } else { None };
            let lm_src_scales = if lm_dtype == "i8" {
                Some(self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?)
            } else { None };
            let lm_src_i4 = if lm_dtype == "i4" {
                Some(self.tensor_u8_by_exact_name(&lm_src_resolved)?)
            } else { None };
            let lm_src_i4_scales = if lm_dtype == "i4" {
                Some(self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?)
            } else { None };
            let lm_src_i2 = if lm_dtype == "i2" {
                Some(self.tensor_u8_by_exact_name(&lm_src_resolved)?)
            } else { None };
            let lm_src_i2_scales = if lm_dtype == "i2" {
                Some(self.tensor_f16_by_exact_name(&format!("{lm_src_resolved}.qscale"))?)
            } else { None };
            let mut buf = vec![0.0f32; vocab];
            if should_parallel_linear_cpu(vocab, hidden) {
                buf.par_iter_mut().enumerate().for_each(|(vid, out_v)| {
                    let mut dot = 0.0f32;
                    if let Some(wf16) = lm_src_f16 {
                        let row = &wf16[vid * hidden..(vid + 1) * hidden];
                        for i in 0..hidden {
                            dot += x_final[i] * f16::from_bits(row[i]).to_f32();
                        }
                    } else if let (Some(wi8), Some(scales)) = (lm_src_i8, lm_src_scales) {
                        let row = &wi8[vid * hidden..(vid + 1) * hidden];
                        let scale = f16::from_bits(scales[vid]).to_f32();
                        for i in 0..hidden {
                            dot += x_final[i] * (row[i] as f32) * scale;
                        }
                    } else if let (Some(wi4), Some(scales)) = (lm_src_i4, lm_src_i4_scales) {
                        let row_stride = hidden.div_ceil(2);
                        let row = &wi4[vid * row_stride..(vid + 1) * row_stride];
                        let scale = f16::from_bits(scales[vid]).to_f32();
                        dot = dot_i4_scaled_row(row, x_final.as_slice(), scale);
                    } else if let (Some(wi2), Some(scales)) = (lm_src_i2, lm_src_i2_scales) {
                        let row_stride = hidden.div_ceil(4);
                        let row = &wi2[vid * row_stride..(vid + 1) * row_stride];
                        let scale = f16::from_bits(scales[vid]).to_f32();
                        dot = dot_i2_scaled_row(row, x_final.as_slice(), scale);
                    }
                    *out_v = dot;
                });
            } else {
                for vid in 0..vocab {
                    let mut dot = 0.0f32;
                    if let Some(wf16) = lm_src_f16 {
                        let row = &wf16[vid * hidden..(vid + 1) * hidden];
                        for i in 0..hidden {
                            dot += x_final[i] * f16::from_bits(row[i]).to_f32();
                        }
                    } else if let (Some(wi8), Some(scales)) = (lm_src_i8, lm_src_scales) {
                        let row = &wi8[vid * hidden..(vid + 1) * hidden];
                        let scale = f16::from_bits(scales[vid]).to_f32();
                        for i in 0..hidden {
                            dot += x_final[i] * (row[i] as f32) * scale;
                        }
                    } else if let (Some(wi4), Some(scales)) = (lm_src_i4, lm_src_i4_scales) {
                        let row_stride = hidden.div_ceil(2);
                        let row = &wi4[vid * row_stride..(vid + 1) * row_stride];
                        let scale = f16::from_bits(scales[vid]).to_f32();
                        dot = dot_i4_scaled_row(row, &x_final, scale);
                    } else if let (Some(wi2), Some(scales)) = (lm_src_i2, lm_src_i2_scales) {
                        let row_stride = hidden.div_ceil(4);
                        let row = &wi2[vid * row_stride..(vid + 1) * row_stride];
                        let scale = f16::from_bits(scales[vid]).to_f32();
                        dot = dot_i2_scaled_row(row, &x_final, scale);
                    } else {
                        dot = f32::NAN;
                    }
                    buf[vid] = dot;
                }
            }
            if buf.iter().any(|v| v.is_nan()) {
                return Err(CoreError::Backend(format!(
                    "unsupported lm dtype {lm_dtype} for {lm_src_resolved}"
                )));
            }
            all_logits = buf;
        }

        // Top-k selection from flat logits (always on CPU – tiny work).
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut min_idx = 0usize;
        let mut min_val = f32::INFINITY;
        for (vid, &raw_dot) in all_logits.iter().enumerate() {
            let dot = if let Some(cap) = self.final_logit_softcapping {
                if cap > 0.0 {
                    cap * (raw_dot / cap).tanh()
                } else {
                    raw_dot
                }
            } else {
                raw_dot
            };
            if top.len() < k {
                top.push((vid as u32, dot));
                if dot < min_val { min_val = dot; min_idx = top.len() - 1; }
            } else if dot > min_val {
                top[min_idx] = (vid as u32, dot);
                min_val = top[0].1; min_idx = 0;
                for (i, &(_, s)) in top.iter().enumerate().skip(1) {
                    if s < min_val { min_val = s; min_idx = i; }
                }
            }
        }
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
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

    fn tensor_u8_by_exact_name(&self, resolved: &str) -> Result<&[u8], CoreError> {
        self.file.tensor_bytes(resolved)
    }

    fn tensor_meta_by_exact_name(&self, resolved: &str) -> Option<&crate::CellmTensorIndex> {
        self.file.tensor_index(resolved)
    }

    fn embed_token(&self, token: u32, out: &mut [f32]) -> Result<(), CoreError> {
        let hidden = out.len();
        let name = "model.embed_tokens.weight";
        let resolved = self.resolve_name(name)?;
        let meta = self.tensor_meta_by_exact_name(&resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {resolved}"))
        })?;
        let vocab = self.cfg.vocab_size;
        let t = (token as usize) % vocab;

        let embed_scale = if self.is_gemma4_text && self.gemma4_disable_embed_scale {
            1.0
        } else {
            (hidden as f32).sqrt()
        };

        match meta.dtype.as_str() {
            "f16" => {
                let embed = self.tensor_f16_by_exact_name(&resolved)?;
                let row = &embed[t * hidden..(t + 1) * hidden];
                for i in 0..hidden {
                    out[i] = f16::from_bits(row[i]).to_f32() * embed_scale;
                }
            }
            "i8" => {
                let embed = self.tensor_i8_by_exact_name(&resolved)?;
                let scales = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                let scale = f16::from_bits(scales[t]).to_f32() * embed_scale;
                let row = &embed[t * hidden..(t + 1) * hidden];
                for i in 0..hidden {
                    out[i] = (row[i] as f32) * scale;
                }
            }
            "i4" => {
                let embed = self.tensor_u8_by_exact_name(&resolved)?;
                let scales = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                let scale = f16::from_bits(scales[t]).to_f32() * embed_scale;
                let row_stride = hidden.div_ceil(2);
                let row = &embed[t * row_stride..(t + 1) * row_stride];
                for i in 0..hidden {
                    out[i] = unpack_i4(row, i) * scale;
                }
            }
            "i2" => {
                let embed = self.tensor_u8_by_exact_name(&resolved)?;
                let scales = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                let scale = f16::from_bits(scales[t]).to_f32() * embed_scale;
                let row_stride = hidden.div_ceil(4);
                let row = &embed[t * row_stride..(t + 1) * row_stride];
                for i in 0..hidden {
                    out[i] = unpack_i2(row, i) * scale;
                }
            }
            other => {
                return Err(CoreError::Backend(format!(
                    "unsupported embed dtype for {name}: {other}"
                )));
            }
        }
        Ok(())
    }

    fn prepare_gemma4_per_layer_inputs(
        &mut self,
        token: Option<u32>,
        hidden_state: &[f32],
    ) -> Result<Option<Vec<f32>>, CoreError> {
        if !self.is_gemma4_text {
            return Ok(None);
        }
        let Some(spec) = &self.per_layer_input else {
            return Ok(None);
        };
        let per_total_dim = spec.per_total_dim;
        let per_layer_dim = spec.per_layer_dim;
        let token_embd_name = spec.token_embd_name.clone();
        let model_proj_name = spec.model_proj_name.clone();
        let proj_norm_name = spec.proj_norm_name.clone();
        if hidden_state.len() != self.cfg.hidden_size {
            return Err(CoreError::Backend(format!(
                "gemma: hidden state len mismatch {} != {}",
                hidden_state.len(),
                self.cfg.hidden_size
            )));
        }

        let mut per_token = vec![0.0f32; per_total_dim];
        if let Some(tok) = token {
            let t = (tok as usize) % self.cfg.vocab_size;
            let scale = if self.gemma4_disable_per_token_embed_scale {
                1.0
            } else {
                (per_layer_dim as f32).sqrt()
            };
            let te_meta = self
                .tensor_meta_by_exact_name(&token_embd_name)
                .ok_or_else(|| CoreError::Backend(format!("unknown tensor {token_embd_name}")))?;
            match te_meta.dtype.as_str() {
                "f16" => {
                    let te = self.tensor_f16_by_exact_name(&token_embd_name)?;
                    let te_row = &te[t * per_total_dim..(t + 1) * per_total_dim];
                    for i in 0..per_total_dim {
                        per_token[i] = f16::from_bits(te_row[i]).to_f32() * scale;
                    }
                }
                "i8" => {
                    let te = self.tensor_i8_by_exact_name(&token_embd_name)?;
                    let te_scales =
                        self.tensor_f16_by_exact_name(&format!("{token_embd_name}.qscale"))?;
                    let row_scale = f16::from_bits(te_scales[t]).to_f32() * scale;
                    let te_row = &te[t * per_total_dim..(t + 1) * per_total_dim];
                    for i in 0..per_total_dim {
                        per_token[i] = (te_row[i] as f32) * row_scale;
                    }
                }
                "i4" => {
                    let te = self.tensor_u8_by_exact_name(&token_embd_name)?;
                    let te_scales =
                        self.tensor_f16_by_exact_name(&format!("{token_embd_name}.qscale"))?;
                    let row_scale = f16::from_bits(te_scales[t]).to_f32() * scale;
                    let row_stride = per_total_dim.div_ceil(2);
                    let te_row = &te[t * row_stride..(t + 1) * row_stride];
                    for i in 0..per_total_dim {
                        per_token[i] = unpack_i4(te_row, i) * row_scale;
                    }
                }
                "i2" => {
                    let te = self.tensor_u8_by_exact_name(&token_embd_name)?;
                    let te_scales =
                        self.tensor_f16_by_exact_name(&format!("{token_embd_name}.qscale"))?;
                    let row_scale = f16::from_bits(te_scales[t]).to_f32() * scale;
                    let row_stride = per_total_dim.div_ceil(4);
                    let te_row = &te[t * row_stride..(t + 1) * row_stride];
                    for i in 0..per_total_dim {
                        per_token[i] = unpack_i2(te_row, i) * row_scale;
                    }
                }
                other => {
                    return Err(CoreError::Backend(format!(
                        "unsupported per-layer token embed dtype for {token_embd_name}: {other}"
                    )));
                }
            }
        }

        let mut projected = vec![0.0f32; per_total_dim];
        self.linear_f16_out_in(
            hidden_state,
            &model_proj_name,
            per_total_dim,
            self.cfg.hidden_size,
            &mut projected,
        )?;
        let proj_scale = 1.0f32 / (self.cfg.hidden_size as f32).sqrt();
        for v in projected.iter_mut() {
            *v *= proj_scale;
        }

        let pn = self.tensor_f16_by_exact_name(&proj_norm_name)?;
        let mut pn_f = vec![0.0f32; per_layer_dim];
        for i in 0..per_layer_dim {
            pn_f[i] = f16::from_bits(pn[i]).to_f32();
        }
        for chunk in projected.chunks_mut(per_layer_dim) {
            self.rms_norm_inplace_segment(chunk, &pn_f, self.cfg.rms_norm_eps);
        }

        if token.is_some() {
            let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
            for i in 0..per_total_dim {
                per_token[i] = (per_token[i] + projected[i]) * inv_sqrt2;
            }
        } else {
            for i in 0..per_total_dim {
                per_token[i] = projected[i];
            }
        }
        Ok(Some(per_token))
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
            let base = f16::from_bits(w[i]).to_f32();
            out[i] = if self.rmsnorm_weight_is_offset {
                base + 1.0
            } else {
                base
            };
        }
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
        let dtype = meta.dtype.clone();
        let disable_metal_i4 = self.is_gemma4_text
            && std::env::var("CELLM_GEMMA4_I4_DISABLE_METAL_LINEAR")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
        let use_metal = self.metal_ops.is_some();
        if use_metal {
            let ctx = self.metal_ops.as_ref().unwrap();
            match dtype.as_str() {
                "f16" => {
                    let w = self.tensor_f16_by_exact_name(&resolved)?;
                    ctx.logits_f16(x, w, out_dim, in_dim, &resolved, out)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                    return Ok(());
                }
                "i8" => {
                    let w = self.tensor_i8_by_exact_name(&resolved)?;
                    let s = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                    ctx.logits_i8(x, w, s, out_dim, in_dim, &resolved, out)
                        .map_err(|e| CoreError::Backend(e.to_string()))?;
                    return Ok(());
                }
                _ => {}
            }
        }

        match dtype.as_str() {
            "f16" => {
                let w = self.tensor_f16_by_exact_name(&resolved)?;
                if w.len() != out_dim * in_dim {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} len mismatch: {} expected {}",
                        w.len(),
                        out_dim * in_dim
                    )));
                }
                cellm_kernels::cpu_kernels::matmul_f16_f32(w, out_dim, in_dim, x, out);
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
                let s = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                if s.len() != out_dim {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} qscale len mismatch: {} expected {}",
                        s.len(),
                        out_dim
                    )));
                }
                cellm_kernels::cpu_kernels::matmul_i8_f32(w, s, out_dim, in_dim, x, out);
            }
            "i4" => {
                let w = self.tensor_u8_by_exact_name(&resolved)?;
                let row_stride = (in_dim + 1) / 2;
                if w.len() != out_dim * row_stride {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} len mismatch: {} expected {}",
                        w.len(),
                        out_dim * row_stride
                    )));
                }
                let s = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                if s.len() != out_dim {
                    return Err(CoreError::Backend(format!(
                        "weight {weight_name} qscale len mismatch: {} expected {}",
                        s.len(),
                        out_dim
                    )));
                }
                cellm_kernels::cpu_kernels::matmul_i4_f32(w, s, out_dim, in_dim, x, out);
            }
            "i2" => {
                let w = self.tensor_u8_by_exact_name(&resolved)?;
                let scales = self.tensor_f16_by_exact_name(&format!("{resolved}.qscale"))?;
                let row_stride = in_dim.div_ceil(4);
                if should_parallel_linear_cpu(out_dim, in_dim) {
                    out.par_iter_mut().enumerate().for_each(|(j, out_j)| {
                        let row = &w[j * row_stride..(j + 1) * row_stride];
                        let scale = f16::from_bits(scales[j]).to_f32();
                        *out_j = dot_i2_scaled_row(row, x, scale);
                    });
                } else {
                    for j in 0..out_dim {
                        let row = &w[j * row_stride..(j + 1) * row_stride];
                        let scale = f16::from_bits(scales[j]).to_f32();
                        let acc = dot_i2_scaled_row(row, x, scale);
                        out[j] = acc;
                    }
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
        let mut candidates = vec![name.to_string()];
        if let Some(base) = name.strip_suffix(".weight") {
            candidates.push(format!("{base}.linear.weight"));
        }

        for cand in candidates {
            if self.file.tensor_index(&cand).is_some() {
                return Ok(cand);
            }
            if !self.tensor_prefix.is_empty() {
                let prefixed = format!("{}{}", self.tensor_prefix, cand);
                if self.file.tensor_index(&prefixed).is_some() {
                    return Ok(prefixed);
                }
            }
            if let Some(suffix) = cand.strip_prefix("model.") {
                let text_model = format!("model.text_model.{suffix}");
                if self.file.tensor_index(&text_model).is_some() {
                    return Ok(text_model);
                }
                let language_model = format!("model.language_model.{suffix}");
                if self.file.tensor_index(&language_model).is_some() {
                    return Ok(language_model);
                }
            }
        }
        Err(CoreError::Backend(format!("unknown tensor {name}")))
    }

    fn rms_norm_inplace_segment(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        let mut sumsq = 0.0f32;
        for &v in x.iter() {
            sumsq += v * v;
        }
        let inv_rms = 1.0f32 / (sumsq / (x.len() as f32) + eps).sqrt();
        for i in 0..x.len() {
            x[i] = x[i] * inv_rms * weight[i];
        }
    }

    fn linear_qkv_f16_out_in(
        &mut self,
        x: &[f32],
        q_weight_name: &str,
        q_out_dim: usize,
        k_weight_name: &str,
        k_out_dim: usize,
        v_weight_name: &str,
        v_out_dim: usize,
        in_dim: usize,
        q_out: &mut [f32],
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<bool, CoreError> {
        if self.metal_ops.is_none() {
            return Ok(false);
        }
        if x.len() != in_dim || q_out.len() != q_out_dim || k_out.len() != k_out_dim || v_out.len() != v_out_dim {
            return Err(CoreError::Backend(format!(
                "gemma qkv dims mismatch: x={} q={} k={} v={} expected in={} q_out={} k_out={} v_out={}",
                x.len(), q_out.len(), k_out.len(), v_out.len(), in_dim, q_out_dim, k_out_dim, v_out_dim
            )));
        }

        let q_resolved = self.resolve_name(q_weight_name)?;
        let k_resolved = self.resolve_name(k_weight_name)?;
        let v_resolved = self.resolve_name(v_weight_name)?;

        let q_meta = self.tensor_meta_by_exact_name(&q_resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {q_resolved}"))
        })?;
        let k_meta = self.tensor_meta_by_exact_name(&k_resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {k_resolved}"))
        })?;
        let v_meta = self.tensor_meta_by_exact_name(&v_resolved).ok_or_else(|| {
            CoreError::Backend(format!("unknown tensor {v_resolved}"))
        })?;

        if q_meta.shape.len() != 2 || k_meta.shape.len() != 2 || v_meta.shape.len() != 2 {
            return Err(CoreError::Backend(format!(
                "gemma qkv fused expects 2D weights, got q={:?} k={:?} v={:?}",
                q_meta.shape, k_meta.shape, v_meta.shape
            )));
        }
        if q_meta.shape != [q_out_dim, in_dim]
            || k_meta.shape != [k_out_dim, in_dim]
            || v_meta.shape != [v_out_dim, in_dim]
        {
            return Err(CoreError::Backend(format!(
                "gemma qkv fused shape mismatch: q={:?} k={:?} v={:?} expected q=[{},{}] k=[{},{}] v=[{},{}]",
                q_meta.shape, k_meta.shape, v_meta.shape, q_out_dim, in_dim, k_out_dim, in_dim, v_out_dim, in_dim
            )));
        }
        if q_meta.dtype != "f16" || k_meta.dtype != "f16" || v_meta.dtype != "f16" {
            return Ok(false);
        }

        let wq = self.tensor_f16_by_exact_name(&q_resolved)?;
        let wk = self.tensor_f16_by_exact_name(&k_resolved)?;
        let wv = self.tensor_f16_by_exact_name(&v_resolved)?;
        let wq = unsafe { std::slice::from_raw_parts(wq.as_ptr(), wq.len()) };
        let wk = unsafe { std::slice::from_raw_parts(wk.as_ptr(), wk.len()) };
        let wv = unsafe { std::slice::from_raw_parts(wv.as_ptr(), wv.len()) };

        self.metal_ops
            .as_mut()
            .unwrap()
            .logits_qkv_f16(
                x,
                wq,
                wk,
                wv,
                q_out_dim,
                k_out_dim,
                v_out_dim,
                in_dim,
                &q_resolved,
                &k_resolved,
                &v_resolved,
                q_out,
                k_out,
                v_out,
            )
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(true)
    }

    fn rms_norm_inplace_no_weight_segment(&self, x: &mut [f32], eps: f32) {
        let mut sumsq = 0.0f32;
        for &v in x.iter() {
            sumsq += v * v;
        }
        let inv_rms = 1.0f32 / (sumsq / (x.len() as f32) + eps).sqrt();
        for v in x.iter_mut() {
            *v *= inv_rms;
        }
    }
}

fn should_parallel_linear_cpu(out_dim: usize, in_dim: usize) -> bool {
    let elems = out_dim.saturating_mul(in_dim);
    elems >= 262_144 && std::thread::available_parallelism().map(|n| n.get() > 1).unwrap_or(false)
}

fn dot_i4_scaled_row(row_packed: &[u8], x: &[f32], scale: f32) -> f32 {
    let mut acc = 0.0f32;
    let mut xi = 0usize;
    for &b in row_packed {
        if xi >= x.len() {
            break;
        }
        let lo = ((b & 0x0f) as i8) - 8;
        acc += x[xi] * ((lo as f32) * scale);
        xi += 1;
        if xi >= x.len() {
            break;
        }
        let hi = ((b >> 4) as i8) - 8;
        acc += x[xi] * ((hi as f32) * scale);
        xi += 1;
    }
    acc
}

fn dot_i2_scaled_row(row_packed: &[u8], x: &[f32], scale: f32) -> f32 {
    let mut acc = 0.0f32;
    let mut xi = 0usize;
    for &b in row_packed {
        for lane in 0..4 {
            if xi >= x.len() {
                return acc;
            }
            let q = (b >> (lane * 2)) & 0x03;
            acc += x[xi] * (dequant_i2(q) * scale);
            xi += 1;
        }
    }
    acc
}

fn gelu_tanh_f32(x: f32) -> f32 {
    let k = 0.797_884_6f32;
    let c = 0.044_715f32;
    0.5f32 * x * (1.0f32 + (k * (x + c * x * x * x)).tanh())
}

fn rope_inplace_rotate_half_f32(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = pos as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + i];
            let x1 = x[base + half + i];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + half + i] = x1 * cos + x0 * sin;
        }
    }
}

fn detect_gemma_prefix(file: &CellmFile) -> Result<String, CoreError> {
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
    if file
        .tensor_index("model.language_model.embed_tokens.weight")
        .is_some()
        && file.tensor_index("model.language_model.norm.weight").is_some()
    {
        return Ok(String::new());
    }
    Err(CoreError::Backend(
        "missing required gemma tensors: model.embed_tokens.weight/model.norm.weight".into(),
    ))
}

fn infer_gemma_layer_attn_specs(
    file: &CellmFile,
    prefix: &str,
    num_layers: usize,
    default_n_heads: usize,
    default_n_kv_heads: usize,
) -> Result<(Vec<GemmaLayerAttnSpec>, usize, usize, usize), CoreError> {
    if default_n_heads == 0 || default_n_kv_heads == 0 {
        return Err(CoreError::Backend(
            "gemma: num_attention_heads/num_key_value_heads must be > 0".into(),
        ));
    }
    let mut specs = Vec::with_capacity(num_layers);
    let mut max_q_dim = 0usize;
    let mut max_kv_dim = 0usize;
    let mut max_ffn_dim = 0usize;

    let resolve_meta = |name: &str| {
        let mut candidates = vec![name.to_string()];
        if let Some(base) = name.strip_suffix(".weight") {
            candidates.push(format!("{base}.linear.weight"));
        }
        for cand in candidates {
            if let Some(t) = file.tensor_index(&cand) {
                return Some(t);
            }
            if !prefix.is_empty() {
                let prefixed = format!("{prefix}{cand}");
                if let Some(t) = file.tensor_index(&prefixed) {
                    return Some(t);
                }
            }
            if let Some(suffix) = cand.strip_prefix("model.") {
                let text_model = format!("model.text_model.{suffix}");
                if let Some(t) = file.tensor_index(&text_model) {
                    return Some(t);
                }
                let language_model = format!("model.language_model.{suffix}");
                if let Some(t) = file.tensor_index(&language_model) {
                    return Some(t);
                }
            }
        }
        None
    };

    for layer in 0..num_layers {
        let q_name = format!("{prefix}model.layers.{layer}.self_attn.q_proj.weight");
        let k_name = format!("{prefix}model.layers.{layer}.self_attn.k_proj.weight");
        let q_meta = resolve_meta(&q_name)
            .ok_or_else(|| CoreError::Backend(format!("gemma: missing tensor {q_name}")))?;
        let k_meta = resolve_meta(&k_name)
            .ok_or_else(|| CoreError::Backend(format!("gemma: missing tensor {k_name}")))?;
        if q_meta.shape.len() != 2 || k_meta.shape.len() != 2 {
            return Err(CoreError::Backend(format!(
                "gemma: expected 2D q_proj/k_proj weights at layer {layer}"
            )));
        }
        let q_out = q_meta.shape[0];
        let k_out = k_meta.shape[0];
        let gate_name = format!("{prefix}model.layers.{layer}.mlp.gate_proj.weight");
        let up_name = format!("{prefix}model.layers.{layer}.mlp.up_proj.weight");
        let down_name = format!("{prefix}model.layers.{layer}.mlp.down_proj.weight");
        let gate_meta = resolve_meta(&gate_name)
            .ok_or_else(|| CoreError::Backend(format!("gemma: missing tensor {gate_name}")))?;
        let up_meta = resolve_meta(&up_name)
            .ok_or_else(|| CoreError::Backend(format!("gemma: missing tensor {up_name}")))?;
        let down_meta = resolve_meta(&down_name)
            .ok_or_else(|| CoreError::Backend(format!("gemma: missing tensor {down_name}")))?;

        let q_norm_name = format!("{prefix}model.layers.{layer}.self_attn.q_norm.weight");
        let k_norm_name = format!("{prefix}model.layers.{layer}.self_attn.k_norm.weight");
        let q_norm_dim = resolve_meta(&q_norm_name)
            .and_then(|t| (t.shape.len() == 1).then_some(t.shape[0]));
        let k_norm_dim = resolve_meta(&k_norm_name)
            .and_then(|t| (t.shape.len() == 1).then_some(t.shape[0]));

        let mut n_heads = default_n_heads;
        let mut n_kv_heads = default_n_kv_heads;

        let head_dim = if let Some(d) = q_norm_dim {
            if d == 0 || q_out % d != 0 {
                return Err(CoreError::Backend(format!(
                    "gemma: invalid q_norm dim at layer {layer}: q_out={q_out} q_norm_dim={d}"
                )));
            }
            n_heads = q_out / d;
            d
        } else if q_out % n_heads == 0 {
            q_out / n_heads
        } else {
            return Err(CoreError::Backend(format!(
                "gemma: q_proj out_dim {q_out} not divisible by n_heads={n_heads} at layer {layer}"
            )));
        };

        let kv_head_dim = if let Some(d) = k_norm_dim {
            if d == 0 || k_out % d != 0 {
                return Err(CoreError::Backend(format!(
                    "gemma: invalid k_norm dim at layer {layer}: k_out={k_out} k_norm_dim={d}"
                )));
            }
            n_kv_heads = k_out / d;
            d
        } else if k_out % n_kv_heads == 0 {
            k_out / n_kv_heads
        } else {
            return Err(CoreError::Backend(format!(
                "gemma: k_proj out_dim {k_out} not divisible by n_kv_heads={n_kv_heads} at layer {layer}"
            )));
        };

        if head_dim != kv_head_dim {
            return Err(CoreError::Backend(format!(
                "gemma: mixed head_dim unsupported at layer {layer} (q_head_dim={head_dim}, kv_head_dim={kv_head_dim})"
            )));
        }

        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * kv_head_dim;
        if q_dim != q_out || kv_dim != k_out {
            return Err(CoreError::Backend(format!(
                "gemma: inferred dims mismatch at layer {layer} (q_dim={q_dim} q_out={q_out} kv_dim={kv_dim} k_out={k_out})"
            )));
        }
        if gate_meta.shape.len() != 2
            || up_meta.shape.len() != 2
            || down_meta.shape.len() != 2
        {
            return Err(CoreError::Backend(format!(
                "gemma: expected 2D MLP weights at layer {layer}"
            )));
        }
        if gate_meta.shape[1] != k_meta.shape[1] || up_meta.shape[1] != k_meta.shape[1] {
            return Err(CoreError::Backend(format!(
                "gemma: MLP input dim mismatch at layer {layer} (gate={:?} up={:?} hidden={})",
                gate_meta.shape,
                up_meta.shape,
                k_meta.shape[1]
            )));
        }
        if down_meta.shape[0] != k_meta.shape[1] || down_meta.shape[1] != gate_meta.shape[0] {
            return Err(CoreError::Backend(format!(
                "gemma: down_proj shape mismatch at layer {layer} (down={:?} gate={:?} hidden={})",
                down_meta.shape,
                gate_meta.shape,
                k_meta.shape[1]
            )));
        }
        if up_meta.shape[0] != gate_meta.shape[0] {
            return Err(CoreError::Backend(format!(
                "gemma: gate/up out_dim mismatch at layer {layer} (gate={} up={})",
                gate_meta.shape[0], up_meta.shape[0]
            )));
        }
        let ffn_dim = gate_meta.shape[0];

        max_q_dim = max_q_dim.max(q_dim);
        max_kv_dim = max_kv_dim.max(kv_dim);
        max_ffn_dim = max_ffn_dim.max(ffn_dim);
        specs.push(GemmaLayerAttnSpec {
            n_heads,
            n_kv_heads,
            head_dim,
            kv_head_dim,
            q_dim,
            kv_dim,
            ffn_dim,
        });
    }
    Ok((specs, max_q_dim, max_kv_dim, max_ffn_dim))
}

fn infer_gemma_per_layer_input_spec(
    file: &CellmFile,
    prefix: &str,
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
) -> Result<Option<GemmaPerLayerInputSpec>, CoreError> {
    let candidates_for = |name: &str| -> Vec<String> {
        let mut out = vec![name.to_string()];
        match name {
            "model.per_layer_token_embd.weight" => {
                out.push("model.embed_tokens_per_layer.weight".to_string());
            }
            "model.per_layer_model_proj.weight" => {
                out.push("model.per_layer_model_projection.weight".to_string());
            }
            "model.per_layer_proj_norm.weight" => {
                out.push("model.per_layer_projection_norm.weight".to_string());
            }
            _ => {}
        }
        out
    };
    let cand = |name: &str| -> Option<String> {
        for base in candidates_for(name) {
            if file.tensor_index(&base).is_some() {
                return Some(base);
            }
            if !prefix.is_empty() {
                let p = format!("{prefix}{base}");
                if file.tensor_index(&p).is_some() {
                    return Some(p);
                }
            }
            if let Some(suffix) = base.strip_prefix("model.") {
                let text_model = format!("model.text_model.{suffix}");
                if file.tensor_index(&text_model).is_some() {
                    return Some(text_model);
                }
                let language_model = format!("model.language_model.{suffix}");
                if file.tensor_index(&language_model).is_some() {
                    return Some(language_model);
                }
            }
        }
        None
    };
    let Some(token_embd_name) = cand("model.per_layer_token_embd.weight") else {
        return Ok(None);
    };
    let model_proj_name = cand("model.per_layer_model_proj.weight").ok_or_else(|| {
        CoreError::Backend("gemma: per_layer_token_embd present but per_layer_model_proj missing".into())
    })?;
    let proj_norm_name = cand("model.per_layer_proj_norm.weight").ok_or_else(|| {
        CoreError::Backend("gemma: per_layer_token_embd present but per_layer_proj_norm missing".into())
    })?;

    let te = file
        .tensor_index(&token_embd_name)
        .ok_or_else(|| CoreError::Backend(format!("gemma: missing tensor {}", token_embd_name)))?;
    let mp = file
        .tensor_index(&model_proj_name)
        .ok_or_else(|| CoreError::Backend(format!("gemma: missing tensor {}", model_proj_name)))?;
    let pn = file
        .tensor_index(&proj_norm_name)
        .ok_or_else(|| CoreError::Backend(format!("gemma: missing tensor {}", proj_norm_name)))?;
    if te.shape.len() != 2 || mp.shape.len() != 2 || pn.shape.len() != 1 {
        return Err(CoreError::Backend(
            "gemma: per_layer tensors must be 2D/2D/1D".into(),
        ));
    }
    if te.shape[0] != vocab_size {
        return Err(CoreError::Backend(format!(
            "gemma: per_layer_token_embd vocab mismatch {} != {}",
            te.shape[0], vocab_size
        )));
    }
    let per_total_dim = te.shape[1];
    let per_layer_dim = pn.shape[0];
    if per_layer_dim == 0 || per_total_dim == 0 || per_total_dim % per_layer_dim != 0 {
        return Err(CoreError::Backend(format!(
            "gemma: invalid per-layer dims total={} per={}",
            per_total_dim, per_layer_dim
        )));
    }
    if per_total_dim / per_layer_dim != num_layers {
        return Err(CoreError::Backend(format!(
            "gemma: per-layer chunk count mismatch {} != num_layers {}",
            per_total_dim / per_layer_dim,
            num_layers
        )));
    }
    if mp.shape[0] != per_total_dim || mp.shape[1] != hidden_size {
        return Err(CoreError::Backend(format!(
            "gemma: per_layer_model_proj shape mismatch {:?}, expected [{},{}]",
            mp.shape, per_total_dim, hidden_size
        )));
    }
    for layer in 0..num_layers {
        let gate_name = cand(&format!("model.layers.{layer}.per_layer_input_gate.weight"))
            .ok_or_else(|| CoreError::Backend(format!(
                "gemma: missing model.layers.{layer}.per_layer_input_gate.weight"
            )))?;
        let gate_meta = file.tensor_index(&gate_name).ok_or_else(|| {
            CoreError::Backend(format!("gemma: missing tensor {}", gate_name))
        })?;
        if gate_meta.shape != vec![per_layer_dim, hidden_size] {
            return Err(CoreError::Backend(format!(
                "gemma: per-layer gate shape mismatch at layer {}: {:?}, expected [{},{}]",
                layer, gate_meta.shape, per_layer_dim, hidden_size
            )));
        }

        let proj_name = cand(&format!("model.layers.{layer}.per_layer_projection.weight"))
            .ok_or_else(|| CoreError::Backend(format!(
                "gemma: missing model.layers.{layer}.per_layer_projection.weight"
            )))?;
        let proj_meta = file.tensor_index(&proj_name).ok_or_else(|| {
            CoreError::Backend(format!("gemma: missing tensor {}", proj_name))
        })?;
        if proj_meta.shape != vec![hidden_size, per_layer_dim] {
            return Err(CoreError::Backend(format!(
                "gemma: per-layer projection shape mismatch at layer {}: {:?}, expected [{},{}]",
                layer, proj_meta.shape, hidden_size, per_layer_dim
            )));
        }

        let post_name = cand(&format!("model.layers.{layer}.post_per_layer_input_norm.weight"))
            .ok_or_else(|| CoreError::Backend(format!(
                "gemma: missing model.layers.{layer}.post_per_layer_input_norm.weight"
            )))?;
        let post_meta = file.tensor_index(&post_name).ok_or_else(|| {
            CoreError::Backend(format!("gemma: missing tensor {}", post_name))
        })?;
        if post_meta.shape != vec![hidden_size] {
            return Err(CoreError::Backend(format!(
                "gemma: post per-layer norm shape mismatch at layer {}: {:?}, expected [{}]",
                layer, post_meta.shape, hidden_size
            )));
        }

        if let Some(scale_name) = cand(&format!("model.layers.{layer}.layer_output_scale.weight")) {
            let scale_meta = file.tensor_index(&scale_name).ok_or_else(|| {
                CoreError::Backend(format!("gemma: missing tensor {}", scale_name))
            })?;
            if scale_meta.shape != vec![1] && scale_meta.shape != vec![hidden_size] {
                return Err(CoreError::Backend(format!(
                    "gemma: layer_output_scale shape mismatch at layer {}: {:?}, expected [1] or [{}]",
                    layer, scale_meta.shape, hidden_size
                )));
            }
        }
    }
    Ok(Some(GemmaPerLayerInputSpec {
        token_embd_name,
        model_proj_name,
        proj_norm_name,
        per_total_dim,
        per_layer_dim,
    }))
}

fn has_tensor_with_gemma_aliases(file: &CellmFile, tensor_prefix: &str, name: &str) -> bool {
    if file.tensor_index(name).is_some() {
        return true;
    }
    if !tensor_prefix.is_empty() {
        let prefixed = format!("{}{}", tensor_prefix, name);
        if file.tensor_index(&prefixed).is_some() {
            return true;
        }
    }
    if let Some(suffix) = name.strip_prefix("model.") {
        let text_model = format!("model.text_model.{suffix}");
        if file.tensor_index(&text_model).is_some() {
            return true;
        }
    }
    false
}

fn source_text_cfg_usize(cfg: Option<&Value>, keys: &[&str]) -> Option<usize> {
    let Value::Object(obj) = cfg? else {
        return None;
    };
    for key in keys {
        match obj.get(*key) {
            Some(Value::Number(n)) => {
                if let Some(v) = n.as_u64() {
                    return Some(v as usize);
                }
            }
            Some(Value::String(s)) => {
                if let Ok(v) = s.parse::<usize>() {
                    return Some(v);
                }
            }
            _ => {}
        }
    }
    None
}

fn source_text_cfg_rope_theta(cfg: Option<&Value>, path: &[&str]) -> Option<f32> {
    let mut cur = cfg?;
    for key in path {
        let Value::Object(obj) = cur else {
            return None;
        };
        cur = obj.get(*key)?;
    }
    match cur {
        Value::Number(n) => n.as_f64().map(|v| v as f32),
        Value::String(s) => s.parse::<f32>().ok(),
        _ => None,
    }
}

fn infer_gemma4_sliding_mask_from_cfg(cfg: Option<&Value>, num_layers: usize) -> Option<Vec<bool>> {
    let Value::Object(obj) = cfg? else {
        return None;
    };
    let Value::Array(layer_types) = obj.get("layer_types")? else {
        return None;
    };
    if layer_types.len() != num_layers {
        return None;
    }
    let mut out = Vec::with_capacity(num_layers);
    for v in layer_types {
        let Value::String(s) = v else {
            return None;
        };
        match s.as_str() {
            "sliding_attention" => out.push(true),
            "full_attention" => out.push(false),
            _ => return None,
        }
    }
    Some(out)
}

fn infer_gemma4_shared_kv_layers(
    file: &CellmFile,
    tensor_prefix: &str,
    num_layers: usize,
    source_text_config: Option<&Value>,
) -> usize {
    if let Ok(v) = std::env::var("CELLM_GEMMA4_SHARED_KV_LAYERS") {
        if let Ok(n) = v.parse::<usize>() {
            return n.min(num_layers);
        }
    }
    if let Some(n) = source_text_cfg_usize(
        source_text_config,
        &[
            "num_kv_shared_layers",
            "kv_shared_layers",
            "shared_kv_layers",
            "num_shared_kv_layers",
        ],
    ) {
        return n.min(num_layers);
    }

    // Infer trailing shared layers from absent K/V projection tensors.
    // If all layers have explicit K/V projection weights, use 0.
    let mut first_missing = num_layers;
    for layer in 0..num_layers {
        let k_name = format!("model.layers.{layer}.self_attn.k_proj.weight");
        let v_name = format!("model.layers.{layer}.self_attn.v_proj.weight");
        let has_k = has_tensor_with_gemma_aliases(file, tensor_prefix, &k_name);
        let has_v = has_tensor_with_gemma_aliases(file, tensor_prefix, &v_name);
        if !(has_k && has_v) {
            first_missing = first_missing.min(layer);
        }
    }
    if first_missing >= num_layers {
        0
    } else {
        num_layers - first_missing
    }
}

fn unpack_i4(packed_row: &[u8], idx: usize) -> f32 {
    let byte = packed_row[idx / 2];
    let nibble = if idx % 2 == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    };
    (nibble as i8 - 8) as f32
}

fn unpack_i2(packed_row: &[u8], idx: usize) -> f32 {
    let byte = packed_row[idx / 4];
    let q = (byte >> ((idx % 4) * 2)) & 0x03;
    dequant_i2(q)
}

fn dequant_i2(q: u8) -> f32 {
    match q & 0x03 {
        0 => -1.5,
        1 => -0.5,
        2 => 0.5,
        _ => 1.5,
    }
}
