// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! cellm-model: `.cellm` model format + minimal runners.

pub mod cellm_file;
pub use cellm_file::{CellmFile, CellmHeader, CellmTensorIndex};

pub mod llama;
pub mod llama_graph;
pub mod gemma;
pub mod gemma_graph;
pub mod qwen;
pub mod granite;
pub mod lfm;
pub mod lfm_encoder;
pub mod privacy_filter;
pub use privacy_filter::PrivacyFilterRunner;
pub mod deepseek_v4;
pub mod batched;

pub use deepseek_v4::DeepSeekV4Runner;

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_original_max_position_embeddings: Option<usize>,
    pub rope_scaling_low_freq_factor: Option<f32>,
    pub rope_scaling_high_freq_factor: Option<f32>,
    pub attention_softcap: f32,

    // DeepSeek-V4 specifics
    pub hc_mult: Option<usize>,
    pub hc_sinkhorn_iters: Option<usize>,
    pub o_groups: Option<usize>,
    pub o_lora_rank: Option<usize>,
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: Option<usize>,
    pub n_routed_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub hc_eps: Option<f32>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 0,
            hidden_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            head_dim: 0,
            intermediate_size: 0,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_max_position_embeddings: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            attention_softcap: 0.0,
            hc_mult: None,
            hc_sinkhorn_iters: None,
            o_groups: None,
            o_lora_rank: None,
            q_lora_rank: None,
            qk_rope_head_dim: None,
            n_routed_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            hc_eps: None,
        }
    }
}
