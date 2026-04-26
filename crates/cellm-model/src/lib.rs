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
}
