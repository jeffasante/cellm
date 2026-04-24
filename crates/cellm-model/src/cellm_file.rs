use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use cellm_core::CoreError;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellmTensorIndex {
    pub name: String,
    pub offset_bytes: u64,
    pub nbytes: u64,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellmHeader {
    pub model_type: String,
    pub source_model_type: Option<String>,
    pub source_safetensors_format: Option<String>,
    pub text_tensor_prefix: Option<String>,
    pub vision_tensor_prefix: Option<String>,
    pub projector_tensor_prefix: Option<String>,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub max_position_embeddings: Option<usize>,
    pub tie_word_embeddings: Option<bool>,
    pub source_torch_dtype: Option<String>,
    pub source_architectures: Option<Vec<String>>,
    pub source_quantization: Option<Value>,
    pub source_quantization_config: Option<Value>,
    pub source_text_config: Option<Value>,
    pub source_vision_config: Option<Value>,
    pub source_projector_config: Option<Value>,
    pub tensors: Vec<CellmTensorIndex>,
}

pub struct CellmFile {
    mmap: Mmap,
    pub header: CellmHeader,
    tensors: HashMap<String, CellmTensorIndex>,
}

impl CellmFile {
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let f = File::open(path)
            .map_err(|e| CoreError::Backend(format!("cellm load: open failed: {e}")))?;
        let mmap = unsafe { Mmap::map(&f) }
            .map_err(|e| CoreError::Backend(format!("cellm load: mmap failed: {e}")))?;

        if mmap.len() < 10 {
            return Err(CoreError::Backend("cellm load: file too small".into()));
        }
        if &mmap[0..5] != b"CELLM" {
            return Err(CoreError::Backend("cellm load: bad magic".into()));
        }
        let version = mmap[5];
        if version != 1 {
            return Err(CoreError::Backend(format!(
                "cellm load: unsupported version {version}"
            )));
        }

        let header_len = u32::from_le_bytes([mmap[6], mmap[7], mmap[8], mmap[9]]) as usize;
        let header_start = 10usize;
        let header_end = header_start + header_len;
        if header_end > mmap.len() {
            return Err(CoreError::Backend(
                "cellm load: header length out of range".into(),
            ));
        }

        let header: CellmHeader = serde_json::from_slice(&mmap[header_start..header_end])
            .map_err(|e| CoreError::Backend(format!("cellm load: header json parse failed: {e}")))?;

        let mut tensors = HashMap::with_capacity(header.tensors.len());
        for t in &header.tensors {
            tensors.insert(t.name.clone(), t.clone());
        }

        // Pre-warm all weight pages into the OS page cache.
        // Without this, each weight tensor access during inference causes a page fault
        // (OS must read the data from the SSD on first access per page).
        // For a 500MB int8 model this adds ~50-300ms to the FIRST prefill token and
        // makes decode unpredictable under memory pressure.
        // Paying the cost once at startup (before inference begins) is far cheaper overall.
        // Metal avoids this by calling preload_weight() for all tensors; we replicate that
        // benefit here for the CPU path.
        {
            // Read one byte per OS page to force all page faults now.
            // Apple Silicon uses 16 KB pages; use 4 KB as safe lower bound on all platforms.
            const PAGE: usize = 4096;
            let _ = (0..mmap.len()).step_by(PAGE).fold(0u8, |acc, i| acc | mmap[i]);
        }

        // Basic bounds check.
        for t in header.tensors.iter() {
            let start = t.offset_bytes as usize;
            let end = start + t.nbytes as usize;
            if end > mmap.len() {
                return Err(CoreError::Backend(format!(
                    "cellm load: tensor {} out of range (end={} file_len={})",
                    t.name,
                    end,
                    mmap.len()
                )));
            }
        }

        Ok(Self { mmap, header, tensors })
    }

    pub fn tensor_index(&self, name: &str) -> Option<&CellmTensorIndex> {
        self.tensors.get(name)
    }

    pub fn all_tensors(&self) -> impl Iterator<Item = (&String, &[u8])> {
        self.tensors.iter().map(move |(name, t)| {
            let start = t.offset_bytes as usize;
            let end = start + t.nbytes as usize;
            (name, &self.mmap[start..end])
        })
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], CoreError> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| CoreError::Backend(format!("cellm: unknown tensor {name}")))?;
        let start = t.offset_bytes as usize;
        let end = start + t.nbytes as usize;
        Ok(&self.mmap[start..end])
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}
