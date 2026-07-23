// Author: Jeffrey Asante (https://jeffasante.github.io/)
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(all(unix, not(target_arch = "wasm32")))]
use std::os::unix::io::FromRawFd;
use std::path::Path;

use cellm_core::CoreError;
#[cfg(not(target_arch = "wasm32"))]
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
    /// Whether Q/K projections use adjacent-pair rotary layout.
    pub rope_interleaved: Option<bool>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub max_position_embeddings: Option<usize>,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_original_max_position_embeddings: Option<usize>,
    pub rope_scaling_low_freq_factor: Option<f32>,
    pub rope_scaling_high_freq_factor: Option<f32>,
    pub tie_word_embeddings: Option<bool>,
    pub source_torch_dtype: Option<String>,
    pub source_architectures: Option<Vec<String>>,
    pub source_quantization: Option<Value>,
    pub source_quantization_config: Option<Value>,
    pub source_text_config: Option<Value>,
    pub source_vision_config: Option<Value>,
    pub source_projector_config: Option<Value>,
    pub tensors: Vec<CellmTensorIndex>,
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

pub struct CellmFile {
    data: CellmData,
    pub header: CellmHeader,
    tensors: HashMap<String, CellmTensorIndex>,
}

enum CellmData {
    #[cfg(not(target_arch = "wasm32"))]
    Mmap(Mmap),
    Owned(Vec<u8>),
}

impl CellmData {
    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            CellmData::Mmap(m) => m,
            CellmData::Owned(v) => v,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(target_os = "android")]
fn open_model_file(path: &Path) -> Result<File, CoreError> {
    use std::os::unix::ffi::OsStrExt;
    let cpath = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|_| CoreError::Backend("cellm load: path contains null".into()))?;
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
    if fd < 0 {
        return Err(CoreError::Backend(format!(
            "cellm load: libc::open failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_os = "android"))]
fn open_model_file(path: &Path) -> Result<File, CoreError> {
    File::open(path).map_err(|e| CoreError::Backend(format!("cellm load: open failed: {e}")))
}

impl CellmFile {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let f = open_model_file(path)?;
        let file_data = match unsafe { Mmap::map(&f) } {
            Ok(m) => CellmData::Mmap(m),
            Err(mmap_err) => {
                use std::io::Read;
                let mut buf = Vec::new();
                #[cfg(target_os = "android")]
                let mut file = {
                    use std::os::unix::ffi::OsStrExt;
                    let cpath = std::ffi::CString::new(path.as_os_str().as_bytes())
                        .map_err(|_| CoreError::Backend("cellm load: path contains null".into()))?;
                    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
                    if fd < 0 {
                        return Err(CoreError::Backend(format!(
                            "cellm load: libc::open reopen failed: {}",
                            std::io::Error::last_os_error()
                        )));
                    }
                    unsafe { std::fs::File::from_raw_fd(fd) }
                };
                #[cfg(not(target_os = "android"))]
                let mut file = File::open(path)
                    .map_err(|e| CoreError::Backend(format!("cellm load: reopen for read failed: {e}")))?;
                file.read_to_end(&mut buf)
                    .map_err(|e| CoreError::Backend(format!("cellm load: read failed (mmap err: {mmap_err}, read err: {e})")))?;
                CellmData::Owned(buf)
            }
        };

        if file_data.as_slice().len() < 10 {
            return Err(CoreError::Backend("cellm load: file too small".into()));
        }
        if &file_data.as_slice()[0..5] != b"CELLM" {
            return Err(CoreError::Backend("cellm load: bad magic".into()));
        }
        let version = file_data.as_slice()[5];
        if version != 1 {
            return Err(CoreError::Backend(format!(
                "cellm load: unsupported version {version}"
            )));
        }

        let header_len = u32::from_le_bytes([file_data.as_slice()[6], file_data.as_slice()[7], file_data.as_slice()[8], file_data.as_slice()[9]]) as usize;
        let header_start = 10usize;
        let header_end = header_start + header_len;
        if header_end > file_data.as_slice().len() {
            return Err(CoreError::Backend(
                "cellm load: header length out of range".into(),
            ));
        }

        let header: CellmHeader = serde_json::from_slice(&file_data.as_slice()[header_start..header_end])
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
            #[cfg(any(target_os = "android", target_os = "linux"))]
            {
                // Use madvise for async non-blocking prefetch on Linux/Android.
                // This avoids blocking the thread for minutes on slow emulator storage.
                let ptr = file_data.as_slice().as_ptr() as *mut libc::c_void;
                let len = file_data.as_slice().len();
                unsafe {
                    libc::madvise(ptr, len, libc::MADV_WILLNEED);
                }
            }
            #[cfg(not(any(target_os = "android", target_os = "linux")))]
            {
                // Read one byte per OS page to force all page faults now.
                // Apple Silicon uses 16 KB pages; use 4 KB as safe lower bound on all platforms.
                const PAGE: usize = 4096;
                let _ = (0..file_data.as_slice().len()).step_by(PAGE).fold(0u8, |acc, i| acc | file_data.as_slice()[i]);
            }
        }

        // Basic bounds check.
        for t in header.tensors.iter() {
            let start = t.offset_bytes as usize;
            let end = start + t.nbytes as usize;
            if end > file_data.as_slice().len() {
                return Err(CoreError::Backend(format!(
                    "cellm load: tensor {} out of range (end={} file_len={})",
                    t.name,
                    end,
                    file_data.as_slice().len()
                )));
            }
        }

        Ok(Self { data: file_data, header, tensors })
    }

    #[cfg(target_arch = "wasm32")]
    pub fn load(_path: &Path) -> Result<Self, CoreError> {
        Err(CoreError::Backend(
            "cellm load: file-based loading not available on WASM. Use CellmFile::load_from_bytes instead.".into(),
        ))
    }

    pub fn tensor_index(&self, name: &str) -> Option<&CellmTensorIndex> {
        self.tensors.get(name)
    }

    pub fn all_tensors(&self) -> impl Iterator<Item = (&String, &[u8])> {
        self.tensors.iter().map(move |(name, t)| {
            let start = t.offset_bytes as usize;
            let end = start + t.nbytes as usize;
            (name, &self.data.as_slice()[start..end])
        })
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], CoreError> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| CoreError::Backend(format!("cellm: unknown tensor {name}")))?;
        let start = t.offset_bytes as usize;
        let end = start + t.nbytes as usize;
        Ok(&self.data.as_slice()[start..end])
    }

    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, CoreError> {
        Self::load_from_vec(bytes.to_vec())
    }

    pub fn load_from_vec(bytes: Vec<u8>) -> Result<Self, CoreError> {
        if bytes.len() < 10 {
            return Err(CoreError::Backend("cellm load: file too small".into()));
        }
        if &bytes[0..5] != b"CELLM" {
            return Err(CoreError::Backend("cellm load: bad magic".into()));
        }
        let version = bytes[5];
        if version != 1 {
            return Err(CoreError::Backend(format!(
                "cellm load: unsupported version {version}"
            )));
        }

        let header_len =
            u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        let header_start = 10usize;
        let header_end = header_start + header_len;
        if header_end > bytes.len() {
            return Err(CoreError::Backend(
                "cellm load: header length out of range".into(),
            ));
        }

        let header: CellmHeader = serde_json::from_slice(&bytes[header_start..header_end])
            .map_err(|e| CoreError::Backend(format!("cellm load: header json parse failed: {e}")))?;

        let mut tensors = HashMap::with_capacity(header.tensors.len());
        for t in &header.tensors {
            tensors.insert(t.name.clone(), t.clone());
        }

        for t in header.tensors.iter() {
            let start = t.offset_bytes as usize;
            let end = start + t.nbytes as usize;
            if end > bytes.len() {
                return Err(CoreError::Backend(format!(
                    "cellm load: tensor {} out of range (end={} file_len={})",
                    t.name,
                    end,
                    bytes.len()
                )));
            }
        }

        Ok(Self {
            data: CellmData::Owned(bytes),
            header,
            tensors,
        })
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}
