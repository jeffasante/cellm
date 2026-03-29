//! cellm Model Converter
//!
//! Converts HuggingFace checkpoints (safetensors) into a memory-mappable `.cellm`
//! file with a JSON header + 64-byte aligned tensor data.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::Command;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use half::f16;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(name = "convert", about = "HuggingFace to cellm format converter")]
struct Args {
    /// Input HuggingFace model directory
    #[arg(short, long)]
    input: PathBuf,

    /// Output .cellm file path
    #[arg(short, long)]
    output: PathBuf,

    /// Target dtype: currently only "f16"
    #[arg(long, default_value = "f16")]
    dtype: String,

    /// Dequantize 4-bit affine weights (uint32 + scales/biases) into f16 weights.
    ///
    /// This can significantly increase output file size.
    #[arg(long, default_value_t = false)]
    dequant_4bit_affine: bool,

    /// Quantize eligible Llama linear weights to per-row symmetric int8 (+ f16 scales).
    ///
    /// This is weight-only quantization and currently targets Llama-family text stacks.
    #[arg(long, default_value_t = false)]
    quantize_int8_symmetric: bool,
}

#[derive(Debug, Deserialize)]
struct HfConfigRoot {
    // Common LLama-style fields (some models put these at the top-level).
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    rms_norm_eps: Option<f32>,
    rope_theta: Option<f32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    model_type: Option<String>,
    torch_dtype: Option<String>,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,

    // Nested configs (e.g. Qwen3.5 multimodal checkpoints).
    text_config: Option<HfTextConfig>,
    architectures: Option<Vec<String>>,

    // Quantization metadata.
    quantization_config: Option<QuantizationConfig>,
    _quantization: Option<Value>,
}

#[derive(Debug, Deserialize, Clone)]
struct HfTextConfig {
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    rms_norm_eps: Option<f32>,
    rope_theta: Option<f32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    model_type: Option<String>,
    torch_dtype: Option<String>,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,

    // Qwen-style RoPE nesting.
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Deserialize, Clone)]
struct RopeParameters {
    rope_theta: Option<f32>,
}

#[derive(Debug, Deserialize, Clone)]
struct QuantizationConfig {
    group_size: Option<usize>,
    bits: Option<usize>,
    mode: Option<String>,
}

#[derive(Debug, Serialize)]
struct CellmHeader {
    model_type: String,
    source_model_type: Option<String>,
    text_tensor_prefix: Option<String>,
    vision_tensor_prefix: Option<String>,
    projector_tensor_prefix: Option<String>,
    vocab_size: usize,
    hidden_dim: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
    source_torch_dtype: Option<String>,
    source_architectures: Option<Vec<String>>,
    source_quantization: Option<Value>,
    source_quantization_config: Option<Value>,
    source_text_config: Option<Value>,
    source_vision_config: Option<Value>,
    source_projector_config: Option<Value>,
    tensors: Vec<CellmTensorIndex>,
}

#[derive(Debug, Serialize)]
struct CellmTensorIndex {
    name: String,
    offset_bytes: u64,
    nbytes: u64,
    shape: Vec<usize>,
    dtype: String,
}

#[derive(Debug, Clone)]
struct TensorPlan {
    name: String,
    shape: Vec<usize>,
    file_idx: usize,
    offset_bytes: u64,
    out_nbytes: usize,
    out_dtype: String,
    op: TensorOp,
}

#[derive(Debug, Clone)]
enum TensorOp {
    CopyAsF16,
    Dequant4Affine { group_size: usize },
    QuantizeI8Data,
    QuantizeI8Scales { weight_name: String },
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();
    if args.dtype.as_str() != "f16" {
        anyhow::bail!("only --dtype f16 is supported right now");
    }

    let input_root = if args.input.is_dir() {
        args.input.clone()
    } else {
        args.input
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| anyhow::anyhow!("input has no parent directory: {:?}", args.input))?
    };

    let config_path = input_root.join("config.json");
    let cfg_bytes =
        std::fs::read(&config_path).with_context(|| format!("read {:?}", config_path))?;
    let hf_cfg: HfConfigRoot =
        serde_json::from_slice(&cfg_bytes).with_context(|| format!("parse {:?}", config_path))?;
    let cfg_json: Value =
        serde_json::from_slice(&cfg_bytes).with_context(|| format!("parse {:?}", config_path))?;

    let text_cfg_json = cfg_json.get("text_config").cloned();
    let vision_cfg_json = cfg_json.get("vision_config").cloned();
    let projector_cfg_json = cfg_json
        .get("connector_config")
        .cloned()
        .or_else(|| cfg_json.get("perceiver_config").cloned());
    let quant_cfg_json = cfg_json.get("quantization_config").cloned();
    let quant_json = cfg_json.get("quantization").cloned();

    let selected = select_text_config(&hf_cfg)
        .with_context(|| "unsupported or incomplete HF config (missing vocab/hidden/layers/etc)")?;

    let safetensors_paths = resolve_safetensors_inputs(&args.input, &input_root)?;
    if safetensors_paths.is_empty() {
        anyhow::bail!("no supported model tensor files found in {:?}", args.input);
    }

    log::info!("Input:  {:?}", args.input);
    log::info!("Output: {:?}", args.output);
    log::info!("Found {} safetensors file(s).", safetensors_paths.len());

    // Pass 1: collect tensor metadata and build a deterministic plan.
    let quant_group_size = hf_cfg
        .quantization_config
        .as_ref()
        .and_then(|q| q.group_size)
        .unwrap_or(64);
    let quant_bits = hf_cfg.quantization_config.as_ref().and_then(|q| q.bits);
    let quant_mode = hf_cfg
        .quantization_config
        .as_ref()
        .and_then(|q| q.mode.as_deref());
    let has_4bit_affine = matches!((quant_bits, quant_mode), (Some(4), Some("affine")));

    if has_4bit_affine && !args.dequant_4bit_affine {
        anyhow::bail!(
            "input appears to be 4-bit affine quantized (uint32 weights + scales/biases). Re-run with --dequant-4bit-affine to expand weights to f16."
        );
    }
    if args.quantize_int8_symmetric && selected.model_type != "llama" {
        anyhow::bail!(
            "--quantize-int8-symmetric is currently supported for llama text stacks only (detected model_type={}).",
            selected.model_type
        );
    }

    let plans = build_tensor_plans(
        &safetensors_paths,
        args.dequant_4bit_affine,
        has_4bit_affine,
        quant_group_size,
        args.quantize_int8_symmetric,
        &selected.model_type,
    )?;
    let (text_tensor_prefix, vision_tensor_prefix, projector_tensor_prefix) =
        infer_tensor_prefixes(&plans);

    // Build header with placeholder offsets (filled after we compute header size).
    let mut header = CellmHeader {
        model_type: selected.model_type,
        source_model_type: hf_cfg.model_type.clone(),
        text_tensor_prefix,
        vision_tensor_prefix,
        projector_tensor_prefix,
        vocab_size: selected.vocab_size,
        hidden_dim: selected.hidden_size,
        intermediate_size: selected.intermediate_size,
        num_layers: selected.num_hidden_layers,
        num_heads: selected.num_attention_heads,
        num_kv_heads: selected.num_key_value_heads,
        rms_norm_eps: selected.rms_norm_eps,
        rope_theta: selected.rope_theta,
        bos_token_id: selected.bos_token_id,
        eos_token_id: selected.eos_token_id,
        max_position_embeddings: selected.max_position_embeddings,
        tie_word_embeddings: selected.tie_word_embeddings,
        source_torch_dtype: selected.source_torch_dtype,
        source_architectures: hf_cfg.architectures.clone(),
        source_quantization: quant_json,
        source_quantization_config: quant_cfg_json,
        source_text_config: text_cfg_json,
        source_vision_config: vision_cfg_json,
        source_projector_config: projector_cfg_json,
        tensors: Vec::with_capacity(plans.len()),
    };

    // First, compute preamble + header size so we can compute absolute offsets.
    // We do this by iterating until the JSON length stabilizes (offsets affect length).
    let mut planned = plans.clone();
    let mut last_header_len = None;
    for _ in 0..3 {
        header.tensors = planned
            .iter()
            .map(|p| CellmTensorIndex {
                name: p.name.clone(),
                offset_bytes: p.offset_bytes,
                nbytes: p.out_nbytes as u64,
                shape: p.shape.clone(),
                dtype: p.out_dtype.clone(),
            })
            .collect();
        let header_bytes = serde_json::to_vec(&header)?;
        let header_len = header_bytes.len();
        if last_header_len == Some(header_len) {
            break;
        }
        last_header_len = Some(header_len);

        let data_start = align_up((5 + 1 + 4 + header_len) as u64, 64);
        let mut cursor = data_start;
        for p in planned.iter_mut() {
            cursor = align_up(cursor, 64);
            p.offset_bytes = cursor;
            cursor += p.out_nbytes as u64;
        }
    }

    // Finalize header with computed offsets.
    header.tensors = planned
        .iter()
        .map(|p| CellmTensorIndex {
            name: p.name.clone(),
            offset_bytes: p.offset_bytes,
            nbytes: p.out_nbytes as u64,
            shape: p.shape.clone(),
            dtype: p.out_dtype.clone(),
        })
        .collect();
    let header_bytes = serde_json::to_vec(&header)?;
    if header_bytes.len() > (u32::MAX as usize) {
        anyhow::bail!("header too large: {} bytes", header_bytes.len());
    }

    // Write output.
    let out = File::create(&args.output).with_context(|| format!("create {:?}", args.output))?;
    let mut w = BufWriter::new(out);

    // Preamble.
    w.write_all(b"CELLM")?;
    w.write_all(&[1u8])?; // version
    w.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
    w.write_all(&header_bytes)?;

    // Align to 64-byte boundary for tensor data.
    let mut pos = (5 + 1 + 4 + header_bytes.len()) as u64;
    let aligned = align_up(pos, 64);
    if aligned > pos {
        w.write_all(&vec![0u8; (aligned - pos) as usize])?;
        pos = aligned;
    }

    // Pass 2: write tensors in the planned order.
    let mmaps = mmap_all(&safetensors_paths)?;
    for p in planned.iter() {
        // Ensure writer is at planned offset (padding if needed).
        if pos < p.offset_bytes {
            w.write_all(&vec![0u8; (p.offset_bytes - pos) as usize])?;
            pos = p.offset_bytes;
        }
        if pos != p.offset_bytes {
            anyhow::bail!(
                "writer position mismatch for {}: pos={} planned={}",
                p.name,
                pos,
                p.offset_bytes
            );
        }

        let st = parse_safetensors_with_hint(&mmaps[p.file_idx], &safetensors_paths[p.file_idx])?;

        match &p.op {
            TensorOp::CopyAsF16 => {
                let t = st.tensor(&p.name).with_context(|| {
                    format!("tensor {} in {:?}", p.name, safetensors_paths[p.file_idx])
                })?;
                match t.dtype() {
                    Dtype::F16 => {
                        w.write_all(t.data())?;
                    }
                    Dtype::BF16 => {
                        write_bf16_as_f16(t.data(), &mut w)?;
                    }
                    Dtype::F32 => {
                        write_f32_as_f16(t.data(), &mut w)?;
                    }
                    other => {
                        anyhow::bail!("unsupported dtype for {}: {:?}", p.name, other);
                    }
                }
            }
            TensorOp::Dequant4Affine { group_size } => {
                let t = st.tensor(&p.name).with_context(|| {
                    format!("tensor {} in {:?}", p.name, safetensors_paths[p.file_idx])
                })?;
                if t.dtype() != Dtype::U32 {
                    anyhow::bail!(
                        "expected u32 packed tensor for {}, got {:?}",
                        p.name,
                        t.dtype()
                    );
                }
                let base = p.name.trim_end_matches(".weight");
                let scales_name = format!("{base}.scales");
                let biases_name = format!("{base}.biases");
                let scales = st.tensor(&scales_name).with_context(|| {
                    format!(
                        "tensor {} (required for dequant) in {:?}",
                        scales_name, safetensors_paths[p.file_idx]
                    )
                })?;
                let biases = st.tensor(&biases_name).with_context(|| {
                    format!(
                        "tensor {} (required for dequant) in {:?}",
                        biases_name, safetensors_paths[p.file_idx]
                    )
                })?;
                dequant_u32_affine_to_f16(
                    t.data(),
                    t.shape(),
                    scales.data(),
                    scales.dtype(),
                    biases.data(),
                    biases.dtype(),
                    *group_size,
                    &mut w,
                )?;
            }
            TensorOp::QuantizeI8Data => {
                let t = st.tensor(&p.name).with_context(|| {
                    format!("tensor {} in {:?}", p.name, safetensors_paths[p.file_idx])
                })?;
                write_quant_i8_data_per_row_symmetric(t.data(), t.shape(), t.dtype(), &mut w)?;
            }
            TensorOp::QuantizeI8Scales { weight_name } => {
                let t = st.tensor(weight_name).with_context(|| {
                    format!("tensor {} in {:?}", weight_name, safetensors_paths[p.file_idx])
                })?;
                write_quant_i8_scales_per_row_symmetric(t.data(), t.shape(), t.dtype(), &mut w)?;
            }
        }

        pos += p.out_nbytes as u64;
        log::debug!(
            "wrote {:>10} bytes @ {}  {}",
            p.out_nbytes,
            p.offset_bytes,
            p.name
        );
    }

    w.flush()?;

    log::info!(
        "Converted {} tensors into {:?}",
        header.tensors.len(),
        args.output
    );
    log::info!("Tip: keep tokenizer.json alongside the .cellm file for now.");
    Ok(())
}

fn find_safetensors(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {:?}", dir))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    out.sort();
    Ok(out)
}

fn resolve_safetensors_inputs(input: &Path, input_root: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        let ext = input.extension().and_then(|s| s.to_str()).unwrap_or_default();
        if ext == "safetensors" {
            return Ok(vec![input.to_path_buf()]);
        }
        if matches!(ext, "bin" | "pt" | "pth") {
            let out = convert_pytorch_file_to_safetensors(input)?;
            return Ok(vec![out]);
        }
        anyhow::bail!(
            "unsupported input file extension {:?}; expected .safetensors/.bin/.pt/.pth",
            input
        );
    }

    let st = find_safetensors(input_root)?;
    if !st.is_empty() {
        return Ok(st);
    }

    for name in ["pytorch_model.bin", "model.bin", "model.pt", "pytorch_model.pt"] {
        let p = input_root.join(name);
        if p.exists() {
            let out = convert_pytorch_file_to_safetensors(&p)?;
            return Ok(vec![out]);
        }
    }

    Ok(Vec::new())
}

fn convert_pytorch_file_to_safetensors(input_file: &Path) -> Result<PathBuf> {
    let out = std::env::temp_dir().join(format!(
        "cellm_pt_import_{}_{}.safetensors",
        std::process::id(),
        chrono_like_millis()
    ));

    let script = r#"
import sys
import torch
from safetensors.torch import save_file

inp, out = sys.argv[1], sys.argv[2]
obj = torch.load(inp, map_location='cpu')
if hasattr(obj, 'state_dict'):
    obj = obj.state_dict()
elif isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
    obj = obj['state_dict']
if not isinstance(obj, dict):
    raise RuntimeError(f'expected state_dict dict, got {type(obj)}')

clean = {}
for k, v in obj.items():
    if torch.is_tensor(v):
        clean[str(k)] = v.detach().cpu().contiguous()

if not clean:
    raise RuntimeError('no tensors found in state_dict')
save_file(clean, out)
print(len(clean))
"#;

    let outp = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(input_file)
        .arg(&out)
        .output()
        .with_context(|| "failed to run python3 for .bin/.pt -> .safetensors conversion")?;
    if !outp.status.success() {
        let stderr = String::from_utf8_lossy(&outp.stderr);
        anyhow::bail!(
            "python conversion failed for {:?}: {}",
            input_file,
            stderr.trim()
        );
    }
    let n = String::from_utf8_lossy(&outp.stdout).trim().to_string();
    log::info!(
        "Imported {:?} via PyTorch -> temporary safetensors {:?} ({} tensors).",
        input_file,
        out,
        n
    );
    Ok(out)
}

fn chrono_like_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn mmap_all(paths: &[PathBuf]) -> Result<Vec<Mmap>> {
    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        let f = File::open(p).with_context(|| format!("open {:?}", p))?;
        let mmap = unsafe { Mmap::map(&f) }.with_context(|| format!("mmap {:?}", p))?;
        out.push(mmap);
    }
    Ok(out)
}

fn parse_safetensors_with_hint<'a>(bytes: &'a [u8], path: &Path) -> Result<SafeTensors<'a>> {
    SafeTensors::deserialize(bytes).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("MetadataIncompleteBuffer") {
            anyhow::anyhow!(
                "parse safetensors {:?}: metadata incomplete (file likely truncated/incomplete). Re-download checkpoint weights and try convert again.",
                path
            )
        } else {
            anyhow::anyhow!("parse safetensors {:?}: {e}", path)
        }
    })
}

fn build_tensor_plans(
    paths: &[PathBuf],
    dequant_4bit_affine: bool,
    has_4bit_affine: bool,
    group_size: usize,
    quantize_int8_symmetric: bool,
    model_type: &str,
) -> Result<Vec<TensorPlan>> {
    let mmaps = mmap_all(paths)?;
    let mut plans: Vec<TensorPlan> = Vec::new();
    let mut name_to_file: HashMap<String, usize> = HashMap::new();

    for (file_idx, mmap) in mmaps.iter().enumerate() {
        let st = parse_safetensors_with_hint(mmap, &paths[file_idx])?;

        let names: Vec<String> = st.names().iter().map(|s| (*s).to_string()).collect();
        let name_set: std::collections::HashSet<&str> =
            names.iter().map(|s| s.as_str()).collect();

        for name in st.names() {
            let t = st.tensor(name)?;
            let shape: Vec<usize> = t.shape().iter().map(|&d| d as usize).collect();
            let mut out_shape = shape.clone();
            let mut out_dtype = "f16".to_string();
            let op = if dequant_4bit_affine
                && has_4bit_affine
                && name.ends_with(".weight")
                && t.dtype() == Dtype::U32
            {
                let base = name.trim_end_matches(".weight");
                let scales = format!("{base}.scales");
                let biases = format!("{base}.biases");
                if name_set.contains(scales.as_str()) && name_set.contains(biases.as_str()) {
                    // Packed u32 holds 8x 4-bit values per element in the last dim.
                    if out_shape.len() != 2 {
                        anyhow::bail!("expected 2D packed weight for {name}, got {:?}", out_shape);
                    }
                    out_shape[1] = out_shape[1] * 8;
                    TensorOp::Dequant4Affine { group_size }
                } else {
                    anyhow::bail!(
                        "found uint32 packed weight {name} but missing {scales} or {biases}"
                    );
                }
            } else if quantize_int8_symmetric
                && model_type == "llama"
                && should_quantize_i8_llama_weight(name, &shape)
            {
                out_dtype = "i8".to_string();
                TensorOp::QuantizeI8Data
            } else {
                TensorOp::CopyAsF16
            };

            // Skip scales/biases tensors when they are consumed by a dequantized weight.
            if dequant_4bit_affine
                && has_4bit_affine
                && (name.ends_with(".scales") || name.ends_with(".biases"))
            {
                let base = name
                    .trim_end_matches(".scales")
                    .trim_end_matches(".biases");
                let w = format!("{base}.weight");
                if name_set.contains(w.as_str()) {
                    let wt = st.tensor(&w)?;
                    if wt.dtype() == Dtype::U32 {
                        continue;
                    }
                }
            }

            let numel = out_shape.iter().product::<usize>();
            let out_nbytes = if out_dtype == "i8" { numel } else { numel * 2 };

            // Validate input dtype early unless we are dequantizing it.
            if !matches!(&op, TensorOp::Dequant4Affine { .. }) {
                match t.dtype() {
                    Dtype::F16 | Dtype::BF16 | Dtype::F32 => {}
                    other => anyhow::bail!("unsupported dtype for {name}: {:?}", other),
                }
            }

            if name_to_file.insert(name.to_string(), file_idx).is_some() {
                anyhow::bail!("duplicate tensor name across safetensors files: {name}");
            }

            plans.push(TensorPlan {
                name: name.to_string(),
                shape: out_shape,
                file_idx,
                offset_bytes: 0,
                out_nbytes,
                out_dtype: out_dtype.clone(),
                op,
            });

            if out_dtype == "i8" {
                let out_dim = shape[0];
                plans.push(TensorPlan {
                    name: format!("{name}.qscale"),
                    shape: vec![out_dim],
                    file_idx,
                    offset_bytes: 0,
                    out_nbytes: out_dim * 2,
                    out_dtype: "f16".to_string(),
                    op: TensorOp::QuantizeI8Scales {
                        weight_name: name.to_string(),
                    },
                });
            }
        }
    }

    plans.sort_by(|a, b| a.name.cmp(&b.name));
    let _ = name_to_file;
    Ok(plans)
}

fn align_up(v: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (v + (align - 1)) & !(align - 1)
}

fn infer_tensor_prefixes(
    plans: &[TensorPlan],
) -> (Option<String>, Option<String>, Option<String>) {
    let has = |prefix: &str| plans.iter().any(|p| p.name.starts_with(prefix));

    let text = if has("model.text_model.") {
        Some("model.text_model.".to_string())
    } else if has("language_model.") {
        Some("language_model.".to_string())
    } else if has("model.") {
        Some("model.".to_string())
    } else {
        None
    };

    let vision = if has("model.vision_model.") {
        Some("model.vision_model.".to_string())
    } else if has("vision_model.") {
        Some("vision_model.".to_string())
    } else {
        None
    };

    let projector = if has("model.connector.") {
        Some("model.connector.".to_string())
    } else if has("connector.") {
        Some("connector.".to_string())
    } else {
        None
    };

    (text, vision, projector)
}

fn should_quantize_i8_llama_weight(name: &str, shape: &[usize]) -> bool {
    if shape.len() != 2 || !name.ends_with(".weight") {
        return false;
    }
    if name.contains("embed_tokens") || name.contains("norm") || name.contains("lm_head") {
        return false;
    }
    name.contains(".self_attn.") || name.contains(".mlp.")
}

fn write_f32_as_f16(bytes: &[u8], w: &mut impl Write) -> Result<()> {
    if bytes.len() % 4 != 0 {
        anyhow::bail!("f32 tensor bytes not multiple of 4");
    }
    for chunk in bytes.chunks_exact(4) {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let h = f16::from_f32(f);
        w.write_all(&h.to_bits().to_le_bytes())?;
    }
    Ok(())
}

fn tensor_data_to_f32(data: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            if data.len() % 4 != 0 {
                anyhow::bail!("f32 tensor bytes not multiple of 4");
            }
            Ok(data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        Dtype::F16 => f16_bytes_to_f32_slice(data),
        Dtype::BF16 => bf16_bytes_to_f32_slice(data),
        other => anyhow::bail!("unsupported dtype for int8 quantization: {other:?}"),
    }
}

fn write_quant_i8_data_per_row_symmetric(
    data: &[u8],
    shape: &[usize],
    dtype: Dtype,
    w: &mut impl Write,
) -> Result<()> {
    if shape.len() != 2 {
        anyhow::bail!("int8 quantization expects 2D weight, got shape={shape:?}");
    }
    let out_dim = shape[0];
    let in_dim = shape[1];
    let values = tensor_data_to_f32(data, dtype)?;
    if values.len() != out_dim * in_dim {
        anyhow::bail!(
            "int8 quantization data length mismatch: {} vs {}",
            values.len(),
            out_dim * in_dim
        );
    }

    for r in 0..out_dim {
        let row = &values[r * in_dim..(r + 1) * in_dim];
        let mut max_abs = 0.0f32;
        for &v in row {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        for &v in row {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            w.write_all(&[q as u8])?;
        }
    }
    Ok(())
}

fn write_quant_i8_scales_per_row_symmetric(
    data: &[u8],
    shape: &[usize],
    dtype: Dtype,
    w: &mut impl Write,
) -> Result<()> {
    if shape.len() != 2 {
        anyhow::bail!("int8 quantization expects 2D weight, got shape={shape:?}");
    }
    let out_dim = shape[0];
    let in_dim = shape[1];
    let values = tensor_data_to_f32(data, dtype)?;
    if values.len() != out_dim * in_dim {
        anyhow::bail!(
            "int8 quantization data length mismatch: {} vs {}",
            values.len(),
            out_dim * in_dim
        );
    }

    for r in 0..out_dim {
        let row = &values[r * in_dim..(r + 1) * in_dim];
        let mut max_abs = 0.0f32;
        for &v in row {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let h = f16::from_f32(scale);
        w.write_all(&h.to_bits().to_le_bytes())?;
    }
    Ok(())
}

fn write_bf16_as_f16(bytes: &[u8], w: &mut impl Write) -> Result<()> {
    if bytes.len() % 2 != 0 {
        anyhow::bail!("bf16 tensor bytes not multiple of 2");
    }
    for chunk in bytes.chunks_exact(2) {
        let b = u16::from_le_bytes([chunk[0], chunk[1]]);
        let f = f32::from_bits((b as u32) << 16);
        let h = f16::from_f32(f);
        w.write_all(&h.to_bits().to_le_bytes())?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct SelectedTextConfig {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
    source_torch_dtype: Option<String>,
}

fn select_text_config(cfg: &HfConfigRoot) -> Result<SelectedTextConfig> {
    let nested = cfg.text_config.as_ref();

    let need = |v: Option<usize>, n: &str| -> Result<usize> {
        v.with_context(|| format!("missing {n}"))
    };

    let vocab_size = need(
        cfg.vocab_size.or_else(|| nested.and_then(|t| t.vocab_size)),
        "vocab_size",
    )?;
    let hidden_size = need(
        cfg.hidden_size.or_else(|| nested.and_then(|t| t.hidden_size)),
        "hidden_size",
    )?;
    let intermediate_size = need(
        cfg.intermediate_size
            .or_else(|| nested.and_then(|t| t.intermediate_size)),
        "intermediate_size",
    )?;
    let num_hidden_layers = need(
        cfg.num_hidden_layers
            .or_else(|| nested.and_then(|t| t.num_hidden_layers)),
        "num_hidden_layers",
    )?;
    let num_attention_heads = need(
        cfg.num_attention_heads
            .or_else(|| nested.and_then(|t| t.num_attention_heads)),
        "num_attention_heads",
    )?;
    let num_key_value_heads = need(
        cfg.num_key_value_heads
            .or_else(|| nested.and_then(|t| t.num_key_value_heads)),
        "num_key_value_heads",
    )?;

    let rms_norm_eps = cfg
        .rms_norm_eps
        .or_else(|| nested.and_then(|t| t.rms_norm_eps))
        .unwrap_or(1e-5);

    let rope_theta = cfg
        .rope_theta
        .or_else(|| nested.and_then(|t| t.rope_theta))
        .or_else(|| nested.and_then(|t| t.rope_parameters.as_ref().and_then(|rp| rp.rope_theta)))
        .unwrap_or(10000.0);

    let model_type = cfg
        .text_config
        .as_ref()
        .and_then(|t| t.model_type.clone())
        .or_else(|| cfg.model_type.clone())
        .unwrap_or_else(|| "unknown".into());

    Ok(SelectedTextConfig {
        model_type,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        rms_norm_eps,
        rope_theta,
        bos_token_id: cfg
            .bos_token_id
            .or_else(|| nested.and_then(|t| t.bos_token_id)),
        eos_token_id: cfg
            .eos_token_id
            .or_else(|| nested.and_then(|t| t.eos_token_id)),
        max_position_embeddings: cfg
            .max_position_embeddings
            .or_else(|| nested.and_then(|t| t.max_position_embeddings)),
        tie_word_embeddings: cfg
            .tie_word_embeddings
            .or_else(|| nested.and_then(|t| t.tie_word_embeddings)),
        source_torch_dtype: cfg
            .torch_dtype
            .clone()
            .or_else(|| nested.and_then(|t| t.torch_dtype.clone())),
    })
}

fn bf16_bytes_to_f32_slice(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        anyhow::bail!("bf16 bytes not multiple of 2");
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let b = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f32::from_bits((b as u32) << 16));
    }
    Ok(out)
}

fn f16_bytes_to_f32_slice(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        anyhow::bail!("f16 bytes not multiple of 2");
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let b = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f16::from_bits(b).to_f32());
    }
    Ok(out)
}

fn dequant_u32_affine_to_f16(
    packed_bytes: &[u8],
    packed_shape: &[usize],
    scales_bytes: &[u8],
    scales_dtype: Dtype,
    biases_bytes: &[u8],
    biases_dtype: Dtype,
    group_size: usize,
    w: &mut impl Write,
) -> Result<()> {
    if packed_shape.len() != 2 {
        anyhow::bail!("dequant: expected packed 2D weight, got shape={packed_shape:?}");
    }
    if group_size == 0 {
        anyhow::bail!("dequant: group_size must be > 0");
    }

    if packed_bytes.len() % 4 != 0 {
        anyhow::bail!("dequant: packed u32 bytes not multiple of 4");
    }
    let out_dim = packed_shape[0];
    let packed_in = packed_shape[1];
    let in_dim = packed_in * 8;
    let groups = (in_dim + group_size - 1) / group_size;

    let scales = match scales_dtype {
        Dtype::BF16 => bf16_bytes_to_f32_slice(scales_bytes)?,
        Dtype::F16 => f16_bytes_to_f32_slice(scales_bytes)?,
        Dtype::F32 => {
            if scales_bytes.len() % 4 != 0 {
                anyhow::bail!("dequant: scales f32 bytes not multiple of 4");
            }
            scales_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        other => anyhow::bail!("dequant: unsupported scales dtype {other:?}"),
    };
    let biases = match biases_dtype {
        Dtype::BF16 => bf16_bytes_to_f32_slice(biases_bytes)?,
        Dtype::F16 => f16_bytes_to_f32_slice(biases_bytes)?,
        Dtype::F32 => {
            if biases_bytes.len() % 4 != 0 {
                anyhow::bail!("dequant: biases f32 bytes not multiple of 4");
            }
            biases_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        other => anyhow::bail!("dequant: unsupported biases dtype {other:?}"),
    };

    if scales.len() != out_dim * groups {
        anyhow::bail!(
            "dequant: scales len mismatch: {} expected {} (out_dim={out_dim} groups={groups})",
            scales.len(),
            out_dim * groups
        );
    }
    if biases.len() != out_dim * groups {
        anyhow::bail!(
            "dequant: biases len mismatch: {} expected {} (out_dim={out_dim} groups={groups})",
            biases.len(),
            out_dim * groups
        );
    }

    // Stream row-by-row to avoid allocating out_dim*in_dim f16 in memory.
    let mut row_out: Vec<u16> = vec![0u16; in_dim];
    for r in 0..out_dim {
        let row_start = r * packed_in;
        let row_words = &packed_bytes[row_start * 4..(row_start + packed_in) * 4];

        for i in 0..in_dim {
            let word = {
                let wi = i / 8;
                let b = &row_words[wi * 4..wi * 4 + 4];
                u32::from_le_bytes([b[0], b[1], b[2], b[3]])
            };
            let nib = (i % 8) * 4;
            let q = ((word >> nib) & 0xF) as u8;
            let g = (i / group_size).min(groups.saturating_sub(1));
            let scale = scales[r * groups + g];
            let bias = biases[r * groups + g];
            let fp = (q as f32) * scale + bias;
            row_out[i] = f16::from_f32(fp).to_bits();
        }

        for bits in &row_out {
            w.write_all(&bits.to_le_bytes())?;
        }
    }
    Ok(())
}
