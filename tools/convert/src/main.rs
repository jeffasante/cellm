//! cellm Model Converter
//!
//! Converts HuggingFace checkpoints (safetensors) into a memory-mappable `.cellm`
//! file with a JSON header + 64-byte aligned tensor data.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::process::Command;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use clap::Parser;
use cellm_model::CellmFile;
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

    /// Quantize eligible text linear weights to per-row symmetric int8 (+ f16 scales).
    ///
    /// This is weight-only quantization and currently targets Llama/Qwen text stacks.
    #[arg(long, default_value_t = false)]
    quantize_int8_symmetric: bool,

    /// Quantize eligible text 2D weights to per-row symmetric int4 packed nibbles (+ f16 scales).
    ///
    /// This is an aggressive weight-only mode for Qwen text stacks.
    #[arg(long, default_value_t = false)]
    quantize_int4_symmetric: bool,

    /// Gemma4-only aggressive INT4 profile:
    /// quantize most 2D text weights (including embeddings / lm_head when present)
    /// while keeping norm-style tensors in f16.
    #[arg(long, default_value_t = false)]
    gemma4_aggressive_int4: bool,

    /// Gemma4-only balanced INT4 profile:
    /// quantize layer attention/MLP projections, keep embeddings/lm_head/PLE+noms in f16.
    #[arg(long, default_value_t = false)]
    gemma4_balanced_int4: bool,

    /// Gemma4-only AWQ-lite INT4 profile:
    /// keep sensitive tensors (embeddings/lm_head/PLE/norms + first/last 2 layers) in f16,
    /// quantize remaining layer projections to int4.
    #[arg(long, default_value_t = false)]
    gemma4_awq_lite_int4: bool,

    /// Gemma4-only mixed profile for mobile experiments:
    /// quantize attention projections to int8 and MLP projections to int4.
    /// Keeps embeddings/lm_head/norm-style tensors in f16.
    #[arg(long, default_value_t = false)]
    gemma4_mixed_i8_i4: bool,

    /// Optional Gemma4 INT4 layer exclusion list (comma-separated), e.g. "0,1,33,34".
    /// Excluded layers stay in f16 for INT4 quantization profiles.
    #[arg(long)]
    gemma4_int4_exclude_layers: Option<String>,

    /// Optional Gemma4 mixed-profile attention INT8 layer list (comma-separated).
    /// When set with --gemma4-mixed-i8-i4, only these attention layers use INT8;
    /// other attention layers fall back to INT4.
    #[arg(long)]
    gemma4_mixed_i8_attn_layers: Option<String>,

    /// Keep only text-stack tensors (drop vision/projector tensors).
    #[arg(long, default_value_t = false)]
    text_only: bool,

    /// Validate GGUF -> cellm conversion with tensor checks and logits parity.
    #[arg(long, default_value_t = false)]
    validate_gguf: bool,

    /// Number of leading f16 values to compare per tensor during validation.
    #[arg(long, default_value_t = 64)]
    validate_tensor_values: usize,

    /// Number of logits rows to sample for parity checks.
    #[arg(long, default_value_t = 128)]
    validate_logits_rows: usize,

    /// Validate a GGUF source against an existing .cellm file without writing output.
    #[arg(long)]
    validate_against: Option<PathBuf>,
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
    bos_token_id: Option<Value>,
    eos_token_id: Option<Value>,
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
    full_attention: Option<RopeParamVariant>,
    sliding_attention: Option<RopeParamVariant>,
}

#[derive(Debug, Deserialize, Clone)]
struct RopeParamVariant {
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
    source_safetensors_format: Option<String>,
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
    header_shape: Vec<usize>,
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
    QuantizeI4Data,
    QuantizeI4Scales { weight_name: String },
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();
    if args.dtype.as_str() != "f16" {
        anyhow::bail!("only --dtype f16 is supported right now");
    }

    if let Some(gguf_path) = resolve_gguf_input(&args.input)? {
        handle_gguf_not_yet_supported(&args, &gguf_path)?;
        return Ok(());
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
    let source_safetensors_format = ensure_supported_safetensors_metadata(&safetensors_paths)?;

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
    if args.quantize_int8_symmetric
        && !(selected.model_type == "llama"
            || selected.model_type == "smollm3"
            || selected.model_type.starts_with("qwen")
            || selected.model_type.starts_with("gemma"))
    {
        anyhow::bail!(
            "--quantize-int8-symmetric is currently supported for llama/smollm3/qwen/gemma text stacks only (detected model_type={}).",
            selected.model_type
        );
    }
    if args.quantize_int4_symmetric && args.quantize_int8_symmetric && !args.gemma4_mixed_i8_i4 {
        anyhow::bail!("choose only one quantization mode: --quantize-int8-symmetric or --quantize-int4-symmetric");
    }
    if (args.gemma4_aggressive_int4 as u8
        + args.gemma4_balanced_int4 as u8
        + args.gemma4_awq_lite_int4 as u8
        + args.gemma4_mixed_i8_i4 as u8) > 1
    {
        anyhow::bail!("choose only one Gemma4 quant profile: --gemma4-aggressive-int4, --gemma4-balanced-int4, --gemma4-awq-lite-int4, or --gemma4-mixed-i8-i4");
    }
    if (args.gemma4_aggressive_int4 || args.gemma4_balanced_int4 || args.gemma4_awq_lite_int4)
        && !args.quantize_int4_symmetric
    {
        anyhow::bail!("Gemma4 INT4 profiles require --quantize-int4-symmetric");
    }
    if args.gemma4_mixed_i8_i4
        && !(args.quantize_int4_symmetric && args.quantize_int8_symmetric)
    {
        anyhow::bail!("--gemma4-mixed-i8-i4 requires both --quantize-int4-symmetric and --quantize-int8-symmetric");
    }
    if args.quantize_int4_symmetric && !(selected.model_type.starts_with("qwen") || selected.model_type.starts_with("gemma")) {
        anyhow::bail!(
            "--quantize-int4-symmetric is currently supported for qwen/gemma text stacks only (detected model_type={}).",
            selected.model_type
        );
    }

    let plans = build_tensor_plans(
        &safetensors_paths,
        args.dequant_4bit_affine,
        has_4bit_affine,
        quant_group_size,
        args.quantize_int8_symmetric,
        args.quantize_int4_symmetric,
        args.gemma4_aggressive_int4,
        args.gemma4_balanced_int4,
        args.gemma4_awq_lite_int4,
        args.gemma4_mixed_i8_i4,
        &parse_layer_index_list(args.gemma4_int4_exclude_layers.as_deref())?,
        &parse_layer_index_list(args.gemma4_mixed_i8_attn_layers.as_deref())?,
        args.text_only,
        &selected.model_type,
        selected.num_hidden_layers,
    )?;
    let (text_tensor_prefix, vision_tensor_prefix, projector_tensor_prefix) =
        infer_tensor_prefixes(&plans);

    // Build header with placeholder offsets (filled after we compute header size).
    let mut header = CellmHeader {
        model_type: selected.model_type,
        source_model_type: hf_cfg.model_type.clone(),
        source_safetensors_format,
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
                shape: p.header_shape.clone(),
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
            shape: p.header_shape.clone(),
            dtype: p.out_dtype.clone(),
        })
        .collect();
    let header_bytes = serde_json::to_vec(&header)?;
    if header_bytes.len() > (u32::MAX as usize) {
        anyhow::bail!("header too large: {} bytes", header_bytes.len());
    }

    // Write output.
    let out = File::create(&args.output).with_context(|| format!("create {:?}", args.output))?;
    let mut w = BufWriter::with_capacity(65536, out);

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
            TensorOp::QuantizeI4Data => {
                let t = st.tensor(&p.name).with_context(|| {
                    format!("tensor {} in {:?}", p.name, safetensors_paths[p.file_idx])
                })?;
                write_quant_i4_data_per_row_symmetric(t.data(), t.shape(), t.dtype(), &mut w)?;
            }
            TensorOp::QuantizeI4Scales { weight_name } => {
                let t = st.tensor(weight_name).with_context(|| {
                    format!("tensor {} in {:?}", weight_name, safetensors_paths[p.file_idx])
                })?;
                write_quant_i4_scales_per_row_symmetric(t.data(), t.shape(), t.dtype(), &mut w)?;
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

fn resolve_gguf_input(input: &Path) -> Result<Option<PathBuf>> {
    if input.is_file() {
        if input.extension().and_then(|s| s.to_str()) == Some("gguf") {
            return Ok(Some(input.to_path_buf()));
        }
        return Ok(None);
    }

    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(input)
        .with_context(|| format!("read_dir {:?}", input))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    ggufs.sort();
    Ok(ggufs.into_iter().next())
}

fn handle_gguf_not_yet_supported(_args: &Args, gguf_path: &Path) -> Result<()> {
    let info = inspect_gguf_tensor_types(gguf_path)?;
    let mut summary: Vec<String> = info
        .type_counts
        .iter()
        .map(|(k, v)| format!("{k} x{v}"))
        .collect();
    summary.sort();
    let summary = summary.join(", ");

    convert_gguf_to_cellm(_args, gguf_path, &summary)
}

#[derive(Debug)]
struct GgufTypeScan {
    type_counts: HashMap<String, usize>,
}

fn inspect_gguf_tensor_types(path: &Path) -> Result<GgufTypeScan> {
    let f = File::open(path).with_context(|| format!("open {:?}", path))?;
    let mmap = unsafe { Mmap::map(&f).with_context(|| format!("mmap {:?}", path))? };
    let mut c = GgufCursor { b: &mmap, p: 0 };

    let magic = c.read_exact(4)?;
    if magic != b"GGUF" {
        anyhow::bail!("not a GGUF file (bad magic) for {:?}", path);
    }
    let version = c.read_u32()?;
    if version < 2 || version > 3 {
        anyhow::bail!("unsupported GGUF version {} in {:?}", version, path);
    }
    let tensor_count = c.read_u64()?;
    let kv_count = c.read_u64()?;

    for _ in 0..kv_count {
        let _key = c.read_string()?;
        skip_gguf_value(&mut c)?;
    }

    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..tensor_count {
        let _name = c.read_string()?;
        let n_dims = c.read_u32()? as usize;
        for _ in 0..n_dims {
            let _ = c.read_u64()?;
        }
        let ggml_type = c.read_u32()?;
        let type_name = ggml_type_name(ggml_type);
        *type_counts.entry(type_name.to_string()).or_insert(0) += 1;
        let _offset = c.read_u64()?;
    }

    Ok(GgufTypeScan { type_counts })
}

fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        _ => "UNKNOWN",
    }
}

struct GgufCursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> GgufCursor<'a> {
    fn read_exact(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            anyhow::bail!("unexpected EOF while parsing GGUF");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }
    fn read_u16(&mut self) -> Result<u16> {
        let mut a = [0u8; 2];
        a.copy_from_slice(self.read_exact(2)?);
        Ok(u16::from_le_bytes(a))
    }
    fn read_u32(&mut self) -> Result<u32> {
        let mut a = [0u8; 4];
        a.copy_from_slice(self.read_exact(4)?);
        Ok(u32::from_le_bytes(a))
    }
    fn read_u64(&mut self) -> Result<u64> {
        let mut a = [0u8; 8];
        a.copy_from_slice(self.read_exact(8)?);
        Ok(u64::from_le_bytes(a))
    }
    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }
    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }
    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }
    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }
    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }
    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }
    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }
    fn read_string(&mut self) -> Result<String> {
        let n = self.read_u64()? as usize;
        let s = self.read_exact(n)?;
        Ok(std::str::from_utf8(s)
            .with_context(|| "invalid UTF-8 in GGUF string")?
            .to_string())
    }
}

fn skip_gguf_value(c: &mut GgufCursor<'_>) -> Result<()> {
    let t = c.read_u32()?;
    skip_gguf_typed_value(c, t)
}

fn skip_gguf_typed_value(c: &mut GgufCursor<'_>, t: u32) -> Result<()> {
    match t {
        0 => {
            let _ = c.read_u8()?;
        }
        1 => {
            let _ = c.read_i8()?;
        }
        2 => {
            let _ = c.read_u16()?;
        }
        3 => {
            let _ = c.read_i16()?;
        }
        4 => {
            let _ = c.read_u32()?;
        }
        5 => {
            let _ = c.read_i32()?;
        }
        6 => {
            let _ = c.read_f32()?;
        }
        7 => {
            let _ = c.read_bool()?;
        }
        8 => {
            let _ = c.read_string()?;
        }
        9 => {
            let elem_t = c.read_u32()?;
            let n = c.read_u64()? as usize;
            for _ in 0..n {
                skip_gguf_typed_value(c, elem_t)?;
            }
        }
        10 => {
            let _ = c.read_u64()?;
        }
        11 => {
            let _ = c.read_i64()?;
        }
        12 => {
            let _ = c.read_f64()?;
        }
        other => anyhow::bail!("unsupported GGUF metadata value type {}", other),
    }
    Ok(())
}

#[derive(Debug, Default)]
struct GgufMeta {
    architecture: Option<String>,
    name: Option<String>,
    file_type: Option<u64>,
    quant_version: Option<u64>,
    alignment: Option<u64>,
    vocab_size: Option<usize>,
    embedding_length: Option<usize>,
    block_count: Option<usize>,
    feed_forward_length: Option<usize>,
    head_count: Option<usize>,
    head_count_kv: Option<usize>,
    rms_norm_eps: Option<f32>,
    rope_freq_base: Option<f32>,
    context_length: Option<usize>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
struct GgufTensorInfo {
    name: String,
    dims: Vec<usize>,
    ggml_type: u32,
    rel_offset: u64,
}

#[derive(Debug)]
struct ParsedGguf {
    version: u32,
    meta: GgufMeta,
    data_start: usize,
    tensors: Vec<GgufTensorInfo>,
    type_counts: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct GgufOutTensor {
    name: String,
    src_name: String,
    shape: Vec<usize>,
    header_shape: Vec<usize>,
    out_dtype: String,
    src_dims: Vec<usize>,
    src_type: u32,
    src_offset: usize,
    src_nbytes: usize,
    out_offset: u64,
    out_nbytes: usize,
    reverse_2d: bool,
    is_q8_head: bool,
    layer_idx: Option<usize>,
    op: TensorOp,
}

fn decode_gguf_all_f16(src: &[u8], src_type: u32, numel: usize) -> Result<Vec<u16>> {
    decode_gguf_prefix_f16(src, src_type, numel, numel)
}

fn convert_gguf_to_cellm(args: &Args, gguf_path: &Path, type_summary: &str) -> Result<()> {
    let f = File::open(gguf_path).with_context(|| format!("open {:?}", gguf_path))?;
    let mmap = unsafe { Mmap::map(&f).with_context(|| format!("mmap {:?}", gguf_path))? };
    let parsed = parse_gguf(&mmap)?;

    let arch = parsed
        .meta
        .architecture
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let (model_type, source_model_type, map_fn, required_fn, ffn_gate_probe_name): (
        &str,
        &str,
        fn(&str) -> Option<(String, bool, Option<usize>)>,
        fn(usize) -> Vec<String>,
        &str,
    ) = match arch.as_str() {
        "llama" => (
            "llama",
            "llama",
            map_gguf_llama_tensor_name,
            required_llama_tensor_names,
            "blk.0.ffn_gate.weight",
        ),
        "gemma4" => (
            "gemma4_text",
            "gemma4",
            map_gguf_gemma4_tensor_name,
            required_gemma4_tensor_names,
            "blk.0.ffn_gate.weight",
        ),
        _ => {
            anyhow::bail!(
                "GGUF architecture {:?} is not supported yet. Current GGUF conversion path supports llama and gemma4.",
                arch
            );
        }
    };

    let mut by_src_name: HashMap<&str, &GgufTensorInfo> = HashMap::new();
    for t in &parsed.tensors {
        by_src_name.insert(t.name.as_str(), t);
    }

    let mut out_tensors: Vec<GgufOutTensor> = Vec::new();
    // Safety valve: Gemma4 int4 conversion can stall on the large per-layer-input tensor family.
    // Skip those tensors in this mode so conversion can complete on constrained machines.
    let skip_gemma4_per_layer =
        arch == "gemma4" && (args.quantize_int4_symmetric || args.quantize_int8_symmetric);
    for src in &parsed.tensors {
        let Some((dst_name, reverse_2d, layer_idx)) = map_fn(&src.name) else {
            continue;
        };
        if skip_gemma4_per_layer
            && (dst_name.starts_with("model.per_layer_") || dst_name.contains(".per_layer_"))
        {
            continue;
        }
        let numel = src
            .dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| anyhow::anyhow!("numel overflow for {}", src.name))?;
        let src_nbytes = ggml_tensor_nbytes(src.ggml_type, numel).ok_or_else(|| {
            anyhow::anyhow!(
                "unsupported GGUF tensor type for {}: {} ({})",
                src.name,
                src.ggml_type,
                ggml_type_name(src.ggml_type)
            )
        })?;
        let src_offset = parsed
            .data_start
            .checked_add(src.rel_offset as usize)
            .ok_or_else(|| anyhow::anyhow!("offset overflow for {}", src.name))?;
        if src_offset + src_nbytes > mmap.len() {
            anyhow::bail!(
                "GGUF tensor {} out of range: offset={} nbytes={} file={}",
                src.name,
                src_offset,
                src_nbytes,
                mmap.len()
            );
        }

        let shape = if reverse_2d {
            if src.dims.len() != 2 {
                anyhow::bail!(
                    "GGUF tensor {} expected 2D for reverse mapping, got {:?}",
                    src.name,
                    src.dims
                );
            }
            vec![src.dims[1], src.dims[0]]
        } else {
            src.dims.clone()
        };
        let out_nbytes = numel
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("output nbytes overflow for {}", src.name))?;

        let mut op = TensorOp::CopyAsF16;
        let mut final_out_nbytes = out_nbytes;

        // Apply quantization if requested and if the tensor is a candidate (usually 2D weights)
        if args.quantize_int4_symmetric
            && shape.len() == 2
            && !is_quantization_excluded_name(&dst_name)
        {
            op = TensorOp::QuantizeI4Data;
            // int4 is packed per row, so odd in_dim needs row-wise ceil(in_dim/2).
            final_out_nbytes = i4_packed_nbytes_for_shape(&shape)?;
            
            // Push the main data tensor
            out_tensors.push(GgufOutTensor {
                src_name: src.name.clone(),
                src_dims: src.dims.clone(),
                name: dst_name.clone(),
                shape: shape.clone(),
                header_shape: shape.clone(),
                out_dtype: "i4".to_string(),
                src_type: src.ggml_type,
                src_offset,
                src_nbytes,
                out_nbytes: final_out_nbytes,
                out_offset: 0,
                reverse_2d,
                is_q8_head: false,
                layer_idx,
                op: TensorOp::QuantizeI4Data,
            });

            // Push a companion scales tensor
            out_tensors.push(GgufOutTensor {
                src_name: src.name.clone(),
                src_dims: src.dims.clone(),
                name: format!("{}.qscale", dst_name),
                shape: shape.clone(), // Use 2D shape for Pass 2 processing
                header_shape: vec![shape[0]], // Use 1D shape for header
                out_dtype: "f16".to_string(),
                src_type: src.ggml_type,
                src_offset,
                src_nbytes,
                out_nbytes: shape[0] * 2, // f16
                out_offset: 0,
                reverse_2d,
                is_q8_head: false,
                layer_idx,
                op: TensorOp::QuantizeI4Scales { weight_name: dst_name },
            });
            continue;
        }

        if args.quantize_int8_symmetric
            && shape.len() == 2
            && !is_quantization_excluded_name(&dst_name)
        {
            op = TensorOp::QuantizeI8Data;
            final_out_nbytes = numel;

            out_tensors.push(GgufOutTensor {
                src_name: src.name.clone(),
                src_dims: src.dims.clone(),
                name: dst_name.clone(),
                shape: shape.clone(),
                header_shape: shape.clone(),
                out_dtype: "i8".to_string(),
                src_type: src.ggml_type,
                src_offset,
                src_nbytes,
                out_nbytes: final_out_nbytes,
                out_offset: 0,
                reverse_2d,
                is_q8_head: false,
                layer_idx,
                op: TensorOp::QuantizeI8Data,
            });

            out_tensors.push(GgufOutTensor {
                src_name: src.name.clone(),
                src_dims: src.dims.clone(),
                name: format!("{}.qscale", dst_name),
                shape: shape.clone(),
                header_shape: vec![shape[0]],
                out_dtype: "f16".to_string(),
                src_type: src.ggml_type,
                src_offset,
                src_nbytes,
                out_nbytes: shape[0] * 2,
                out_offset: 0,
                reverse_2d,
                is_q8_head: false,
                layer_idx,
                op: TensorOp::QuantizeI8Scales { weight_name: dst_name },
            });
            continue;
        }

        out_tensors.push(GgufOutTensor {
            src_name: src.name.clone(),
            src_dims: src.dims.clone(),
            name: dst_name,
            shape: shape.clone(),
            header_shape: shape,
            out_dtype: "f16".to_string(),
            src_type: src.ggml_type,
            src_offset,
            src_nbytes,
            out_nbytes: final_out_nbytes,
            out_offset: 0,
            reverse_2d,
            is_q8_head: false,
            layer_idx,
            op,
        });
    }

    out_tensors.sort_by(|a, b| a.name.cmp(&b.name));

    if out_tensors.is_empty() {
        anyhow::bail!("no GGUF tensors were mapped for architecture={} from {:?}", arch, gguf_path);
    }

    let num_layers = parsed.meta.block_count.unwrap_or_else(|| {
        out_tensors
            .iter()
            .filter_map(|t| t.layer_idx)
            .max()
            .map(|v| v + 1)
            .unwrap_or(0)
    });
    if num_layers == 0 {
        anyhow::bail!("failed to infer layer count from GGUF (arch={})", arch);
    }

    let mapped_names: std::collections::HashSet<&str> =
        out_tensors.iter().map(|t| t.name.as_str()).collect();
    let mut missing: Vec<String> = Vec::new();
    let mut required = required_fn(num_layers);
    if skip_gemma4_per_layer {
        required.retain(|n| !(n.starts_with("model.per_layer_") || n.contains(".per_layer_")));
    }
    for req in required {
        if !mapped_names.contains(req.as_str()) {
            missing.push(req);
        }
    }
    if !missing.is_empty() {
        missing.sort();
        let preview = missing
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!(
            "mapped GGUF is missing required tensors for {} (showing up to 20): {}{}",
            arch,
            preview,
            if missing.len() > 20 { " ..." } else { "" }
        );
    }

    let embed = by_src_name
        .get("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("missing token_embd.weight in GGUF"))?;
    if embed.dims.len() != 2 {
        anyhow::bail!("token_embd.weight expected 2D, got {:?}", embed.dims);
    }
    let vocab_size = parsed.meta.vocab_size.unwrap_or(embed.dims[1]);
    let hidden_dim = parsed.meta.embedding_length.unwrap_or(embed.dims[0]);
    let intermediate_size = parsed.meta.feed_forward_length.unwrap_or_else(|| {
        by_src_name
            .get(ffn_gate_probe_name)
            .and_then(|t| t.dims.get(1).copied())
            .unwrap_or(0)
    });
    if intermediate_size == 0 {
        anyhow::bail!("failed to infer intermediate_size from GGUF (arch={})", arch);
    }

    let num_heads = parsed
        .meta
        .head_count
        .ok_or_else(|| anyhow::anyhow!("missing attention.head_count in GGUF metadata (arch={})", arch))?;
    let num_kv_heads = parsed.meta.head_count_kv.unwrap_or(num_heads);
    let rms_norm_eps = parsed.meta.rms_norm_eps.unwrap_or(1e-5);
    let rope_theta = parsed.meta.rope_freq_base.unwrap_or(10000.0);

    let tie_word_embeddings = Some(by_src_name.get("output.weight").is_none());
    let source_quantization = Some(serde_json::json!({
        "source_format": "gguf",
        "gguf_version": parsed.version,
        "file_type": parsed.meta.file_type,
        "quantization_version": parsed.meta.quant_version,
        "tensor_types": parsed.type_counts,
    }));

    // Compute output offsets.
    let mut header = CellmHeader {
        model_type: model_type.to_string(),
        source_model_type: Some(source_model_type.to_string()),
        source_safetensors_format: None,
        text_tensor_prefix: None,
        vision_tensor_prefix: None,
        projector_tensor_prefix: None,
        vocab_size,
        hidden_dim,
        intermediate_size,
        num_layers,
        num_heads,
        num_kv_heads,
        rms_norm_eps,
        rope_theta,
        bos_token_id: parsed.meta.bos_token_id,
        eos_token_id: parsed.meta.eos_token_id,
        max_position_embeddings: parsed.meta.context_length,
        tie_word_embeddings,
        source_torch_dtype: Some("mixed_f16_bf16_f32_qk_from_gguf".to_string()),
        source_architectures: Some(vec![arch.clone()]),
        source_quantization,
        source_quantization_config: None,
        source_text_config: None,
        source_vision_config: None,
        source_projector_config: None,
        tensors: Vec::new(),
    };

    let mut planned = out_tensors.clone();
    let mut last_header_len = None;
    for _ in 0..3 {
        header.tensors = planned
            .iter()
            .map(|p| CellmTensorIndex {
                name: p.name.clone(),
                offset_bytes: p.out_offset,
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
            p.out_offset = cursor;
            cursor += p.out_nbytes as u64;
        }
    }
    header.tensors = planned
        .iter()
        .map(|p| CellmTensorIndex {
            name: p.name.clone(),
            offset_bytes: p.out_offset,
            nbytes: p.out_nbytes as u64,
            shape: p.shape.clone(),
            dtype: p.out_dtype.clone(),
        })
        .collect();
    let header_bytes = serde_json::to_vec(&header)?;

    if let Some(validate_path) = args.validate_against.as_ref() {
        validate_gguf_conversion(
            gguf_path,
            &mmap,
            &parsed,
            &planned,
            validate_path,
            args.validate_tensor_values.max(1),
            args.validate_logits_rows.max(1),
        )?;
        log::info!(
            "Validation-only mode succeeded for GGUF {:?} against {:?}",
            gguf_path,
            validate_path
        );
        return Ok(());
    }

    let out = File::create(&args.output).with_context(|| format!("create {:?}", args.output))?;
    let mut w = BufWriter::with_capacity(65536, out);
    w.write_all(b"CELLM")?;
    w.write_all(&[1u8])?;
    w.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
    w.write_all(&header_bytes)?;
    let mut pos = (5 + 1 + 4 + header_bytes.len()) as u64;
    let aligned = align_up(pos, 64);
    if aligned > pos {
        w.write_all(&vec![0u8; (aligned - pos) as usize])?;
        pos = aligned;
    }

    log::info!("Pass 2: Writing {} tensors to {:?}...", planned.len(), args.output);
    for (idx, p) in planned.iter().enumerate() {
        if idx % 10 == 0 || p.out_nbytes > 100_000_000 {
            log::info!("[{:3}/{}] Writing {:<40} ({} MB)...", idx + 1, out_tensors.len(), p.name, p.out_nbytes / 1024 / 1024);
        }
        if pos < p.out_offset {
            w.write_all(&vec![0u8; (p.out_offset - pos) as usize])?;
            pos = p.out_offset;
        }
        let src = &mmap[p.src_offset..p.src_offset + p.src_nbytes];
        let numel = p
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| anyhow::anyhow!("numel overflow for {}", p.name))?;
        
        match &p.op {
            TensorOp::CopyAsF16 => {
                write_gguf_tensor_as_f16(src, p.src_type, numel, &mut w)?;
            }
            TensorOp::QuantizeI4Data => {
                // First decode GGUF specialized quant to f16, then re-quant to cellm int4
                let f16_data = decode_gguf_all_f16(src, p.src_type, numel)?;
                write_quant_i4_data_per_row_symmetric(cast_slice(&f16_data), &p.shape, Dtype::F16, &mut w)?;
            }
            TensorOp::QuantizeI4Scales { weight_name: _ } => {
                // Similar for scales
                let f16_data = decode_gguf_all_f16(src, p.src_type, numel)?;
                write_quant_i4_scales_per_row_symmetric(cast_slice(&f16_data), &p.shape, Dtype::F16, &mut w)?;
            }
            TensorOp::QuantizeI8Data => {
                let f16_data = decode_gguf_all_f16(src, p.src_type, numel)?;
                write_quant_i8_data_per_row_symmetric(cast_slice(&f16_data), &p.shape, Dtype::F16, &mut w)?;
            }
            TensorOp::QuantizeI8Scales { weight_name: _ } => {
                let f16_data = decode_gguf_all_f16(src, p.src_type, numel)?;
                write_quant_i8_scales_per_row_symmetric(cast_slice(&f16_data), &p.shape, Dtype::F16, &mut w)?;
            }
            _ => {
                // Fallback to f16 for now
                write_gguf_tensor_as_f16(src, p.src_type, numel, &mut w)?;
            }
        }
        
        pos += p.out_nbytes as u64;
    }
    w.flush()?;

    log::info!(
        "Converted GGUF {:?} -> {:?} ({} tensors). Types: {}",
        gguf_path,
        args.output,
        header.tensors.len(),
        type_summary
    );
    if args.validate_gguf {
        validate_gguf_conversion(
            gguf_path,
            &mmap,
            &parsed,
            &planned,
            &args.output,
            args.validate_tensor_values.max(1),
            args.validate_logits_rows.max(1),
        )?;
    }
    Ok(())
}

fn validate_gguf_conversion(
    gguf_path: &Path,
    gguf_mmap: &[u8],
    _parsed: &ParsedGguf,
    planned: &[GgufOutTensor],
    out_path: &Path,
    tensor_values: usize,
    logits_rows: usize,
) -> Result<()> {
    log::info!(
        "Validation: GGUF tensor checks for {:?} against {:?}",
        gguf_path,
        out_path
    );
    let cellm = CellmFile::load(out_path)?;

    for t in planned {
        let meta = cellm
            .tensor_index(&t.name)
            .ok_or_else(|| anyhow::anyhow!("validation: missing output tensor {}", t.name))?;
        if meta.dtype != "f16" {
            anyhow::bail!(
                "validation: tensor {} dtype mismatch: {} != f16",
                t.name,
                meta.dtype
            );
        }
        if meta.shape != t.shape {
            anyhow::bail!(
                "validation: tensor {} shape mismatch: {:?} != {:?}",
                t.name,
                meta.shape,
                t.shape
            );
        }
        if meta.nbytes as usize != t.out_nbytes {
            anyhow::bail!(
                "validation: tensor {} nbytes mismatch: {} != {}",
                t.name,
                meta.nbytes,
                t.out_nbytes
            );
        }

        let src = &gguf_mmap[t.src_offset..t.src_offset + t.src_nbytes];
        let out_bytes = cellm.tensor_bytes(&t.name)?;
        let out_u16: &[u16] = cast_slice(out_bytes);
        let expected = decode_gguf_prefix_f16(src, t.src_type, t.shape.iter().product(), tensor_values)?;
        let got_len = expected.len().min(out_u16.len());
        let mut max_abs = 0.0f32;
        let mut mean_abs = 0.0f32;
        for i in 0..got_len {
            let a = f16::from_bits(expected[i]).to_f32();
            let b = f16::from_bits(out_u16[i]).to_f32();
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            mean_abs += d;
        }
        if got_len > 0 {
            mean_abs /= got_len as f32;
        }
        if max_abs > 1e-3 {
            anyhow::bail!(
                "validation: tensor {} prefix mismatch: max_abs={} mean_abs={} ({} values)",
                t.name,
                max_abs,
                mean_abs,
                got_len
            );
        }
    }

    validate_gguf_logits_parity(gguf_mmap, planned, &cellm, logits_rows)?;
    log::info!("Validation: GGUF conversion parity checks passed.");
    Ok(())
}

fn validate_gguf_logits_parity(
    gguf_mmap: &[u8],
    planned: &[GgufOutTensor],
    cellm: &CellmFile,
    logits_rows: usize,
) -> Result<()> {
    let lm = planned
        .iter()
        .find(|t| t.name == "lm_head.weight")
        .or_else(|| planned.iter().find(|t| t.name == "model.embed_tokens.weight"))
        .ok_or_else(|| anyhow::anyhow!("validation: missing lm_head/embed tensor for logits parity"))?;

    if lm.shape.len() != 2 {
        anyhow::bail!("validation: logits tensor {} is not 2D", lm.name);
    }
    if !lm.reverse_2d {
        anyhow::bail!(
            "validation: logits tensor {} expected reverse_2d mapping from GGUF",
            lm.name
        );
    }

    let vocab = lm.shape[0];
    let hidden = lm.shape[1];
    let rows = logits_rows.min(vocab).max(1);
    let src_row_ne0 = *lm
        .src_dims
        .first()
        .ok_or_else(|| anyhow::anyhow!("validation: missing source dims for {}", lm.src_name))?;
    if src_row_ne0 != hidden {
        anyhow::bail!(
            "validation: source row width mismatch for {}: {} != hidden {}",
            lm.src_name,
            src_row_ne0,
            hidden
        );
    }

    let src_row_nbytes = ggml_tensor_nbytes(lm.src_type, src_row_ne0).ok_or_else(|| {
        anyhow::anyhow!(
            "validation: unsupported source dtype {} for {}",
            lm.src_type,
            lm.src_name
        )
    })?;

    let mut x = vec![0.0f32; hidden];
    for (i, xi) in x.iter_mut().enumerate() {
        let a = (i as f32 * 0.017).sin();
        let b = (i as f32 * 0.023).cos();
        *xi = 0.5 * a + 0.5 * b;
    }

    let out_bytes = cellm.tensor_bytes(&lm.name)?;
    let out_u16: &[u16] = cast_slice(out_bytes);
    if out_u16.len() != vocab * hidden {
        anyhow::bail!(
            "validation: output tensor {} len mismatch: {} != {}",
            lm.name,
            out_u16.len(),
            vocab * hidden
        );
    }

    let src_all = &gguf_mmap[lm.src_offset..lm.src_offset + lm.src_nbytes];
    let mut max_abs = 0.0f32;
    let mut mean_abs = 0.0f32;
    let mut src_top = (0usize, f32::NEG_INFINITY);
    let mut dst_top = (0usize, f32::NEG_INFINITY);
    for row in 0..rows {
        let src_off = row
            .checked_mul(src_row_nbytes)
            .ok_or_else(|| anyhow::anyhow!("validation: source row offset overflow"))?;
        let src_row = &src_all[src_off..src_off + src_row_nbytes];
        let src_vals = decode_gguf_prefix_f32(src_row, lm.src_type, hidden, hidden)?;
        let dst_row = &out_u16[row * hidden..(row + 1) * hidden];

        let mut src_dot = 0.0f32;
        let mut dst_dot = 0.0f32;
        for i in 0..hidden {
            src_dot += src_vals[i] * x[i];
            dst_dot += f16::from_bits(dst_row[i]).to_f32() * x[i];
        }
        if src_dot > src_top.1 {
            src_top = (row, src_dot);
        }
        if dst_dot > dst_top.1 {
            dst_top = (row, dst_dot);
        }
        let d = (src_dot - dst_dot).abs();
        max_abs = max_abs.max(d);
        mean_abs += d;
    }
    mean_abs /= rows as f32;

    if max_abs > 5e-2 {
        anyhow::bail!(
            "validation: logits parity failed for {}: max_abs={} mean_abs={} src_top={} dst_top={}",
            lm.name,
            max_abs,
            mean_abs,
            src_top.0,
            dst_top.0
        );
    }

    log::info!(
        "Validation logits parity: tensor={} rows={} max_abs={} mean_abs={} top_row(src={},dst={})",
        lm.name,
        rows,
        max_abs,
        mean_abs,
        src_top.0,
        dst_top.0
    );
    Ok(())
}


fn decode_gguf_prefix_f16(
    src: &[u8],
    src_type: u32,
    numel: usize,
    want: usize,
) -> Result<Vec<u16>> {
    let vals = decode_gguf_prefix_f32(src, src_type, numel, want)?;
    Ok(vals.into_iter().map(|v| f16::from_f32(v).to_bits()).collect())
}

fn decode_gguf_prefix_f32(
    src: &[u8],
    src_type: u32,
    numel: usize,
    want: usize,
) -> Result<Vec<f32>> {
    let target = want.min(numel);
    let mut out = Vec::with_capacity(target);
    match src_type {
        0 => {
            // F32
            if src.len() % 4 != 0 {
                anyhow::bail!("decode prefix f32: bad source length {}", src.len());
            }
            let n = (src.len() / 4).min(target);
            for i in 0..n {
                let off = i * 4;
                let bits = u32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]]);
                out.push(f32::from_bits(bits));
            }
        }
        1 => {
            // F16
            if src.len() % 2 != 0 {
                anyhow::bail!("decode prefix f16: bad source length {}", src.len());
            }
            let n = (src.len() / 2).min(target);
            for i in 0..n {
                let off = i * 2;
                out.push(read_f16_le_as_f32(src[off], src[off + 1]));
            }
        }
        30 => {
            // BF16
            if src.len() % 2 != 0 {
                anyhow::bail!("decode prefix bf16: bad source length {}", src.len());
            }
            let n = (src.len() / 2).min(target);
            for i in 0..n {
                let off = i * 2;
                let bits = u16::from_le_bytes([src[off], src[off + 1]]);
                out.push(f32::from_bits((bits as u32) << 16));
            }
        }
        2 => decode_q4_0_prefix(src, target, &mut out)?,
        3 => decode_q4_1_prefix(src, target, &mut out)?,
        10 => decode_qk_prefix(src, target, 84, dequantize_q2k_block, &mut out)?,
        11 => decode_qk_prefix(src, target, 110, dequantize_q3k_block, &mut out)?,
        12 => decode_qk_prefix(src, target, 144, dequantize_q4k_block, &mut out)?,
        13 => decode_qk_prefix(src, target, 176, dequantize_q5k_block, &mut out)?,
        14 => decode_qk_prefix(src, target, 210, dequantize_q6k_block, &mut out)?,
        other => anyhow::bail!("decode prefix: unsupported GGUF type {}", other),
    }
    if out.len() < target {
        anyhow::bail!(
            "decode prefix: insufficient decoded values {} < {}",
            out.len(),
            target
        );
    }
    out.truncate(target);
    Ok(out)
}

fn decode_q4_0_prefix(src: &[u8], target: usize, out: &mut Vec<f32>) -> Result<()> {
    let bs = 18usize;
    if src.len() % bs != 0 {
        anyhow::bail!("Q4_0 decode prefix: bad source length {}", src.len());
    }
    let nblk = src.len() / bs;
    for bi in 0..nblk {
        if out.len() >= target {
            break;
        }
        let off = bi * bs;
        let d = read_f16_le_as_f32(src[off], src[off + 1]);
        let q = &src[off + 2..off + bs];
        for i in 0..16usize {
            if out.len() >= target {
                break;
            }
            let b = q[i];
            let q0 = (b & 0x0F) as i32 - 8;
            out.push(d * q0 as f32);
            if out.len() >= target {
                break;
            }
            let q1 = (b >> 4) as i32 - 8;
            out.push(d * q1 as f32);
        }
    }
    Ok(())
}

fn decode_q4_1_prefix(src: &[u8], target: usize, out: &mut Vec<f32>) -> Result<()> {
    let bs = 20usize;
    if src.len() % bs != 0 {
        anyhow::bail!("Q4_1 decode prefix: bad source length {}", src.len());
    }
    let nblk = src.len() / bs;
    for bi in 0..nblk {
        if out.len() >= target {
            break;
        }
        let off = bi * bs;
        let d = read_f16_le_as_f32(src[off], src[off + 1]);
        let m = read_f16_le_as_f32(src[off + 2], src[off + 3]);
        let q = &src[off + 4..off + bs];
        for i in 0..16usize {
            if out.len() >= target {
                break;
            }
            let b = q[i];
            let q0 = (b & 0x0F) as f32;
            out.push(d * q0 + m);
            if out.len() >= target {
                break;
            }
            let q1 = (b >> 4) as f32;
            out.push(d * q1 + m);
        }
    }
    Ok(())
}

fn decode_qk_prefix(
    src: &[u8],
    target: usize,
    block_size: usize,
    decode_block: fn(&[u8], &mut [f32]) -> Result<()>,
    out: &mut Vec<f32>,
) -> Result<()> {
    if src.len() % block_size != 0 {
        anyhow::bail!(
            "QK decode prefix: bad source length {} for block {}",
            src.len(),
            block_size
        );
    }
    let nblk = src.len() / block_size;
    let mut vals = vec![0.0f32; 256];
    for bi in 0..nblk {
        if out.len() >= target {
            break;
        }
        let off = bi * block_size;
        decode_block(&src[off..off + block_size], &mut vals)?;
        let need = target - out.len();
        let take = need.min(vals.len());
        out.extend_from_slice(&vals[..take]);
    }
    Ok(())
}

fn required_llama_tensor_names(num_layers: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(2 + num_layers * 9);
    out.push("model.embed_tokens.weight".to_string());
    out.push("model.norm.weight".to_string());
    for i in 0..num_layers {
        out.push(format!("model.layers.{i}.input_layernorm.weight"));
        out.push(format!("model.layers.{i}.self_attn.q_proj.weight"));
        out.push(format!("model.layers.{i}.self_attn.k_proj.weight"));
        out.push(format!("model.layers.{i}.self_attn.v_proj.weight"));
        out.push(format!("model.layers.{i}.self_attn.o_proj.weight"));
        out.push(format!("model.layers.{i}.post_attention_layernorm.weight"));
        out.push(format!("model.layers.{i}.mlp.gate_proj.weight"));
        out.push(format!("model.layers.{i}.mlp.up_proj.weight"));
        out.push(format!("model.layers.{i}.mlp.down_proj.weight"));
    }
    out
}

fn required_gemma_tensor_names(num_layers: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(2 + num_layers * 13);
    out.push("model.embed_tokens.weight".to_string());
    out.push("model.norm.weight".to_string());
    for i in 0..num_layers {
        out.push(format!("model.layers.{i}.input_layernorm.weight"));
        out.push(format!("model.layers.{i}.self_attn.q_proj.weight"));
        out.push(format!("model.layers.{i}.self_attn.k_proj.weight"));
        out.push(format!("model.layers.{i}.self_attn.v_proj.weight"));
        out.push(format!("model.layers.{i}.self_attn.o_proj.weight"));
        out.push(format!("model.layers.{i}.self_attn.q_norm.weight"));
        out.push(format!("model.layers.{i}.self_attn.k_norm.weight"));
        out.push(format!("model.layers.{i}.post_attention_layernorm.weight"));
        out.push(format!("model.layers.{i}.pre_feedforward_layernorm.weight"));
        out.push(format!("model.layers.{i}.mlp.gate_proj.weight"));
        out.push(format!("model.layers.{i}.mlp.up_proj.weight"));
        out.push(format!("model.layers.{i}.mlp.down_proj.weight"));
        out.push(format!("model.layers.{i}.post_feedforward_layernorm.weight"));
    }
    out
}

fn required_gemma4_tensor_names(num_layers: usize) -> Vec<String> {
    let mut out = required_gemma_tensor_names(num_layers);
    out.push("model.per_layer_token_embd.weight".to_string());
    out.push("model.per_layer_model_proj.weight".to_string());
    out.push("model.per_layer_proj_norm.weight".to_string());
    for i in 0..num_layers {
        out.push(format!("model.layers.{i}.per_layer_input_gate.weight"));
        out.push(format!("model.layers.{i}.per_layer_projection.weight"));
        out.push(format!("model.layers.{i}.post_per_layer_input_norm.weight"));
        out.push(format!("model.layers.{i}.layer_output_scale.weight"));
    }
    out
}

fn map_gguf_llama_tensor_name(name: &str) -> Option<(String, bool, Option<usize>)> {
    match name {
        "token_embd.weight" => {
            return Some(("model.embed_tokens.weight".to_string(), true, None))
        }
        "output_norm.weight" => return Some(("model.norm.weight".to_string(), false, None)),
        "output.weight" => return Some(("lm_head.weight".to_string(), true, None)),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (layer_s, suffix) = rest.split_once('.')?;
    let layer_idx: usize = layer_s.parse().ok()?;
    let mapped = match suffix {
        "attn_norm.weight" => ("input_layernorm.weight", false),
        "attn_q.weight" => ("self_attn.q_proj.weight", true),
        "attn_k.weight" => ("self_attn.k_proj.weight", true),
        "attn_v.weight" => ("self_attn.v_proj.weight", true),
        "attn_output.weight" => ("self_attn.o_proj.weight", true),
        "ffn_norm.weight" => ("post_attention_layernorm.weight", false),
        "ffn_gate.weight" => ("mlp.gate_proj.weight", true),
        "ffn_up.weight" => ("mlp.up_proj.weight", true),
        "ffn_down.weight" => ("mlp.down_proj.weight", true),
        _ => return None,
    };
    Some((
        format!("model.layers.{layer_idx}.{}", mapped.0),
        mapped.1,
        Some(layer_idx),
    ))
}

fn map_gguf_gemma4_tensor_name(name: &str) -> Option<(String, bool, Option<usize>)> {
    match name {
        "token_embd.weight" => {
            return Some(("model.embed_tokens.weight".to_string(), true, None))
        }
        "per_layer_token_embd.weight" => {
            return Some(("model.per_layer_token_embd.weight".to_string(), true, None))
        }
        "per_layer_model_proj.weight" => {
            return Some(("model.per_layer_model_proj.weight".to_string(), true, None))
        }
        "per_layer_proj_norm.weight" => {
            return Some(("model.per_layer_proj_norm.weight".to_string(), false, None))
        }
        "output_norm.weight" => return Some(("model.norm.weight".to_string(), false, None)),
        "output.weight" => return Some(("lm_head.weight".to_string(), true, None)),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (layer_s, suffix) = rest.split_once('.')?;
    let layer_idx: usize = layer_s.parse().ok()?;
    let mapped = match suffix {
        "attn_norm.weight" => ("input_layernorm.weight", false),
        "attn_q.weight" => ("self_attn.q_proj.weight", true),
        "attn_k.weight" => ("self_attn.k_proj.weight", true),
        "attn_v.weight" => ("self_attn.v_proj.weight", true),
        "attn_output.weight" => ("self_attn.o_proj.weight", true),
        "attn_q_norm.weight" => ("self_attn.q_norm.weight", false),
        "attn_k_norm.weight" => ("self_attn.k_norm.weight", false),
        "post_attention_norm.weight" => ("post_attention_layernorm.weight", false),
        "ffn_norm.weight" => ("pre_feedforward_layernorm.weight", false),
        "ffn_gate.weight" => ("mlp.gate_proj.weight", true),
        "ffn_up.weight" => ("mlp.up_proj.weight", true),
        "ffn_down.weight" => ("mlp.down_proj.weight", true),
        "post_ffw_norm.weight" => ("post_feedforward_layernorm.weight", false),
        "inp_gate.weight" => ("per_layer_input_gate.weight", true),
        "proj.weight" => ("per_layer_projection.weight", true),
        "post_norm.weight" => ("post_per_layer_input_norm.weight", false),
        "layer_output_scale.weight" => ("layer_output_scale.weight", false),
        _ => return None,
    };
    Some((
        format!("model.layers.{layer_idx}.{}", mapped.0),
        mapped.1,
        Some(layer_idx),
    ))
}

fn parse_gguf(bytes: &[u8]) -> Result<ParsedGguf> {
    let mut c = GgufCursor { b: bytes, p: 0 };
    let magic = c.read_exact(4)?;
    if magic != b"GGUF" {
        anyhow::bail!("not a GGUF file (bad magic)");
    }
    let version = c.read_u32()?;
    if version < 2 || version > 3 {
        anyhow::bail!("unsupported GGUF version {}", version);
    }
    let tensor_count = c.read_u64()? as usize;
    let kv_count = c.read_u64()? as usize;

    let mut meta = GgufMeta::default();
    for _ in 0..kv_count {
        let key = c.read_string()?;
        let t = c.read_u32()?;
        match key.as_str() {
            "general.architecture" => meta.architecture = Some(read_gguf_string_typed(&mut c, t)?),
            "general.name" => meta.name = Some(read_gguf_string_typed(&mut c, t)?),
            "general.file_type" => meta.file_type = Some(read_gguf_u64_typed(&mut c, t)?),
            "general.quantization_version" => {
                meta.quant_version = Some(read_gguf_u64_typed(&mut c, t)?)
            }
            "general.alignment" => meta.alignment = Some(read_gguf_u64_typed(&mut c, t)?),
            "llama.vocab_size" | "gemma4.vocab_size" => {
                meta.vocab_size = Some(read_gguf_u64_typed(&mut c, t)? as usize)
            }
            "llama.embedding_length" | "gemma4.embedding_length" => {
                meta.embedding_length = Some(read_gguf_u64_typed(&mut c, t)? as usize)
            }
            "llama.block_count" | "gemma4.block_count" => {
                meta.block_count = Some(read_gguf_u64_typed(&mut c, t)? as usize)
            }
            "llama.feed_forward_length" | "gemma4.feed_forward_length" => {
                meta.feed_forward_length = Some(read_gguf_u64_typed(&mut c, t)? as usize)
            }
            "llama.attention.head_count" | "gemma4.attention.head_count" => {
                meta.head_count = Some(read_gguf_u64_typed(&mut c, t)? as usize)
            }
            "llama.attention.head_count_kv" | "gemma4.attention.head_count_kv" => {
                meta.head_count_kv = Some(read_gguf_u64_typed(&mut c, t)? as usize)
            }
            "llama.attention.layer_norm_rms_epsilon" | "gemma4.attention.layer_norm_rms_epsilon" => {
                meta.rms_norm_eps = Some(read_gguf_f32_typed(&mut c, t)?)
            }
            "llama.rope.freq_base" | "gemma4.rope.freq_base" => {
                meta.rope_freq_base = Some(read_gguf_f32_typed(&mut c, t)?)
            }
            "llama.context_length" | "gemma4.context_length" => {
                meta.context_length = Some(read_gguf_u64_typed(&mut c, t)? as usize)
            }
            "tokenizer.ggml.bos_token_id" => {
                meta.bos_token_id = Some(read_gguf_u64_typed(&mut c, t)? as u32)
            }
            "tokenizer.ggml.eos_token_id" => {
                meta.eos_token_id = Some(read_gguf_u64_typed(&mut c, t)? as u32)
            }
            _ => skip_gguf_typed_value(&mut c, t)?,
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..tensor_count {
        let name = c.read_string()?;
        let n_dims = c.read_u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(c.read_u64()? as usize);
        }
        let ggml_type = c.read_u32()?;
        let rel_offset = c.read_u64()?;
        *type_counts
            .entry(ggml_type_name(ggml_type).to_string())
            .or_insert(0) += 1;
        tensors.push(GgufTensorInfo {
            name,
            dims,
            ggml_type,
            rel_offset,
        });
    }

    let alignment = meta.alignment.unwrap_or(32).max(1);
    let data_start = align_up(c.p as u64, alignment) as usize;
    if data_start > bytes.len() {
        anyhow::bail!(
            "GGUF data start {} exceeds file length {}",
            data_start,
            bytes.len()
        );
    }

    Ok(ParsedGguf {
        version,
        meta,
        data_start,
        tensors,
        type_counts,
    })
}

fn read_gguf_string_typed(c: &mut GgufCursor<'_>, t: u32) -> Result<String> {
    if t != 8 {
        anyhow::bail!("expected GGUF string (type=8), got type={}", t);
    }
    c.read_string()
}

fn read_gguf_u64_typed(c: &mut GgufCursor<'_>, t: u32) -> Result<u64> {
    match t {
        0 => Ok(c.read_u8()? as u64),
        1 => Ok(c.read_i8()? as i64 as u64),
        2 => Ok(c.read_u16()? as u64),
        3 => Ok(c.read_i16()? as i64 as u64),
        4 => Ok(c.read_u32()? as u64),
        5 => Ok(c.read_i32()? as i64 as u64),
        9 => {
            let elem_t = c.read_u32()?;
            let n = c.read_u64()? as usize;
            if n == 0 {
                anyhow::bail!("expected non-empty GGUF numeric array, got empty array");
            }
            let first = read_gguf_u64_typed(c, elem_t)?;
            for _ in 1..n {
                skip_gguf_typed_value(c, elem_t)?;
            }
            Ok(first)
        }
        10 => Ok(c.read_u64()?),
        11 => Ok(c.read_i64()? as u64),
        other => anyhow::bail!("expected GGUF integer scalar, got type={}", other),
    }
}

fn read_gguf_f32_typed(c: &mut GgufCursor<'_>, t: u32) -> Result<f32> {
    match t {
        6 => c.read_f32(),
        12 => Ok(c.read_f64()? as f32),
        0 | 1 | 2 | 3 | 4 | 5 | 10 | 11 => Ok(read_gguf_u64_typed(c, t)? as f32),
        other => anyhow::bail!("expected GGUF numeric scalar, got type={}", other),
    }
}

fn ggml_type_elem_size(t: u32) -> Option<usize> {
    match t {
        0 => Some(4), // F32
        1 => Some(2), // F16
        30 => Some(2), // BF16
        _ => None,
    }
}

fn ggml_tensor_nbytes(t: u32, numel: usize) -> Option<usize> {
    // Legacy Q4_0 uses 32-element blocks.
    if t == 2 || t == 3 {
        let qk = 32usize;
        if numel % qk != 0 {
            return None;
        }
        let nblk = numel / qk;
        let bs = if t == 2 { 18 } else { 20 };
        return nblk.checked_mul(bs);
    }
    if let Some(elem) = ggml_type_elem_size(t) {
        return numel.checked_mul(elem);
    }
    // K-quants use 256-element blocks.
    let qk = 256usize;
    if numel % qk != 0 {
        return None;
    }
    let nblk = numel / qk;
    match t {
        10 => nblk.checked_mul(84), // Q2_K block size
        11 => nblk.checked_mul(110), // Q3_K block size
        12 => nblk.checked_mul(144), // Q4_K block size
        13 => nblk.checked_mul(176), // Q5_K block size
        14 => nblk.checked_mul(210), // Q6_K block size
        _ => None,
    }
}

fn write_gguf_tensor_as_f16<W: Write>(
    src: &[u8],
    src_type: u32,
    numel: usize,
    w: &mut W,
) -> Result<()> {
    match src_type {
        2 => {
            dequant_q4_0_to_f16(src, numel, w)?;
        }
        3 => {
            dequant_q4_1_to_f16(src, numel, w)?;
        }
        1 => {
            w.write_all(src)?;
        }
        30 => {
            write_bf16_as_f16(src, w)?;
        }
        0 => {
            write_f32_as_f16(src, w)?;
        }
        12 => {
            dequant_q4k_to_f16(src, numel, w)?;
        }
        13 => {
            dequant_q5k_to_f16(src, numel, w)?;
        }
        14 => {
            dequant_q6k_to_f16(src, numel, w)?;
        }
        10 => {
            dequant_q2k_to_f16(src, numel, w)?;
        }
        11 => {
            dequant_q3k_to_f16(src, numel, w)?;
        }
        other => {
            anyhow::bail!(
                "unsupported GGUF source tensor type for f16 conversion: {} ({})",
                other,
                ggml_type_name(other)
            );
        }
    }
    Ok(())
}

fn dequant_q4_0_to_f16<W: Write>(src: &[u8], numel: usize, w: &mut W) -> Result<()> {
    let qk = 32usize;
    if numel % qk != 0 {
        anyhow::bail!("Q4_0 tensor numel {} is not divisible by {}", numel, qk);
    }
    let bs = 18usize; // fp16 scale + 16 packed nibbles for 32 values
    let nblk = numel / qk;
    if src.len() != nblk * bs {
        anyhow::bail!(
            "Q4_0 source size mismatch: got {} bytes, expected {}",
            src.len(),
            nblk * bs
        );
    }
    let mut out_f16 = vec![0u16; qk];
    for bi in 0..nblk {
        let off = bi * bs;
        let d = f16::from_bits(u16::from_le_bytes([src[off], src[off + 1]])).to_f32();
        let qbytes = &src[off + 2..off + bs];
        for i in 0..16 {
            let b = qbytes[i];
            let q0 = (b & 0x0F) as i32 - 8;
            let q1 = (b >> 4) as i32 - 8;
            out_f16[2 * i] = f16::from_f32(d * (q0 as f32)).to_bits();
            out_f16[2 * i + 1] = f16::from_f32(d * (q1 as f32)).to_bits();
        }
        w.write_all(cast_slice(&out_f16))?;
    }
    Ok(())
}

fn dequant_q4_1_to_f16<W: Write>(src: &[u8], numel: usize, w: &mut W) -> Result<()> {
    let qk = 32usize;
    if numel % qk != 0 {
        anyhow::bail!("Q4_1 tensor numel {} is not divisible by {}", numel, qk);
    }
    let bs = 20usize; // fp16 d + fp16 m + 16 packed nibbles
    let nblk = numel / qk;
    if src.len() != nblk * bs {
        anyhow::bail!(
            "Q4_1 source size mismatch: got {} bytes, expected {}",
            src.len(),
            nblk * bs
        );
    }
    let mut out_f16 = vec![0u16; qk];
    for bi in 0..nblk {
        let off = bi * bs;
        let d = f16::from_bits(u16::from_le_bytes([src[off], src[off + 1]])).to_f32();
        let m = f16::from_bits(u16::from_le_bytes([src[off + 2], src[off + 3]])).to_f32();
        let qbytes = &src[off + 4..off + bs];
        for i in 0..16 {
            let b = qbytes[i];
            let q0 = (b & 0x0F) as f32;
            let q1 = (b >> 4) as f32;
            out_f16[2 * i] = f16::from_f32(d * q0 + m).to_bits();
            out_f16[2 * i + 1] = f16::from_f32(d * q1 + m).to_bits();
        }
        w.write_all(cast_slice(&out_f16))?;
    }
    Ok(())
}

fn dequant_q2k_to_f16<W: Write>(src: &[u8], numel: usize, w: &mut W) -> Result<()> {
    let qk = 256usize;
    if numel % qk != 0 {
        anyhow::bail!("Q2_K tensor numel {} is not divisible by {}", numel, qk);
    }
    let bs = 84usize;
    let nblk = numel / qk;
    if src.len() != nblk * bs {
        anyhow::bail!(
            "Q2_K source size mismatch: got {} bytes, expected {}",
            src.len(),
            nblk * bs
        );
    }
    let mut out_f16 = vec![0u16; qk];
    let mut vals = vec![0f32; qk];
    for bi in 0..nblk {
        let off = bi * bs;
        let b = &src[off..off + bs];
        dequantize_q2k_block(b, &mut vals)?;
        for i in 0..qk {
            out_f16[i] = f16::from_f32(vals[i]).to_bits();
        }
        w.write_all(cast_slice(&out_f16))?;
    }
    Ok(())
}

fn dequant_q3k_to_f16<W: Write>(src: &[u8], numel: usize, w: &mut W) -> Result<()> {
    let qk = 256usize;
    if numel % qk != 0 {
        anyhow::bail!("Q3_K tensor numel {} is not divisible by {}", numel, qk);
    }
    let bs = 110usize;
    let nblk = numel / qk;
    if src.len() != nblk * bs {
        anyhow::bail!(
            "Q3_K source size mismatch: got {} bytes, expected {}",
            src.len(),
            nblk * bs
        );
    }
    let mut out_f16 = vec![0u16; qk];
    let mut vals = vec![0f32; qk];
    for bi in 0..nblk {
        let off = bi * bs;
        let b = &src[off..off + bs];
        dequantize_q3k_block(b, &mut vals)?;
        for i in 0..qk {
            out_f16[i] = f16::from_f32(vals[i]).to_bits();
        }
        w.write_all(cast_slice(&out_f16))?;
    }
    Ok(())
}

fn dequant_q4k_to_f16<W: Write>(src: &[u8], numel: usize, w: &mut W) -> Result<()> {
    let qk = 256usize;
    if numel % qk != 0 {
        anyhow::bail!("Q4_K tensor numel {} is not divisible by {}", numel, qk);
    }
    let bs = 144usize;
    let nblk = numel / qk;
    if src.len() != nblk * bs {
        anyhow::bail!(
            "Q4_K source size mismatch: got {} bytes, expected {}",
            src.len(),
            nblk * bs
        );
    }
    let mut out_f16 = vec![0u16; qk];
    let mut vals = vec![0f32; qk];
    for bi in 0..nblk {
        let off = bi * bs;
        let b = &src[off..off + bs];
        dequantize_q4k_block(b, &mut vals)?;
        for i in 0..qk {
            out_f16[i] = f16::from_f32(vals[i]).to_bits();
        }
        w.write_all(cast_slice(&out_f16))?;
    }
    Ok(())
}

fn dequant_q6k_to_f16<W: Write>(src: &[u8], numel: usize, w: &mut W) -> Result<()> {
    let qk = 256usize;
    if numel % qk != 0 {
        anyhow::bail!("Q6_K tensor numel {} is not divisible by {}", numel, qk);
    }
    let bs = 210usize;
    let nblk = numel / qk;
    if src.len() != nblk * bs {
        anyhow::bail!(
            "Q6_K source size mismatch: got {} bytes, expected {}",
            src.len(),
            nblk * bs
        );
    }
    let mut out_f16 = vec![0u16; qk];
    let mut vals = vec![0f32; qk];
    for bi in 0..nblk {
        let off = bi * bs;
        let b = &src[off..off + bs];
        dequantize_q6k_block(b, &mut vals)?;
        for i in 0..qk {
            out_f16[i] = f16::from_f32(vals[i]).to_bits();
        }
        w.write_all(cast_slice(&out_f16))?;
    }
    Ok(())
}

fn dequant_q5k_to_f16<W: Write>(src: &[u8], numel: usize, w: &mut W) -> Result<()> {
    let qk = 256usize;
    if numel % qk != 0 {
        anyhow::bail!("Q5_K tensor numel {} is not divisible by {}", numel, qk);
    }
    let bs = 176usize;
    let nblk = numel / qk;
    if src.len() != nblk * bs {
        anyhow::bail!(
            "Q5_K source size mismatch: got {} bytes, expected {}",
            src.len(),
            nblk * bs
        );
    }
    let mut out_f16 = vec![0u16; qk];
    let mut vals = vec![0f32; qk];
    for bi in 0..nblk {
        let off = bi * bs;
        let b = &src[off..off + bs];
        dequantize_q5k_block(b, &mut vals)?;
        for i in 0..qk {
            out_f16[i] = f16::from_f32(vals[i]).to_bits();
        }
        w.write_all(cast_slice(&out_f16))?;
    }
    Ok(())
}

fn read_f16_le_as_f32(lo: u8, hi: u8) -> f32 {
    f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32()
}

fn dequantize_q2k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != 84 || out.len() != 256 {
        anyhow::bail!(
            "bad Q2_K block sizes: block={} out={}",
            block.len(),
            out.len()
        );
    }
    let d = read_f16_le_as_f32(block[0], block[1]);
    let dmin = read_f16_le_as_f32(block[2], block[3]);
    let scales = &block[4..20];
    let mut q = &block[20..84];
    let mut y = 0usize;
    let mut is = 0usize;
    for _ in (0..256usize).step_by(128) {
        let mut shift = 0usize;
        for _ in 0..4usize {
            let sc0 = scales[is];
            is += 1;
            let dl0 = d * ((sc0 & 0x0F) as f32);
            let ml0 = dmin * ((sc0 >> 4) as f32);
            for l in 0..16usize {
                let qv = ((q[l] >> shift) & 0x03) as f32;
                out[y] = dl0 * qv - ml0;
                y += 1;
            }

            let sc1 = scales[is];
            is += 1;
            let dl1 = d * ((sc1 & 0x0F) as f32);
            let ml1 = dmin * ((sc1 >> 4) as f32);
            for l in 0..16usize {
                let qv = ((q[l + 16] >> shift) & 0x03) as f32;
                out[y] = dl1 * qv - ml1;
                y += 1;
            }
            shift += 2;
        }
        q = &q[32..];
    }
    Ok(())
}

fn dequantize_q3k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != 110 || out.len() != 256 {
        anyhow::bail!(
            "bad Q3_K block sizes: block={} out={}",
            block.len(),
            out.len()
        );
    }
    let d_all = read_f16_le_as_f32(block[108], block[109]);
    let mut q = &block[32..96];
    let hm = &block[0..32];
    let scales_raw = &block[96..108];

    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([scales_raw[0], scales_raw[1], scales_raw[2], scales_raw[3]]);
    aux[1] = u32::from_le_bytes([scales_raw[4], scales_raw[5], scales_raw[6], scales_raw[7]]);
    aux[2] = u32::from_le_bytes([scales_raw[8], scales_raw[9], scales_raw[10], scales_raw[11]]);
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let mut scales = [0i8; 16];
    for i in 0..4usize {
        let b = aux[i].to_le_bytes();
        for j in 0..4usize {
            scales[i * 4 + j] = b[j] as i8;
        }
    }

    let mut m: u8 = 1;
    let mut is = 0usize;
    let mut y = 0usize;
    for _ in (0..256usize).step_by(128) {
        let mut shift = 0usize;
        for _ in 0..4usize {
            let dl0 = d_all * ((scales[is] as f32) - 32.0);
            is += 1;
            for l in 0..16usize {
                let qv = ((q[l] >> shift) & 0x03) as i8;
                let h = if (hm[l] & m) != 0 { 0 } else { 4 };
                out[y] = dl0 * ((qv - h) as f32);
                y += 1;
            }

            let dl1 = d_all * ((scales[is] as f32) - 32.0);
            is += 1;
            for l in 0..16usize {
                let qv = ((q[l + 16] >> shift) & 0x03) as i8;
                let h = if (hm[l + 16] & m) != 0 { 0 } else { 4 };
                out[y] = dl1 * ((qv - h) as f32);
                y += 1;
            }
            shift += 2;
            m = m.wrapping_shl(1);
        }
        q = &q[32..];
    }
    Ok(())
}

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

fn dequantize_q4k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != 144 || out.len() != 256 {
        anyhow::bail!(
            "bad Q4_K block sizes: block={} out={}",
            block.len(),
            out.len()
        );
    }
    let d = read_f16_le_as_f32(block[0], block[1]);
    let dmin = read_f16_le_as_f32(block[2], block[3]);
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut is = 0usize;
    let mut qoff = 0usize;
    for j in (0..256).step_by(64) {
        let (sc0, m0) = get_scale_min_k4(is, scales);
        let (sc1, m1) = get_scale_min_k4(is + 1, scales);
        let d1 = d * (sc0 as f32);
        let d2 = d * (sc1 as f32);
        let min1 = dmin * (m0 as f32);
        let min2 = dmin * (m1 as f32);
        for l in 0..32usize {
            let q = qs[qoff + l];
            let q0 = (q & 0x0F) as f32;
            let q1 = (q >> 4) as f32;
            out[j + l] = d1 * q0 - min1;
            out[j + 32 + l] = d2 * q1 - min2;
        }
        qoff += 32;
        is += 2;
    }
    Ok(())
}

fn dequantize_q6k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != 210 || out.len() != 256 {
        anyhow::bail!(
            "bad Q6_K block sizes: block={} out={}",
            block.len(),
            out.len()
        );
    }
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = read_f16_le_as_f32(block[208], block[209]);

    for n in (0..256).step_by(128) {
        let ql_off = n / 2;
        let qh_off = n / 4;
        let sc_off = n / 16;
        for l in 0..32usize {
            let is = l / 16;
            let ql_l = ql[ql_off + l];
            let ql_l32 = ql[ql_off + l + 32];
            let qh_l = qh[qh_off + l];

            let q0 = ((ql_l & 0x0F) | (((qh_l >> 0) & 0x03) << 4)) as i16 - 32;
            let q1 = ((ql_l32 & 0x0F) | (((qh_l >> 2) & 0x03) << 4)) as i16 - 32;
            let q2 = ((ql_l >> 4) | (((qh_l >> 4) & 0x03) << 4)) as i16 - 32;
            let q3 = ((ql_l32 >> 4) | (((qh_l >> 6) & 0x03) << 4)) as i16 - 32;

            let s0 = (sc[sc_off + is] as i8) as f32;
            let s1 = (sc[sc_off + is + 2] as i8) as f32;
            let s2 = (sc[sc_off + is + 4] as i8) as f32;
            let s3 = (sc[sc_off + is + 6] as i8) as f32;

            out[n + l] = d * s0 * (q0 as f32);
            out[n + l + 32] = d * s1 * (q1 as f32);
            out[n + l + 64] = d * s2 * (q2 as f32);
            out[n + l + 96] = d * s3 * (q3 as f32);
        }
    }
    Ok(())
}

fn dequantize_q5k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != 176 || out.len() != 256 {
        anyhow::bail!(
            "bad Q5_K block sizes: block={} out={}",
            block.len(),
            out.len()
        );
    }
    let d = read_f16_le_as_f32(block[0], block[1]);
    let dmin = read_f16_le_as_f32(block[2], block[3]);
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut is = 0usize;
    let mut qoff = 0usize;
    for j in (0..256usize).step_by(64) {
        let (sc0, m0) = get_scale_min_k4(is, scales);
        let (sc1, m1) = get_scale_min_k4(is + 1, scales);
        let d1 = d * (sc0 as f32);
        let d2 = d * (sc1 as f32);
        let min1 = dmin * (m0 as f32);
        let min2 = dmin * (m1 as f32);
        for l in 0..32usize {
            let q = qs[qoff + l];
            let idx0 = j + l;
            let idx1 = j + 32 + l;
            let h0 = ((qh[idx0 >> 3] >> (idx0 & 7)) & 1) as u8;
            let h1 = ((qh[idx1 >> 3] >> (idx1 & 7)) & 1) as u8;
            let q0 = ((q & 0x0F) | (h0 << 4)) as f32;
            let q1 = ((q >> 4) | (h1 << 4)) as f32;
            out[idx0] = d1 * q0 - min1;
            out[idx1] = d2 * q1 - min2;
        }
        qoff += 32;
        is += 2;
    }
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
try:
    obj = torch.load(inp, map_location='cpu', weights_only=False)
except TypeError:
    # Older torch versions without weights_only kwarg.
    obj = torch.load(inp, map_location='cpu')
if hasattr(obj, 'state_dict'):
    obj = obj.state_dict()
elif isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
    obj = obj['state_dict']
elif isinstance(obj, dict) and 'model_state_dict' in obj and isinstance(obj['model_state_dict'], dict):
    obj = obj['model_state_dict']
if not isinstance(obj, dict):
    raise RuntimeError(f'expected state_dict dict, got {type(obj)}')

clean = {}
for k, v in obj.items():
    if torch.is_tensor(v):
        t = v
        # Handle torchao quantized tensor wrappers that don't expose a valid storage ptr
        # until dequantized/materialized.
        if hasattr(t, "dequantize"):
            try:
                t = t.dequantize()
            except Exception:
                pass
        clean[str(k)] = t.detach().cpu().contiguous()

if not clean:
    raise RuntimeError('no tensors found in state_dict')
save_file(clean, out)
print(len(clean))
"#;

    let python_bin = std::env::var("CELLM_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let outp = Command::new(&python_bin)
        .arg("-c")
        .arg(script)
        .arg(input_file)
        .arg(&out)
        .output()
        .with_context(|| format!("failed to run {python_bin} for .bin/.pt -> .safetensors conversion"))?;
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

fn ensure_supported_safetensors_metadata(paths: &[PathBuf]) -> Result<Option<String>> {
    let mut detected_format: Option<String> = None;
    for path in paths {
        let format = read_safetensors_metadata_format(path)?;
        if let Some(fmt) = format.as_deref() {
            if detected_format.is_none() {
                detected_format = Some(fmt.to_string());
            }
        }
        if matches!(format.as_deref(), Some("mlx")) {
            anyhow::bail!(
                "unsupported source safetensors format=\"mlx\" in {:?}. \
This checkpoint was exported for MLX and is not layout-compatible with cellm conversion yet. \
Use the original non-MLX Hugging Face checkpoint (for example, Qwen/Qwen3.5-0.8B) and re-run convert.",
                path
            );
        }
    }
    Ok(detected_format)
}

fn read_safetensors_metadata_format(path: &Path) -> Result<Option<String>> {
    let mut f = File::open(path).with_context(|| format!("open {:?}", path))?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)
        .with_context(|| format!("read safetensors header length from {:?}", path))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)
        .with_context(|| format!("read safetensors header from {:?}", path))?;
    let header_json: Value =
        serde_json::from_slice(&header).with_context(|| format!("parse safetensors header {:?}", path))?;
    Ok(header_json
        .get("__metadata__")
        .and_then(|m| m.get("format"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string()))
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
    quantize_int4_symmetric: bool,
    gemma4_aggressive_int4: bool,
    gemma4_balanced_int4: bool,
    gemma4_awq_lite_int4: bool,
    gemma4_mixed_i8_i4: bool,
    gemma4_int4_exclude_layers: &std::collections::HashSet<usize>,
    gemma4_mixed_i8_attn_layers: &std::collections::HashSet<usize>,
    text_only: bool,
    model_type: &str,
    model_num_layers: usize,
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
            if text_only && !is_text_tensor_name(name) {
                continue;
            }
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
                && should_quantize_i8_weight(
                    model_type,
                    name,
                    &shape,
                    gemma4_mixed_i8_i4,
                    gemma4_int4_exclude_layers,
                    gemma4_mixed_i8_attn_layers,
                )
            {
                out_dtype = "i8".to_string();
                TensorOp::QuantizeI8Data
            } else if quantize_int4_symmetric
                && should_quantize_i4_weight(
                    model_type,
                    name,
                    &shape,
                    gemma4_aggressive_int4,
                    gemma4_balanced_int4,
                    gemma4_awq_lite_int4,
                    gemma4_mixed_i8_i4,
                    gemma4_int4_exclude_layers,
                    gemma4_mixed_i8_attn_layers,
                    model_num_layers,
                )
            {
                out_dtype = "i4".to_string();
                TensorOp::QuantizeI4Data
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
            let out_nbytes = if out_dtype == "i8" {
                numel
            } else if out_dtype == "i4" {
                i4_packed_nbytes_for_shape(&out_shape)?
            } else {
                numel * 2
            };

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
                shape: out_shape.clone(),
                header_shape: out_shape,
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
                    header_shape: vec![out_dim],
                    file_idx,
                    offset_bytes: 0,
                    out_nbytes: out_dim * 2,
                    out_dtype: "f16".to_string(),
                    op: TensorOp::QuantizeI8Scales {
                        weight_name: name.to_string(),
                    },
                });
            } else if out_dtype == "i4" {
                let out_dim = shape[0];
                plans.push(TensorPlan {
                    name: format!("{name}.qscale"),
                    shape: vec![out_dim],
                    header_shape: vec![out_dim],
                    file_idx,
                    offset_bytes: 0,
                    out_nbytes: out_dim * 2,
                    out_dtype: "f16".to_string(),
                    op: TensorOp::QuantizeI4Scales {
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

fn i4_packed_nbytes_for_shape(shape: &[usize]) -> Result<usize> {
    if shape.len() != 2 {
        anyhow::bail!("i4 packed byte size expects 2D shape, got {shape:?}");
    }
    let rows = shape[0];
    let cols = shape[1];
    rows.checked_mul(cols.div_ceil(2))
        .ok_or_else(|| anyhow::anyhow!("i4 packed byte size overflow for shape={shape:?}"))
}

fn is_quantization_excluded_name(name: &str) -> bool {
    name.contains("norm")
        || name.contains("per_layer_model_proj")
        || name.contains("lm_head")
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

fn is_text_tensor_name(name: &str) -> bool {
    name.starts_with("model.language_model.")
        || name.starts_with("language_model.model.")
        || name.starts_with("model.text_model.")
        || name.starts_with("text_model.")
        || name == "lm_head.weight"
        || name == "model.lm_head.weight"
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

fn should_quantize_i8_qwen_weight(name: &str, shape: &[usize]) -> bool {
    if shape.len() != 2 || !name.ends_with(".weight") {
        return false;
    }
    let in_qwen_layer = name.contains("language_model.model.layers.")
        || name.contains("model.layers.")
        || name.contains("model.text_model.layers.")
        || name.contains("model.language_model.layers.")
        || name == "model.embed_tokens.weight"
        || name == "lm_head.weight"
        || name == "model.lm_head.weight";
    if !in_qwen_layer {
        return false;
    }
    if name.contains("norm")
        || name.contains("conv1d")
    {
        return false;
    }
    // Qwen3.5 linear-attention projections are highly sensitive to quantization
    // in the current runner. Keep them in f16 for parity; quantize the rest.
    if name.contains(".linear_attn.") {
        return false;
    }
    true
}

fn should_quantize_i8_gemma_weight(name: &str, shape: &[usize]) -> bool {
    if shape.len() != 2 || !name.ends_with(".weight") {
        return false;
    }
    let in_gemma_layer = name.contains("model.layers.")
        || name.contains("model.text_model.layers.")
        || name.contains("model.language_model.layers.");
    if !in_gemma_layer {
        return false;
    }
    if name.contains("embed_tokens")
        || name.contains("lm_head")
        || name.contains("norm")
        || name.contains("q_norm")
        || name.contains("k_norm")
    {
        return false;
    }
    name.contains(".self_attn.") || name.contains(".mlp.")
}

fn should_quantize_i4_gemma_weight(name: &str, shape: &[usize]) -> bool {
    if shape.len() != 2 || !name.ends_with(".weight") {
        return false;
    }
    // Mixed strategy for Gemma: quantize only MLP linears, keep
    // attention projections + embeddings + lm_head in higher precision.
    if !name.contains(".mlp.") {
        return false;
    }
    name.starts_with("model.layers.")
        || name.starts_with("model.language_model.")
        || name.starts_with("model.text_model.")
}

fn should_quantize_i4_qwen_weight(name: &str, shape: &[usize]) -> bool {
    if shape.len() != 2 || !name.ends_with(".weight") {
        return false;
    }
    // Mixed strategy for Qwen: quantize MLP linears, keep attention projections
    // and embeddings in f16 for better generation quality at modest size.
    if name.contains(".mlp.") {
        return name.starts_with("model.layers.")
            || name.starts_with("model.language_model.")
            || name.starts_with("language_model.model.")
            || name.starts_with("model.text_model.");
    }
    false
}

fn should_quantize_i8_weight(
    model_type: &str,
    name: &str,
    shape: &[usize],
    gemma4_mixed_i8_i4: bool,
    gemma4_int4_exclude_layers: &std::collections::HashSet<usize>,
    gemma4_mixed_i8_attn_layers: &std::collections::HashSet<usize>,
) -> bool {
    if model_type == "llama" || model_type == "smollm3" {
        return should_quantize_i8_llama_weight(name, shape);
    }
    if model_type.starts_with("qwen") {
        return should_quantize_i8_qwen_weight(name, shape);
    }
    if model_type.starts_with("gemma") {
        if model_type.starts_with("gemma4") && gemma4_mixed_i8_i4 {
            if shape.len() != 2 || !name.ends_with(".weight") {
                return false;
            }
            if name.contains("norm")
                || name.contains("q_norm")
                || name.contains("k_norm")
                || name.contains("embed_tokens")
                || name.contains("lm_head")
                || name.contains("per_layer_token_embd")
                || name.contains("embed_tokens_per_layer")
                || name.contains("per_layer_model_proj")
                || name.contains("per_layer_proj_norm")
            {
                return false;
            }
            if !(name.starts_with("model.layers.")
                || name.starts_with("model.language_model.layers.")
                || name.starts_with("model.text_model.layers."))
            {
                return false;
            }
            if let Some(i) = extract_layer_index(name) {
                if gemma4_int4_exclude_layers.contains(&i) {
                    return false;
                }
                if !gemma4_mixed_i8_attn_layers.is_empty()
                    && !gemma4_mixed_i8_attn_layers.contains(&i)
                {
                    return false;
                }
            }
            return name.contains(".self_attn.");
        }
        return should_quantize_i8_gemma_weight(name, shape);
    }
    false
}

fn should_quantize_i4_weight(
    model_type: &str,
    name: &str,
    shape: &[usize],
    gemma4_aggressive_int4: bool,
    gemma4_balanced_int4: bool,
    gemma4_awq_lite_int4: bool,
    gemma4_mixed_i8_i4: bool,
    gemma4_int4_exclude_layers: &std::collections::HashSet<usize>,
    gemma4_mixed_i8_attn_layers: &std::collections::HashSet<usize>,
    model_num_layers: usize,
) -> bool {
    if model_type.starts_with("qwen") {
        return should_quantize_i4_qwen_weight(name, shape);
    }
    if model_type.starts_with("gemma") {
        if model_type.starts_with("gemma4") && gemma4_aggressive_int4 {
            if shape.len() != 2 || !name.ends_with(".weight") {
                return false;
            }
            // Keep normalization-like / special routing tensors in f16.
            if name.contains("norm")
                || name.contains("q_norm")
                || name.contains("k_norm")
                || name.contains("rope_freqs")
                || name.contains("layer_output_scale")
                || name.contains("per_layer_proj_norm")
            {
                return false;
            }
            // Restrict to text stack tensors only.
            let is_text_stack = name.starts_with("model.layers.")
                || name.starts_with("model.language_model.")
                || name.starts_with("model.text_model.")
                || name.contains("embed_tokens")
                || name.contains("per_layer_token_embd")
                || name.contains("per_layer_model_proj")
                || name.contains("lm_head");
            if let Some(i) = extract_layer_index(name) {
                if gemma4_int4_exclude_layers.contains(&i) {
                    return false;
                }
            }
            return is_text_stack;
        }
        if model_type.starts_with("gemma4") && gemma4_balanced_int4 {
            if shape.len() != 2 || !name.ends_with(".weight") {
                return false;
            }
            if name.contains("norm")
                || name.contains("q_norm")
                || name.contains("k_norm")
                || name.contains("rope_freqs")
                || name.contains("layer_output_scale")
                || name.contains("embed_tokens")
                || name.contains("lm_head")
                || name.contains("per_layer_token_embd")
                || name.contains("embed_tokens_per_layer")
                || name.contains("per_layer_model_proj")
                || name.contains("per_layer_proj_norm")
            {
                return false;
            }
            // Balanced profile: quantize layer projections only.
            let is_proj = name.starts_with("model.layers.")
                || name.starts_with("model.language_model.layers.")
                || name.starts_with("model.text_model.layers.");
            if !is_proj {
                return false;
            }
            if let Some(i) = extract_layer_index(name) {
                if gemma4_int4_exclude_layers.contains(&i) {
                    return false;
                }
            }
            return true;
        }
        if model_type.starts_with("gemma4") && gemma4_awq_lite_int4 {
            if shape.len() != 2 || !name.ends_with(".weight") {
                return false;
            }
            // Keep critical tensors in higher precision.
            if name.contains("norm")
                || name.contains("q_norm")
                || name.contains("k_norm")
                || name.contains("rope_freqs")
                || name.contains("layer_output_scale")
                || name.contains("embed_tokens")
                || name.contains("lm_head")
                || name.contains("per_layer_token_embd")
                || name.contains("embed_tokens_per_layer")
                || name.contains("per_layer_model_proj")
                || name.contains("per_layer_proj_norm")
            {
                return false;
            }
            // Restrict to decoder layer projections.
            if !(name.starts_with("model.layers.")
                || name.starts_with("model.language_model.layers.")
                || name.starts_with("model.text_model.layers."))
            {
                return false;
            }
            // AWQ-lite sensitivity heuristic: keep first/last 2 layers in f16.
            let idx = extract_layer_index(name);
            if let Some(i) = idx {
                if gemma4_int4_exclude_layers.contains(&i) {
                    return false;
                }
                if i < 2 || (model_num_layers >= 2 && i + 2 >= model_num_layers) {
                    return false;
                }
            }
            return true;
        }
        if model_type.starts_with("gemma4") && gemma4_mixed_i8_i4 {
            if shape.len() != 2 || !name.ends_with(".weight") {
                return false;
            }
            if name.contains("norm")
                || name.contains("q_norm")
                || name.contains("k_norm")
                || name.contains("embed_tokens")
                || name.contains("lm_head")
                || name.contains("per_layer_token_embd")
                || name.contains("embed_tokens_per_layer")
                || name.contains("per_layer_model_proj")
                || name.contains("per_layer_proj_norm")
            {
                return false;
            }
            if !(name.starts_with("model.layers.")
                || name.starts_with("model.language_model.layers.")
                || name.starts_with("model.text_model.layers."))
            {
                return false;
            }
            if let Some(i) = extract_layer_index(name) {
                if gemma4_int4_exclude_layers.contains(&i) {
                    return false;
                }
            }
            if name.contains(".mlp.") {
                return true;
            }
            if name.contains(".self_attn.") {
                if let Some(i) = extract_layer_index(name) {
                    return !gemma4_mixed_i8_attn_layers.contains(&i);
                }
                return true;
            }
            return false;
        }
        return should_quantize_i4_gemma_weight(name, shape);
    }
    false
}

fn extract_layer_index(name: &str) -> Option<usize> {
    for pfx in ["model.layers.", "model.language_model.layers.", "model.text_model.layers."] {
        if let Some(rest) = name.strip_prefix(pfx) {
            let idx_str = rest.split('.').next()?;
            return idx_str.parse::<usize>().ok();
        }
    }
    None
}

fn parse_layer_index_list(s: Option<&str>) -> Result<std::collections::HashSet<usize>> {
    let mut out = std::collections::HashSet::new();
    let Some(raw) = s else {
        return Ok(out);
    };
    for part in raw.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let v: usize = p
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid layer index in --gemma4-int4-exclude-layers: {p}"))?;
        out.insert(v);
    }
    Ok(out)
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

fn write_quant_i4_data_per_row_symmetric(
    data: &[u8],
    shape: &[usize],
    dtype: Dtype,
    w: &mut impl Write,
) -> Result<()> {
    if shape.len() != 2 {
        anyhow::bail!("int4 quantization expects 2D weight, got shape={shape:?}");
    }
    let out_dim = shape[0];
    let in_dim = shape[1];
    let values = tensor_data_to_f32(data, dtype)?;
    if values.len() != out_dim * in_dim {
        anyhow::bail!(
            "int4 quantization data length mismatch: {} vs {}",
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
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        for i in (0..in_dim).step_by(2) {
            let q0 = (row[i] / scale).round().clamp(-7.0, 7.0) as i8;
            let q1 = if i + 1 < in_dim {
                (row[i + 1] / scale).round().clamp(-7.0, 7.0) as i8
            } else {
                0
            };
            let n0 = (q0 + 8) as u8 & 0x0f;
            let n1 = (q1 + 8) as u8 & 0x0f;
            w.write_all(&[n0 | (n1 << 4)])?;
        }
    }
    Ok(())
}

fn write_quant_i4_scales_per_row_symmetric(
    data: &[u8],
    shape: &[usize],
    dtype: Dtype,
    w: &mut impl Write,
) -> Result<()> {
    if shape.len() != 2 {
        anyhow::bail!("int4 quantization expects 2D weight, got shape={shape:?}");
    }
    let out_dim = shape[0];
    let in_dim = shape[1];
    let values = tensor_data_to_f32(data, dtype)?;
    if values.len() != out_dim * in_dim {
        anyhow::bail!(
            "int4 quantization data length mismatch: {} vs {}",
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
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
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
    let parse_token_id = |v: Option<&Value>| -> Option<u32> {
        let val = v?;
        match val {
            Value::Number(n) => n.as_u64().and_then(|x| u32::try_from(x).ok()),
            Value::Array(arr) => arr
                .iter()
                .find_map(|x| x.as_u64().and_then(|n| u32::try_from(n).ok())),
            _ => None,
        }
    };

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

    let text_model_type = cfg
        .text_config
        .as_ref()
        .and_then(|t| t.model_type.as_deref());
    let top_model_type = cfg.model_type.as_deref();
    let is_gemma4_text = text_model_type
        .or(top_model_type)
        .map(|s| s.starts_with("gemma4"))
        .unwrap_or(false);

    let rope_theta = if is_gemma4_text {
        cfg.rope_theta
            .or_else(|| nested.and_then(|t| t.rope_theta))
            .or_else(|| {
                nested.and_then(|t| {
                    t.rope_parameters
                        .as_ref()
                        .and_then(|rp| rp.full_attention.as_ref())
                        .and_then(|v| v.rope_theta)
                })
            })
            .or_else(|| {
                nested.and_then(|t| {
                    t.rope_parameters
                        .as_ref()
                        .and_then(|rp| rp.rope_theta)
                })
            })
            .or_else(|| {
                nested.and_then(|t| {
                    t.rope_parameters
                        .as_ref()
                        .and_then(|rp| rp.sliding_attention.as_ref())
                        .and_then(|v| v.rope_theta)
                })
            })
            .unwrap_or(10000.0)
    } else {
        cfg.rope_theta
            .or_else(|| nested.and_then(|t| t.rope_theta))
            .or_else(|| nested.and_then(|t| t.rope_parameters.as_ref().and_then(|rp| rp.rope_theta)))
            .unwrap_or(10000.0)
    };

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
        bos_token_id: parse_token_id(cfg.bos_token_id.as_ref())
            .or_else(|| nested.and_then(|t| t.bos_token_id)),
        eos_token_id: parse_token_id(cfg.eos_token_id.as_ref())
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
