// Author: Jeffrey Asante (https://jeffasante.github.io/)
use std::fs::File;
use std::io::{BufWriter, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use cellm_model::{CellmFile, CellmTensorIndex};
use half::f16;
use bytemuck::cast_slice;

#[derive(ValueEnum, Clone, Copy, Debug)]
enum QuantMode {
    I4,
    I2,
    I8,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value = "i4")]
    mode: QuantMode,
    #[arg(long, default_value = "128")]
    group_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let f = CellmFile::load(&args.input)?;
    println!("Loaded model: {:?}", args.input);

    let mut new_header = f.header.clone();
    new_header.tensors.clear();

    // We'll write to a temporary file first for the data, then assemble.
    let temp_path = args.output.with_extension("tmp_data");
    let mut data_writer = BufWriter::new(File::create(&temp_path)?);
    let mut current_offset = 0u64;
    let mut new_tensors = Vec::new();

    for t_idx in &f.header.tensors {
        let name = &t_idx.name;
        let data = f.tensor_bytes(name)?;
        let shape = &t_idx.shape;
        
        let should_quant = shape.len() == 2 && name.ends_with(".weight") && !name.contains("norm") && !name.contains("embed") && !name.contains("lm_head");

        if should_quant {
            println!("Quantizing {} (shape={:?}) to {:?}", name, shape, args.mode);
            let vals = cast_slice::<u8, u16>(data);
            let f32_vals: Vec<f32> = vals.iter().map(|&v| f16::from_bits(v).to_f32()).collect();
            
            let out_dim = shape[0];
            let in_dim = shape[1];

            let start_off = current_offset;
            match args.mode {
                QuantMode::I4 => {
                    let gs = args.group_size;
                    for r in 0..out_dim {
                        let row = &f32_vals[r * in_dim..(r + 1) * in_dim];
                        for i in (0..in_dim).step_by(2) {
                            let group_idx = i / gs;
                            let group_start = group_idx * gs;
                            let group_end = (group_start + gs).min(in_dim);
                            let group = &row[group_start..group_end];
                            
                            let mut max_abs = 0.0f32;
                            for &v in group { if v.abs() > max_abs { max_abs = v.abs(); } }
                            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };

                            let q0 = (row[i] / scale).round().clamp(-7.0, 7.0) as i8;
                            let q1 = if i + 1 < in_dim { (row[i+1] / scale).round().clamp(-7.0, 7.0) as i8 } else { 0 };
                            let n0 = (q0 + 8) as u8 & 0x0f;
                            let n1 = (q1 + 8) as u8 & 0x0f;
                            data_writer.write_all(&[n0 | (n1 << 4)])?;
                        }
                    }
                    let nbytes = (out_dim * (in_dim.div_ceil(2))) as u64;
                    new_tensors.push(CellmTensorIndex {
                        name: name.clone(),
                        offset_bytes: start_off,
                        nbytes,
                        shape: shape.clone(),
                        dtype: "i4".into(),
                    });
                    current_offset += nbytes;

                    // Write scales
                    let scale_off = current_offset;
                    for r in 0..out_dim {
                        let row = &f32_vals[r * in_dim..(r + 1) * in_dim];
                        for g in 0..in_dim.div_ceil(gs) {
                            let group_start = g * gs;
                            let group_end = (group_start + gs).min(in_dim);
                            let group = &row[group_start..group_end];
                            let mut max_abs = 0.0f32;
                            for &v in group { if v.abs() > max_abs { max_abs = v.abs(); } }
                            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
                            data_writer.write_all(&f16::from_f32(scale).to_bits().to_le_bytes())?;
                        }
                    }
                    let n_groups_per_row = in_dim.div_ceil(gs);
                    let scale_nbytes = (out_dim * n_groups_per_row * 2) as u64;
                    new_tensors.push(CellmTensorIndex {
                        name: format!("{}.qscale", name),
                        offset_bytes: scale_off,
                        nbytes: scale_nbytes,
                        shape: vec![out_dim, n_groups_per_row],
                        dtype: "f16".into(),
                    });
                    current_offset += scale_nbytes;
                }
                QuantMode::I2 => {
                    for r in 0..out_dim {
                        let row = &f32_vals[r * in_dim..(r + 1) * in_dim];
                        let mut max_abs = 0.0f32;
                        for &v in row { if v.abs() > max_abs { max_abs = v.abs(); } }
                        let scale = if max_abs > 0.0 { max_abs / 1.5 } else { 1.0 };
                        for i in (0..in_dim).step_by(4) {
                            let mut packed = 0u8;
                            for lane in 0..4 {
                                let idx = i + lane;
                                let q = if idx < in_dim {
                                    let v = row[idx] / scale;
                                    if v < -1.0 { 0 } else if v < 0.0 { 1 } else if v < 1.0 { 2 } else { 3 }
                                } else { 1 };
                                packed |= (q & 0x03) << (lane * 2);
                            }
                            data_writer.write_all(&[packed])?;
                        }
                    }
                    let nbytes = (out_dim * (in_dim.div_ceil(4))) as u64;
                    new_tensors.push(CellmTensorIndex {
                        name: name.clone(),
                        offset_bytes: start_off,
                        nbytes,
                        shape: shape.clone(),
                        dtype: "i2".into(),
                    });
                    current_offset += nbytes;

                    // Write scales
                    let scale_off = current_offset;
                    for r in 0..out_dim {
                        let row = &f32_vals[r * in_dim..(r + 1) * in_dim];
                        let mut max_abs = 0.0f32;
                        for &v in row { if v.abs() > max_abs { max_abs = v.abs(); } }
                        let scale = if max_abs > 0.0 { max_abs / 1.5 } else { 1.0 };
                        data_writer.write_all(&f16::from_f32(scale).to_bits().to_le_bytes())?;
                    }
                    let scale_nbytes = (out_dim * 2) as u64;
                    new_tensors.push(CellmTensorIndex {
                        name: format!("{}.qscale", name),
                        offset_bytes: scale_off,
                        nbytes: scale_nbytes,
                        shape: vec![out_dim],
                        dtype: "f16".into(),
                    });
                    current_offset += scale_nbytes;
                }
                QuantMode::I8 => {
                    for r in 0..out_dim {
                        let row = &f32_vals[r * in_dim..(r + 1) * in_dim];
                        let mut max_abs = 0.0f32;
                        for &v in row { if v.abs() > max_abs { max_abs = v.abs(); } }
                        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                        for i in 0..in_dim {
                            let q = (row[i] / scale).round().clamp(-127.0, 127.0) as i8;
                            data_writer.write_all(&[q as u8])?;
                        }
                    }
                    let nbytes = (out_dim * in_dim) as u64;
                    new_tensors.push(CellmTensorIndex {
                        name: name.clone(),
                        offset_bytes: start_off,
                        nbytes,
                        shape: shape.clone(),
                        dtype: "i8".into(),
                    });
                    current_offset += nbytes;

                    // Write scales
                    let scale_off = current_offset;
                    for r in 0..out_dim {
                        let row = &f32_vals[r * in_dim..(r + 1) * in_dim];
                        let mut max_abs = 0.0f32;
                        for &v in row { if v.abs() > max_abs { max_abs = v.abs(); } }
                        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                        data_writer.write_all(&f16::from_f32(scale).to_bits().to_le_bytes())?;
                    }
                    let scale_nbytes = (out_dim * 2) as u64;
                    new_tensors.push(CellmTensorIndex {
                        name: format!("{}.qscale", name),
                        offset_bytes: scale_off,
                        nbytes: scale_nbytes,
                        shape: vec![out_dim],
                        dtype: "f16".into(),
                    });
                    current_offset += scale_nbytes;
                }
            }
        } else {
            let start_off = current_offset;
            data_writer.write_all(data)?;
            new_tensors.push(CellmTensorIndex {
                name: name.clone(),
                offset_bytes: start_off,
                nbytes: data.len() as u64,
                shape: shape.clone(),
                dtype: t_idx.dtype.clone(),
            });
            current_offset += data.len() as u64;
        }

        let pad = (64 - (current_offset % 64)) % 64;
        if pad > 0 {
            data_writer.write_all(&vec![0u8; pad as usize])?;
            current_offset += pad;
        }
    }

    data_writer.flush()?;
    drop(data_writer);

    // Final Assembly
    new_header.tensors = new_tensors;
    let mut header_bytes = serde_json::to_vec(&new_header)?;
    let mut header_len = header_bytes.len();
    
    // Iterate until header length is stable (offsets change JSON size)
    for _ in 0..10 {
        let header_end = 10 + header_len as u64;
        let header_pad = (64 - (header_end % 64)) % 64;
        let data_start = header_end + header_pad;

        let mut temp_tensors = new_header.tensors.clone();
        for t in &mut temp_tensors {
            t.offset_bytes += data_start;
        }
        let mut temp_header = new_header.clone();
        temp_header.tensors = temp_tensors;
        let new_bytes = serde_json::to_vec(&temp_header)?;
        if new_bytes.len() == header_len {
            header_bytes = new_bytes;
            break;
        }
        header_len = new_bytes.len();
    }

    let header_end = 10 + header_bytes.len() as u64;
    let header_pad = (64 - (header_end % 64)) % 64;

    let mut final_file = File::create(&args.output)?;
    final_file.write_all(b"CELLM\x01")?;
    final_file.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
    final_file.write_all(&header_bytes)?;
    if header_pad > 0 {
        final_file.write_all(&vec![0u8; header_pad as usize])?;
    }
    
    let mut temp_data = File::open(&temp_path)?;
    std::io::copy(&mut temp_data, &mut final_file)?;
    
    std::fs::remove_file(temp_path)?;

    println!("Quantized model saved to {:?}", args.output);
    Ok(())
}
