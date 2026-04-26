// Author: Jeffrey Asante (https://jeffasante.github.io/)
use cellm_core::{kv_cache::DeviceKvStorage, CoreError, KvCacheLayout, KvCacheReadView, KvCacheView};
use half::f16;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions,
};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::rc::autoreleasepool;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::sync::Mutex;

// Compiled Metal library cache — compiled once per process lifetime.
#[cfg(any(target_os = "macos", target_os = "ios"))]
static KV_CACHE_LIB_CACHE: Mutex<Option<Library>> = Mutex::new(None);

use crate::BlockAllocator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvStorageKind {
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvEncodingKind {
    /// Current default: store K/V as f16.
    F16,
    /// Experimental TurboQuant mode.
    ///
    /// Note: current implementation keeps storage in f16 while plumbing the config;
    /// quantized backing is wired in a follow-up patch.
    TurboQuant,
}

/// Physical paged KV cache storage.
///
/// Owns:
/// - a `BlockAllocator` (block id free-list)
/// - device-specific K and V buffers
#[derive(Debug)]
pub struct CpuKvStorage {
    k: Vec<f16>,
    v: Vec<f16>,
}

#[derive(Debug)]
pub struct CpuTurboQuantKvStorage {
    k_q: Vec<u8>,
    v_q: Vec<u8>,
    k_scale: Vec<f32>,
    v_scale: Vec<f32>,
    k_res_sign: Vec<u8>,
    v_res_sign: Vec<u8>,
    k_res_scale: Vec<f32>,
    v_res_scale: Vec<f32>,
    row_bytes: usize,
    kv_dim: usize,
    n_kv_heads: usize,
    head_dim: usize,
    tq: TurboQuantParams,
}

#[derive(Debug, Clone)]
struct TurboQuantParams {
    rot_sign: Vec<f32>,
    qjl_sign: Vec<f32>,
    outlier_mask: Vec<bool>,
    bit_offsets: Vec<u32>,
    row_bits: u32,
    row_bytes: usize,
    inv_sqrt_head_dim: f32,
}

impl TurboQuantParams {
    const BASE_BITS: u8 = 8;
    const OUTLIER_BITS: u8 = 8;

    fn new(kv_dim: usize, n_kv_heads: usize, head_dim: usize, seed: u64) -> Self {
        let mut s1 = seed;
        let mut s2 = seed ^ 0x9E37_79B9_7F4A_7C15u64;
        let mut rot_sign = Vec::with_capacity(kv_dim);
        let mut qjl_sign = Vec::with_capacity(kv_dim);
        let mut outlier_mask = vec![false; kv_dim];
        // Outlier mask is selected per head to avoid mixing cross-head statistics.
        let outliers_per_head = (head_dim / 4).max(1);
        for h in 0..n_kv_heads {
            for i in 0..head_dim {
                let g = h * head_dim + i;
                let r1 = splitmix64_next(&mut s1);
                let r2 = splitmix64_next(&mut s2);
                rot_sign.push(if (r1 & 1) == 0 { 1.0 } else { -1.0 });
                qjl_sign.push(if (r2 & 1) == 0 { 1.0 } else { -1.0 });
                if i < outliers_per_head {
                    outlier_mask[g] = true;
                }
            }
        }
        let mut bit_offsets = vec![0u32; kv_dim];
        let mut running = 0u32;
        for i in 0..kv_dim {
            bit_offsets[i] = running;
            running += if outlier_mask[i] {
                Self::OUTLIER_BITS as u32
            } else {
                Self::BASE_BITS as u32
            };
        }
        let row_bits = running;
        let row_bytes = row_bits.div_ceil(8) as usize;
        Self {
            rot_sign,
            qjl_sign,
            outlier_mask,
            bit_offsets,
            row_bits,
            row_bytes,
            inv_sqrt_head_dim: 1.0f32 / (head_dim.max(1) as f32).sqrt(),
        }
    }

    #[inline]
    fn bits_for_dim(&self, dim: usize) -> u8 {
        if self.outlier_mask.get(dim).copied().unwrap_or(false) {
            Self::OUTLIER_BITS
        } else {
            Self::BASE_BITS
        }
    }

    #[inline]
    fn codebook(bits: u8) -> &'static [f32] {
        // Lloyd-Max-like fixed centroids for near-Gaussian coordinates.
        const CB3: [f32; 8] = [-1.874, -1.204, -0.675, -0.225, 0.225, 0.675, 1.204, 1.874];
        const CB4: [f32; 16] = [
            -2.300, -1.710, -1.290, -0.970, -0.700, -0.460, -0.240, -0.070,
            0.070, 0.240, 0.460, 0.700, 0.970, 1.290, 1.710, 2.300,
        ];
        match bits {
            4 => &CB4,
            _ => &CB3,
        }
    }

    #[inline]
    fn nearest_codebook_index(bits: u8, value: f32) -> u8 {
        if bits > 4 {
            let levels = 1u32 << bits.min(8);
            let min_v = -3.0f32;
            let max_v = 3.0f32;
            let step = (max_v - min_v) / (levels.saturating_sub(1) as f32);
            let idx = ((value.clamp(min_v, max_v) - min_v) / step).round() as i32;
            return idx.clamp(0, (levels as i32) - 1) as u8;
        }
        let cb = Self::codebook(bits);
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for (i, &c) in cb.iter().enumerate() {
            let d = (value - c).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx as u8
    }

    #[inline]
    fn dequant_centroid(bits: u8, idx: u8) -> f32 {
        if bits > 4 {
            let levels = 1u32 << bits.min(8);
            let min_v = -3.0f32;
            let max_v = 3.0f32;
            let step = (max_v - min_v) / (levels.saturating_sub(1) as f32);
            let i = (idx as u32).min(levels.saturating_sub(1));
            return min_v + (i as f32) * step;
        }
        let cb = Self::codebook(bits);
        cb[(idx as usize).min(cb.len().saturating_sub(1))]
    }

    #[inline]
    fn get_q_idx(&self, row: &[u8], dim: usize) -> u8 {
        let bits = self.bits_for_dim(dim) as u32;
        let mut remain = bits;
        let mut bit_pos = self.bit_offsets[dim] as usize;
        let mut out = 0u32;
        let mut out_shift = 0u32;
        while remain > 0 {
            let byte_idx = bit_pos / 8;
            let intra = (bit_pos % 8) as u32;
            let take = remain.min(8 - intra);
            let mask = ((1u32 << take) - 1) << intra;
            let chunk = ((row[byte_idx] as u32) & mask) >> intra;
            out |= chunk << out_shift;
            out_shift += take;
            remain -= take;
            bit_pos += take as usize;
        }
        out as u8
    }

    #[inline]
    fn set_q_idx(&self, row: &mut [u8], dim: usize, val: u8) {
        let bits = self.bits_for_dim(dim) as u32;
        let mut remain = bits;
        let mut bit_pos = self.bit_offsets[dim] as usize;
        let mut in_shift = 0u32;
        while remain > 0 {
            let byte_idx = bit_pos / 8;
            let intra = (bit_pos % 8) as u32;
            let take = remain.min(8 - intra);
            let mask = ((1u32 << take) - 1) << intra;
            let chunk = ((val as u32) >> in_shift) & ((1u32 << take) - 1);
            let cur = row[byte_idx] as u32;
            row[byte_idx] = ((cur & !mask) | (chunk << intra)) as u8;
            in_shift += take;
            remain -= take;
            bit_pos += take as usize;
        }
    }

    #[inline]
    fn is_uniform_8bit(&self, kv_dim: usize) -> bool {
        self.row_bits as usize == kv_dim.saturating_mul(8)
    }
}

#[inline]
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15u64);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9u64);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EBu64);
    z ^ (z >> 31)
}

#[inline]
fn fwht_inplace(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1usize;
    while h < n {
        let stride = h * 2;
        let mut i = 0usize;
        while i < n {
            for j in 0..h {
                let a = x[i + j];
                let b = x[i + j + h];
                x[i + j] = a + b;
                x[i + j + h] = a - b;
            }
            i += stride;
        }
        h *= 2;
    }
}

#[inline]
fn rotate_head_forward(
    src: &[f32],
    rot_sign: &[f32],
    out: &mut [f32],
    inv_sqrt_head_dim: f32,
) {
    debug_assert_eq!(src.len(), rot_sign.len());
    debug_assert_eq!(src.len(), out.len());
    for i in 0..src.len() {
        out[i] = src[i] * rot_sign[i];
    }
    if src.len().is_power_of_two() {
        fwht_inplace(out);
        for v in out.iter_mut() {
            *v *= inv_sqrt_head_dim;
        }
    }
}

#[inline]
fn rotate_head_inverse_inplace(x: &mut [f32], rot_sign: &[f32], inv_sqrt_head_dim: f32) {
    debug_assert_eq!(x.len(), rot_sign.len());
    if x.len().is_power_of_two() {
        fwht_inplace(x);
        for v in x.iter_mut() {
            *v *= inv_sqrt_head_dim;
        }
    }
    for i in 0..x.len() {
        x[i] *= rot_sign[i];
    }
}

#[inline]
fn turboquant_compress_row_f32(
    src: &[f32],
    dst_q_packed: &mut [u8],
    dst_res_sign: &mut [u8],
    scale_out: &mut f32,
    res_scale_out: &mut f32,
    n_kv_heads: usize,
    head_dim: usize,
    tq: &TurboQuantParams,
) {
    debug_assert_eq!(src.len(), n_kv_heads * head_dim);
    debug_assert_eq!(src.len(), dst_res_sign.len());
    debug_assert_eq!(dst_q_packed.len(), tq.row_bytes);
    let d = src.len();
    if d == 0 {
        *scale_out = 1.0;
        *res_scale_out = 0.0;
        return;
    }
    dst_q_packed.fill(0);
    let mut rotated = vec![0.0f32; d];
    for h in 0..n_kv_heads {
        let off = h * head_dim;
        rotate_head_forward(
            &src[off..off + head_dim],
            &tq.rot_sign[off..off + head_dim],
            &mut rotated[off..off + head_dim],
            tq.inv_sqrt_head_dim,
        );
    }
    let mut norm2 = 0.0f32;
    for &xr in rotated.iter() {
        norm2 += xr * xr;
    }
    let scale = norm2.sqrt().max(1e-12);
    *scale_out = scale;
    let inv = 1.0f32 / scale;
    let mut res2 = 0.0f32;
    for (dim, &xr) in rotated.iter().enumerate() {
        let xr_unit = xr * inv;
        let bits = tq.bits_for_dim(dim);
        let q_idx = TurboQuantParams::nearest_codebook_index(bits, xr_unit);
        let deq = TurboQuantParams::dequant_centroid(bits, q_idx);
        tq.set_q_idx(dst_q_packed, dim, q_idx);
        let residual = xr_unit - deq;
        let sketch = tq.qjl_sign[dim] * residual;
        dst_res_sign[dim] = if sketch >= 0.0 { 1 } else { 0 };
        res2 += residual * residual;
    }
    *res_scale_out = (res2 / d as f32).sqrt();
}

#[inline]
fn turboquant_compress_row_f16(
    src: &[f16],
    dst_q_packed: &mut [u8],
    dst_res_sign: &mut [u8],
    scale_out: &mut f32,
    res_scale_out: &mut f32,
    n_kv_heads: usize,
    head_dim: usize,
    tq: &TurboQuantParams,
) {
    debug_assert_eq!(src.len(), n_kv_heads * head_dim);
    debug_assert_eq!(src.len(), dst_res_sign.len());
    debug_assert_eq!(dst_q_packed.len(), tq.row_bytes);
    let d = src.len();
    if d == 0 {
        *scale_out = 1.0;
        *res_scale_out = 0.0;
        return;
    }
    dst_q_packed.fill(0);
    let mut src_f32 = vec![0.0f32; d];
    for i in 0..d {
        src_f32[i] = src[i].to_f32();
    }
    let mut rotated = vec![0.0f32; d];
    for h in 0..n_kv_heads {
        let off = h * head_dim;
        rotate_head_forward(
            &src_f32[off..off + head_dim],
            &tq.rot_sign[off..off + head_dim],
            &mut rotated[off..off + head_dim],
            tq.inv_sqrt_head_dim,
        );
    }
    let mut norm2 = 0.0f32;
    for &xr in rotated.iter() {
        norm2 += xr * xr;
    }
    let scale = norm2.sqrt().max(1e-12);
    *scale_out = scale;
    let inv = 1.0f32 / scale;
    let mut res2 = 0.0f32;
    for (dim, &xr) in rotated.iter().enumerate() {
        let xr_unit = xr * inv;
        let bits = tq.bits_for_dim(dim);
        let q_idx = TurboQuantParams::nearest_codebook_index(bits, xr_unit);
        let deq = TurboQuantParams::dequant_centroid(bits, q_idx);
        tq.set_q_idx(dst_q_packed, dim, q_idx);
        let residual = xr_unit - deq;
        let sketch = tq.qjl_sign[dim] * residual;
        dst_res_sign[dim] = if sketch >= 0.0 { 1 } else { 0 };
        res2 += residual * residual;
    }
    *res_scale_out = (res2 / d as f32).sqrt();
}

#[inline]
fn turboquant_decode_rotated_value(row_packed: &[u8], scale: f32, dim: usize, tq: &TurboQuantParams) -> f32 {
    debug_assert_eq!(row_packed.len(), tq.row_bytes);
    debug_assert!((tq.bit_offsets[dim] + tq.bits_for_dim(dim) as u32) <= tq.row_bits);
    let q_idx = tq.get_q_idx(row_packed, dim);
    let bits = tq.bits_for_dim(dim);
    TurboQuantParams::dequant_centroid(bits, q_idx) * scale
}

#[inline]
fn turboquant_decode_head_to_original(
    row_packed: &[u8],
    scale: f32,
    head: usize,
    head_dim: usize,
    tq: &TurboQuantParams,
    out: &mut [f32],
) {
    let off = head * head_dim;
    for i in 0..head_dim {
        out[i] = turboquant_decode_rotated_value(row_packed, scale, off + i, tq);
    }
    rotate_head_inverse_inplace(out, &tq.rot_sign[off..off + head_dim], tq.inv_sqrt_head_dim);
}

#[inline]
fn quantize_symmetric_i8(src: &[f32], dst: &mut [i8]) -> f32 {
    debug_assert_eq!(src.len(), dst.len());
    let mut max_abs = 0.0f32;
    for &v in src {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let scale = (max_abs / 127.0).max(1e-8);
    let inv = 1.0f32 / scale;
    for i in 0..src.len() {
        dst[i] = (src[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

#[inline]
fn turboquant_dot_rotated_head_int8_uniform8(
    row_packed: &[u8],
    kv_h: usize,
    head_dim: usize,
    q_i8: &[i8],
    q_scale: f32,
    k_scale: f32,
) -> f32 {
    // For uniform-8 packing, centroid(idx) = -3 + idx * (6/255).
    // Compute in integer domain with exact affine reconstruction to minimize bias.
    let off = kv_h.saturating_mul(head_dim);
    let mut sum_qidx = 0i32;
    let mut sum_q = 0i32;
    for i in 0..head_dim {
        let qi = q_i8[i] as i32;
        let idx = row_packed[off + i] as i32;
        sum_qidx += qi * idx;
        sum_q += qi;
    }
    let min_v = -3.0f32;
    let k_step = 6.0f32 / 255.0f32;
    let acc = (sum_qidx as f32) * k_step + (sum_q as f32) * min_v;
    acc * q_scale * k_scale
}

impl CpuTurboQuantKvStorage {
    fn new(total_elems: usize, n_kv_heads: usize, head_dim: usize) -> Result<Self, CoreError> {
        let kv_dim = n_kv_heads.saturating_mul(head_dim);
        if kv_dim == 0 {
            return Err(CoreError::Backend(
                "kv cache cpu turboquant: kv_dim must be > 0".into(),
            ));
        }
        if total_elems % kv_dim != 0 {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: total_elems {} not divisible by kv_dim {}",
                total_elems, kv_dim
            )));
        }
        let tokens = total_elems / kv_dim;
        let tq = TurboQuantParams::new(kv_dim, n_kv_heads, head_dim, 0xCE11_6D4Bu64);
        let row_bytes = tq.row_bytes;
        Ok(Self {
            k_q: vec![0u8; tokens.saturating_mul(row_bytes)],
            v_q: vec![0u8; tokens.saturating_mul(row_bytes)],
            k_scale: vec![1.0f32; tokens],
            v_scale: vec![1.0f32; tokens],
            k_res_sign: vec![1u8; total_elems],
            v_res_sign: vec![1u8; total_elems],
            k_res_scale: vec![0.0f32; tokens],
            v_res_scale: vec![0.0f32; tokens],
            row_bytes,
            kv_dim,
            n_kv_heads,
            head_dim,
            tq,
        })
    }

    #[inline]
    fn token_index(&self, base: usize, len: usize) -> Result<usize, CoreError> {
        if len != self.kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: token len mismatch {} expected {}",
                len, self.kv_dim
            )));
        }
        if base % self.kv_dim != 0 {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: base {} is not token-aligned (kv_dim={})",
                base, self.kv_dim
            )));
        }
        let tok = base / self.kv_dim;
        if tok >= self.k_scale.len() || tok >= self.v_scale.len() {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: token index {} out of range {}",
                tok,
                self.k_scale.len()
            )));
        }
        Ok(tok)
    }

    #[inline]
    fn row_slice_mut<'a>(buf: &'a mut [u8], tok: usize, row_bytes: usize) -> Result<&'a mut [u8], CoreError> {
        let off = tok.saturating_mul(row_bytes);
        let end = off.saturating_add(row_bytes);
        if end > buf.len() {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: row out of bounds tok={} row_bytes={} cap={}",
                tok, row_bytes, buf.len()
            )));
        }
        Ok(&mut buf[off..end])
    }

    #[inline]
    fn row_slice<'a>(buf: &'a [u8], tok: usize, row_bytes: usize) -> Result<&'a [u8], CoreError> {
        let off = tok.saturating_mul(row_bytes);
        let end = off.saturating_add(row_bytes);
        if end > buf.len() {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: row out of bounds tok={} row_bytes={} cap={}",
                tok, row_bytes, buf.len()
            )));
        }
        Ok(&buf[off..end])
    }

}

impl DeviceKvStorage for CpuKvStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn write_token_f16(
        &mut self,
        base: usize,
        k_src: &[f16],
        v_src: &[f16],
    ) -> Result<(), CoreError> {
        let kv_dim = k_src.len();
        if v_src.len() != kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: expected matching k/v len, got {}/{}",
                kv_dim,
                v_src.len()
            )));
        }
        let end = base.saturating_add(kv_dim);
        if end > self.k.len() || end > self.v.len() {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: write out of bounds base={base} len={kv_dim} cap={}",
                self.k.len()
            )));
        }
        self.k[base..end].copy_from_slice(k_src);
        self.v[base..end].copy_from_slice(v_src);
        Ok(())
    }

    fn write_token_f32(
        &mut self,
        base: usize,
        k_src: &[f32],
        v_src: &[f32],
    ) -> Result<(), CoreError> {
        let kv_dim = k_src.len();
        if v_src.len() != kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: expected matching k/v len, got {}/{}",
                kv_dim,
                v_src.len()
            )));
        }
        let end = base.saturating_add(kv_dim);
        if end > self.k.len() || end > self.v.len() {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: write out of bounds base={base} len={kv_dim} cap={}",
                self.k.len()
            )));
        }
        for i in 0..kv_dim {
            self.k[base + i] = f16::from_f32(k_src[i]);
            self.v[base + i] = f16::from_f32(v_src[i]);
        }
        Ok(())
    }

    fn read_token_f16(
        &self,
        base: usize,
        k_out: &mut [f16],
        v_out: &mut [f16],
    ) -> Result<(), CoreError> {
        let kv_dim = k_out.len();
        if v_out.len() != kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: expected matching k/v len, got {}/{}",
                kv_dim,
                v_out.len()
            )));
        }
        let end = base.saturating_add(kv_dim);
        if end > self.k.len() || end > self.v.len() {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: read out of bounds base={base} len={kv_dim} cap={}",
                self.k.len()
            )));
        }
        k_out.copy_from_slice(&self.k[base..end]);
        v_out.copy_from_slice(&self.v[base..end]);
        Ok(())
    }

    fn read_token_f32(
        &self,
        base: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError> {
        let kv_dim = k_out.len();
        if v_out.len() != kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: expected matching k/v len, got {}/{}",
                kv_dim,
                v_out.len()
            )));
        }
        let end = base.saturating_add(kv_dim);
        if end > self.k.len() || end > self.v.len() {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: read out of bounds base={base} len={kv_dim} cap={}",
                self.k.len()
            )));
        }
        for i in 0..kv_dim {
            k_out[i] = self.k[base + i].to_f32();
            v_out[i] = self.v[base + i].to_f32();
        }
        Ok(())
    }

    fn gather_tokens_f32(
        &self,
        bases: &[usize],
        kv_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError> {
        if k_out.len() != v_out.len() {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: gather output len mismatch k={} v={}",
                k_out.len(),
                v_out.len()
            )));
        }
        let need = bases.len().saturating_mul(kv_dim);
        if k_out.len() != need {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: gather output len {} expected {}",
                k_out.len(),
                need
            )));
        }
        for (t, &base) in bases.iter().enumerate() {
            let end = base.saturating_add(kv_dim);
            if end > self.k.len() || end > self.v.len() {
                return Err(CoreError::Backend(format!(
                    "kv cache cpu storage: gather out of bounds base={base} len={kv_dim} cap={}",
                    self.k.len()
                )));
            }
            let dst = t * kv_dim;
            for i in 0..kv_dim {
                k_out[dst + i] = self.k[base + i].to_f32();
                v_out[dst + i] = self.v[base + i].to_f32();
            }
        }
        Ok(())
    }

    fn attention_single_token_gqa_f32(
        &self,
        bases: &[usize],
        q: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        attn_scale: Option<f32>,
        soft_cap: Option<f32>,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        if q.len() != n_heads.saturating_mul(head_dim) {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: attn q len {} expected {}",
                q.len(),
                n_heads.saturating_mul(head_dim)
            )));
        }
        if out.len() != n_heads.saturating_mul(head_dim) {
            return Err(CoreError::Backend(format!(
                "kv cache cpu storage: attn out len {} expected {}",
                out.len(),
                n_heads.saturating_mul(head_dim)
            )));
        }

        let seq = bases.len();
        out.fill(0.0);
        if seq == 0 {
            return Ok(());
        }

        let kv_dim = n_kv_heads.saturating_mul(head_dim);
        for &base in bases {
            let end = base.saturating_add(kv_dim);
            if end > self.k.len() || end > self.v.len() {
                return Err(CoreError::Backend(format!(
                    "kv cache cpu storage: attn out of bounds base={base} len={kv_dim} cap={}",
                    self.k.len()
                )));
            }
        }

        let scale = attn_scale.unwrap_or_else(|| 1.0f32 / (head_dim as f32).sqrt());
        let group_size = (n_heads / n_kv_heads).max(1);
        let mut scores = vec![0.0f32; seq];

        for h in 0..n_heads {
            let kv_h = (h / group_size).min(n_kv_heads.saturating_sub(1));
            let qh = &q[h * head_dim..(h + 1) * head_dim];

            // score_t = dot(q_h, k_t,kv_h) * scale
            for (t, &base) in bases.iter().enumerate() {
                let kt_base = base + kv_h * head_dim;
                let kt = &self.k[kt_base..kt_base + head_dim];
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += qh[i] * kt[i].to_f32();
                }
                let mut s = dot * scale;
                if let Some(cap) = soft_cap {
                    s = (s / cap).tanh() * cap;
                }
                scores[t] = s;
            }
            softmax_f32_inplace_cpu_local(&mut scores);

            // out_h += sum_t softmax(score_t) * v_t,kv_h
            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            for (t, &base) in bases.iter().enumerate() {
                let vt_base = base + kv_h * head_dim;
                let vt = &self.v[vt_base..vt_base + head_dim];
                let w = scores[t];
                for i in 0..head_dim {
                    out_h[i] += w * vt[i].to_f32();
                }
            }
        }
        Ok(())
    }
}

impl DeviceKvStorage for CpuTurboQuantKvStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn write_token_f16(
        &mut self,
        base: usize,
        k_src: &[f16],
        v_src: &[f16],
    ) -> Result<(), CoreError> {
        let len = k_src.len();
        if v_src.len() != len {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: expected matching k/v len, got {}/{}",
                len,
                v_src.len()
            )));
        }
        if len > self.kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: token len mismatch {} exceeds {}",
                len, self.kv_dim
            )));
        }
        let tok = self.token_index(base, self.kv_dim)?;
        let row_bytes = self.row_bytes;
        let k_dst = Self::row_slice_mut(&mut self.k_q, tok, row_bytes)?;
        let v_dst = Self::row_slice_mut(&mut self.v_q, tok, row_bytes)?;
        let k_res = &mut self.k_res_sign[base..base + self.kv_dim];
        let v_res = &mut self.v_res_sign[base..base + self.kv_dim];
        let mut k_row = vec![f16::from_f32(0.0); self.kv_dim];
        let mut v_row = vec![f16::from_f32(0.0); self.kv_dim];
        k_row[..len].copy_from_slice(k_src);
        v_row[..len].copy_from_slice(v_src);
        let mut k_scale = 1.0f32;
        let mut v_scale = 1.0f32;
        let mut k_res_scale = 0.0f32;
        let mut v_res_scale = 0.0f32;
        turboquant_compress_row_f16(
            &k_row,
            k_dst,
            k_res,
            &mut k_scale,
            &mut k_res_scale,
            self.n_kv_heads,
            self.head_dim,
            &self.tq,
        );
        turboquant_compress_row_f16(
            &v_row,
            v_dst,
            v_res,
            &mut v_scale,
            &mut v_res_scale,
            self.n_kv_heads,
            self.head_dim,
            &self.tq,
        );
        self.k_scale[tok] = k_scale;
        self.v_scale[tok] = v_scale;
        self.k_res_scale[tok] = k_res_scale;
        self.v_res_scale[tok] = v_res_scale;
        Ok(())
    }

    fn write_token_f32(
        &mut self,
        base: usize,
        k_src: &[f32],
        v_src: &[f32],
    ) -> Result<(), CoreError> {
        let len = k_src.len();
        if v_src.len() != len {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: expected matching k/v len, got {}/{}",
                len,
                v_src.len()
            )));
        }
        if len > self.kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: token len mismatch {} exceeds {}",
                len, self.kv_dim
            )));
        }
        let tok = self.token_index(base, self.kv_dim)?;
        let row_bytes = self.row_bytes;
        let k_dst = Self::row_slice_mut(&mut self.k_q, tok, row_bytes)?;
        let v_dst = Self::row_slice_mut(&mut self.v_q, tok, row_bytes)?;
        let k_res = &mut self.k_res_sign[base..base + self.kv_dim];
        let v_res = &mut self.v_res_sign[base..base + self.kv_dim];
        let mut k_row = vec![0.0f32; self.kv_dim];
        let mut v_row = vec![0.0f32; self.kv_dim];
        k_row[..len].copy_from_slice(k_src);
        v_row[..len].copy_from_slice(v_src);
        let mut k_scale = 1.0f32;
        let mut v_scale = 1.0f32;
        let mut k_res_scale = 0.0f32;
        let mut v_res_scale = 0.0f32;
        turboquant_compress_row_f32(
            &k_row,
            k_dst,
            k_res,
            &mut k_scale,
            &mut k_res_scale,
            self.n_kv_heads,
            self.head_dim,
            &self.tq,
        );
        turboquant_compress_row_f32(
            &v_row,
            v_dst,
            v_res,
            &mut v_scale,
            &mut v_res_scale,
            self.n_kv_heads,
            self.head_dim,
            &self.tq,
        );
        self.k_scale[tok] = k_scale;
        self.v_scale[tok] = v_scale;
        self.k_res_scale[tok] = k_res_scale;
        self.v_res_scale[tok] = v_res_scale;
        Ok(())
    }

    fn read_token_f16(
        &self,
        base: usize,
        k_out: &mut [f16],
        v_out: &mut [f16],
    ) -> Result<(), CoreError> {
        let len = k_out.len();
        if v_out.len() != len {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: expected matching k/v len, got {}/{}",
                len,
                v_out.len()
            )));
        }
        let tok = self.token_index(base, len)?;
        let ks = self.k_scale[tok];
        let vs = self.v_scale[tok];
        let k_row = Self::row_slice(&self.k_q, tok, self.row_bytes)?;
        let v_row = Self::row_slice(&self.v_q, tok, self.row_bytes)?;
        for h in 0..self.n_kv_heads {
            let off = h * self.head_dim;
            let mut tmp = vec![0.0f32; self.head_dim];
            turboquant_decode_head_to_original(k_row, ks, h, self.head_dim, &self.tq, &mut tmp);
            for i in 0..self.head_dim {
                k_out[off + i] = f16::from_f32(tmp[i]);
            }
            turboquant_decode_head_to_original(v_row, vs, h, self.head_dim, &self.tq, &mut tmp);
            for i in 0..self.head_dim {
                v_out[off + i] = f16::from_f32(tmp[i]);
            }
        }
        Ok(())
    }

    fn read_token_f32(
        &self,
        base: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError> {
        let len = k_out.len();
        if v_out.len() != len {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: expected matching k/v len, got {}/{}",
                len,
                v_out.len()
            )));
        }
        let tok = self.token_index(base, len)?;
        let ks = self.k_scale[tok];
        let vs = self.v_scale[tok];
        let k_row = Self::row_slice(&self.k_q, tok, self.row_bytes)?;
        let v_row = Self::row_slice(&self.v_q, tok, self.row_bytes)?;
        for h in 0..self.n_kv_heads {
            let off = h * self.head_dim;
            let mut tmp = vec![0.0f32; self.head_dim];
            turboquant_decode_head_to_original(k_row, ks, h, self.head_dim, &self.tq, &mut tmp);
            for i in 0..self.head_dim {
                k_out[off + i] = tmp[i];
            }
            turboquant_decode_head_to_original(v_row, vs, h, self.head_dim, &self.tq, &mut tmp);
            for i in 0..self.head_dim {
                v_out[off + i] = tmp[i];
            }
        }
        Ok(())
    }

    fn gather_tokens_f32(
        &self,
        bases: &[usize],
        kv_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError> {
        if kv_dim != self.kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: gather kv_dim {} expected {}",
                kv_dim, self.kv_dim
            )));
        }
        if k_out.len() != v_out.len() {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: gather output len mismatch k={} v={}",
                k_out.len(),
                v_out.len()
            )));
        }
        let need = bases.len().saturating_mul(kv_dim);
        if k_out.len() != need {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: gather output len {} expected {}",
                k_out.len(),
                need
            )));
        }
        for (t, &base) in bases.iter().enumerate() {
            let tok = self.token_index(base, kv_dim)?;
            let ks = self.k_scale[tok];
            let vs = self.v_scale[tok];
            let dst = t * kv_dim;
            let k_row = Self::row_slice(&self.k_q, tok, self.row_bytes)?;
            let v_row = Self::row_slice(&self.v_q, tok, self.row_bytes)?;
            for h in 0..self.n_kv_heads {
                let off = h * self.head_dim;
                let mut tmp = vec![0.0f32; self.head_dim];
                turboquant_decode_head_to_original(k_row, ks, h, self.head_dim, &self.tq, &mut tmp);
                for i in 0..self.head_dim {
                    k_out[dst + off + i] = tmp[i];
                }
                turboquant_decode_head_to_original(v_row, vs, h, self.head_dim, &self.tq, &mut tmp);
                for i in 0..self.head_dim {
                    v_out[dst + off + i] = tmp[i];
                }
            }
        }
        Ok(())
    }

    fn attention_single_token_gqa_f32(
        &self,
        bases: &[usize],
        q: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        attn_scale: Option<f32>,
        soft_cap: Option<f32>,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        if q.len() != n_heads.saturating_mul(head_dim) {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: attn q len {} expected {}",
                q.len(),
                n_heads.saturating_mul(head_dim)
            )));
        }
        if out.len() != n_heads.saturating_mul(head_dim) {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: attn out len {} expected {}",
                out.len(),
                n_heads.saturating_mul(head_dim)
            )));
        }

        let seq = bases.len();
        out.fill(0.0);
        if seq == 0 {
            return Ok(());
        }
        if n_kv_heads > self.n_kv_heads || head_dim > self.head_dim {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: attention shape mismatch n_kv_heads={} > {} or head_dim={} > {}",
                n_kv_heads, self.n_kv_heads, head_dim, self.head_dim
            )));
        }
        for &base in bases {
            if base % self.kv_dim != 0 {
                return Err(CoreError::Backend(format!(
                    "kv cache cpu turboquant: base {} is not token-aligned (kv_dim={})",
                    base, self.kv_dim
                )));
            }
            let end = base.saturating_add(self.kv_dim);
            if end > self.k_res_sign.len() {
                return Err(CoreError::Backend(format!(
                    "kv cache cpu turboquant: attention base out of bounds base={} kv_dim={} cap={}",
                    base, self.kv_dim, self.k_res_sign.len()
                )));
            }
        }

        let scale = attn_scale.unwrap_or_else(|| 1.0f32 / (head_dim as f32).sqrt());
        let group_size = (n_heads / n_kv_heads).max(1);
        let mut scores = vec![0.0f32; seq];
        let mut q_i8 = vec![0i8; head_dim];
        let use_int8_dot = self.tq.is_uniform_8bit(self.kv_dim)
            && std::env::var("CELLM_TURBOQ_INT8_DOT")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
        let use_qjl_corr = std::env::var("CELLM_TURBOQ_QJL_CORR")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let use_uniform8_decode = self.tq.is_uniform_8bit(self.kv_dim);
        let q8_min = -3.0f32;
        let q8_step = 6.0f32 / 255.0f32;
        let mut q_rot = vec![0.0f32; head_dim];
        let mut v_acc_rot = vec![0.0f32; head_dim];
        let mut k_scales = vec![0.0f32; seq];
        let mut v_scales = vec![0.0f32; seq];
        let mut k_res_scales = vec![0.0f32; seq];
        let mut row_offs = vec![0usize; seq];
        let mut sign_bases = vec![0usize; seq];
        for (t, &base) in bases.iter().enumerate() {
            let tok = base / self.kv_dim;
            k_scales[t] = self.k_scale[tok];
            v_scales[t] = self.v_scale[tok];
            k_res_scales[t] = self.k_res_scale[tok];
            row_offs[t] = tok.saturating_mul(self.row_bytes);
            sign_bases[t] = base;
        }

        for h in 0..n_heads {
            let kv_h = (h / group_size).min(n_kv_heads.saturating_sub(1));
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            rotate_head_forward(
                qh,
                &self.tq.rot_sign[kv_h * head_dim..(kv_h + 1) * head_dim],
                &mut q_rot,
                self.tq.inv_sqrt_head_dim,
            );
            let q_scale = if use_int8_dot {
                quantize_symmetric_i8(&q_rot, &mut q_i8)
            } else {
                1.0
            };
            for (t, &base) in bases.iter().enumerate() {
                let _ = base;
                let ks = k_scales[t];
                let row_off = row_offs[t];
                let k_row = &self.k_q[row_off..row_off + self.row_bytes];
                let mut dot = if use_int8_dot {
                    turboquant_dot_rotated_head_int8_uniform8(k_row, kv_h, head_dim, &q_i8, q_scale, ks)
                } else {
                    0.0f32
                };
                if !use_int8_dot {
                    for i in 0..head_dim {
                        let dim = kv_h * head_dim + i;
                        let rot_q = q_rot[i];
                        if use_uniform8_decode {
                            let idx = k_row[dim] as f32;
                            let deq = (q8_min + idx * q8_step) * ks;
                            dot += rot_q * deq;
                        } else {
                            let deq = turboquant_decode_rotated_value(k_row, ks, dim, &self.tq);
                            dot += rot_q * deq;
                        }
                    }
                }
                if use_qjl_corr {
                    let sign_base = sign_bases[t] + kv_h * head_dim;
                    let mut corr = 0.0f32;
                    for i in 0..head_dim {
                        let dim = kv_h * head_dim + i;
                        let sign = if self.k_res_sign[sign_base + i] == 0 { -1.0 } else { 1.0 };
                        corr += (self.tq.qjl_sign[dim] * q_rot[i]) * sign;
                    }
                    dot += (k_res_scales[t] / head_dim as f32) * corr;
                }
                let mut s = dot * scale;
                if let Some(cap) = soft_cap {
                    s = (s / cap).tanh() * cap;
                }
                scores[t] = s;
            }
            softmax_f32_inplace_cpu_local(&mut scores);

            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            v_acc_rot.fill(0.0);
            for t in 0..seq {
                let vs = v_scales[t];
                let row_off = row_offs[t];
                let v_row = &self.v_q[row_off..row_off + self.row_bytes];
                let w = scores[t];
                for i in 0..head_dim {
                    let dim = kv_h * head_dim + i;
                    let v_rot = if use_uniform8_decode {
                        let idx = v_row[dim] as f32;
                        (q8_min + idx * q8_step) * vs
                    } else {
                        turboquant_decode_rotated_value(v_row, vs, dim, &self.tq)
                    };
                    v_acc_rot[i] += w * v_rot;
                }
            }
            rotate_head_inverse_inplace(
                &mut v_acc_rot,
                &self.tq.rot_sign[kv_h * head_dim..(kv_h + 1) * head_dim],
                self.tq.inv_sqrt_head_dim,
            );
            out_h.copy_from_slice(&v_acc_rot);
        }
        Ok(())
    }
}

fn softmax_f32_inplace_cpu_local(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_v = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0f32 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Debug)]
pub struct MetalKvStorage {
    encoding: KvEncodingKind,
    kv_dim: usize,
    n_kv_heads: usize,
    head_dim: usize,
    row_bytes: usize,
    queue: CommandQueue,
    _lib: Library,
    pso_write_f16: ComputePipelineState,
    pso_write_f32: ComputePipelineState,
    pso_read_f16: ComputePipelineState,
    pso_read_f32: ComputePipelineState,
    pso_gather_f32: ComputePipelineState,
    pso_attn_single_gqa_f32: ComputePipelineState,
    _pso_attn_single_gqa_q8_f32: ComputePipelineState,
    k: Buffer,
    v: Buffer,
    k_q: Option<Buffer>,
    v_q: Option<Buffer>,
    k_scale: Option<Buffer>,
    v_scale: Option<Buffer>,
    k_res_sign: Option<Buffer>,
    v_res_sign: Option<Buffer>,
    k_res_scale: Option<Buffer>,
    v_res_scale: Option<Buffer>,
    tq: Option<TurboQuantParams>,
    len: usize,
    scratch: Mutex<MetalScratch>,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Debug, Default)]
struct MetalScratch {
    k_f16: Option<Buffer>,
    v_f16: Option<Buffer>,
    k_f32: Option<Buffer>,
    v_f32: Option<Buffer>,
    q_f32: Option<Buffer>,
    out_f32: Option<Buffer>,
    bases_u32: Option<Buffer>,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalKvStorage {
    pub fn new(
        total_elems: usize,
        n_kv_heads: usize,
        head_dim: usize,
        encoding: KvEncodingKind,
    ) -> Result<Self, CoreError> {
        let kv_dim = n_kv_heads.saturating_mul(head_dim);
        if kv_dim == 0 {
            return Err(CoreError::Backend("kv cache metal storage: kv_dim must be > 0".into()));
        }
        if total_elems % kv_dim != 0 {
            return Err(CoreError::Backend(format!(
                "kv cache metal storage: total_elems {} not divisible by kv_dim {}",
                total_elems, kv_dim
            )));
        }
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| CoreError::Backend("kv cache metal storage: no Metal device found".into()))?;
        let queue = device.new_command_queue();
        let src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void kv_write_f16(
            device half* k_cache [[buffer(0)]],
            device half* v_cache [[buffer(1)]],
            device const half* k_in [[buffer(2)]],
            device const half* v_in [[buffer(3)]],
            constant uint& base [[buffer(4)]],
            constant uint& len [[buffer(5)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= len) return;
            uint idx = base + gid;
            k_cache[idx] = k_in[gid];
            v_cache[idx] = v_in[gid];
        }

        kernel void kv_write_f32(
            device half* k_cache [[buffer(0)]],
            device half* v_cache [[buffer(1)]],
            device const float* k_in [[buffer(2)]],
            device const float* v_in [[buffer(3)]],
            constant uint& base [[buffer(4)]],
            constant uint& len [[buffer(5)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= len) return;
            uint idx = base + gid;
            k_cache[idx] = half(clamp(k_in[gid], -65504.0f, 65504.0f));
            v_cache[idx] = half(clamp(v_in[gid], -65504.0f, 65504.0f));
        }

        kernel void kv_read_f16(
            device const half* k_cache [[buffer(0)]],
            device const half* v_cache [[buffer(1)]],
            device half* k_out [[buffer(2)]],
            device half* v_out [[buffer(3)]],
            constant uint& base [[buffer(4)]],
            constant uint& len [[buffer(5)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= len) return;
            uint idx = base + gid;
            k_out[gid] = k_cache[idx];
            v_out[gid] = v_cache[idx];
        }

        kernel void kv_read_f32(
            device const half* k_cache [[buffer(0)]],
            device const half* v_cache [[buffer(1)]],
            device float* k_out [[buffer(2)]],
            device float* v_out [[buffer(3)]],
            constant uint& base [[buffer(4)]],
            constant uint& len [[buffer(5)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= len) return;
            uint idx = base + gid;
            k_out[gid] = float(k_cache[idx]);
            v_out[gid] = float(v_cache[idx]);
        }

        kernel void kv_gather_f32(
            device const half* k_cache [[buffer(0)]],
            device const half* v_cache [[buffer(1)]],
            device const uint* bases [[buffer(2)]],
            device float* k_out [[buffer(3)]],
            device float* v_out [[buffer(4)]],
            constant uint& kv_dim [[buffer(5)]],
            constant uint& total [[buffer(6)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= total) return;
            uint tok = gid / kv_dim;
            uint i = gid - tok * kv_dim;
            uint src = bases[tok] + i;
            k_out[gid] = float(k_cache[src]);
            v_out[gid] = float(v_cache[src]);
        }

        kernel void kv_attn_single_gqa_f32(
            device const half* k_cache [[buffer(0)]],
            device const half* v_cache [[buffer(1)]],
            device const uint* bases [[buffer(2)]],
            device const float* q [[buffer(3)]],
            device float* out [[buffer(4)]],
            constant uint& n_heads [[buffer(5)]],
            constant uint& n_kv_heads [[buffer(6)]],
            constant uint& head_dim [[buffer(7)]],
            constant uint& seq [[buffer(8)]],
            constant float& scale [[buffer(9)]],
            constant float& soft_cap [[buffer(10)]],
            uint gid [[thread_position_in_grid]],
            uint tid [[thread_position_in_threadgroup]],
            uint head_idx [[threadgroup_position_in_grid]]
        ) {
            if (head_idx >= n_heads) return;
            uint group_size = n_heads / n_kv_heads;
            uint kv_h = head_idx / group_size;
            const device float* qh = q + head_idx * head_dim;

            float max_score = -INFINITY;
            float denom = 0.0f;
            // v_acc accumulates partial V sums; each thread covers head_dim/32 elements.
            // 32 slots supports head_dim up to 1024 (32 threads × 32 slots = 1024).
            float v_acc[32];
            for (uint j = 0; j < 32; j++) v_acc[j] = 0.0f;

            threadgroup float dots[32];
            threadgroup float shared_score;

            for (uint t = 0; t < seq; ++t) {
                uint base = bases[t] + kv_h * head_dim;
                float pdot = 0.0f;
                for (uint i = tid; i < head_dim; i += 32) pdot += qh[i] * float(k_cache[base + i]);
                dots[tid] = pdot;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid == 0) {
                    float dot = 0.0f;
                    for (int k = 0; k < 32; ++k) dot += dots[k];
                    float s = dot * scale;
                    if (soft_cap > 0.0f) s = tanh(s / soft_cap) * soft_cap;
                    shared_score = s;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                float score = shared_score;
                float old_max = max_score;
                max_score = max(max_score, score);
                float exp_prev = (old_max == -INFINITY) ? 0.0f : exp(old_max - max_score);
                float exp_curr = exp(score - max_score);
                
                denom = denom * exp_prev + exp_curr;
                for (uint i = tid, j = 0; i < head_dim; i += 32, ++j) {
                    v_acc[j] = v_acc[j] * exp_prev + exp_curr * float(v_cache[base + i]);
                }
            }

            for (uint i = tid, j = 0; i < head_dim; i += 32, ++j) {
                out[head_idx * head_dim + i] = v_acc[j] / denom;
            }
        }

        kernel void kv_attn_single_gqa_q8_f32(
            device const char* k_q [[buffer(0)]],
            device const char* v_q [[buffer(1)]],
            device const float* k_scales [[buffer(2)]],
            device const float* v_scales [[buffer(3)]],
            device const uint* bases [[buffer(4)]],
            device const float* q [[buffer(5)]],
            device float* out [[buffer(6)]],
            constant uint& seq [[buffer(7)]],
            constant uint& n_heads [[buffer(8)]],
            constant uint& n_kv_heads [[buffer(9)]],
            constant uint& head_dim [[buffer(10)]],
            constant uint& kv_dim [[buffer(11)]],
            constant float& scale [[buffer(12)]],
            constant float& soft_cap [[buffer(13)]],
            uint h [[thread_position_in_grid]]
        ) {
            if (h >= n_heads) return;
            uint group_size = max((uint)1, n_heads / max((uint)1, n_kv_heads));
            uint kv_h = min(h / group_size, max((uint)0, n_kv_heads - 1));
            const device float* qh = q + h * head_dim;

            float max_score = -INFINITY;
            for (uint t = 0; t < seq; ++t) {
                uint base_tok = bases[t];
                uint tok_idx = base_tok / kv_dim;
                uint base = base_tok + kv_h * head_dim;
                float ks = k_scales[tok_idx];
                float dot = 0.0f;
                for (uint i = 0; i < head_dim; ++i) {
                    dot += qh[i] * (float(k_q[base + i]) * ks);
                }
                float s = dot * scale;
                if (soft_cap > 0.0f) s = tanh(s / soft_cap) * soft_cap;
                if (s > max_score) max_score = s;
            }

            float denom = 0.0f;
            for (uint t = 0; t < seq; ++t) {
                uint base_tok = bases[t];
                uint tok_idx = base_tok / kv_dim;
                uint base = base_tok + kv_h * head_dim;
                float ks = k_scales[tok_idx];
                float dot = 0.0f;
                for (uint i = 0; i < head_dim; ++i) {
                    dot += qh[i] * (float(k_q[base + i]) * ks);
                }
                float s = dot * scale;
                if (soft_cap > 0.0f) s = tanh(s / soft_cap) * soft_cap;
                denom += exp(s - max_score);
            }
            denom = max(denom, 1e-12f);

            for (uint i = 0; i < head_dim; ++i) {
                out[h * head_dim + i] = 0.0f;
            }
            for (uint t = 0; t < seq; ++t) {
                uint base_tok = bases[t];
                uint tok_idx = base_tok / kv_dim;
                uint base = base_tok + kv_h * head_dim;
                float ks = k_scales[tok_idx];
                float vs = v_scales[tok_idx];
                float dot = 0.0f;
                for (uint i = 0; i < head_dim; ++i) {
                    dot += qh[i] * (float(k_q[base + i]) * ks);
                }
                float s = dot * scale;
                if (soft_cap > 0.0f) s = tanh(s / soft_cap) * soft_cap;
                float w = exp(s - max_score) / denom;
                for (uint i = 0; i < head_dim; ++i) {
                    out[h * head_dim + i] += w * (float(v_q[base + i]) * vs);
                }
            }
        }
        "#;
        let lib = {
            let mut guard = KV_CACHE_LIB_CACHE.lock().unwrap();
            if guard.is_none() {
                let options = metal::CompileOptions::new();
                options.set_fast_math_enabled(true);
                *guard = Some(
                    device
                        .new_library_with_source(src, &options)
                        .map_err(|e| CoreError::Backend(format!("kv cache metal storage: compile failed: {e:?}")))?
                );
            }
            guard.as_ref().unwrap().clone()
        };
        let pso_write_f16 = build_pso(&device, &lib, "kv_write_f16")?;
        let pso_write_f32 = build_pso(&device, &lib, "kv_write_f32")?;
        let pso_read_f16 = build_pso(&device, &lib, "kv_read_f16")?;
        let pso_read_f32 = build_pso(&device, &lib, "kv_read_f32")?;
        let pso_gather_f32 = build_pso(&device, &lib, "kv_gather_f32")?;
        let pso_attn_single_gqa_f32 = build_pso(&device, &lib, "kv_attn_single_gqa_f32")?;
        let pso_attn_single_gqa_q8_f32 = build_pso(&device, &lib, "kv_attn_single_gqa_q8_f32")?;
        let bytes = (total_elems * std::mem::size_of::<f16>()) as u64;
        let k = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let v = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let z = f16::from_f32(0.0);
        let k_ptr = k.contents() as *mut f16;
        let v_ptr = v.contents() as *mut f16;
        if k_ptr.is_null() || v_ptr.is_null() {
            return Err(CoreError::Backend(
                "kv cache metal storage: null buffer contents".into(),
            ));
        }
        let k_slice = unsafe { std::slice::from_raw_parts_mut(k_ptr, total_elems) };
        let v_slice = unsafe { std::slice::from_raw_parts_mut(v_ptr, total_elems) };
        k_slice.fill(z);
        v_slice.fill(z);
        let (k_q, v_q, k_scale, v_scale, k_res_sign, v_res_sign, k_res_scale, v_res_scale, tq, row_bytes) =
            if encoding == KvEncodingKind::TurboQuant {
            let tokens = total_elems / kv_dim;
            let tq = TurboQuantParams::new(kv_dim, n_kv_heads, head_dim, 0xCE11_6D4Bu64);
            let row_bytes = tq.row_bytes;
            let bytes_q = (tokens.saturating_mul(row_bytes)) as u64;
            let bytes_s = (tokens * std::mem::size_of::<f32>()) as u64;
            let kq = device.new_buffer(bytes_q, MTLResourceOptions::StorageModeShared);
            let vq = device.new_buffer(bytes_q, MTLResourceOptions::StorageModeShared);
            let bytes_sign = total_elems as u64;
            let krs = device.new_buffer(bytes_sign, MTLResourceOptions::StorageModeShared);
            let vrs = device.new_buffer(bytes_sign, MTLResourceOptions::StorageModeShared);
            let ks = device.new_buffer(bytes_s, MTLResourceOptions::StorageModeShared);
            let vs = device.new_buffer(bytes_s, MTLResourceOptions::StorageModeShared);
            let krs_scale = device.new_buffer(bytes_s, MTLResourceOptions::StorageModeShared);
            let vrs_scale = device.new_buffer(bytes_s, MTLResourceOptions::StorageModeShared);
            let kq_ptr = kq.contents() as *mut u8;
            let vq_ptr = vq.contents() as *mut u8;
            let krs_ptr = krs.contents() as *mut u8;
            let vrs_ptr = vrs.contents() as *mut u8;
            let ks_ptr = ks.contents() as *mut f32;
            let vs_ptr = vs.contents() as *mut f32;
            let krs_scale_ptr = krs_scale.contents() as *mut f32;
            let vrs_scale_ptr = vrs_scale.contents() as *mut f32;
            if kq_ptr.is_null()
                || vq_ptr.is_null()
                || krs_ptr.is_null()
                || vrs_ptr.is_null()
                || ks_ptr.is_null()
                || vs_ptr.is_null()
                || krs_scale_ptr.is_null()
                || vrs_scale_ptr.is_null()
            {
                return Err(CoreError::Backend(
                    "kv cache metal storage: null turboquant buffer contents".into(),
                ));
            }
            if (kq.length() as usize) < tokens.saturating_mul(row_bytes)
                || (vq.length() as usize) < tokens.saturating_mul(row_bytes)
                || (krs.length() as usize) < total_elems
                || (vrs.length() as usize) < total_elems
                || (ks.length() as usize) < tokens.saturating_mul(std::mem::size_of::<f32>())
                || (vs.length() as usize) < tokens.saturating_mul(std::mem::size_of::<f32>())
                || (krs_scale.length() as usize) < tokens.saturating_mul(std::mem::size_of::<f32>())
                || (vrs_scale.length() as usize) < tokens.saturating_mul(std::mem::size_of::<f32>())
            {
                return Err(CoreError::Backend(
                    "kv cache metal storage: turboquant buffer length mismatch".into(),
                ));
            }
            unsafe {
                std::slice::from_raw_parts_mut(kq_ptr, tokens.saturating_mul(row_bytes)).fill(0);
                std::slice::from_raw_parts_mut(vq_ptr, tokens.saturating_mul(row_bytes)).fill(0);
                std::slice::from_raw_parts_mut(krs_ptr, total_elems).fill(1u8);
                std::slice::from_raw_parts_mut(vrs_ptr, total_elems).fill(1u8);
                std::slice::from_raw_parts_mut(ks_ptr, tokens).fill(1.0f32);
                std::slice::from_raw_parts_mut(vs_ptr, tokens).fill(1.0f32);
                std::slice::from_raw_parts_mut(krs_scale_ptr, tokens).fill(0.0f32);
                std::slice::from_raw_parts_mut(vrs_scale_ptr, tokens).fill(0.0f32);
            }
            (
                Some(kq),
                Some(vq),
                Some(ks),
                Some(vs),
                Some(krs),
                Some(vrs),
                Some(krs_scale),
                Some(vrs_scale),
                Some(tq),
                row_bytes,
            )
        } else {
            (None, None, None, None, None, None, None, None, None, 0usize)
        };
        Ok(Self {
            encoding,
            kv_dim,
            n_kv_heads,
            head_dim,
            row_bytes,
            queue,
            _lib: lib,
            pso_write_f16,
            pso_write_f32,
            pso_read_f16,
            pso_read_f32,
            pso_gather_f32,
            pso_attn_single_gqa_f32,
            _pso_attn_single_gqa_q8_f32: pso_attn_single_gqa_q8_f32,
            k,
            v,
            k_q,
            v_q,
            k_scale,
            v_scale,
            k_res_sign,
            v_res_sign,
            k_res_scale,
            v_res_scale,
            tq,
            len: total_elems,
            scratch: Mutex::new(MetalScratch::default()),
        })
    }

    fn check_range(&self, base: usize, len: usize) -> Result<(), CoreError> {
        let end = base.saturating_add(len);
        if end > self.len {
            return Err(CoreError::Backend(format!(
                "kv cache metal storage: out of bounds base={base} len={len} cap={}",
                self.len
            )));
        }
        Ok(())
    }

    fn check_u32(&self, v: usize, what: &str) -> Result<u32, CoreError> {
        u32::try_from(v).map_err(|_| {
            CoreError::Backend(format!("kv cache metal storage: {what} exceeds u32: {v}"))
        })
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl DeviceKvStorage for MetalKvStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn write_token_f16(
        &mut self,
        base: usize,
        k_src: &[f16],
        v_src: &[f16],
    ) -> Result<(), CoreError> {
        if self.encoding == KvEncodingKind::TurboQuant {
            if k_src.len() != v_src.len() {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: expected matching k/v len, got {}/{}",
                    k_src.len(),
                    v_src.len()
                )));
            }
            self.check_range(base, self.kv_dim)?;
            if base % self.kv_dim != 0 || k_src.len() > self.kv_dim {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: write must be token-aligned and no wider than a token (base={base}, len={}, kv_dim={})",
                    k_src.len(),
                    self.kv_dim
                )));
            }
            let tok = base / self.kv_dim;
            let row_off = tok.saturating_mul(self.row_bytes);
            let kq = self.k_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_q".into()))?;
            let vq = self.v_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_q".into()))?;
            let krs = self.k_res_sign.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_res_sign".into()))?;
            let vrs = self.v_res_sign.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_res_sign".into()))?;
            let ks = self.k_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_scale".into()))?;
            let vs = self.v_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_scale".into()))?;
            let krs_scale = self.k_res_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_res_scale".into()))?;
            let vrs_scale = self.v_res_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_res_scale".into()))?;
            let tq = self.tq.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing params".into()))?;
            let k_ptr = kq.contents() as *mut u8;
            let v_ptr = vq.contents() as *mut u8;
            let krs_ptr = krs.contents() as *mut u8;
            let vrs_ptr = vrs.contents() as *mut u8;
            let ks_ptr = ks.contents() as *mut f32;
            let vs_ptr = vs.contents() as *mut f32;
            let krs_scale_ptr = krs_scale.contents() as *mut f32;
            let vrs_scale_ptr = vrs_scale.contents() as *mut f32;
            if k_ptr.is_null()
                || v_ptr.is_null()
                || krs_ptr.is_null()
                || vrs_ptr.is_null()
                || ks_ptr.is_null()
                || vs_ptr.is_null()
                || krs_scale_ptr.is_null()
                || vrs_scale_ptr.is_null()
            {
                return Err(CoreError::Backend("kv cache metal turboquant: null buffer contents".into()));
            }
            let k_dst = unsafe { std::slice::from_raw_parts_mut(k_ptr.add(row_off), self.row_bytes) };
            let v_dst = unsafe { std::slice::from_raw_parts_mut(v_ptr.add(row_off), self.row_bytes) };
            let k_res_dst = unsafe { std::slice::from_raw_parts_mut(krs_ptr.add(base), self.kv_dim) };
            let v_res_dst = unsafe { std::slice::from_raw_parts_mut(vrs_ptr.add(base), self.kv_dim) };
            let mut k_row = vec![f16::from_f32(0.0); self.kv_dim];
            let mut v_row = vec![f16::from_f32(0.0); self.kv_dim];
            k_row[..k_src.len()].copy_from_slice(k_src);
            v_row[..v_src.len()].copy_from_slice(v_src);
            let mut k_scale = 1.0f32;
            let mut v_scale = 1.0f32;
            let mut k_residual_scale = 0.0f32;
            let mut v_residual_scale = 0.0f32;
            turboquant_compress_row_f16(
                &k_row,
                k_dst,
                k_res_dst,
                &mut k_scale,
                &mut k_residual_scale,
                self.n_kv_heads,
                self.head_dim,
                tq,
            );
            turboquant_compress_row_f16(
                &v_row,
                v_dst,
                v_res_dst,
                &mut v_scale,
                &mut v_residual_scale,
                self.n_kv_heads,
                self.head_dim,
                tq,
            );
            unsafe {
                *ks_ptr.add(tok) = k_scale;
                *vs_ptr.add(tok) = v_scale;
                *krs_scale_ptr.add(tok) = k_residual_scale;
                *vrs_scale_ptr.add(tok) = v_residual_scale;
            }
            return Ok(());
        }
        if k_src.len() != v_src.len() {
            return Err(CoreError::Backend(format!(
                "kv cache metal storage: expected matching k/v len, got {}/{}",
                k_src.len(),
                v_src.len()
            )));
        }
        self.check_range(base, k_src.len())?;
        let device = self.queue.device();
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| CoreError::Backend("kv cache metal write_f16: scratch lock poisoned".into()))?;
        ensure_buffer_f16(device, &mut scratch.k_f16, k_src.len(), "k_f16")?;
        ensure_buffer_f16(device, &mut scratch.v_f16, v_src.len(), "v_f16")?;
        let k_in = scratch
            .k_f16
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal write_f16: missing k_f16 scratch".into()))?;
        let v_in = scratch
            .v_f16
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal write_f16: missing v_f16 scratch".into()))?;
        write_buf_f16(k_in, k_src, "k_f16")?;
        write_buf_f16(v_in, v_src, "v_f16")?;
        let base_u32 = self.check_u32(base, "base")?;
        let len_u32 = self.check_u32(k_src.len(), "len")?;
        let base_ptr = (&base_u32 as *const u32).cast();
        let len_ptr = (&len_u32 as *const u32).cast();
        dispatch_1d(&self.queue, &self.pso_write_f16, k_src.len() as u64, |enc| {
            enc.set_buffer(0, Some(&self.k), 0);
            enc.set_buffer(1, Some(&self.v), 0);
            enc.set_buffer(2, Some(&k_in), 0);
            enc.set_buffer(3, Some(&v_in), 0);
            enc.set_bytes(4, std::mem::size_of::<u32>() as u64, base_ptr);
            enc.set_bytes(5, std::mem::size_of::<u32>() as u64, len_ptr);
        })
        .map_err(|e| CoreError::Backend(format!("kv cache metal write_f16 failed: {e}")))
    }

    fn write_token_f32(
        &mut self,
        base: usize,
        k_src: &[f32],
        v_src: &[f32],
    ) -> Result<(), CoreError> {
        if self.encoding == KvEncodingKind::TurboQuant {
            if k_src.len() != v_src.len() {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: expected matching k/v len, got {}/{}",
                    k_src.len(),
                    v_src.len()
                )));
            }
            self.check_range(base, self.kv_dim)?;
            if base % self.kv_dim != 0 || k_src.len() > self.kv_dim {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: write must be token-aligned and no wider than a token (base={base}, len={}, kv_dim={})",
                    k_src.len(),
                    self.kv_dim
                )));
            }
            let tok = base / self.kv_dim;
            let row_off = tok.saturating_mul(self.row_bytes);
            let kq = self.k_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_q".into()))?;
            let vq = self.v_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_q".into()))?;
            let krs = self.k_res_sign.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_res_sign".into()))?;
            let vrs = self.v_res_sign.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_res_sign".into()))?;
            let ks = self.k_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_scale".into()))?;
            let vs = self.v_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_scale".into()))?;
            let krs_scale = self.k_res_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_res_scale".into()))?;
            let vrs_scale = self.v_res_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_res_scale".into()))?;
            let tq = self.tq.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing params".into()))?;
            let k_ptr = kq.contents() as *mut u8;
            let v_ptr = vq.contents() as *mut u8;
            let krs_ptr = krs.contents() as *mut u8;
            let vrs_ptr = vrs.contents() as *mut u8;
            let ks_ptr = ks.contents() as *mut f32;
            let vs_ptr = vs.contents() as *mut f32;
            let krs_scale_ptr = krs_scale.contents() as *mut f32;
            let vrs_scale_ptr = vrs_scale.contents() as *mut f32;
            if k_ptr.is_null()
                || v_ptr.is_null()
                || krs_ptr.is_null()
                || vrs_ptr.is_null()
                || ks_ptr.is_null()
                || vs_ptr.is_null()
                || krs_scale_ptr.is_null()
                || vrs_scale_ptr.is_null()
            {
                return Err(CoreError::Backend("kv cache metal turboquant: null buffer contents".into()));
            }
            let k_dst = unsafe { std::slice::from_raw_parts_mut(k_ptr.add(row_off), self.row_bytes) };
            let v_dst = unsafe { std::slice::from_raw_parts_mut(v_ptr.add(row_off), self.row_bytes) };
            let k_res_dst = unsafe { std::slice::from_raw_parts_mut(krs_ptr.add(base), self.kv_dim) };
            let v_res_dst = unsafe { std::slice::from_raw_parts_mut(vrs_ptr.add(base), self.kv_dim) };
            let mut k_row = vec![0.0f32; self.kv_dim];
            let mut v_row = vec![0.0f32; self.kv_dim];
            k_row[..k_src.len()].copy_from_slice(k_src);
            v_row[..v_src.len()].copy_from_slice(v_src);
            let mut k_scale = 1.0f32;
            let mut v_scale = 1.0f32;
            let mut k_residual_scale = 0.0f32;
            let mut v_residual_scale = 0.0f32;
            turboquant_compress_row_f32(
                &k_row,
                k_dst,
                k_res_dst,
                &mut k_scale,
                &mut k_residual_scale,
                self.n_kv_heads,
                self.head_dim,
                tq,
            );
            turboquant_compress_row_f32(
                &v_row,
                v_dst,
                v_res_dst,
                &mut v_scale,
                &mut v_residual_scale,
                self.n_kv_heads,
                self.head_dim,
                tq,
            );
            unsafe {
                *ks_ptr.add(tok) = k_scale;
                *vs_ptr.add(tok) = v_scale;
                *krs_scale_ptr.add(tok) = k_residual_scale;
                *vrs_scale_ptr.add(tok) = v_residual_scale;
            }
            return Ok(());
        }
        if k_src.len() != v_src.len() {
            return Err(CoreError::Backend(format!(
                "kv cache metal storage: expected matching k/v len, got {}/{}",
                k_src.len(),
                v_src.len()
            )));
        }
        self.check_range(base, k_src.len())?;
        let device = self.queue.device();
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| CoreError::Backend("kv cache metal write_f32: scratch lock poisoned".into()))?;
        ensure_buffer_f32(device, &mut scratch.k_f32, k_src.len(), "k_f32")?;
        ensure_buffer_f32(device, &mut scratch.v_f32, v_src.len(), "v_f32")?;
        let k_in = scratch
            .k_f32
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal write_f32: missing k_f32 scratch".into()))?;
        let v_in = scratch
            .v_f32
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal write_f32: missing v_f32 scratch".into()))?;
        write_buf_f32(k_in, k_src, "k_f32")?;
        write_buf_f32(v_in, v_src, "v_f32")?;
        let base_u32 = self.check_u32(base, "base")?;
        let len_u32 = self.check_u32(k_src.len(), "len")?;
        let base_ptr = (&base_u32 as *const u32).cast();
        let len_ptr = (&len_u32 as *const u32).cast();
        dispatch_1d(&self.queue, &self.pso_write_f32, k_src.len() as u64, |enc| {
            enc.set_buffer(0, Some(&self.k), 0);
            enc.set_buffer(1, Some(&self.v), 0);
            enc.set_buffer(2, Some(&k_in), 0);
            enc.set_buffer(3, Some(&v_in), 0);
            enc.set_bytes(4, std::mem::size_of::<u32>() as u64, base_ptr);
            enc.set_bytes(5, std::mem::size_of::<u32>() as u64, len_ptr);
        })
        .map_err(|e| CoreError::Backend(format!("kv cache metal write_f32 failed: {e}")))
    }

    fn read_token_f16(
        &self,
        base: usize,
        k_out: &mut [f16],
        v_out: &mut [f16],
    ) -> Result<(), CoreError> {
        if self.encoding == KvEncodingKind::TurboQuant {
            let len = k_out.len();
            if v_out.len() != len {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: expected matching k/v len, got {}/{}",
                    len,
                    v_out.len()
                )));
            }
            self.check_range(base, len)?;
            if base % self.kv_dim != 0 || len != self.kv_dim {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: read must be token-aligned (base={base}, len={len}, kv_dim={})",
                    self.kv_dim
                )));
            }
            let tok = base / self.kv_dim;
            let row_off = tok.saturating_mul(self.row_bytes);
            let kq = self.k_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_q".into()))?;
            let vq = self.v_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_q".into()))?;
            let ks = self.k_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_scale".into()))?;
            let vs = self.v_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_scale".into()))?;
            let tq = self.tq.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing params".into()))?;
            let k_ptr = kq.contents() as *const u8;
            let v_ptr = vq.contents() as *const u8;
            let ks_ptr = ks.contents() as *const f32;
            let vs_ptr = vs.contents() as *const f32;
            if k_ptr.is_null() || v_ptr.is_null() || ks_ptr.is_null() || vs_ptr.is_null() {
                return Err(CoreError::Backend("kv cache metal turboquant: null buffer contents".into()));
            }
            let k_row = unsafe { std::slice::from_raw_parts(k_ptr.add(row_off), self.row_bytes) };
            let v_row = unsafe { std::slice::from_raw_parts(v_ptr.add(row_off), self.row_bytes) };
            let ks_val = unsafe { *ks_ptr.add(tok) };
            let vs_val = unsafe { *vs_ptr.add(tok) };
            for h in 0..self.n_kv_heads {
                let off = h * self.head_dim;
                let mut tmp = vec![0.0f32; self.head_dim];
                turboquant_decode_head_to_original(k_row, ks_val, h, self.head_dim, tq, &mut tmp);
                for i in 0..self.head_dim {
                    k_out[off + i] = f16::from_f32(tmp[i]);
                }
                turboquant_decode_head_to_original(v_row, vs_val, h, self.head_dim, tq, &mut tmp);
                for i in 0..self.head_dim {
                    v_out[off + i] = f16::from_f32(tmp[i]);
                }
            }
            return Ok(());
        }
        if k_out.len() != v_out.len() {
            return Err(CoreError::Backend(format!(
                "kv cache metal storage: expected matching k/v len, got {}/{}",
                k_out.len(),
                v_out.len()
            )));
        }
        self.check_range(base, k_out.len())?;
        let device = self.queue.device();
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| CoreError::Backend("kv cache metal read_f16: scratch lock poisoned".into()))?;
        ensure_buffer_f16(device, &mut scratch.k_f16, k_out.len(), "k_f16")?;
        ensure_buffer_f16(device, &mut scratch.v_f16, v_out.len(), "v_f16")?;
        let k_tmp = scratch
            .k_f16
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal read_f16: missing k_f16 scratch".into()))?;
        let v_tmp = scratch
            .v_f16
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal read_f16: missing v_f16 scratch".into()))?;
        let base_u32 = self.check_u32(base, "base")?;
        let len_u32 = self.check_u32(k_out.len(), "len")?;
        let base_ptr = (&base_u32 as *const u32).cast();
        let len_ptr = (&len_u32 as *const u32).cast();
        dispatch_1d(&self.queue, &self.pso_read_f16, k_out.len() as u64, |enc| {
            enc.set_buffer(0, Some(&self.k), 0);
            enc.set_buffer(1, Some(&self.v), 0);
            enc.set_buffer(2, Some(&k_tmp), 0);
            enc.set_buffer(3, Some(&v_tmp), 0);
            enc.set_bytes(4, std::mem::size_of::<u32>() as u64, base_ptr);
            enc.set_bytes(5, std::mem::size_of::<u32>() as u64, len_ptr);
        })
        .map_err(|e| CoreError::Backend(format!("kv cache metal read_f16 failed: {e}")))?;
        read_buf_f16(k_tmp, k_out, "k_f16")?;
        read_buf_f16(v_tmp, v_out, "v_f16")?;
        Ok(())
    }

    fn read_token_f32(
        &self,
        base: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError> {
        if self.encoding == KvEncodingKind::TurboQuant {
            let len = k_out.len();
            if v_out.len() != len {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: expected matching k/v len, got {}/{}",
                    len,
                    v_out.len()
                )));
            }
            self.check_range(base, len)?;
            if base % self.kv_dim != 0 || len != self.kv_dim {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: read must be token-aligned (base={base}, len={len}, kv_dim={})",
                    self.kv_dim
                )));
            }
            let tok = base / self.kv_dim;
            let row_off = tok.saturating_mul(self.row_bytes);
            let kq = self.k_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_q".into()))?;
            let vq = self.v_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_q".into()))?;
            let ks = self.k_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_scale".into()))?;
            let vs = self.v_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_scale".into()))?;
            let tq = self.tq.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing params".into()))?;
            let k_ptr = kq.contents() as *const u8;
            let v_ptr = vq.contents() as *const u8;
            let ks_ptr = ks.contents() as *const f32;
            let vs_ptr = vs.contents() as *const f32;
            if k_ptr.is_null() || v_ptr.is_null() || ks_ptr.is_null() || vs_ptr.is_null() {
                return Err(CoreError::Backend("kv cache metal turboquant: null buffer contents".into()));
            }
            let k_row = unsafe { std::slice::from_raw_parts(k_ptr.add(row_off), self.row_bytes) };
            let v_row = unsafe { std::slice::from_raw_parts(v_ptr.add(row_off), self.row_bytes) };
            let ks_val = unsafe { *ks_ptr.add(tok) };
            let vs_val = unsafe { *vs_ptr.add(tok) };
            for h in 0..self.n_kv_heads {
                let off = h * self.head_dim;
                let mut tmp = vec![0.0f32; self.head_dim];
                turboquant_decode_head_to_original(k_row, ks_val, h, self.head_dim, tq, &mut tmp);
                for i in 0..self.head_dim {
                    k_out[off + i] = tmp[i];
                }
                turboquant_decode_head_to_original(v_row, vs_val, h, self.head_dim, tq, &mut tmp);
                for i in 0..self.head_dim {
                    v_out[off + i] = tmp[i];
                }
            }
            return Ok(());
        }
        if k_out.len() != v_out.len() {
            return Err(CoreError::Backend(format!(
                "kv cache metal storage: expected matching k/v len, got {}/{}",
                k_out.len(),
                v_out.len()
            )));
        }
        self.check_range(base, k_out.len())?;
        let device = self.queue.device();
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| CoreError::Backend("kv cache metal read_f32: scratch lock poisoned".into()))?;
        ensure_buffer_f32(device, &mut scratch.k_f32, k_out.len(), "k_f32")?;
        ensure_buffer_f32(device, &mut scratch.v_f32, v_out.len(), "v_f32")?;
        let k_tmp = scratch
            .k_f32
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal read_f32: missing k_f32 scratch".into()))?;
        let v_tmp = scratch
            .v_f32
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal read_f32: missing v_f32 scratch".into()))?;
        let base_u32 = self.check_u32(base, "base")?;
        let len_u32 = self.check_u32(k_out.len(), "len")?;
        let base_ptr = (&base_u32 as *const u32).cast();
        let len_ptr = (&len_u32 as *const u32).cast();
        dispatch_1d(&self.queue, &self.pso_read_f32, k_out.len() as u64, |enc| {
            enc.set_buffer(0, Some(&self.k), 0);
            enc.set_buffer(1, Some(&self.v), 0);
            enc.set_buffer(2, Some(&k_tmp), 0);
            enc.set_buffer(3, Some(&v_tmp), 0);
            enc.set_bytes(4, std::mem::size_of::<u32>() as u64, base_ptr);
            enc.set_bytes(5, std::mem::size_of::<u32>() as u64, len_ptr);
        })
        .map_err(|e| CoreError::Backend(format!("kv cache metal read_f32 failed: {e}")))?;
        read_buf_f32(k_tmp, k_out, "k_f32")?;
        read_buf_f32(v_tmp, v_out, "v_f32")?;
        Ok(())
    }

    fn gather_tokens_f32(
        &self,
        bases: &[usize],
        kv_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError> {
        if self.encoding == KvEncodingKind::TurboQuant {
            if kv_dim != self.kv_dim {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: gather kv_dim {} expected {}",
                    kv_dim, self.kv_dim
                )));
            }
            if k_out.len() != v_out.len() {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: gather output len mismatch k={} v={}",
                    k_out.len(),
                    v_out.len()
                )));
            }
            let total = bases.len().saturating_mul(kv_dim);
            if k_out.len() != total {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: gather output len {} expected {}",
                    k_out.len(),
                    total
                )));
            }
            let kq = self
                .k_q
                .as_ref()
                .ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_q".into()))?;
            let vq = self
                .v_q
                .as_ref()
                .ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_q".into()))?;
            let ks = self.k_scale.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing k_scale".into())
            })?;
            let vs = self.v_scale.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing v_scale".into())
            })?;
            let tq = self.tq.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing params".into())
            })?;
            let k_ptr = kq.contents() as *const u8;
            let v_ptr = vq.contents() as *const u8;
            let ks_ptr = ks.contents() as *const f32;
            let vs_ptr = vs.contents() as *const f32;
            if k_ptr.is_null() || v_ptr.is_null() || ks_ptr.is_null() || vs_ptr.is_null() {
                return Err(CoreError::Backend(
                    "kv cache metal turboquant: null buffer contents".into(),
                ));
            }
            for (t, &base) in bases.iter().enumerate() {
                self.check_range(base, kv_dim)?;
                let tok = base / kv_dim;
                let row_off = tok.saturating_mul(self.row_bytes);
                let k_scale = unsafe { *ks_ptr.add(tok) };
                let v_scale = unsafe { *vs_ptr.add(tok) };
                let dst = t * kv_dim;
                let k_row = unsafe { std::slice::from_raw_parts(k_ptr.add(row_off), self.row_bytes) };
                let v_row = unsafe { std::slice::from_raw_parts(v_ptr.add(row_off), self.row_bytes) };
                for h in 0..self.n_kv_heads {
                    let off = h * self.head_dim;
                    let mut tmp = vec![0.0f32; self.head_dim];
                    turboquant_decode_head_to_original(k_row, k_scale, h, self.head_dim, tq, &mut tmp);
                    for i in 0..self.head_dim {
                        k_out[dst + off + i] = tmp[i];
                    }
                    turboquant_decode_head_to_original(v_row, v_scale, h, self.head_dim, tq, &mut tmp);
                    for i in 0..self.head_dim {
                        v_out[dst + off + i] = tmp[i];
                    }
                }
            }
            return Ok(());
        }
        if k_out.len() != v_out.len() {
            return Err(CoreError::Backend(format!(
                "kv cache metal storage: gather output len mismatch k={} v={}",
                k_out.len(),
                v_out.len()
            )));
        }
        let total = bases.len().saturating_mul(kv_dim);
        if k_out.len() != total {
            return Err(CoreError::Backend(format!(
                "kv cache metal storage: gather output len {} expected {}",
                k_out.len(),
                total
            )));
        }
        for &b in bases {
            self.check_range(b, kv_dim)?;
        }
        let device = self.queue.device();
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| CoreError::Backend("kv cache metal gather_f32: scratch lock poisoned".into()))?;
        let bases_u32: Result<Vec<u32>, CoreError> = bases
            .iter()
            .map(|&b| self.check_u32(b, "base"))
            .collect();
        let bases_u32 = bases_u32?;
        ensure_buffer_u32(device, &mut scratch.bases_u32, bases_u32.len(), "bases_u32")?;
        ensure_buffer_f32(device, &mut scratch.k_f32, k_out.len(), "k_f32")?;
        ensure_buffer_f32(device, &mut scratch.v_f32, v_out.len(), "v_f32")?;
        let bases_buf = scratch
            .bases_u32
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal gather_f32: missing bases scratch".into()))?;
        let k_tmp = scratch
            .k_f32
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal gather_f32: missing k_f32 scratch".into()))?;
        let v_tmp = scratch
            .v_f32
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal gather_f32: missing v_f32 scratch".into()))?;
        write_buf_u32(bases_buf, &bases_u32, "bases_u32")?;
        let kv_dim_u32 = self.check_u32(kv_dim, "kv_dim")?;
        let total_u32 = self.check_u32(total, "total")?;
        let kv_dim_ptr = (&kv_dim_u32 as *const u32).cast();
        let total_ptr = (&total_u32 as *const u32).cast();
        dispatch_1d(&self.queue, &self.pso_gather_f32, total as u64, |enc| {
            enc.set_buffer(0, Some(&self.k), 0);
            enc.set_buffer(1, Some(&self.v), 0);
            enc.set_buffer(2, Some(&bases_buf), 0);
            enc.set_buffer(3, Some(&k_tmp), 0);
            enc.set_buffer(4, Some(&v_tmp), 0);
            enc.set_bytes(5, std::mem::size_of::<u32>() as u64, kv_dim_ptr);
            enc.set_bytes(6, std::mem::size_of::<u32>() as u64, total_ptr);
        })
        .map_err(|e| CoreError::Backend(format!("kv cache metal gather_f32 failed: {e}")))?;
        read_buf_f32(k_tmp, k_out, "k_f32")?;
        read_buf_f32(v_tmp, v_out, "v_f32")?;
        Ok(())
    }

    fn attention_single_token_gqa_f32(
        &self,
        bases: &[usize],
        q: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        attn_scale: Option<f32>,
        soft_cap: Option<f32>,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        if q.len() != n_heads.saturating_mul(head_dim) {
            return Err(CoreError::Backend(format!(
                "kv cache metal storage: attn q len {} expected {}",
                q.len(),
                n_heads.saturating_mul(head_dim)
            )));
        }
        if out.len() != n_heads.saturating_mul(head_dim) {
            return Err(CoreError::Backend(format!(
                "kv cache metal storage: attn out len {} expected {}",
                out.len(),
                n_heads.saturating_mul(head_dim)
            )));
        }
        let seq = bases.len();
        if seq == 0 {
            out.fill(0.0);
            return Ok(());
        }
        for &b in bases {
            self.check_range(b, n_kv_heads.saturating_mul(head_dim))?;
        }

        let device = self.queue.device();
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| CoreError::Backend("kv cache metal attn gqa: scratch lock poisoned".into()))?;
        let bases_u32: Result<Vec<u32>, CoreError> = bases
            .iter()
            .map(|&b| self.check_u32(b, "base"))
            .collect();
        let bases_u32 = bases_u32?;
        ensure_buffer_u32(device, &mut scratch.bases_u32, bases_u32.len(), "bases_u32")?;
        ensure_buffer_f32(device, &mut scratch.q_f32, q.len(), "q_f32")?;
        ensure_buffer_f32(device, &mut scratch.out_f32, out.len(), "out_f32")?;
        let bases_buf = scratch
            .bases_u32
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal attn gqa: missing bases scratch".into()))?;
        let q_buf = scratch
            .q_f32
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal attn gqa: missing q_f32 scratch".into()))?;
        let out_buf = scratch
            .out_f32
            .as_ref()
            .ok_or_else(|| CoreError::Backend("kv cache metal attn gqa: missing out_f32 scratch".into()))?;
        write_buf_u32(bases_buf, &bases_u32, "bases_u32")?;
        write_buf_f32(q_buf, q, "q_f32")?;

        let seq_u32 = self.check_u32(seq, "seq")?;
        let n_heads_u32 = self.check_u32(n_heads, "n_heads")?;
        let n_kv_heads_u32 = self.check_u32(n_kv_heads, "n_kv_heads")?;
        let head_dim_u32 = self.check_u32(head_dim, "head_dim")?;
        let scale = attn_scale.unwrap_or_else(|| 1.0f32 / (head_dim as f32).sqrt());
        let seq_ptr = (&seq_u32 as *const u32).cast();
        let n_heads_ptr = (&n_heads_u32 as *const u32).cast();
        let n_kv_heads_ptr = (&n_kv_heads_u32 as *const u32).cast();
        let head_dim_ptr = (&head_dim_u32 as *const u32).cast();
        let scale_ptr = (&scale as *const f32).cast();
        if self.encoding == KvEncodingKind::TurboQuant {
            let group_size = (n_heads / n_kv_heads).max(1);
            let tq = self.tq.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing params".into())
            })?;
            let k_q = self.k_q.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing k_q".into())
            })?;
            let v_q = self.v_q.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing v_q".into())
            })?;
            let k_scale = self.k_scale.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing k_scale".into())
            })?;
            let v_scale = self.v_scale.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing v_scale".into())
            })?;
            let k_res_sign = self.k_res_sign.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing k_res_sign".into())
            })?;
            let k_res_scale = self.k_res_scale.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing k_res_scale".into())
            })?;
            let kq_ptr = k_q.contents() as *const u8;
            let vq_ptr = v_q.contents() as *const u8;
            let ks_ptr = k_scale.contents() as *const f32;
            let vs_ptr = v_scale.contents() as *const f32;
            let ksgn_ptr = k_res_sign.contents() as *const u8;
            let krs_ptr = k_res_scale.contents() as *const f32;
            if kq_ptr.is_null()
                || vq_ptr.is_null()
                || ks_ptr.is_null()
                || vs_ptr.is_null()
                || ksgn_ptr.is_null()
                || krs_ptr.is_null()
            {
                return Err(CoreError::Backend(
                    "kv cache metal turboquant: null attention buffer contents".into(),
                ));
            }
            let mut scores = vec![0.0f32; seq];
            let mut q_i8 = vec![0i8; head_dim];
            let use_int8_dot = tq.is_uniform_8bit(self.kv_dim)
                && std::env::var("CELLM_TURBOQ_INT8_DOT")
                    .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                    .unwrap_or(true);
            let use_qjl_corr = std::env::var("CELLM_TURBOQ_QJL_CORR")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
            let use_uniform8_decode = tq.is_uniform_8bit(self.kv_dim);
            let q8_min = -3.0f32;
            let q8_step = 6.0f32 / 255.0f32;
            let mut q_rot = vec![0.0f32; head_dim];
            let mut v_acc_rot = vec![0.0f32; head_dim];
            let mut k_scales = vec![0.0f32; seq];
            let mut v_scales = vec![0.0f32; seq];
            let mut k_res_scales = vec![0.0f32; seq];
            let mut row_offs = vec![0usize; seq];
            let mut sign_bases = vec![0usize; seq];
            for (t, &base) in bases.iter().enumerate() {
                let tok = base / self.kv_dim;
                k_scales[t] = unsafe { *ks_ptr.add(tok) };
                v_scales[t] = unsafe { *vs_ptr.add(tok) };
                k_res_scales[t] = unsafe { *krs_ptr.add(tok) };
                row_offs[t] = tok.saturating_mul(self.row_bytes);
                sign_bases[t] = base;
            }
            out.fill(0.0);
            for h in 0..n_heads {
                let kv_h = (h / group_size).min(n_kv_heads.saturating_sub(1));
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                rotate_head_forward(
                    qh,
                    &tq.rot_sign[kv_h * head_dim..(kv_h + 1) * head_dim],
                    &mut q_rot,
                    tq.inv_sqrt_head_dim,
                );
                let q_scale = if use_int8_dot {
                    quantize_symmetric_i8(&q_rot, &mut q_i8)
                } else {
                    1.0
                };
                for (t, &base) in bases.iter().enumerate() {
                    let _ = base;
                    let ks = k_scales[t];
                    let row_off = row_offs[t];
                    let k_row = unsafe { std::slice::from_raw_parts(kq_ptr.add(row_off), self.row_bytes) };
                    let mut dot = if use_int8_dot {
                        turboquant_dot_rotated_head_int8_uniform8(k_row, kv_h, head_dim, &q_i8, q_scale, ks)
                    } else {
                        0.0f32
                    };
                    if !use_int8_dot {
                        for i in 0..head_dim {
                            let dim = kv_h * head_dim + i;
                            let rot_q = q_rot[i];
                            let deq = if use_uniform8_decode {
                                let idx = k_row[dim] as f32;
                                (q8_min + idx * q8_step) * ks
                            } else {
                                turboquant_decode_rotated_value(k_row, ks, dim, tq)
                            };
                            dot += rot_q * deq;
                        }
                    }
                    if use_qjl_corr {
                        let sign_base = sign_bases[t] + kv_h * head_dim;
                        let mut corr = 0.0f32;
                        for i in 0..head_dim {
                            let dim = kv_h * head_dim + i;
                            let sign = if unsafe { *ksgn_ptr.add(sign_base + i) } == 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            corr += (tq.qjl_sign[dim] * q_rot[i]) * sign;
                        }
                        dot += (k_res_scales[t] / head_dim as f32) * corr;
                    }
                    let mut s = dot * scale;
                    if let Some(cap) = soft_cap {
                        s = (s / cap).tanh() * cap;
                    }
                    scores[t] = s;
                }
                softmax_f32_inplace_cpu_local(&mut scores);
                let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
                v_acc_rot.fill(0.0);
                for t in 0..seq {
                    let vs = v_scales[t];
                    let row_off = row_offs[t];
                    let v_row = unsafe { std::slice::from_raw_parts(vq_ptr.add(row_off), self.row_bytes) };
                    let w = scores[t];
                    for i in 0..head_dim {
                        let dim = kv_h * head_dim + i;
                        let v_rot = if use_uniform8_decode {
                            let idx = v_row[dim] as f32;
                            (q8_min + idx * q8_step) * vs
                        } else {
                            turboquant_decode_rotated_value(v_row, vs, dim, tq)
                        };
                        v_acc_rot[i] += w * v_rot;
                    }
                }
                rotate_head_inverse_inplace(
                    &mut v_acc_rot,
                    &tq.rot_sign[kv_h * head_dim..(kv_h + 1) * head_dim],
                    tq.inv_sqrt_head_dim,
                );
                out_h.copy_from_slice(&v_acc_rot);
            }
        } else {
            let soft_cap_val = soft_cap.unwrap_or(0.0f32);
            let soft_cap_ptr = (&soft_cap_val as *const f32).cast();
            
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso_attn_single_gqa_f32);
            
            enc.set_buffer(0, Some(&self.k), 0);
            enc.set_buffer(1, Some(&self.v), 0);
            enc.set_buffer(2, Some(&bases_buf), 0);
            enc.set_buffer(3, Some(&q_buf), 0);
            enc.set_buffer(4, Some(&out_buf), 0);
            enc.set_bytes(5, 4, n_heads_ptr);
            enc.set_bytes(6, 4, n_kv_heads_ptr);
            enc.set_bytes(7, 4, head_dim_ptr);
            enc.set_bytes(8, 4, seq_ptr);
            enc.set_bytes(9, 4, scale_ptr);
            enc.set_bytes(10, 4, soft_cap_ptr);

            let threads_per_group = 32;
            let grid_size = metal::MTLSize {
                width: (n_heads * threads_per_group) as u64,
                height: 1,
                depth: 1,
            };
            let group_size = metal::MTLSize {
                width: threads_per_group as u64,
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid_size, group_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        if self.encoding != KvEncodingKind::TurboQuant {
            read_buf_f32(out_buf, out, "out_f32")?;
        }
        Ok(())
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn build_pso(
    device: &Device,
    lib: &Library,
    name: &str,
) -> Result<ComputePipelineState, CoreError> {
    let f = lib
        .get_function(name, None)
        .map_err(|e| CoreError::Backend(format!("kv cache metal: missing function {name}: {e:?}")))?;
    device
        .new_compute_pipeline_state_with_function(&f)
        .map_err(|e| CoreError::Backend(format!("kv cache metal: build pipeline {name} failed: {e:?}")))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ensure_buffer_f16(
    device: &metal::DeviceRef,
    slot: &mut Option<Buffer>,
    elems: usize,
    label: &str,
) -> Result<(), CoreError> {
    let need_bytes = elems.saturating_mul(std::mem::size_of::<f16>());
    if slot
        .as_ref()
        .map(|b| (b.length() as usize) >= need_bytes)
        .unwrap_or(false)
    {
        return Ok(());
    }
    let bytes = need_bytes as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    if buf.contents().is_null() {
        return Err(CoreError::Backend(format!(
            "kv cache metal: null {label} scratch buffer"
        )));
    }
    *slot = Some(buf);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ensure_buffer_f32(
    device: &metal::DeviceRef,
    slot: &mut Option<Buffer>,
    elems: usize,
    label: &str,
) -> Result<(), CoreError> {
    let need_bytes = elems.saturating_mul(std::mem::size_of::<f32>());
    if slot
        .as_ref()
        .map(|b| (b.length() as usize) >= need_bytes)
        .unwrap_or(false)
    {
        return Ok(());
    }
    let bytes = need_bytes as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    if buf.contents().is_null() {
        return Err(CoreError::Backend(format!(
            "kv cache metal: null {label} scratch buffer"
        )));
    }
    *slot = Some(buf);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ensure_buffer_u32(
    device: &metal::DeviceRef,
    slot: &mut Option<Buffer>,
    elems: usize,
    label: &str,
) -> Result<(), CoreError> {
    let need_bytes = elems.saturating_mul(std::mem::size_of::<u32>());
    if slot
        .as_ref()
        .map(|b| (b.length() as usize) >= need_bytes)
        .unwrap_or(false)
    {
        return Ok(());
    }
    let bytes = need_bytes as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    if buf.contents().is_null() {
        return Err(CoreError::Backend(format!(
            "kv cache metal: null {label} scratch buffer"
        )));
    }
    *slot = Some(buf);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn write_buf_f16(buf: &Buffer, src: &[f16], label: &str) -> Result<(), CoreError> {
    let ptr = buf.contents() as *mut f16;
    if ptr.is_null() {
        return Err(CoreError::Backend(format!(
            "kv cache metal: null {label} scratch contents"
        )));
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, src.len()) };
    dst.copy_from_slice(src);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn write_buf_f32(buf: &Buffer, src: &[f32], label: &str) -> Result<(), CoreError> {
    let ptr = buf.contents() as *mut f32;
    if ptr.is_null() {
        return Err(CoreError::Backend(format!(
            "kv cache metal: null {label} scratch contents"
        )));
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, src.len()) };
    dst.copy_from_slice(src);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn write_buf_u32(buf: &Buffer, src: &[u32], label: &str) -> Result<(), CoreError> {
    let ptr = buf.contents() as *mut u32;
    if ptr.is_null() {
        return Err(CoreError::Backend(format!(
            "kv cache metal: null {label} scratch contents"
        )));
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, src.len()) };
    dst.copy_from_slice(src);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn read_buf_f16(buf: &Buffer, dst: &mut [f16], label: &str) -> Result<(), CoreError> {
    let ptr = buf.contents() as *const f16;
    if ptr.is_null() {
        return Err(CoreError::Backend(format!(
            "kv cache metal: null {label} scratch contents"
        )));
    }
    let src = unsafe { std::slice::from_raw_parts(ptr, dst.len()) };
    dst.copy_from_slice(src);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn read_buf_f32(buf: &Buffer, dst: &mut [f32], label: &str) -> Result<(), CoreError> {
    let ptr = buf.contents() as *const f32;
    if ptr.is_null() {
        return Err(CoreError::Backend(format!(
            "kv cache metal: null {label} scratch contents"
        )));
    }
    let src = unsafe { std::slice::from_raw_parts(ptr, dst.len()) };
    dst.copy_from_slice(src);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn dispatch_1d(
    queue: &CommandQueue,
    pso: &ComputePipelineState,
    threads: u64,
    bind: impl FnOnce(&metal::ComputeCommandEncoderRef),
) -> Result<(), CoreError> {
    autoreleasepool(|| {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        bind(enc);
        let w = pso.thread_execution_width() as u64;
        let tg = metal::MTLSize {
            width: w.max(1).min(threads.max(1)),
            height: 1,
            depth: 1,
        };
        let grid = metal::MTLSize {
            width: threads.max(1),
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    })
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[derive(Debug)]
pub struct MetalKvStorage;

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalKvStorage {
    pub fn new(
        _total_elems: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _encoding: KvEncodingKind,
    ) -> Result<Self, CoreError> {
        Err(CoreError::Backend(
            "kv cache metal storage is only supported on Apple platforms".into(),
        ))
    }
}

#[derive(Debug)]
pub struct KVCache {
    layout: KvCacheLayout,
    allocator: BlockAllocator,
    storage: Box<dyn DeviceKvStorage>,
    encoding: KvEncodingKind,
}

impl KVCache {
    pub fn new(layout: KvCacheLayout) -> Result<Self, CoreError> {
        Self::new_with_kind_and_encoding(layout, KvStorageKind::Cpu, KvEncodingKind::F16)
    }

    pub fn new_with_kind(layout: KvCacheLayout, kind: KvStorageKind) -> Result<Self, CoreError> {
        Self::new_with_kind_and_encoding(layout, kind, KvEncodingKind::F16)
    }

    pub fn new_with_kind_and_encoding(
        layout: KvCacheLayout,
        kind: KvStorageKind,
        encoding: KvEncodingKind,
    ) -> Result<Self, CoreError> {
        if layout.total_blocks == 0 {
            return Err(CoreError::Backend("kv cache: total_blocks must be > 0".into()));
        }
        if layout.tokens_per_block == 0 {
            return Err(CoreError::Backend(
                "kv cache: tokens_per_block must be > 0".into(),
            ));
        }
        if layout.num_layers == 0 || layout.num_kv_heads == 0 || layout.head_dim == 0 {
            return Err(CoreError::Backend(
                "kv cache: num_layers/num_kv_heads/head_dim must be > 0".into(),
            ));
        }

        let total_elems = layout.total_elems();
        let storage: Box<dyn DeviceKvStorage> = match (kind, encoding) {
            (KvStorageKind::Cpu, KvEncodingKind::F16) => Box::new(CpuKvStorage {
                k: vec![f16::from_f32(0.0); total_elems],
                v: vec![f16::from_f32(0.0); total_elems],
            }),
            (KvStorageKind::Cpu, KvEncodingKind::TurboQuant) => {
                Box::new(CpuTurboQuantKvStorage::new(
                    total_elems,
                    layout.num_kv_heads,
                    layout.head_dim,
                )?)
            }
            (KvStorageKind::Metal, enc) => Box::new(MetalKvStorage::new(
                total_elems,
                layout.num_kv_heads,
                layout.head_dim,
                enc,
            )?),
        };

        Ok(Self {
            allocator: BlockAllocator::new(layout.total_blocks),
            layout,
            storage,
            encoding,
        })
    }

    pub fn layout(&self) -> KvCacheLayout {
        self.layout
    }

    pub fn storage(&self) -> &dyn DeviceKvStorage {
        self.storage.as_ref()
    }

    pub fn encoding(&self) -> KvEncodingKind {
        self.encoding
    }

    pub fn allocator_mut(&mut self) -> &mut BlockAllocator {
        &mut self.allocator
    }

    pub fn allocator(&self) -> &BlockAllocator {
        &self.allocator
    }

    pub fn view_mut(&mut self) -> KvCacheView<'_> {
        KvCacheView {
            layout: self.layout,
            storage: self.storage.as_mut(),
        }
    }

    pub fn view(&self) -> KvCacheReadView<'_> {
        KvCacheReadView {
            layout: self.layout,
            storage: self.storage.as_ref(),
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalKvStorage {
    pub fn k_buffer(&self) -> &metal::BufferRef { &self.k }
    pub fn v_buffer(&self) -> &metal::BufferRef { &self.v }

    /// Appends the single-token GQA kernel to an active command encoder (NO syncing).
    pub fn encode_attention(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        bases_buf: &metal::BufferRef,
        bases_offset: u64,
        q_buf: &metal::BufferRef,
        out_buf: &metal::BufferRef,
        seq: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attn_scale: Option<f32>,
        soft_cap: Option<f32>,
    ) {
        if self.encoding == KvEncodingKind::TurboQuant {
            eprintln!("[cellm-cache] WARNING: TurboQuant KV cache is not yet supported in the Metal fused graph path. Falling back to non-fused execution (this is much slower). Please use --kv-encoding f16 for maximum GPU performance.");
            return;
        }
        let scale = attn_scale.unwrap_or_else(|| 1.0f32 / (head_dim as f32).sqrt());
        let seq_ptr = (&seq as *const u32).cast();
        let n_heads_ptr = (&n_heads as *const u32).cast();
        let n_kv_heads_ptr = (&n_kv_heads as *const u32).cast();
        let head_dim_ptr = (&head_dim as *const u32).cast();
        let scale_ptr = (&scale as *const f32).cast();
        let soft_cap_val = soft_cap.unwrap_or(0.0f32);
        let soft_cap_ptr = (&soft_cap_val as *const f32).cast();

        enc.set_compute_pipeline_state(&self.pso_attn_single_gqa_f32);
        enc.set_buffer(0, Some(&self.k), 0);
        enc.set_buffer(1, Some(&self.v), 0);
        enc.set_buffer(2, Some(bases_buf), bases_offset);
        enc.set_buffer(3, Some(q_buf), 0);
        enc.set_buffer(4, Some(out_buf), 0);
        enc.set_bytes(5, std::mem::size_of::<u32>() as u64, n_heads_ptr);
        enc.set_bytes(6, std::mem::size_of::<u32>() as u64, n_kv_heads_ptr);
        enc.set_bytes(7, std::mem::size_of::<u32>() as u64, head_dim_ptr);
        enc.set_bytes(8, std::mem::size_of::<u32>() as u64, seq_ptr);
        enc.set_bytes(9, std::mem::size_of::<f32>() as u64, scale_ptr);
        enc.set_bytes(10, std::mem::size_of::<f32>() as u64, soft_cap_ptr);

        let threads_per_tg = 32;
        let tg = metal::MTLSize { width: threads_per_tg, height: 1, depth: 1 };
        let grid = metal::MTLSize { width: n_heads as u64, height: 1, depth: 1 };
        enc.dispatch_thread_groups(grid, tg);
    }
    
    pub fn encode_write_token_f32(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        base: usize,
        k_buf: &metal::BufferRef,
        v_buf: &metal::BufferRef,
        kv_dim: usize,
    ) {
        if self.encoding == KvEncodingKind::TurboQuant {
            panic!("Llama fused graph + kv-encoding=turboquant is not implemented yet (graph-side quantized write path missing)");
        }
        let items = kv_dim as u32;
        let items_ptr = (&items as *const u32).cast();
        let base_u32 = base as u32;
        let base_ptr = (&base_u32 as *const u32).cast();

        enc.set_compute_pipeline_state(&self.pso_write_f32);
        enc.set_buffer(0, Some(&self.k), 0);
        enc.set_buffer(1, Some(&self.v), 0);
        enc.set_buffer(2, Some(k_buf), 0);
        enc.set_buffer(3, Some(v_buf), 0);
        enc.set_bytes(4, 4, base_ptr);
        enc.set_bytes(5, 4, items_ptr);
        
        let threads = kv_dim as u64;
        let w = self.pso_write_f32.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }
}
