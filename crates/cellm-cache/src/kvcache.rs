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
    k_q: Vec<i8>,
    v_q: Vec<i8>,
    k_scale: Vec<f32>,
    v_scale: Vec<f32>,
    kv_dim: usize,
}

impl CpuTurboQuantKvStorage {
    fn new(total_elems: usize, kv_dim: usize) -> Result<Self, CoreError> {
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
        Ok(Self {
            k_q: vec![0i8; total_elems],
            v_q: vec![0i8; total_elems],
            k_scale: vec![1.0f32; tokens],
            v_scale: vec![1.0f32; tokens],
            kv_dim,
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
        let end = base.saturating_add(len);
        if end > self.k_q.len() || end > self.v_q.len() {
            return Err(CoreError::Backend(format!(
                "kv cache cpu turboquant: out of bounds base={base} len={len} cap={}",
                self.k_q.len()
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
    fn quantize_row(src: &[f32], dst: &mut [i8]) -> f32 {
        debug_assert_eq!(src.len(), dst.len());
        let mut max_abs = 0.0f32;
        for &x in src {
            let ax = x.abs();
            if ax > max_abs {
                max_abs = ax;
            }
        }
        if max_abs <= 0.0 {
            dst.fill(0);
            return 1.0;
        }
        let scale = max_abs / 127.0f32;
        let inv = 1.0f32 / scale;
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            let q = (s * inv).round().clamp(-127.0, 127.0);
            *d = q as i8;
        }
        scale
    }

    #[inline]
    fn quantize_row_f16(src: &[f16], dst: &mut [i8]) -> f32 {
        debug_assert_eq!(src.len(), dst.len());
        let mut max_abs = 0.0f32;
        for &x in src {
            let ax = x.to_f32().abs();
            if ax > max_abs {
                max_abs = ax;
            }
        }
        if max_abs <= 0.0 {
            dst.fill(0);
            return 1.0;
        }
        let scale = max_abs / 127.0f32;
        let inv = 1.0f32 / scale;
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            let q = (s.to_f32() * inv).round().clamp(-127.0, 127.0);
            *d = q as i8;
        }
        scale
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

        let scale = 1.0f32 / (head_dim as f32).sqrt();
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
                scores[t] = dot * scale;
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
        let tok = self.token_index(base, len)?;
        let k_dst = &mut self.k_q[base..base + len];
        let v_dst = &mut self.v_q[base..base + len];
        self.k_scale[tok] = Self::quantize_row_f16(k_src, k_dst);
        self.v_scale[tok] = Self::quantize_row_f16(v_src, v_dst);
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
        let tok = self.token_index(base, len)?;
        let k_dst = &mut self.k_q[base..base + len];
        let v_dst = &mut self.v_q[base..base + len];
        self.k_scale[tok] = Self::quantize_row(k_src, k_dst);
        self.v_scale[tok] = Self::quantize_row(v_src, v_dst);
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
        let k_src = &self.k_q[base..base + len];
        let v_src = &self.v_q[base..base + len];
        for i in 0..len {
            k_out[i] = f16::from_f32(k_src[i] as f32 * ks);
            v_out[i] = f16::from_f32(v_src[i] as f32 * vs);
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
        let k_src = &self.k_q[base..base + len];
        let v_src = &self.v_q[base..base + len];
        for i in 0..len {
            k_out[i] = k_src[i] as f32 * ks;
            v_out[i] = v_src[i] as f32 * vs;
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
            for i in 0..kv_dim {
                k_out[dst + i] = self.k_q[base + i] as f32 * ks;
                v_out[dst + i] = self.v_q[base + i] as f32 * vs;
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
        let kv_dim = n_kv_heads.saturating_mul(head_dim);
        for &base in bases {
            let _ = self.token_index(base, kv_dim)?;
        }

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let group_size = (n_heads / n_kv_heads).max(1);
        let mut scores = vec![0.0f32; seq];

        for h in 0..n_heads {
            let kv_h = (h / group_size).min(n_kv_heads.saturating_sub(1));
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            for (t, &base) in bases.iter().enumerate() {
                let tok = base / self.kv_dim;
                let ks = self.k_scale[tok];
                let kt_base = base + kv_h * head_dim;
                let kt = &self.k_q[kt_base..kt_base + head_dim];
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += qh[i] * (kt[i] as f32 * ks);
                }
                scores[t] = dot * scale;
            }
            softmax_f32_inplace_cpu_local(&mut scores);

            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            for (t, &base) in bases.iter().enumerate() {
                let tok = base / self.kv_dim;
                let vs = self.v_scale[tok];
                let vt_base = base + kv_h * head_dim;
                let vt = &self.v_q[vt_base..vt_base + head_dim];
                let w = scores[t];
                for i in 0..head_dim {
                    out_h[i] += w * (vt[i] as f32 * vs);
                }
            }
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

#[inline]
fn quantize_row_f32_to_i8_scale(src: &[f32], dst: &mut [i8]) -> f32 {
    debug_assert_eq!(src.len(), dst.len());
    let mut max_abs = 0.0f32;
    for &x in src {
        let ax = x.abs();
        if ax > max_abs {
            max_abs = ax;
        }
    }
    if max_abs <= 0.0 {
        dst.fill(0);
        return 1.0;
    }
    let scale = max_abs / 127.0f32;
    let inv = 1.0f32 / scale;
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let q = (s * inv).round().clamp(-127.0, 127.0);
        *d = q as i8;
    }
    scale
}

#[inline]
fn quantize_row_f16_to_i8_scale(src: &[f16], dst: &mut [i8]) -> f32 {
    debug_assert_eq!(src.len(), dst.len());
    let mut max_abs = 0.0f32;
    for &x in src {
        let ax = x.to_f32().abs();
        if ax > max_abs {
            max_abs = ax;
        }
    }
    if max_abs <= 0.0 {
        dst.fill(0);
        return 1.0;
    }
    let scale = max_abs / 127.0f32;
    let inv = 1.0f32 / scale;
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let q = (s.to_f32() * inv).round().clamp(-127.0, 127.0);
        *d = q as i8;
    }
    scale
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Debug)]
pub struct MetalKvStorage {
    encoding: KvEncodingKind,
    kv_dim: usize,
    queue: CommandQueue,
    _lib: Library,
    pso_write_f16: ComputePipelineState,
    pso_write_f32: ComputePipelineState,
    pso_read_f16: ComputePipelineState,
    pso_read_f32: ComputePipelineState,
    pso_gather_f32: ComputePipelineState,
    pso_attn_single_gqa_f32: ComputePipelineState,
    pso_attn_single_gqa_q8_f32: ComputePipelineState,
    k: Buffer,
    v: Buffer,
    k_q: Option<Buffer>,
    v_q: Option<Buffer>,
    k_scale: Option<Buffer>,
    v_scale: Option<Buffer>,
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
    pub fn new(total_elems: usize, kv_dim: usize, encoding: KvEncodingKind) -> Result<Self, CoreError> {
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
            k_cache[idx] = half(k_in[gid]);
            v_cache[idx] = half(v_in[gid]);
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
            constant uint& seq [[buffer(5)]],
            constant uint& n_heads [[buffer(6)]],
            constant uint& n_kv_heads [[buffer(7)]],
            constant uint& head_dim [[buffer(8)]],
            constant float& scale [[buffer(9)]],
            uint h [[thread_position_in_grid]]
        ) {
            if (h >= n_heads) return;
            uint group_size = max((uint)1, n_heads / max((uint)1, n_kv_heads));
            uint kv_h = min(h / group_size, max((uint)0, n_kv_heads - 1));
            const device float* qh = q + h * head_dim;

            float max_score = -INFINITY;
            for (uint t = 0; t < seq; ++t) {
                uint base = bases[t] + kv_h * head_dim;
                float dot = 0.0f;
                for (uint i = 0; i < head_dim; ++i) {
                    dot += qh[i] * float(k_cache[base + i]);
                }
                float s = dot * scale;
                if (s > max_score) max_score = s;
            }

            float denom = 0.0f;
            for (uint t = 0; t < seq; ++t) {
                uint base = bases[t] + kv_h * head_dim;
                float dot = 0.0f;
                for (uint i = 0; i < head_dim; ++i) {
                    dot += qh[i] * float(k_cache[base + i]);
                }
                denom += exp(dot * scale - max_score);
            }
            denom = max(denom, 1e-12f);

            for (uint i = 0; i < head_dim; ++i) {
                out[h * head_dim + i] = 0.0f;
            }
            for (uint t = 0; t < seq; ++t) {
                uint base = bases[t] + kv_h * head_dim;
                float dot = 0.0f;
                for (uint i = 0; i < head_dim; ++i) {
                    dot += qh[i] * float(k_cache[base + i]);
                }
                float w = exp(dot * scale - max_score) / denom;
                for (uint i = 0; i < head_dim; ++i) {
                    out[h * head_dim + i] += w * float(v_cache[base + i]);
                }
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
                denom += exp(dot * scale - max_score);
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
                float w = exp(dot * scale - max_score) / denom;
                for (uint i = 0; i < head_dim; ++i) {
                    out[h * head_dim + i] += w * (float(v_q[base + i]) * vs);
                }
            }
        }
        "#;
        let options = metal::CompileOptions::new();
        let lib = device
            .new_library_with_source(src, &options)
            .map_err(|e| CoreError::Backend(format!("kv cache metal storage: compile failed: {e:?}")))?;
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
        let (k_q, v_q, k_scale, v_scale) = if encoding == KvEncodingKind::TurboQuant {
            let bytes_q = total_elems as u64;
            let tokens = total_elems / kv_dim;
            let bytes_s = (tokens * std::mem::size_of::<f32>()) as u64;
            let kq = device.new_buffer(bytes_q, MTLResourceOptions::StorageModeShared);
            let vq = device.new_buffer(bytes_q, MTLResourceOptions::StorageModeShared);
            let ks = device.new_buffer(bytes_s, MTLResourceOptions::StorageModeShared);
            let vs = device.new_buffer(bytes_s, MTLResourceOptions::StorageModeShared);
            let kq_ptr = kq.contents() as *mut i8;
            let vq_ptr = vq.contents() as *mut i8;
            let ks_ptr = ks.contents() as *mut f32;
            let vs_ptr = vs.contents() as *mut f32;
            if kq_ptr.is_null() || vq_ptr.is_null() || ks_ptr.is_null() || vs_ptr.is_null() {
                return Err(CoreError::Backend(
                    "kv cache metal storage: null turboquant buffer contents".into(),
                ));
            }
            unsafe {
                std::slice::from_raw_parts_mut(kq_ptr, total_elems).fill(0);
                std::slice::from_raw_parts_mut(vq_ptr, total_elems).fill(0);
                std::slice::from_raw_parts_mut(ks_ptr, tokens).fill(1.0f32);
                std::slice::from_raw_parts_mut(vs_ptr, tokens).fill(1.0f32);
            }
            (Some(kq), Some(vq), Some(ks), Some(vs))
        } else {
            (None, None, None, None)
        };
        Ok(Self {
            encoding,
            kv_dim,
            queue,
            _lib: lib,
            pso_write_f16,
            pso_write_f32,
            pso_read_f16,
            pso_read_f32,
            pso_gather_f32,
            pso_attn_single_gqa_f32,
            pso_attn_single_gqa_q8_f32,
            k,
            v,
            k_q,
            v_q,
            k_scale,
            v_scale,
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
            self.check_range(base, k_src.len())?;
            if base % self.kv_dim != 0 || k_src.len() != self.kv_dim {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: write must be token-aligned (base={base}, len={}, kv_dim={})",
                    k_src.len(),
                    self.kv_dim
                )));
            }
            let tok = base / self.kv_dim;
            let kq = self.k_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_q".into()))?;
            let vq = self.v_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_q".into()))?;
            let ks = self.k_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_scale".into()))?;
            let vs = self.v_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_scale".into()))?;
            let k_ptr = kq.contents() as *mut i8;
            let v_ptr = vq.contents() as *mut i8;
            let ks_ptr = ks.contents() as *mut f32;
            let vs_ptr = vs.contents() as *mut f32;
            if k_ptr.is_null() || v_ptr.is_null() || ks_ptr.is_null() || vs_ptr.is_null() {
                return Err(CoreError::Backend("kv cache metal turboquant: null buffer contents".into()));
            }
            let k_dst = unsafe { std::slice::from_raw_parts_mut(k_ptr.add(base), k_src.len()) };
            let v_dst = unsafe { std::slice::from_raw_parts_mut(v_ptr.add(base), v_src.len()) };
            let k_scale = quantize_row_f16_to_i8_scale(k_src, k_dst);
            let v_scale = quantize_row_f16_to_i8_scale(v_src, v_dst);
            unsafe {
                *ks_ptr.add(tok) = k_scale;
                *vs_ptr.add(tok) = v_scale;
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
            self.check_range(base, k_src.len())?;
            if base % self.kv_dim != 0 || k_src.len() != self.kv_dim {
                return Err(CoreError::Backend(format!(
                    "kv cache metal turboquant: write must be token-aligned (base={base}, len={}, kv_dim={})",
                    k_src.len(),
                    self.kv_dim
                )));
            }
            let tok = base / self.kv_dim;
            let kq = self.k_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_q".into()))?;
            let vq = self.v_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_q".into()))?;
            let ks = self.k_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_scale".into()))?;
            let vs = self.v_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_scale".into()))?;
            let k_ptr = kq.contents() as *mut i8;
            let v_ptr = vq.contents() as *mut i8;
            let ks_ptr = ks.contents() as *mut f32;
            let vs_ptr = vs.contents() as *mut f32;
            if k_ptr.is_null() || v_ptr.is_null() || ks_ptr.is_null() || vs_ptr.is_null() {
                return Err(CoreError::Backend("kv cache metal turboquant: null buffer contents".into()));
            }
            let k_dst = unsafe { std::slice::from_raw_parts_mut(k_ptr.add(base), k_src.len()) };
            let v_dst = unsafe { std::slice::from_raw_parts_mut(v_ptr.add(base), v_src.len()) };
            let k_scale = quantize_row_f32_to_i8_scale(k_src, k_dst);
            let v_scale = quantize_row_f32_to_i8_scale(v_src, v_dst);
            unsafe {
                *ks_ptr.add(tok) = k_scale;
                *vs_ptr.add(tok) = v_scale;
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
            let kq = self.k_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_q".into()))?;
            let vq = self.v_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_q".into()))?;
            let ks = self.k_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_scale".into()))?;
            let vs = self.v_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_scale".into()))?;
            let k_ptr = kq.contents() as *const i8;
            let v_ptr = vq.contents() as *const i8;
            let ks_ptr = ks.contents() as *const f32;
            let vs_ptr = vs.contents() as *const f32;
            if k_ptr.is_null() || v_ptr.is_null() || ks_ptr.is_null() || vs_ptr.is_null() {
                return Err(CoreError::Backend("kv cache metal turboquant: null buffer contents".into()));
            }
            let k_src = unsafe { std::slice::from_raw_parts(k_ptr.add(base), len) };
            let v_src = unsafe { std::slice::from_raw_parts(v_ptr.add(base), len) };
            let ks_val = unsafe { *ks_ptr.add(tok) };
            let vs_val = unsafe { *vs_ptr.add(tok) };
            for i in 0..len {
                k_out[i] = f16::from_f32(k_src[i] as f32 * ks_val);
                v_out[i] = f16::from_f32(v_src[i] as f32 * vs_val);
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
            let kq = self.k_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_q".into()))?;
            let vq = self.v_q.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_q".into()))?;
            let ks = self.k_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_scale".into()))?;
            let vs = self.v_scale.as_ref().ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_scale".into()))?;
            let k_ptr = kq.contents() as *const i8;
            let v_ptr = vq.contents() as *const i8;
            let ks_ptr = ks.contents() as *const f32;
            let vs_ptr = vs.contents() as *const f32;
            if k_ptr.is_null() || v_ptr.is_null() || ks_ptr.is_null() || vs_ptr.is_null() {
                return Err(CoreError::Backend("kv cache metal turboquant: null buffer contents".into()));
            }
            let k_src = unsafe { std::slice::from_raw_parts(k_ptr.add(base), len) };
            let v_src = unsafe { std::slice::from_raw_parts(v_ptr.add(base), len) };
            let ks_val = unsafe { *ks_ptr.add(tok) };
            let vs_val = unsafe { *vs_ptr.add(tok) };
            for i in 0..len {
                k_out[i] = k_src[i] as f32 * ks_val;
                v_out[i] = v_src[i] as f32 * vs_val;
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
            let k_ptr = kq.contents() as *const i8;
            let v_ptr = vq.contents() as *const i8;
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
                let k_scale = unsafe { *ks_ptr.add(tok) };
                let v_scale = unsafe { *vs_ptr.add(tok) };
                let dst = t * kv_dim;
                for i in 0..kv_dim {
                    let kqv = unsafe { *k_ptr.add(base + i) } as f32;
                    let vqv = unsafe { *v_ptr.add(base + i) } as f32;
                    k_out[dst + i] = kqv * k_scale;
                    v_out[dst + i] = vqv * v_scale;
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
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let seq_ptr = (&seq_u32 as *const u32).cast();
        let n_heads_ptr = (&n_heads_u32 as *const u32).cast();
        let n_kv_heads_ptr = (&n_kv_heads_u32 as *const u32).cast();
        let head_dim_ptr = (&head_dim_u32 as *const u32).cast();
        let scale_ptr = (&scale as *const f32).cast();
        if self.encoding == KvEncodingKind::TurboQuant {
            let kv_dim_u32 = self.check_u32(self.kv_dim, "kv_dim")?;
            let kv_dim_ptr = (&kv_dim_u32 as *const u32).cast();
            let k_q = self
                .k_q
                .as_ref()
                .ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing k_q".into()))?;
            let v_q = self
                .v_q
                .as_ref()
                .ok_or_else(|| CoreError::Backend("kv cache metal turboquant: missing v_q".into()))?;
            let k_scale = self.k_scale.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing k_scale".into())
            })?;
            let v_scale = self.v_scale.as_ref().ok_or_else(|| {
                CoreError::Backend("kv cache metal turboquant: missing v_scale".into())
            })?;
            dispatch_1d(
                &self.queue,
                &self.pso_attn_single_gqa_q8_f32,
                n_heads as u64,
                |enc| {
                    enc.set_buffer(0, Some(k_q), 0);
                    enc.set_buffer(1, Some(v_q), 0);
                    enc.set_buffer(2, Some(k_scale), 0);
                    enc.set_buffer(3, Some(v_scale), 0);
                    enc.set_buffer(4, Some(&bases_buf), 0);
                    enc.set_buffer(5, Some(&q_buf), 0);
                    enc.set_buffer(6, Some(&out_buf), 0);
                    enc.set_bytes(7, std::mem::size_of::<u32>() as u64, seq_ptr);
                    enc.set_bytes(8, std::mem::size_of::<u32>() as u64, n_heads_ptr);
                    enc.set_bytes(9, std::mem::size_of::<u32>() as u64, n_kv_heads_ptr);
                    enc.set_bytes(10, std::mem::size_of::<u32>() as u64, head_dim_ptr);
                    enc.set_bytes(11, std::mem::size_of::<u32>() as u64, kv_dim_ptr);
                    enc.set_bytes(12, std::mem::size_of::<f32>() as u64, scale_ptr);
                },
            )
            .map_err(|e| CoreError::Backend(format!("kv cache metal attn gqa q8 failed: {e}")))?;
        } else {
            dispatch_1d(&self.queue, &self.pso_attn_single_gqa_f32, n_heads as u64, |enc| {
                enc.set_buffer(0, Some(&self.k), 0);
                enc.set_buffer(1, Some(&self.v), 0);
                enc.set_buffer(2, Some(&bases_buf), 0);
                enc.set_buffer(3, Some(&q_buf), 0);
                enc.set_buffer(4, Some(&out_buf), 0);
                enc.set_bytes(5, std::mem::size_of::<u32>() as u64, seq_ptr);
                enc.set_bytes(6, std::mem::size_of::<u32>() as u64, n_heads_ptr);
                enc.set_bytes(7, std::mem::size_of::<u32>() as u64, n_kv_heads_ptr);
                enc.set_bytes(8, std::mem::size_of::<u32>() as u64, head_dim_ptr);
                enc.set_bytes(9, std::mem::size_of::<f32>() as u64, scale_ptr);
            })
            .map_err(|e| CoreError::Backend(format!("kv cache metal attn gqa failed: {e}")))?;
        }

        read_buf_f32(out_buf, out, "out_f32")?;
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
    pub fn new(_total_elems: usize, _kv_dim: usize, _encoding: KvEncodingKind) -> Result<Self, CoreError> {
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
                Box::new(CpuTurboQuantKvStorage::new(total_elems, layout.kv_dim())?)
            }
            (KvStorageKind::Metal, enc) => Box::new(MetalKvStorage::new(total_elems, layout.kv_dim(), enc)?),
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
    ) {
        if self.encoding == KvEncodingKind::TurboQuant {
            panic!("Llama fused graph + kv-encoding=turboquant is not implemented yet (graph-side quantized write path missing)");
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let seq_ptr = (&seq as *const u32).cast();
        let n_heads_ptr = (&n_heads as *const u32).cast();
        let n_kv_heads_ptr = (&n_kv_heads as *const u32).cast();
        let head_dim_ptr = (&head_dim as *const u32).cast();
        let scale_ptr = (&scale as *const f32).cast();

        enc.set_compute_pipeline_state(&self.pso_attn_single_gqa_f32);
        enc.set_buffer(0, Some(&self.k), 0);
        enc.set_buffer(1, Some(&self.v), 0);
        enc.set_buffer(2, Some(bases_buf), bases_offset);
        enc.set_buffer(3, Some(q_buf), 0);
        enc.set_buffer(4, Some(out_buf), 0);
        enc.set_bytes(5, std::mem::size_of::<u32>() as u64, seq_ptr);
        enc.set_bytes(6, std::mem::size_of::<u32>() as u64, n_heads_ptr);
        enc.set_bytes(7, std::mem::size_of::<u32>() as u64, n_kv_heads_ptr);
        enc.set_bytes(8, std::mem::size_of::<u32>() as u64, head_dim_ptr);
        enc.set_bytes(9, std::mem::size_of::<f32>() as u64, scale_ptr);

        let threads = n_heads as u64;
        let w = self.pso_attn_single_gqa_f32.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
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
