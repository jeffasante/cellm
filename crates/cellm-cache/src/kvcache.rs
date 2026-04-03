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
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Debug)]
pub struct MetalKvStorage {
    queue: CommandQueue,
    _lib: Library,
    pso_write_f16: ComputePipelineState,
    pso_write_f32: ComputePipelineState,
    pso_read_f16: ComputePipelineState,
    pso_read_f32: ComputePipelineState,
    pso_gather_f32: ComputePipelineState,
    pso_attn_single_gqa_f32: ComputePipelineState,
    k: Buffer,
    v: Buffer,
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
    pub fn new(total_elems: usize) -> Result<Self, CoreError> {
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
        Ok(Self {
            queue,
            _lib: lib,
            pso_write_f16,
            pso_write_f32,
            pso_read_f16,
            pso_read_f32,
            pso_gather_f32,
            pso_attn_single_gqa_f32,
            k,
            v,
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
    pub fn new(_total_elems: usize) -> Result<Self, CoreError> {
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
}

impl KVCache {
    pub fn new(layout: KvCacheLayout) -> Result<Self, CoreError> {
        Self::new_with_kind(layout, KvStorageKind::Cpu)
    }

    pub fn new_with_kind(layout: KvCacheLayout, kind: KvStorageKind) -> Result<Self, CoreError> {
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
        let storage: Box<dyn DeviceKvStorage> = match kind {
            KvStorageKind::Cpu => Box::new(CpuKvStorage {
                k: vec![f16::from_f32(0.0); total_elems],
                v: vec![f16::from_f32(0.0); total_elems],
            }),
            KvStorageKind::Metal => Box::new(MetalKvStorage::new(total_elems)?),
        };

        Ok(Self {
            allocator: BlockAllocator::new(layout.total_blocks),
            layout,
            storage,
        })
    }

    pub fn layout(&self) -> KvCacheLayout {
        self.layout
    }

    pub fn storage(&self) -> &dyn DeviceKvStorage {
        self.storage.as_ref()
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
