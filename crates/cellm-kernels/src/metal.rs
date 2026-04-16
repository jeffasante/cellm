#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::rc::autoreleasepool;
use std::sync::Mutex;
use std::collections::HashMap;

pub struct MetalKernels;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub type MetalBuffer = Buffer;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalBuffer;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalMatmul {
    pub queue: CommandQueue,
    pub _lib: Library,
    pub pso: ComputePipelineState,
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalMatmul;

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalKernels {
    pub fn smoke_test_add_f32() -> anyhow::Result<()> {
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Metal: no device found (system_default and all() empty). \
                     If you are in a restricted/sandboxed shell, re-run outside sandbox."
                )
            })?;
        let queue = device.new_command_queue();

        let src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void add_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            out[id] = a[id] + b[id];
        }
        "#;

        let (lib, pso) = build_pipeline(&device, src, "add_f32")?;

        let n = 1024usize;
        let bytes = (n * std::mem::size_of::<f32>()) as u64;
        let a = upload_f32(&device, &(0..n).map(|i| i as f32).collect::<Vec<_>>())?;
        let b = upload_f32(&device, &(0..n).map(|i| (2 * i) as f32).collect::<Vec<_>>())?;
        let out = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&a), 0);
        enc.set_buffer(1, Some(&b), 0);
        enc.set_buffer(2, Some(&out), 0);
        let w = pso.thread_execution_width() as u64;
        let tg = MTLSize { width: w.min(n as u64), height: 1, depth: 1 };
        let grid = MTLSize { width: n as u64, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Validate.
        let out_ptr = out.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, n) };
        for i in 0..n {
            let expected = (i as f32) + (2 * i) as f32;
            let got = out_slice[i];
            if (got - expected).abs() > 1e-5 {
                anyhow::bail!("Metal add_f32 mismatch at {i}: got={got} expected={expected}");
            }
        }

        let _ = lib;
        Ok(())
    }

    pub fn create_matmul() -> anyhow::Result<MetalMatmul> {
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| anyhow::anyhow!("Metal: no device found"))?;
        let queue = device.new_command_queue();
        let src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void matmul_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            constant uint& m [[buffer(3)]],
            constant uint& n [[buffer(4)]],
            constant uint& k [[buffer(5)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint row = gid.y;
            uint col = gid.x;
            if (row >= m || col >= n) return;
            float acc = 0.0f;
            for (uint kk = 0; kk < k; ++kk) {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            out[row * n + col] = acc;
        }
        "#;
        let (lib, pso) = build_pipeline(&device, src, "matmul_f32")?;
        Ok(MetalMatmul {
            queue,
            _lib: lib,
            pso,
        })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalKernels {
    pub fn smoke_test_add_f32() -> anyhow::Result<()> {
        anyhow::bail!("MetalKernels only supported on macOS/iOS")
    }
    pub fn create_matmul() -> anyhow::Result<MetalMatmul> {
        anyhow::bail!("Metal matmul only supported on Apple platforms")
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalMatmul {
    pub fn upload_f32(&self, src: &[f32]) -> anyhow::Result<MetalBuffer> {
        upload_f32(self.queue.device(), src)
    }

    pub fn matmul_row_major_f32(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        if a.len() != m * k || b.len() != k * n || out.len() != m * n {
            anyhow::bail!("Metal matmul shape mismatch");
        }
        autoreleasepool(|| {
            let device = self.queue.device();
            let a_buf = upload_f32(device, a)?;
            let b_buf = upload_f32(device, b)?;
            self.matmul_row_major_f32_with_b_buffer(&a_buf, m, k, &b_buf, n, out)
        })
    }

    pub fn matmul_row_major_f32_with_b_buffer(
        &self,
        a_buf: &MetalBuffer,
        m: usize,
        k: usize,
        b_buf: &MetalBuffer,
        n: usize,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        if out.len() != m * n { anyhow::bail!("Metal matmul output size mismatch"); }
        autoreleasepool(|| {
            let out_bytes = (out.len() * 4) as u64;
            let device = self.queue.device();
            let out_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
            
            let m_u32 = m as u32; let n_u32 = n as u32; let k_u32 = k as u32;

            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso);
            enc.set_buffer(0, Some(a_buf), 0);
            enc.set_buffer(1, Some(b_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, 4, (&m_u32 as *const u32).cast());
            enc.set_bytes(4, 4, (&n_u32 as *const u32).cast());
            enc.set_bytes(5, 4, (&k_u32 as *const u32).cast());
            
            let w = self.pso.thread_execution_width() as u64;
            let h = self.pso.max_total_threads_per_threadgroup() as u64 / w;
            let tg = MTLSize { width: w, height: h, depth: 1 };
            let grid = MTLSize { width: n as u64, height: m as u64, depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let out_ptr = out_buf.contents() as *const f32;
            out.copy_from_slice(unsafe { std::slice::from_raw_parts(out_ptr, out.len()) });
            Ok(())
        })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalMatmul {
    pub fn upload_f32(&self, _: &[f32]) -> anyhow::Result<MetalBuffer> { anyhow::bail!("No Metal") }
    pub fn matmul_row_major_f32(&self, _: &[f32], _: usize, _: usize, _: &[f32], _: usize, _: &mut [f32]) -> anyhow::Result<()> { anyhow::bail!("No Metal") }
    pub fn matmul_row_major_f32_with_b_buffer(&self, _: &MetalBuffer, _: usize, _: usize, _: &MetalBuffer, _: usize, _: &mut [f32]) -> anyhow::Result<()> { anyhow::bail!("No Metal") }
}

const ELEM_OPS_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32_inplace(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) a[gid] += b[gid];
}

kernel void silu_mul_f32_inplace(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        float x = a[gid];
        float silu = x / (1.0f + exp(-x));
        a[gid] = silu * b[gid];
    }
}

kernel void rms_norm_f16w(
    device const float*  x         [[buffer(0)]],
    device const ushort* w         [[buffer(1)]],
    device       float*  out       [[buffer(2)]],
    constant     uint&   n         [[buffer(3)]],
    constant     float&  eps       [[buffer(4)]],
    constant     uint&   w_add_one [[buffer(5)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgsize[[threads_per_threadgroup]],
    threadgroup float* shared[[threadgroup(0)]]
) {
    float local_sq = 0.0f;
    for (uint i = tid; i < n; i += tgsize) {
        float v = x[i];
        local_sq += v * v;
    }
    float inv_n = 1.0f / float(n);
    shared[tid] = (tid < tgsize) ? (local_sq * inv_n) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared[0] + eps);
    for (uint i = tid; i < n; i += tgsize) {
        float wf = float(as_type<half>(w[i]));
        if (w_add_one) wf += 1.0f;
        out[i] = x[i] * inv_rms * wf;
    }
}

kernel void rope_adj_f32(
    device       float* x         [[buffer(0)]],
    constant     uint&  n_heads   [[buffer(1)]],
    constant     uint&  head_dim  [[buffer(2)]],
    constant     uint&  pos       [[buffer(3)]],
    constant     float& theta     [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_hd = head_dim >> 1;
    uint h = gid / half_hd;
    uint dim = gid % half_hd;
    if (h >= n_heads) return;
    float inv_freq = pow(theta, -(2.0f * float(dim)) / float(head_dim));
    float angle = float(pos) * inv_freq;
    float c = cos(angle); float s = sin(angle);
    uint base = h * head_dim + dim * 2;
    float x0 = x[base]; float x1 = x[base + 1];
    x[base]     = x0 * c - x1 * s;
    x[base + 1] = x0 * s + x1 * c;
}

kernel void rope_half_f32(
    device       float* x           [[buffer(0)]],
    constant     uint&  n_heads     [[buffer(1)]],
    constant     uint&  head_dim    [[buffer(2)]],
    constant     uint&  rotary_dim  [[buffer(3)]],
    constant     uint&  pos         [[buffer(4)]],
    constant     float& theta       [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_rd = rotary_dim >> 1;
    uint h = gid / half_rd;
    uint i = gid % half_rd;
    if (h >= n_heads) return;
    float inv_freq = pow(theta, -(2.0f * float(i)) / float(rotary_dim));
    float angle = float(pos) * inv_freq;
    float c = cos(angle); float s = sin(angle);
    uint base = h * head_dim;
    float x0 = x[base + i];
    float x1 = x[base + half_rd + i];
    x[base + i]           = x0 * c - x1 * s;
    x[base + half_rd + i] = x1 * c + x0 * s;
}

kernel void mv_f16(
    device const ushort* A       [[buffer(0)]],
    device const float*  x       [[buffer(1)]],
    device const float*  bias    [[buffer(2)]],
    device       float*  out     [[buffer(3)]],
    constant     uint&   rows    [[buffer(4)]],
    constant     uint&   cols    [[buffer(5)]],
    constant     uint&   has_bias [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    device const ushort* row_ptr = A + gid * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
    if (has_bias) acc += bias[gid];
    out[gid] = acc;
}

kernel void mv_i8(
    device const char*   A       [[buffer(0)]],
    device const ushort* scl     [[buffer(1)]],
    device const float*  x       [[buffer(2)]],
    device const float*  bias    [[buffer(3)]],
    device       float*  out     [[buffer(4)]],
    constant     uint&   rows    [[buffer(5)]],
    constant     uint&   cols    [[buffer(6)]],
    constant     uint&   has_bias [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    float scale = float(as_type<half>(scl[gid]));
    device const char* row_ptr = A + gid * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) acc += x[i] * float(row_ptr[i]);
    float res = acc * scale;
    if (has_bias) res += bias[gid];
    out[gid] = res;
}

kernel void mv_qkv_i8(
    device const char*   w_q      [[buffer(0)]],
    device const ushort* s_q      [[buffer(1)]],
    device const char*   w_k      [[buffer(2)]],
    device const ushort* s_k      [[buffer(3)]],
    device const char*   w_v      [[buffer(4)]],
    device const ushort* s_v      [[buffer(5)]],
    device const float*  x        [[buffer(6)]],
    device const float*  b_q      [[buffer(7)]],
    device const float*  b_k      [[buffer(8)]],
    device const float*  b_v      [[buffer(9)]],
    device       float*  q_out    [[buffer(10)]],
    device       float*  k_out    [[buffer(11)]],
    device       float*  v_out    [[buffer(12)]],
    constant     uint&   rows_q   [[buffer(13)]],
    constant     uint&   rows_kv  [[buffer(14)]],
    constant     uint&   cols     [[buffer(15)]],
    constant     uint&   has_bias [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = rows_q + rows_kv + rows_kv;
    if (gid >= total) return;
    if (gid < rows_q) {
        float scale = float(as_type<half>(s_q[gid]));
        device const char* row_ptr = w_q + gid * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(row_ptr[i]);
        float res = acc * scale;
        if (has_bias) res += b_q[gid];
        q_out[gid] = res;
        return;
    }
    uint off = gid - rows_q;
    if (off < rows_kv) {
        float scale = float(as_type<half>(s_k[off]));
        device const char* row_ptr = w_k + off * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(row_ptr[i]);
        float res = acc * scale;
        if (has_bias) res += b_k[off];
        k_out[off] = res;
        return;
    }
    off -= rows_kv;
    float scale = float(as_type<half>(s_v[off]));
    device const char* row_ptr = w_v + off * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) acc += x[i] * float(row_ptr[i]);
    float res = acc * scale;
    if (has_bias) res += b_v[off];
    v_out[off] = res;
}

kernel void mv_qkv_f16(
    device const ushort* w_q      [[buffer(0)]],
    device const ushort* w_k      [[buffer(1)]],
    device const ushort* w_v      [[buffer(2)]],
    device const float*  x        [[buffer(3)]],
    device const float*  b_q      [[buffer(4)]],
    device const float*  b_k      [[buffer(5)]],
    device const float*  b_v      [[buffer(6)]],
    device       float*  q_out    [[buffer(7)]],
    device       float*  k_out    [[buffer(8)]],
    device       float*  v_out    [[buffer(9)]],
    constant     uint&   rows_q   [[buffer(10)]],
    constant     uint&   rows_kv  [[buffer(11)]],
    constant     uint&   cols     [[buffer(12)]],
    constant     uint&   has_bias [[buffer(13)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = rows_q + rows_kv + rows_kv;
    if (gid >= total) return;
    if (gid < rows_q) {
        device const ushort* row_ptr = w_q + gid * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
        if (has_bias) acc += b_q[gid];
        q_out[gid] = acc;
        return;
    }
    uint off = gid - rows_q;
    if (off < rows_kv) {
        device const ushort* row_ptr = w_k + off * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
        if (has_bias) acc += b_k[off];
        k_out[off] = acc;
        return;
    }
    off -= rows_kv;
    device const ushort* row_ptr = w_v + off * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
    if (has_bias) acc += b_v[off];
    v_out[off] = acc;
}

kernel void mv2_f16(
    device const ushort* A0      [[buffer(0)]],
    device const ushort* A1      [[buffer(1)]],
    device const float*  x       [[buffer(2)]],
    device       float*  out0    [[buffer(3)]],
    device       float*  out1    [[buffer(4)]],
    constant     uint&   rows0   [[buffer(5)]],
    constant     uint&   rows1   [[buffer(6)]],
    constant     uint&   cols    [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = rows0 + rows1;
    if (gid >= total) return;
    if (gid < rows0) {
        device const ushort* row_ptr = A0 + gid * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
        out0[gid] = acc;
    } else {
        uint off = gid - rows0;
        device const ushort* row_ptr = A1 + off * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
        out1[off] = acc;
    }
}

kernel void mv2_i8(
    device const char*   w0   [[buffer(0)]],
    device const char*   w1   [[buffer(1)]],
    device const ushort* s0   [[buffer(2)]],
    device const ushort* s1   [[buffer(3)]],
    device const float*  x    [[buffer(4)]],
    device       float*  o0   [[buffer(5)]],
    device       float*  o1   [[buffer(6)]],
    constant     uint&   r0   [[buffer(7)]],
    constant     uint&   r1   [[buffer(8)]],
    constant     uint&   c    [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = r0 + r1;
    if (gid >= total) return;
    if (gid < r0) {
        float scale = float(as_type<half>(s0[gid]));
        device const char* row_ptr = w0 + gid * c;
        float acc = 0.0f;
        for (uint i = 0; i < c; ++i) acc += x[i] * float(row_ptr[i]);
        o0[gid] = acc * scale;
    } else {
        uint off = gid - r0;
        float scale = float(as_type<half>(s1[off]));
        device const char* row_ptr = w1 + off * c;
        float acc = 0.0f;
        for (uint i = 0; i < c; ++i) acc += x[i] * float(row_ptr[i]);
        o1[off] = acc * scale;
    }
}
"#;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalOps {
    pub device: Device,
    pub queue: CommandQueue,
    pub _lib: Library,
    pub pso_rms_norm: ComputePipelineState,
    pub pso_rope_adj: ComputePipelineState,
    pub pso_rope_half: ComputePipelineState,
    pub pso_mv_f16: ComputePipelineState,
    pub pso_mv_i8: ComputePipelineState,
    pub pso_mv_qkv_f16: ComputePipelineState,
    pub pso_mv_qkv_i8: ComputePipelineState,
    pub pso_mv2_f16: ComputePipelineState,
    pub pso_mv2_i8: ComputePipelineState,
    pub pso_add_f32: ComputePipelineState,
    pub pso_silu_mul_f32: ComputePipelineState,
    x_buf: Mutex<Option<Buffer>>,
    out_buf: Mutex<Option<Buffer>>,
    tensor_cache: Mutex<HashMap<String, Buffer>>,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl Clone for MetalOps {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            queue: self.queue.clone(),
            _lib: self._lib.clone(),
            pso_rms_norm: self.pso_rms_norm.clone(),
            pso_rope_adj: self.pso_rope_adj.clone(),
            pso_rope_half: self.pso_rope_half.clone(),
            pso_mv_f16: self.pso_mv_f16.clone(),
            pso_mv_i8: self.pso_mv_i8.clone(),
            pso_mv_qkv_f16: self.pso_mv_qkv_f16.clone(),
            pso_mv_qkv_i8: self.pso_mv_qkv_i8.clone(),
            pso_mv2_f16: self.pso_mv2_f16.clone(),
            pso_mv2_i8: self.pso_mv2_i8.clone(),
            pso_add_f32: self.pso_add_f32.clone(),
            pso_silu_mul_f32: self.pso_silu_mul_f32.clone(),
            x_buf: Mutex::new(None), // New scratch for new clone
            out_buf: Mutex::new(None),
            tensor_cache: Mutex::new(HashMap::new()), // Could share, but safer to have own cache
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalOps;

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalOps {
    pub fn create() -> anyhow::Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device"))?;
        let queue = device.new_command_queue();
        let options = metal::CompileOptions::new();
        let lib = device.new_library_with_source(ELEM_OPS_SHADER, &options).map_err(|e| anyhow::anyhow!("Compile failed: {e:?}"))?;
        
        let pso_rms_norm = build_pso_ops(&device, &lib, "rms_norm_f16w")?;
        let pso_rope_adj = build_pso_ops(&device, &lib, "rope_adj_f32")?;
        let pso_rope_half = build_pso_ops(&device, &lib, "rope_half_f32")?;
        let pso_mv_f16 = build_pso_ops(&device, &lib, "mv_f16")?;
        let pso_mv_i8 = build_pso_ops(&device, &lib, "mv_i8")?;
        let pso_mv_qkv_f16 = build_pso_ops(&device, &lib, "mv_qkv_f16")?;
        let pso_mv_qkv_i8 = build_pso_ops(&device, &lib, "mv_qkv_i8")?;
        let pso_mv2_f16 = build_pso_ops(&device, &lib, "mv2_f16")?;
        let pso_mv2_i8 = build_pso_ops(&device, &lib, "mv2_i8")?;
        let pso_add_f32 = build_pso_ops(&device, &lib, "add_f32_inplace")?;
        let pso_silu_mul_f32 = build_pso_ops(&device, &lib, "silu_mul_f32_inplace")?;

        Ok(Self {
            device, queue, _lib: lib,
            pso_rms_norm, pso_rope_adj, pso_rope_half, pso_mv_f16, pso_mv_i8, pso_mv_qkv_f16, pso_mv_qkv_i8, pso_mv2_f16, pso_mv2_i8,
            pso_add_f32, pso_silu_mul_f32,
            x_buf: Mutex::new(None), out_buf: Mutex::new(None),
            tensor_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn rms_norm_f16w(&self, x: &[f32], w_f16: &[u16], eps: f32, w_add_one: bool, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        let n = x.len();
        let w_key = format!("rmsnorm.w.{cache_key}");
        if !self.tensor_cache.lock().unwrap().contains_key(&w_key) {
            self.tensor_cache.lock().unwrap().insert(w_key.clone(), upload_u16(&self.device, w_f16)?);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let wb = cache.get(&w_key).unwrap();
        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), n)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.lock().unwrap(), n)?;
        let xb_lock = self.x_buf.lock().unwrap(); let xb = xb_lock.as_ref().unwrap();
        let ob_lock = self.out_buf.lock().unwrap(); let ob = ob_lock.as_ref().unwrap();
        write_f32(xb, x)?;
               autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_rms_norm_f16w(enc, xb, wb, ob, n, eps, w_add_one);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }

    pub fn encode_rms_norm_f16w(&self, enc: &metal::ComputeCommandEncoderRef, x: &Buffer, w: &Buffer, out: &Buffer, n: usize, eps: f32, w_add_one: bool) {
        let n32 = n as u32; let add = w_add_one as u32;
        enc.set_compute_pipeline_state(&self.pso_rms_norm);
        enc.set_buffer(0, Some(x), 0); enc.set_buffer(1, Some(w), 0); enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, (&n32 as *const u32).cast());
        enc.set_bytes(4, 4, (&eps as *const f32).cast());
        enc.set_bytes(5, 4, (&add as *const u32).cast());
        let tgsize = 256;
        enc.set_threadgroup_memory_length(0, (tgsize * 4) as u64);
        enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: tgsize, height: 1, depth: 1 });
    }

    pub fn rope_half_f32(&self, x: &mut [f32], n_heads: usize, head_dim: usize, rotary_dim: usize, pos: usize, theta: f32) -> anyhow::Result<()> {
        let n = x.len(); ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), n)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_rope_half_f32(enc, xb, n_heads, head_dim, rotary_dim, pos, theta);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(xb, x)
    }

    pub fn encode_rope_half_f32(&self, enc: &metal::ComputeCommandEncoderRef, x: &Buffer, n_heads: usize, head_dim: usize, rotary_dim: usize, pos: usize, theta: f32) {
        let nh = n_heads as u32; let hd = head_dim as u32; let rd = rotary_dim as u32; let p = pos as u32;
        enc.set_compute_pipeline_state(&self.pso_rope_half);
        enc.set_buffer(0, Some(x), 0);
        enc.set_bytes(1, 4, (&nh as *const u32).cast());
        enc.set_bytes(2, 4, (&hd as *const u32).cast());
        enc.set_bytes(3, 4, (&rd as *const u32).cast());
        enc.set_bytes(4, 4, (&p as *const u32).cast());
        enc.set_bytes(5, 4, (&theta as *const f32).cast());
        let threads = (n_heads * (rotary_dim / 2)) as u64;
        let w = self.pso_rope_half.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn rope_adj_f32(&self, x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) -> anyhow::Result<()> {
        let n = x.len(); ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), n)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_rope_adj_f32(enc, xb, n_heads, head_dim, pos, theta);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(xb, x)
    }

    pub fn encode_rope_adj_f32(&self, enc: &metal::ComputeCommandEncoderRef, x: &Buffer, n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
        let nh = n_heads as u32; let hd = head_dim as u32; let p = pos as u32;
        enc.set_compute_pipeline_state(&self.pso_rope_adj);
        enc.set_buffer(0, Some(x), 0);
        enc.set_bytes(1, 4, (&nh as *const u32).cast());
        enc.set_bytes(2, 4, (&hd as *const u32).cast());
        enc.set_bytes(3, 4, (&p as *const u32).cast());
        enc.set_bytes(4, 4, (&theta as *const f32).cast());
        let threads = (n_heads * (head_dim / 2)) as u64;
        let w = self.pso_rope_adj.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_mv_f16_bias(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::Buffer, x: &metal::Buffer, b: Option<&metal::Buffer>, out: &metal::Buffer, rs: usize, cs: usize) {
        let r = rs as u32; let c = cs as u32; let hb = b.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_f16);
        enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(x), 0);
        if let Some(bb) = b { enc.set_buffer(2, Some(bb), 0); }
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4, 4, (&r as *const u32).cast());
        enc.set_bytes(5, 4, (&c as *const u32).cast());
        enc.set_bytes(6, 4, (&hb as *const u32).cast());
        let w = self.pso_mv_f16.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: rs as u64, height: 1, depth: 1 }, MTLSize { width: w.min(rs as u64), height: 1, depth: 1 });
    }

    pub fn encode_mv_i8_bias(&self, enc: &metal::ComputeCommandEncoderRef, w: &metal::Buffer, s: &metal::Buffer, x: &metal::Buffer, b: Option<&metal::Buffer>, out: &metal::Buffer, rs: usize, cs: usize) {
        let r = rs as u32; let c = cs as u32; let hb = b.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_i8);
        enc.set_buffer(0, Some(w), 0); enc.set_buffer(1, Some(s), 0); enc.set_buffer(2, Some(x), 0);
        if let Some(bb) = b { enc.set_buffer(3, Some(bb), 0); }
        enc.set_buffer(4, Some(out), 0);
        enc.set_bytes(5, 4, (&r as *const u32).cast());
        enc.set_bytes(6, 4, (&c as *const u32).cast());
        enc.set_bytes(7, 4, (&hb as *const u32).cast());
        let ww = self.pso_mv_i8.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: rs as u64, height: 1, depth: 1 }, MTLSize { width: ww.min(rs as u64), height: 1, depth: 1 });
    }

    pub fn encode_qkv_f16_bias(&self, enc: &metal::ComputeCommandEncoderRef, wq: &metal::Buffer, wk: &metal::Buffer, wv: &metal::Buffer, x: &metal::Buffer, bq: Option<&metal::Buffer>, bk: Option<&metal::Buffer>, bv: Option<&metal::Buffer>, qo: &metal::Buffer, ko: &metal::Buffer, vo: &metal::Buffer, rq: usize, rkv: usize, c: usize) {
        let rq32 = rq as u32; let rkv32 = rkv as u32; let c32 = c as u32; let hb = bq.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_qkv_f16);
        enc.set_buffer(0, Some(wq), 0); enc.set_buffer(1, Some(wk), 0); enc.set_buffer(2, Some(wv), 0); enc.set_buffer(3, Some(x), 0);
        if let Some(b) = bq { enc.set_buffer(4, Some(b), 0); }
        if let Some(b) = bk { enc.set_buffer(5, Some(b), 0); }
        if let Some(b) = bv { enc.set_buffer(6, Some(b), 0); }
        enc.set_buffer(7, Some(qo), 0); enc.set_buffer(8, Some(ko), 0); enc.set_buffer(9, Some(vo), 0);
        enc.set_bytes(10, 4, (&rq32 as *const u32).cast());
        enc.set_bytes(11, 4, (&rkv32 as *const u32).cast());
        enc.set_bytes(12, 4, (&c32 as *const u32).cast());
        enc.set_bytes(13, 4, (&hb as *const u32).cast());
        let threads = (rq + rkv + rkv) as u64;
        let w = self.pso_mv_qkv_f16.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_qkv_i8_bias(&self, enc: &metal::ComputeCommandEncoderRef, wq: &metal::Buffer, sq: &metal::Buffer, wk: &metal::Buffer, sk: &metal::Buffer, wv: &metal::Buffer, sv: &metal::Buffer, x: &metal::Buffer, bq: Option<&metal::Buffer>, bk: Option<&metal::Buffer>, bv: Option<&metal::Buffer>, qo: &metal::Buffer, ko: &metal::Buffer, vo: &metal::Buffer, rq: usize, rkv: usize, c: usize) {
        let rq32 = rq as u32; let rkv32 = rkv as u32; let c32 = c as u32; let hb = bq.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_qkv_i8);
        enc.set_buffer(0, Some(wq), 0); enc.set_buffer(1, Some(sq), 0);
        enc.set_buffer(2, Some(wk), 0); enc.set_buffer(3, Some(sk), 0);
        enc.set_buffer(4, Some(wv), 0); enc.set_buffer(5, Some(sv), 0);
        enc.set_buffer(6, Some(x), 0);
        if let Some(b) = bq { enc.set_buffer(7, Some(b), 0); }
        if let Some(b) = bk { enc.set_buffer(8, Some(b), 0); }
        if let Some(b) = bv { enc.set_buffer(9, Some(b), 0); }
        enc.set_buffer(10, Some(qo), 0); enc.set_buffer(11, Some(ko), 0); enc.set_buffer(12, Some(vo), 0);
        enc.set_bytes(13, 4, (&rq32 as *const u32).cast());
        enc.set_bytes(14, 4, (&rkv32 as *const u32).cast());
        enc.set_bytes(15, 4, (&c32 as *const u32).cast());
        enc.set_bytes(16, 4, (&hb as *const u32).cast());
        let threads = (rq + rkv + rkv) as u64;
        let w = self.pso_mv_qkv_i8.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_add_f32_inplace(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::BufferRef, b: &metal::BufferRef, n: usize) {
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.pso_add_f32);
        enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(b), 0);
        enc.set_bytes(2, 4, (&n32 as *const u32).cast());
        let w = self.pso_add_f32.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: n as u64, height: 1, depth: 1 }, MTLSize { width: w.min(n as u64), height: 1, depth: 1 });
    }

    pub fn encode_silu_mul_f32_inplace(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::BufferRef, b: &metal::BufferRef, n: usize) {
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.pso_silu_mul_f32);
        enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(b), 0);
        enc.set_bytes(2, 4, (&n32 as *const u32).cast());
        let w = self.pso_silu_mul_f32.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: n as u64, height: 1, depth: 1 }, MTLSize { width: w.min(n as u64), height: 1, depth: 1 });
    }

    pub fn logits_qkv_f16(&self, x: &[f32], q_w: &[u16], k_w: &[u16], v_w: &[u16], q_dim: usize, k_dim: usize, v_dim: usize, hidden: usize, q_key: &str, k_key: &str, v_key: &str, q_out: &mut [f32], k_out: &mut [f32], v_out: &mut [f32]) -> anyhow::Result<()> {
        self.logits_f16(x, q_w, q_dim, hidden, q_key, q_out)?;
        self.logits_f16(x, k_w, k_dim, hidden, k_key, k_out)?;
        self.logits_f16(x, v_w, v_dim, hidden, v_key, v_out)?;
        Ok(())
    }

    pub fn logits_f16(&self, x: &[f32], embed_f16: &[u16], vocab: usize, hidden: usize, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        if !self.tensor_cache.lock().unwrap().contains_key(cache_key) {
            self.tensor_cache.lock().unwrap().insert(cache_key.to_string(), upload_u16(&self.device, embed_f16)?);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let eb = cache.get(cache_key).unwrap();
        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), hidden)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.lock().unwrap(), vocab)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.lock().unwrap(); let ob = ob_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_mv_f16_bias(enc, eb, xb, None, ob, vocab, hidden);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }

    pub fn logits_i8(&self, x: &[f32], embed_i8: &[i8], scales_f16: &[u16], vocab: usize, hidden: usize, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        let w_key = format!("logits.w.{cache_key}");
        let s_key = format!("logits.s.{cache_key}");
        if !self.tensor_cache.lock().unwrap().contains_key(&w_key) {
            self.tensor_cache.lock().unwrap().insert(w_key.clone(), upload_i8(&self.device, embed_i8)?);
            self.tensor_cache.lock().unwrap().insert(s_key.clone(), upload_u16(&self.device, scales_f16)?);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let eb = cache.get(&w_key).unwrap();
        let sb = cache.get(&s_key).unwrap();
        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), hidden)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.lock().unwrap(), vocab)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.lock().unwrap(); let ob = ob_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_mv_i8_bias(enc, eb, sb, xb, None, ob, vocab, hidden);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }

    // Compat stubs for old callers
    pub fn encode_mv_f16(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::Buffer, x: &metal::Buffer, out: &metal::Buffer, rs: usize, cs: usize) {
        self.encode_mv_f16_bias(enc, a, x, None, out, rs, cs);
    }
    pub fn encode_mv_i8(&self, enc: &metal::ComputeCommandEncoderRef, w: &metal::Buffer, s: &metal::Buffer, x: &metal::Buffer, out: &metal::Buffer, rs: usize, cs: usize) {
        self.encode_mv_i8_bias(enc, w, s, x, None, out, rs, cs);
    }
    pub fn encode_qkv_f16(&self, enc: &metal::ComputeCommandEncoderRef, w_q: &metal::Buffer, w_k: &metal::Buffer, w_v: &metal::Buffer, x_buf: &metal::Buffer, q_out: &metal::Buffer, k_out: &metal::Buffer, v_out: &metal::Buffer, rows_q: usize, rows_kv: usize, cols: usize) {
        self.encode_qkv_f16_bias(enc, w_q, w_k, w_v, x_buf, None, None, None, q_out, k_out, v_out, rows_q, rows_kv, cols);
    }
    pub fn encode_qkv_i8(&self, enc: &metal::ComputeCommandEncoderRef, w_q: &metal::Buffer, s_q: &metal::Buffer, w_k: &metal::Buffer, s_k: &metal::Buffer, w_v: &metal::Buffer, s_v: &metal::Buffer, x_buf: &metal::Buffer, q_out: &metal::Buffer, k_out: &metal::Buffer, v_out: &metal::Buffer, rows_q: usize, rows_kv: usize, cols: usize) {
        self.encode_qkv_i8_bias(enc, w_q, s_q, w_k, s_k, w_v, s_v, x_buf, None, None, None, q_out, k_out, v_out, rows_q, rows_kv, cols);
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn build_pipeline(device: &Device, src: &str, fn_name: &str) -> anyhow::Result<(Library, ComputePipelineState)> {
    let options = metal::CompileOptions::new();
    let lib = device.new_library_with_source(src, &options).map_err(|e| anyhow::anyhow!("Compile failed: {e:?}"))?;
    let func = lib.get_function(fn_name, None).map_err(|e| anyhow::anyhow!("Missing fn {fn_name}: {e:?}"))?;
    let pso = device.new_compute_pipeline_state_with_function(&func).map_err(|e| anyhow::anyhow!("PSO failed: {e:?}"))?;
    Ok((lib, pso))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn build_pso_ops(device: &metal::DeviceRef, lib: &Library, name: &str) -> anyhow::Result<ComputePipelineState> {
    let f = lib.get_function(name, None).map_err(|_| anyhow::anyhow!("Missing function {name}"))?;
    device.new_compute_pipeline_state_with_function(&f).map_err(|e| anyhow::anyhow!("PSO {name} failed: {e:?}"))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn upload_f32(device: &metal::DeviceRef, src: &[f32]) -> anyhow::Result<Buffer> {
    let b = device.new_buffer((src.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let ptr = b.contents() as *mut f32; if ptr.is_null() { anyhow::bail!("Upload f32 failed"); }
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()); }
    Ok(b)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn upload_u16(device: &metal::DeviceRef, src: &[u16]) -> anyhow::Result<Buffer> {
    let b = device.new_buffer((src.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
    let ptr = b.contents() as *mut u16; if ptr.is_null() { anyhow::bail!("Upload u16 failed"); }
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()); }
    Ok(b)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn upload_i8(device: &metal::DeviceRef, src: &[i8]) -> anyhow::Result<Buffer> {
    let b = device.new_buffer(src.len() as u64, MTLResourceOptions::StorageModeShared);
    let ptr = b.contents() as *mut i8; if ptr.is_null() { anyhow::bail!("Upload i8 failed"); }
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()); }
    Ok(b)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ensure_buf_f32(device: &metal::DeviceRef, slot: &mut Option<Buffer>, elems: usize) -> anyhow::Result<()> {
    let need = (elems * 4) as u64;
    if slot.as_ref().map(|b| b.length() >= need).unwrap_or(false) { return Ok(()); }
    let b = device.new_buffer(need, MTLResourceOptions::StorageModeShared);
    if b.contents().is_null() { anyhow::bail!("Scratch f32 failed"); }
    *slot = Some(b); Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn write_f32(buf: &Buffer, src: &[f32]) -> anyhow::Result<()> {
    let ptr = buf.contents() as *mut f32; if ptr.is_null() { anyhow::bail!("Write f32 null"); }
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()); }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn read_f32(buf: &Buffer, dst: &mut [f32]) -> anyhow::Result<()> {
    let ptr = buf.contents() as *const f32; if ptr.is_null() { anyhow::bail!("Read f32 null"); }
    unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()); }
    Ok(())
}
