#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::rc::autoreleasepool;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::cell::RefCell;
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
        let a = make_buf_f32(&device, n, |i| i as f32)?;
        let b = make_buf_f32(&device, n, |i| (2 * i) as f32)?;
        let out = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);

        dispatch_1d(&queue, &pso, n as u64, |enc| {
            enc.set_buffer(0, Some(&a), 0);
            enc.set_buffer(1, Some(&b), 0);
            enc.set_buffer(2, Some(&out), 0);
        })?;

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

        // Keep lib referenced so the pipeline stays valid in debug builds.
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
        anyhow::bail!("MetalKernels only supported on macOS in this crate build")
    }

    pub fn create_matmul() -> anyhow::Result<MetalMatmul> {
        anyhow::bail!("Metal matmul only supported on Apple platforms")
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalMatmul {
    pub fn upload_f32(&self, src: &[f32]) -> anyhow::Result<MetalBuffer> {
        let device = self.queue.device();
        make_buf_from_f32(device, src)
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
            let a_buf = make_buf_from_f32(device, a)?;
            let b_buf = make_buf_from_f32(device, b)?;
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
        if out.len() != m * n {
            anyhow::bail!("Metal matmul shape mismatch for output");
        }
        autoreleasepool(|| {
            let out_bytes = (out.len() * std::mem::size_of::<f32>()) as u64;
            let device = self.queue.device();
            let out_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
            let m_u32 = m as u32;
            let n_u32 = n as u32;
            let k_u32 = k as u32;
            let m_buf = make_buf_from_u32(device, &[m_u32])?;
            let n_buf = make_buf_from_u32(device, &[n_u32])?;
            let k_buf = make_buf_from_u32(device, &[k_u32])?;

            dispatch_2d(&self.queue, &self.pso, n as u64, m as u64, |enc| {
                enc.set_buffer(0, Some(a_buf), 0);
                enc.set_buffer(1, Some(b_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_buffer(3, Some(&m_buf), 0);
                enc.set_buffer(4, Some(&n_buf), 0);
                enc.set_buffer(5, Some(&k_buf), 0);
            })?;

            let out_ptr = out_buf.contents() as *const f32;
            let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, out.len()) };
            out.copy_from_slice(out_slice);
            Ok(())
        })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalMatmul {
    pub fn upload_f32(&self, _src: &[f32]) -> anyhow::Result<MetalBuffer> {
        anyhow::bail!("Metal matmul only supported on Apple platforms")
    }

    pub fn matmul_row_major_f32(
        &self,
        _a: &[f32],
        _m: usize,
        _k: usize,
        _b: &[f32],
        _n: usize,
        _out: &mut [f32],
    ) -> anyhow::Result<()> {
        anyhow::bail!("Metal matmul only supported on Apple platforms")
    }

    pub fn matmul_row_major_f32_with_b_buffer(
        &self,
        _a_buf: &MetalBuffer,
        _m: usize,
        _k: usize,
        _b_buf: &MetalBuffer,
        _n: usize,
        _out: &mut [f32],
    ) -> anyhow::Result<()> {
        anyhow::bail!("Metal matmul only supported on Apple platforms")
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn build_pipeline(device: &Device, src: &str, fn_name: &str) -> anyhow::Result<(Library, ComputePipelineState)> {
    let options = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(src, &options)
        .map_err(|e| anyhow::anyhow!("Metal: failed to compile library: {e:?}"))?;
    let func = lib
        .get_function(fn_name, None)
        .map_err(|e| anyhow::anyhow!("Metal: missing function {fn_name}: {e:?}"))?;
    let pso = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("Metal: failed to build pipeline: {e:?}"))?;
    Ok((lib, pso))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn make_buf_f32(
    device: &Device,
    n: usize,
    f: impl Fn(usize) -> f32,
) -> anyhow::Result<Buffer> {
    let bytes = (n * std::mem::size_of::<f32>()) as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    let ptr = buf.contents() as *mut f32;
    if ptr.is_null() {
        anyhow::bail!("Metal: buffer contents is null");
    }
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, n) };
    for i in 0..n {
        slice[i] = f(i);
    }
    Ok(buf)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn make_buf_from_f32(device: &metal::DeviceRef, src: &[f32]) -> anyhow::Result<Buffer> {
    let bytes = (std::mem::size_of_val(src)) as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    let ptr = buf.contents() as *mut f32;
    if ptr.is_null() {
        anyhow::bail!("Metal: buffer contents is null");
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, src.len()) };
    dst.copy_from_slice(src);
    Ok(buf)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn make_buf_from_u32(device: &metal::DeviceRef, src: &[u32]) -> anyhow::Result<Buffer> {
    let bytes = (std::mem::size_of_val(src)) as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    let ptr = buf.contents() as *mut u32;
    if ptr.is_null() {
        anyhow::bail!("Metal: buffer contents is null");
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, src.len()) };
    dst.copy_from_slice(src);
    Ok(buf)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn dispatch_1d(
    queue: &CommandQueue,
    pso: &ComputePipelineState,
    threads: u64,
    bind: impl FnOnce(&metal::ComputeCommandEncoderRef),
) -> anyhow::Result<()> {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    bind(enc);

    let w = pso.thread_execution_width() as u64;
    let tg = metal::MTLSize {
        width: w.min(threads),
        height: 1,
        depth: 1,
    };
    let grid = metal::MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn dispatch_2d(
    queue: &CommandQueue,
    pso: &ComputePipelineState,
    width: u64,
    height: u64,
    bind: impl FnOnce(&metal::ComputeCommandEncoderRef),
) -> anyhow::Result<()> {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    bind(enc);
    let w = pso.thread_execution_width() as u64;
    let tg = metal::MTLSize {
        width: w.max(1).min(width.max(1)),
        height: 1,
        depth: 1,
    };
    let grid = metal::MTLSize {
        width,
        height,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(())
}

// MetalOps – element-wise kernels: RMSNorm · RoPE · matrix-vector logits
const ELEM_OPS_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Basic add: a += b
kernel void add_f32_inplace(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) a[gid] += b[gid];
}

// SiLU * up: a = (a / (1.0 + exp(-a))) * b
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

// RMSNorm with f16 weights.
// Dispatched as 1 threadgroup of `tgsize` threads, threadgroup mem = tgsize*4 bytes.
kernel void rms_norm_f16w(
    device const float*  x        [[buffer(0)]],
    device const ushort* w        [[buffer(1)]],
    device       float*  out      [[buffer(2)]],
    constant     uint&   n        [[buffer(3)]],
    constant     float&  eps      [[buffer(4)]],
    constant     uint&   w_add_one[[buffer(5)]],
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
    shared[tid] = (tid < n) ? (local_sq * inv_n) : 0.0f;
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

// RoPE – adjacent interleaved pairs: (0,1),(2,3),...
// gid in [0, n_heads * head_dim/2)
kernel void rope_adj_f32(
    device       float* x         [[buffer(0)]],
    constant     uint&  n_heads   [[buffer(1)]],
    constant     uint&  head_dim  [[buffer(2)]],
    constant     uint&  pos       [[buffer(3)]],
    constant     float& theta     [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_hd = head_dim >> 1;
    uint h   = gid / half_hd;
    uint dim = gid % half_hd;
    if (h >= n_heads) return;
    float inv_freq = pow(theta, -(2.0f * float(dim)) / float(head_dim));
    float angle    = float(pos) * inv_freq;
    float cos_a    = cos(angle);
    float sin_a    = sin(angle);
    uint  base     = h * head_dim + dim * 2;
    float x0 = x[base];
    float x1 = x[base + 1];
    x[base]     = x0 * cos_a - x1 * sin_a;
    x[base + 1] = x0 * sin_a + x1 * cos_a;
}

// RoPE – half-split pairs: x[i] and x[half+i] (Qwen-style partial RoPE).
// gid in [0, n_heads * rotary_dim/2)
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
    float angle    = float(pos) * inv_freq;
    float cos_a    = cos(angle);
    float sin_a    = sin(angle);
    uint base      = h * head_dim;
    float x0 = x[base + i];
    float x1 = x[base + half_rd + i];
    x[base + i]          = x0 * cos_a - x1 * sin_a;
    x[base + half_rd + i] = x1 * cos_a + x0 * sin_a;
}

// Matrix-vector product, f16 weight matrix.
// out[row] = dot(A_f16[row*cols .. (row+1)*cols], x[0..cols])
// One thread per output row.
kernel void mv_f16(
    device const ushort* A    [[buffer(0)]],
    device const float*  x    [[buffer(1)]],
    device       float*  out  [[buffer(2)]],
    constant     uint&   rows [[buffer(3)]],
    constant     uint&   cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    device const ushort* row_ptr = A + uint(gid) * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i)
        acc += x[i] * float(as_type<half>(row_ptr[i]));
    out[gid] = acc;
}

// Matrix-vector product, i8 weight + per-row f16 scale.
// out[row] = scale[row] * dot(A_i8[row*cols ..], x)
kernel void mv_i8(
    device const char*   A     [[buffer(0)]],
    device const ushort* scl   [[buffer(1)]],
    device const float*  x     [[buffer(2)]],
    device       float*  out   [[buffer(3)]],
    constant     uint&   rows  [[buffer(4)]],
    constant     uint&   cols  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    float scale = float(as_type<half>(scl[gid]));
    device const char* row_ptr = A + uint(gid) * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i)
        acc += x[i] * float(row_ptr[i]);
    out[gid] = acc * scale;
}

// Fused QKV projection (f16 weights).
// q_out[r] = dot(A_q[r], x), k_out[r] = dot(A_k[r], x), v_out[r] = dot(A_v[r], x)
kernel void mv_qkv_f16(
    device const ushort* A_q      [[buffer(0)]],
    device const ushort* A_k      [[buffer(1)]],
    device const ushort* A_v      [[buffer(2)]],
    device const float*  x        [[buffer(3)]],
    device       float*  q_out    [[buffer(4)]],
    device       float*  k_out    [[buffer(5)]],
    device       float*  v_out    [[buffer(6)]],
    constant     uint&   rows_q   [[buffer(7)]],
    constant     uint&   rows_kv  [[buffer(8)]],
    constant     uint&   cols     [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = rows_q + rows_kv + rows_kv;
    if (gid >= total) return;
    if (gid < rows_q) {
        device const ushort* row_ptr = A_q + gid * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) {
            acc += x[i] * float(as_type<half>(row_ptr[i]));
        }
        q_out[gid] = acc;
        return;
    }
    uint off = gid - rows_q;
    if (off < rows_kv) {
        device const ushort* row_ptr = A_k + off * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) {
            acc += x[i] * float(as_type<half>(row_ptr[i]));
        }
        k_out[off] = acc;
        return;
    }
    off -= rows_kv;
    device const ushort* row_ptr = A_v + off * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) {
        acc += x[i] * float(as_type<half>(row_ptr[i]));
    }
    v_out[off] = acc;
}

// Fused dual matrix-vector (f16 weights): out0 and out1 in one launch.
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
        for (uint i = 0; i < cols; ++i) {
            acc += x[i] * float(as_type<half>(row_ptr[i]));
        }
        out0[gid] = acc;
        return;
    }
    uint off = gid - rows0;
    device const ushort* row_ptr = A1 + off * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) {
        acc += x[i] * float(as_type<half>(row_ptr[i]));
    }
    out1[off] = acc;
}

// Fused dual matrix-vector (i8 weights): out0 and out1 in one launch.
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

// Apple platform implementation

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
    pub pso_mv2_f16: ComputePipelineState,
    pub pso_mv2_i8: ComputePipelineState,
    pub pso_add_f32: ComputePipelineState,
    pub pso_silu_mul_f32: ComputePipelineState,
    // Scratch buffers
    x_buf: RefCell<Option<Buffer>>,
    w_buf: RefCell<Option<Buffer>>,
    out_buf: RefCell<Option<Buffer>>,
    /// Cache large tensor uploads by name.
    tensor_cache: Mutex<HashMap<String, Buffer>>,
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalOps;

// Apple impl
#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalOps {
    pub fn create() -> anyhow::Result<Self> {
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| anyhow::anyhow!("MetalOps: no Metal device found"))?;
        let queue = device.new_command_queue();
        let options = metal::CompileOptions::new();
        let lib = device
            .new_library_with_source(ELEM_OPS_SHADER, &options)
            .map_err(|e| anyhow::anyhow!("MetalOps: compile failed: {e:?}"))?;
        let pso_rms_norm  = build_pso_ops(&device, &lib, "rms_norm_f16w")?;
        let pso_rope_adj  = build_pso_ops(&device, &lib, "rope_adj_f32")?;
        let pso_rope_half = build_pso_ops(&device, &lib, "rope_half_f32")?;
        let pso_mv_f16    = build_pso_ops(&device, &lib, "mv_f16")?;
        let pso_mv_i8     = build_pso_ops(&device, &lib, "mv_i8")?;
        let pso_mv_qkv_f16= build_pso_ops(&device, &lib, "mv_qkv_f16")?;
        let pso_mv2_f16   = build_pso_ops(&device, &lib, "mv2_f16")?;
        let pso_mv2_i8    = build_pso_ops(&device, &lib, "mv2_i8")?;
        let pso_add_f32   = build_pso_ops(&device, &lib, "add_f32_inplace")?;
        let pso_silu_mul_f32 = build_pso_ops(&device, &lib, "silu_mul_f32_inplace")?;
        Ok(Self {
            device, queue, _lib: lib,
            pso_rms_norm, pso_rope_adj, pso_rope_half, pso_mv_f16, pso_mv_i8, pso_mv_qkv_f16, pso_mv2_f16, pso_mv2_i8,
            pso_add_f32, pso_silu_mul_f32,
            x_buf: RefCell::new(None), w_buf: RefCell::new(None), out_buf: RefCell::new(None),
            tensor_cache: Mutex::new(HashMap::new()),
        })
    }

    // RMSNorm 
    /// x: f32 input, w_f16: raw f16 weights, w_add_one: add 1.0 to weights (Gemma).
    pub fn rms_norm_f16w(
        &self,
        x: &[f32],
        w_f16: &[u16],
        eps: f32,
        w_add_one: bool,
        cache_key: &str,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        let n = x.len();
        let w_key = format!("rmsnorm.w.{}", cache_key);
        {
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(&w_key) { cache.insert(w_key.clone(), upload_u16(&self.device, w_f16)?); }
        }
        let cache = self.tensor_cache.lock().unwrap();
        let wb = cache.get(&w_key).unwrap();

        ensure_buf_f32(&self.device, &mut *self.x_buf.borrow_mut(), n)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.borrow_mut(), n)?;

        let xb_ref = self.x_buf.borrow(); let xb = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.borrow(); let ob = ob_ref.as_ref().unwrap();

        write_f32(xb, x)?;

        let n_u32    = n as u32;
        let add_one  = w_add_one as u32;
        let n_ptr    = (&n_u32   as *const u32).cast();
        let eps_ptr  = (&eps     as *const f32).cast();
        let add_ptr  = (&add_one as *const u32).cast();

        // Threadgroup size: largest power-of-2 ≤ 1024 that covers n.
        let tg_size = next_pow2_min(n, 1024) as u64;
        let tg_mem  = tg_size * 4; // float per thread

        autoreleasepool(|| {
            let cb  = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso_rms_norm);
            enc.set_threadgroup_memory_length(0, tg_mem);
            enc.set_buffer(0, Some(xb), 0);
            enc.set_buffer(1, Some(wb), 0);
            enc.set_buffer(2, Some(ob), 0);
            enc.set_bytes(3, 4, n_ptr);
            enc.set_bytes(4, 4, eps_ptr);
            enc.set_bytes(5, 4, add_ptr);
            // 1 threadgroup of tg_size threads
            let tg = metal::MTLSize { width: tg_size, height: 1, depth: 1 };
            enc.dispatch_thread_groups(metal::MTLSize { width: 1, height: 1, depth: 1 }, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        read_f32(ob, out)?;
        Ok(())
    }

    // RoPE (adjacent-pair interleaved) 
    pub fn rope_adj_f32(
        &self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        pos: usize,
        theta: f32,
    ) -> anyhow::Result<()> {
        let n = x.len();
        ensure_buf_f32(&self.device, &mut *self.x_buf.borrow_mut(), n)?;
        let xb_ref = self.x_buf.borrow(); let xb = xb_ref.as_ref().unwrap();
        write_f32(xb, x)?;

        let threads = (n_heads * head_dim / 2) as u64;
        let nh  = n_heads  as u32;
        let hd  = head_dim as u32;
        let p   = pos      as u32;
        let nh_ptr = (&nh as *const u32).cast();
        let hd_ptr = (&hd as *const u32).cast();
        let p_ptr  = (&p  as *const u32).cast();
        let th_ptr = (&theta as *const f32).cast();

        autoreleasepool(|| {
            let cb  = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso_rope_adj);
            enc.set_buffer(0, Some(xb), 0);
            enc.set_bytes(1, 4, nh_ptr);
            enc.set_bytes(2, 4, hd_ptr);
            enc.set_bytes(3, 4, p_ptr);
            enc.set_bytes(4, 4, th_ptr);
            let w   = self.pso_rope_adj.thread_execution_width() as u64;
            let tg  = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
            let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        read_f32(xb, x)?;
        Ok(())
    }

    // RoPE (half-split partial, Qwen-style)
    pub fn rope_half_f32(
        &self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos: usize,
        theta: f32,
    ) -> anyhow::Result<()> {
        let n = x.len();
        ensure_buf_f32(&self.device, &mut *self.x_buf.borrow_mut(), n)?;
        let xb_ref = self.x_buf.borrow(); let xb = xb_ref.as_ref().unwrap();
        write_f32(xb, x)?;

        let threads = (n_heads * rotary_dim / 2) as u64;
        let nh  = n_heads    as u32;
        let hd  = head_dim   as u32;
        let rd  = rotary_dim as u32;
        let p   = pos        as u32;
        let nh_ptr = (&nh as *const u32).cast();
        let hd_ptr = (&hd as *const u32).cast();
        let rd_ptr = (&rd as *const u32).cast();
        let p_ptr  = (&p  as *const u32).cast();
        let th_ptr = (&theta as *const f32).cast();

        autoreleasepool(|| {
            let cb  = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso_rope_half);
            enc.set_buffer(0, Some(xb), 0);
            enc.set_bytes(1, 4, nh_ptr);
            enc.set_bytes(2, 4, hd_ptr);
            enc.set_bytes(3, 4, rd_ptr);
            enc.set_bytes(4, 4, p_ptr);
            enc.set_bytes(5, 4, th_ptr);
            let w    = self.pso_rope_half.thread_execution_width() as u64;
            let tg   = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
            let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        read_f32(xb, x)?;
        Ok(())
    }

    // Logits: all-vocab matrix-vector (f16 weights)─
    /// Computes logits[v] = dot(embed_f16[v*hidden..], x_final) for every v.
    /// The embedding table is uploaded once and cached by `cache_key`.
    pub fn logits_f16(
        &self,
        x: &[f32],
        embed_f16: &[u16],
        vocab: usize,
        hidden: usize,
        cache_key: &str,
        logits_out: &mut [f32],
    ) -> anyhow::Result<()> {
        // Upload embedding table once; reuse on subsequent tokens.
        if !self.tensor_cache.lock().unwrap().contains_key(cache_key) {
            let buf = upload_u16(&self.device, embed_f16)?;
            self.tensor_cache.lock().unwrap().insert(cache_key.to_string(), buf);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let embed_buf = cache.get(cache_key).unwrap();

        ensure_buf_f32(&self.device, &mut *self.x_buf.borrow_mut(), hidden)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.borrow_mut(), vocab)?;
        let xb_ref = self.x_buf.borrow(); let xb  = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.borrow(); let ob  = ob_ref.as_ref().unwrap();
        write_f32(xb, x)?;

        let rows_u32 = vocab  as u32;
        let cols_u32 = hidden as u32;
        let rows_ptr = (&rows_u32 as *const u32).cast();
        let cols_ptr = (&cols_u32 as *const u32).cast();

        let threads = vocab as u64;
        autoreleasepool(|| {
            let cb  = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso_mv_f16);
            enc.set_buffer(0, Some(embed_buf), 0);
            enc.set_buffer(1, Some(xb),        0);
            enc.set_buffer(2, Some(ob),        0);
            enc.set_bytes(3, 4, rows_ptr);
            enc.set_bytes(4, 4, cols_ptr);
            let w    = self.pso_mv_f16.thread_execution_width() as u64;
            let tg   = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
            let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        read_f32(ob, logits_out)?;
        Ok(())
    }

    // Logits: all-vocab matrix-vector (i8 + scale weights)─
    pub fn logits_i8(
        &self,
        x: &[f32],
        embed_i8: &[i8],
        scales_f16: &[u16],
        vocab: usize,
        hidden: usize,
        cache_key: &str,
        logits_out: &mut [f32],
    ) -> anyhow::Result<()> {
        let wkey   = format!("{cache_key}.w");
        let skey   = format!("{cache_key}.s");

        if !self.tensor_cache.lock().unwrap().contains_key(&wkey) {
            let wbuf = upload_i8(&self.device, embed_i8)?;
            self.tensor_cache.lock().unwrap().insert(wkey.clone(), wbuf);
            let sbuf = upload_u16(&self.device, scales_f16)?;
            self.tensor_cache.lock().unwrap().insert(skey.clone(), sbuf);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let wbuf = cache.get(&wkey).unwrap();
        let sbuf = cache.get(&skey).unwrap();

        ensure_buf_f32(&self.device, &mut *self.x_buf.borrow_mut(), hidden)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.borrow_mut(), vocab)?;
        let xb_ref = self.x_buf.borrow(); let xb = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.borrow(); let ob = ob_ref.as_ref().unwrap();
        write_f32(xb, x)?;

        let rows_u32 = vocab  as u32;
        let cols_u32 = hidden as u32;
        let rows_ptr = (&rows_u32 as *const u32).cast();
        let cols_ptr = (&cols_u32 as *const u32).cast();
        let threads  = vocab as u64;

        autoreleasepool(|| {
            let cb  = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso_mv_i8);
            enc.set_buffer(0, Some(wbuf), 0);
            enc.set_buffer(1, Some(sbuf), 0);
            enc.set_buffer(2, Some(xb),   0);
            enc.set_buffer(3, Some(ob),   0);
            enc.set_bytes(4, 4, rows_ptr);
            enc.set_bytes(5, 4, cols_ptr);
            let w    = self.pso_mv_i8.thread_execution_width() as u64;
            let tg   = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
            let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        read_f32(ob, logits_out)?;
        Ok(())
    }

    pub fn logits_qkv_f16(
        &mut self,
        x: &[f32],
        w_q_f16: &[u16],
        w_k_f16: &[u16],
        w_v_f16: &[u16],
        rows_q: usize,
        rows_k: usize,
        rows_v: usize,
        cols: usize,
        cache_key_q: &str,
        cache_key_k: &str,
        cache_key_v: &str,
        q_out: &mut [f32],
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> anyhow::Result<()> {
        if rows_k != rows_v {
            anyhow::bail!("logits_qkv_f16: rows_k ({rows_k}) must equal rows_v ({rows_v})");
        }
        if x.len() != cols || q_out.len() != rows_q || k_out.len() != rows_k || v_out.len() != rows_v {
            anyhow::bail!(
                "logits_qkv_f16 dims mismatch: x={} q_out={} k_out={} v_out={} expected cols={} rows_q={} rows_k={} rows_v={}",
                x.len(),
                q_out.len(),
                k_out.len(),
                v_out.len(),
                cols,
                rows_q,
                rows_k,
                rows_v
            );
        }
        if w_q_f16.len() != rows_q * cols || w_k_f16.len() != rows_k * cols || w_v_f16.len() != rows_v * cols {
            anyhow::bail!(
                "logits_qkv_f16 weight dims mismatch: q={} k={} v={} expected q={} k={} v={}",
                w_q_f16.len(),
                w_k_f16.len(),
                w_v_f16.len(),
                rows_q * cols,
                rows_k * cols,
                rows_v * cols
            );
        }

        if !self.tensor_cache.lock().unwrap().contains_key(cache_key_q) {
            self.tensor_cache.lock().unwrap().insert(cache_key_q.to_string(), upload_u16(&self.device, w_q_f16)?);
        }
        if !self.tensor_cache.lock().unwrap().contains_key(cache_key_k) {
            self.tensor_cache.lock().unwrap().insert(cache_key_k.to_string(), upload_u16(&self.device, w_k_f16)?);
        }
        if !self.tensor_cache.lock().unwrap().contains_key(cache_key_v) {
            self.tensor_cache.lock().unwrap().insert(cache_key_v.to_string(), upload_u16(&self.device, w_v_f16)?);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let wq = cache.get(cache_key_q).unwrap();
        let wk = cache.get(cache_key_k).unwrap();
        let wv = cache.get(cache_key_v).unwrap();

        ensure_buf_f32(&self.device, &mut *self.x_buf.borrow_mut(), cols)?;
        let xb_ref = self.x_buf.borrow(); let xb = xb_ref.as_ref().unwrap();
        write_f32(xb, x)?;

        let q_bytes = (rows_q * std::mem::size_of::<f32>()) as u64;
        let kv_bytes = (rows_k * std::mem::size_of::<f32>()) as u64;
        let q_buf = self.device.new_buffer(q_bytes, MTLResourceOptions::StorageModeShared);
        let k_buf = self.device.new_buffer(kv_bytes, MTLResourceOptions::StorageModeShared);
        let v_buf = self.device.new_buffer(kv_bytes, MTLResourceOptions::StorageModeShared);

        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_qkv_f16(
                &enc,
                wq,
                wk,
                wv,
                xb,
                &q_buf,
                &k_buf,
                &v_buf,
                rows_q,
                rows_k,
                cols,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        read_f32(&q_buf, q_out)?;
        read_f32(&k_buf, k_out)?;
        read_f32(&v_buf, v_out)?;
        Ok(())
    }

    pub fn logits_mv2_i8(
        &self,
        x: &[f32],
        k0: &str,
        k1: &str,
        v: usize,
        h: usize,
        o0: &mut [f32],
        o1: &mut [f32],
    ) -> anyhow::Result<()> {
        let w0_key = format!("{k0}.w"); let s0_key = format!("{k0}.s");
        let w1_key = format!("{k1}.w"); let s1_key = format!("{k1}.s");
        let cache = self.tensor_cache.lock().unwrap();
        let wb0 = cache.get(&w0_key).ok_or_else(|| anyhow::anyhow!("k0-w missing"))?;
        let sb0 = cache.get(&s0_key).ok_or_else(|| anyhow::anyhow!("k0-s missing"))?;
        let wb1 = cache.get(&w1_key).ok_or_else(|| anyhow::anyhow!("k1-w missing"))?;
        let sb1 = cache.get(&s1_key).ok_or_else(|| anyhow::anyhow!("k1-s missing"))?;

        ensure_buf_f32(&self.device, &mut *self.x_buf.borrow_mut(), h)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.borrow_mut(), v * 2)?;
        let xb_ref = self.x_buf.borrow(); let xb = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.borrow(); let ob = ob_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_mv2_i8(enc, wb0, wb1, sb0, sb1, xb, ob, ob, v, v, h);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        unsafe {
            let ptr = ob.contents() as *const f32;
            o0.copy_from_slice(std::slice::from_raw_parts(ptr, v));
            o1.copy_from_slice(std::slice::from_raw_parts(ptr.add(v), v));
        }
        Ok(())
    }

    pub fn encode_mv2_i8(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w0: &metal::BufferRef,
        w1: &metal::BufferRef,
        s0: &metal::BufferRef,
        s1: &metal::BufferRef,
        x: &metal::BufferRef,
        o0: &metal::BufferRef,
        o1: &metal::BufferRef,
        r0: usize,
        r1: usize,
        c: usize,
    ) {
        let r0_32 = r0 as u32; let r1_32 = r1 as u32; let c_32 = c as u32;
        enc.set_compute_pipeline_state(&self.pso_mv2_i8);
        enc.set_buffer(0, Some(w0), 0); enc.set_buffer(1, Some(w1), 0);
        enc.set_buffer(2, Some(s0), 0); enc.set_buffer(3, Some(s1), 0);
        enc.set_buffer(4, Some(x), 0); enc.set_buffer(5, Some(o0), 0); enc.set_buffer(6, Some(o1), 0);
        enc.set_bytes(7, 4, (&r0_32 as *const u32).cast());
        enc.set_bytes(8, 4, (&r1_32 as *const u32).cast());
        enc.set_bytes(9, 4, (&c_32 as *const u32).cast());
        let threads = (r0 + r1) as u64;
        let w = self.pso_mv2_i8.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_rms_norm_f16w(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        w_buf: &Buffer,
        out_buf: &Buffer,
        n: usize,
        eps: f32,
        add_residual: bool,
    ) {
        let tgsize = 256;
        let n_ptr = (&(n as u32) as *const u32).cast();
        let eps_ptr = (&eps as *const f32).cast();
        let add_res_val = if add_residual { 1u32 } else { 0u32 };
        let add_res_ptr = (&add_res_val as *const u32).cast();

        enc.set_compute_pipeline_state(&self.pso_rms_norm);
        enc.set_buffer(0, Some(x_buf), 0);
        enc.set_buffer(1, Some(w_buf), 0);
        enc.set_buffer(2, Some(out_buf), 0);
        enc.set_bytes(3, 4, n_ptr);
        enc.set_bytes(4, 4, eps_ptr);
        enc.set_bytes(5, 4, add_res_ptr);
        enc.set_threadgroup_memory_length(0, (tgsize * 4) as u64);

        let tg = metal::MTLSize { width: tgsize as u64, height: 1, depth: 1 };
        let grid = metal::MTLSize { width: tgsize as u64, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_rope_adj_f32(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        n_heads: usize,
        head_dim: usize,
        pos: usize,
        theta: f32,
    ) {
        let threads = (n_heads * head_dim / 2) as u64;
        let nh = n_heads as u32;
        let hd = head_dim as u32;
        let p = pos as u32;
        let nh_ptr = (&nh as *const u32).cast();
        let hd_ptr = (&hd as *const u32).cast();
        let p_ptr = (&p as *const u32).cast();
        let th_ptr = (&theta as *const f32).cast();

        enc.set_compute_pipeline_state(&self.pso_rope_adj);
        enc.set_buffer(0, Some(x_buf), 0);
        enc.set_bytes(1, 4, nh_ptr);
        enc.set_bytes(2, 4, hd_ptr);
        enc.set_bytes(3, 4, p_ptr);
        enc.set_bytes(4, 4, th_ptr);

        let w = self.pso_rope_adj.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_mv_f16(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        a_buf: &metal::BufferRef,
        x_buf: &metal::BufferRef,
        out_buf: &metal::BufferRef,
        rows: usize,
        cols: usize,
    ) {
        let r32 = rows as u32;
        let c32 = cols as u32;
        let r_ptr = (&r32 as *const u32).cast();
        let c_ptr = (&c32 as *const u32).cast();

        enc.set_compute_pipeline_state(&self.pso_mv_f16);
        enc.set_buffer(0, Some(a_buf), 0);
        enc.set_buffer(1, Some(x_buf), 0);
        enc.set_buffer(2, Some(out_buf), 0);
        enc.set_bytes(3, 4, r_ptr);
        enc.set_bytes(4, 4, c_ptr);

        let threads = rows as u64;
        let w = self.pso_mv_f16.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_qkv_f16(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w_q: &metal::BufferRef,
        w_k: &metal::BufferRef,
        w_v: &metal::BufferRef,
        x_buf: &metal::BufferRef,
        q_out: &metal::BufferRef,
        k_out: &metal::BufferRef,
        v_out: &metal::BufferRef,
        rows_q: usize,
        rows_kv: usize,
        cols: usize,
    ) {
        let rq = rows_q as u32;
        let rkv = rows_kv as u32;
        let c = cols as u32;
        let rq_ptr = (&rq as *const u32).cast();
        let rkv_ptr = (&rkv as *const u32).cast();
        let c_ptr = (&c as *const u32).cast();

        enc.set_compute_pipeline_state(&self.pso_mv_qkv_f16);
        enc.set_buffer(0, Some(w_q), 0);
        enc.set_buffer(1, Some(w_k), 0);
        enc.set_buffer(2, Some(w_v), 0);
        enc.set_buffer(3, Some(x_buf), 0);
        enc.set_buffer(4, Some(q_out), 0);
        enc.set_buffer(5, Some(k_out), 0);
        enc.set_buffer(6, Some(v_out), 0);
        enc.set_bytes(7, 4, rq_ptr);
        enc.set_bytes(8, 4, rkv_ptr);
        enc.set_bytes(9, 4, c_ptr);

        let threads = (rows_q + rows_kv + rows_kv) as u64;
        let w = self.pso_mv_qkv_f16.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_mv2_f16(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        w0: &metal::BufferRef,
        w1: &metal::BufferRef,
        x_buf: &metal::BufferRef,
        out0: &metal::BufferRef,
        out1: &metal::BufferRef,
        rows0: usize,
        rows1: usize,
        cols: usize,
    ) {
        let r0 = rows0 as u32;
        let r1 = rows1 as u32;
        let c = cols as u32;
        let r0_ptr = (&r0 as *const u32).cast();
        let r1_ptr = (&r1 as *const u32).cast();
        let c_ptr = (&c as *const u32).cast();

        enc.set_compute_pipeline_state(&self.pso_mv2_f16);
        enc.set_buffer(0, Some(w0), 0);
        enc.set_buffer(1, Some(w1), 0);
        enc.set_buffer(2, Some(x_buf), 0);
        enc.set_buffer(3, Some(out0), 0);
        enc.set_buffer(4, Some(out1), 0);
        enc.set_bytes(5, 4, r0_ptr);
        enc.set_bytes(6, 4, r1_ptr);
        enc.set_bytes(7, 4, c_ptr);

        let threads = (rows0 + rows1) as u64;
        let w = self.pso_mv2_f16.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_add_f32_inplace(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        a_buf: &metal::BufferRef,
        b_buf: &metal::BufferRef,
        n: usize,
    ) {
        let n_u32 = n as u32;
        let n_ptr = (&n_u32 as *const u32).cast();
        enc.set_compute_pipeline_state(&self.pso_add_f32);
        enc.set_buffer(0, Some(a_buf), 0);
        enc.set_buffer(1, Some(b_buf), 0);
        enc.set_bytes(2, 4, n_ptr);
        
        let threads = n as u64;
        let w = self.pso_add_f32.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_silu_mul_f32_inplace(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        a_buf: &metal::BufferRef,
        b_buf: &metal::BufferRef,
        n: usize,
    ) {
        let n_u32 = n as u32;
        let n_ptr = (&n_u32 as *const u32).cast();
        enc.set_compute_pipeline_state(&self.pso_silu_mul_f32);
        enc.set_buffer(0, Some(a_buf), 0);
        enc.set_buffer(1, Some(b_buf), 0);
        enc.set_bytes(2, 4, n_ptr);
        
        let threads = n as u64;
        let w = self.pso_silu_mul_f32.thread_execution_width() as u64;
        let tg = metal::MTLSize { width: w.min(threads), height: 1, depth: 1 };
        let grid = metal::MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }
}

// Helper functions for MetalOps

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn build_pso_ops(
    device: &Device,
    lib: &Library,
    name: &str,
) -> anyhow::Result<ComputePipelineState> {
    let f = lib
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("MetalOps: missing fn {name}: {e:?}"))?;
    device
        .new_compute_pipeline_state_with_function(&f)
        .map_err(|e| anyhow::anyhow!("MetalOps: pipeline {name} failed: {e:?}"))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn next_pow2_min(n: usize, cap: usize) -> usize {
    let mut p = 1usize;
    while p < n && p < cap { p <<= 1; }
    p.min(cap)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ensure_buf_f32(device: &Device, slot: &mut Option<Buffer>, elems: usize) -> anyhow::Result<()> {
    let need = (elems * 4) as u64;
    if slot.as_ref().map(|b| b.length() >= need).unwrap_or(false) { return Ok(()); }
    let b = device.new_buffer(need, MTLResourceOptions::StorageModeShared);
    if b.contents().is_null() { anyhow::bail!("MetalOps: null f32 scratch"); }
    *slot = Some(b);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ensure_buf_u16(device: &Device, slot: &mut Option<Buffer>, elems: usize) -> anyhow::Result<()> {
    let need = (elems * 2) as u64;
    if slot.as_ref().map(|b| b.length() >= need).unwrap_or(false) { return Ok(()); }
    let b = device.new_buffer(need, MTLResourceOptions::StorageModeShared);
    if b.contents().is_null() { anyhow::bail!("MetalOps: null u16 scratch"); }
    *slot = Some(b);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn write_f32(buf: &Buffer, src: &[f32]) -> anyhow::Result<()> {
    let ptr = buf.contents() as *mut f32;
    if ptr.is_null() { anyhow::bail!("MetalOps: null buf in write_f32"); }
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, src.len()) };
    dst.copy_from_slice(src);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn write_u16(buf: &Buffer, src: &[u16]) -> anyhow::Result<()> {
    let ptr = buf.contents() as *mut u16;
    if ptr.is_null() { anyhow::bail!("MetalOps: null buf in write_u16"); }
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, src.len()) };
    dst.copy_from_slice(src);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn read_f32(buf: &Buffer, dst: &mut [f32]) -> anyhow::Result<()> {
    let ptr = buf.contents() as *const f32;
    if ptr.is_null() { anyhow::bail!("MetalOps: null buf in read_f32"); }
    let src = unsafe { std::slice::from_raw_parts(ptr, dst.len()) };
    dst.copy_from_slice(src);
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn upload_i8(device: &Device, src: &[i8]) -> anyhow::Result<Buffer> {
    let bytes = src.len() as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    if buf.contents().is_null() { anyhow::bail!("MetalOps: null i8 upload buf"); }
    let dst = unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut i8, src.len()) };
    dst.copy_from_slice(src);
    Ok(buf)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn upload_u16(device: &Device, src: &[u16]) -> anyhow::Result<Buffer> {
    let bytes = (src.len() * 2) as u64;
    let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    if buf.contents().is_null() { anyhow::bail!("MetalOps: null u16 upload buf"); }
    let dst = unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut u16, src.len()) };
    dst.copy_from_slice(src);
    Ok(buf)
}

// Non-Apple stub
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalOps {
    pub fn create() -> anyhow::Result<Self> {
        anyhow::bail!("MetalOps only supported on Apple platforms")
    }
    pub fn rms_norm_f16w(&mut self, _: &[f32], _: &[u16], _: f32, _: bool, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available")
    }
    pub fn rope_adj_f32(&mut self, _: &mut [f32], _: usize, _: usize, _: usize, _: f32) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available")
    }
    pub fn rope_half_f32(&mut self, _: &mut [f32], _: usize, _: usize, _: usize, _: usize, _: f32) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available")
    }
    pub fn logits_f16(&mut self, _: &[f32], _: &[u16], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available")
    }
    pub fn logits_i8(&mut self, _: &[f32], _: &[i8], _: &[u16], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available")
    }
    pub fn logits_qkv_f16(&mut self, _: &[f32], _: &[u16], _: &[u16], _: &[u16], _: usize, _: usize, _: usize, _: usize, _: &str, _: &str, _: &str, _: &mut [f32], _: &mut [f32], _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available")
    }
}
