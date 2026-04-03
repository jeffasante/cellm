#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::rc::autoreleasepool;

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
    shared[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared[0] / float(n) + eps);
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
    pub pso_add_f32: ComputePipelineState,
    pub pso_silu_mul_f32: ComputePipelineState,
    x_buf: Option<Buffer>,
    w_buf: Option<Buffer>,
    out_buf: Option<Buffer>,
    /// Cache large tensor uploads (e.g. embedding table) by name.
    tensor_cache: std::collections::HashMap<String, Buffer>,
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
        let pso_add_f32   = build_pso_ops(&device, &lib, "add_f32_inplace")?;
        let pso_silu_mul_f32 = build_pso_ops(&device, &lib, "silu_mul_f32_inplace")?;
        Ok(Self {
            device, queue, _lib: lib,
            pso_rms_norm, pso_rope_adj, pso_rope_half, pso_mv_f16, pso_mv_i8,
            pso_add_f32, pso_silu_mul_f32,
            x_buf: None, w_buf: None, out_buf: None,
            tensor_cache: std::collections::HashMap::new(),
        })
    }

    // RMSNorm 
    /// x: f32 input, w_f16: raw f16 weights, w_add_one: add 1.0 to weights (Gemma).
    pub fn rms_norm_f16w(
        &mut self,
        x: &[f32],
        w_f16: &[u16],
        eps: f32,
        w_add_one: bool,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        let n = x.len();
        ensure_buf_f32(&self.device, &mut self.x_buf, n)?;
        ensure_buf_u16(&self.device, &mut self.w_buf, n)?;
        ensure_buf_f32(&self.device, &mut self.out_buf, n)?;

        let xb = self.x_buf.as_ref().unwrap();
        let wb = self.w_buf.as_ref().unwrap();
        let ob = self.out_buf.as_ref().unwrap();

        write_f32(xb, x)?;
        write_u16(wb, w_f16)?;

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
        &mut self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        pos: usize,
        theta: f32,
    ) -> anyhow::Result<()> {
        let n = x.len();
        ensure_buf_f32(&self.device, &mut self.x_buf, n)?;
        let xb = self.x_buf.as_ref().unwrap();
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
        &mut self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos: usize,
        theta: f32,
    ) -> anyhow::Result<()> {
        let n = x.len();
        ensure_buf_f32(&self.device, &mut self.x_buf, n)?;
        let xb = self.x_buf.as_ref().unwrap();
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
        &mut self,
        x: &[f32],
        embed_f16: &[u16],
        vocab: usize,
        hidden: usize,
        cache_key: &str,
        logits_out: &mut [f32],
    ) -> anyhow::Result<()> {
        // Upload embedding table once; reuse on subsequent tokens.
        if !self.tensor_cache.contains_key(cache_key) {
            let bytes = (embed_f16.len() * 2) as u64;
            let buf = self.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
            let ptr = buf.contents() as *mut u16;
            if ptr.is_null() {
                anyhow::bail!("MetalOps logits_f16: null buffer");
            }
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr, embed_f16.len()) };
            dst.copy_from_slice(embed_f16);
            self.tensor_cache.insert(cache_key.to_string(), buf);
        }
        let embed_buf = self.tensor_cache.get(cache_key).unwrap();

        ensure_buf_f32(&self.device, &mut self.x_buf, hidden)?;
        ensure_buf_f32(&self.device, &mut self.out_buf, vocab)?;
        let xb  = self.x_buf.as_ref().unwrap();
        let ob  = self.out_buf.as_ref().unwrap();
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
        &mut self,
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

        if !self.tensor_cache.contains_key(&wkey) {
            let wbuf = upload_i8(&self.device, embed_i8)?;
            self.tensor_cache.insert(wkey.clone(), wbuf);
            let sbuf = upload_u16(&self.device, scales_f16)?;
            self.tensor_cache.insert(skey.clone(), sbuf);
        }
        let wbuf = self.tensor_cache.get(&wkey).unwrap();
        let sbuf = self.tensor_cache.get(&skey).unwrap();

        ensure_buf_f32(&self.device, &mut self.x_buf, hidden)?;
        ensure_buf_f32(&self.device, &mut self.out_buf, vocab)?;
        let xb = self.x_buf.as_ref().unwrap();
        let ob = self.out_buf.as_ref().unwrap();
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
        enc.set_bytes(2, 4, n_ptr);
        enc.set_bytes(3, 4, eps_ptr);
        enc.set_buffer(4, Some(out_buf), 0);
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
}
