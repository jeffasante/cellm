// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! WebGPU compute-kernel backend for cellm.
//
// Provides GPU-accelerated matmul, RMS norm, RoPE, and softmax kernels
// via WGSL compute shaders dispatched through the `wgpu` crate.
// Targets `wasm32-unknown-unknown` with WebGPU and also works on native
// platforms (Vulkan/Metal/DX12) through wgpu's cross-platform layer.


// ---------------------------------------------------------------------------
// WGSL shader sources (embedded as constants)
// ---------------------------------------------------------------------------

/// Matrix-vector multiply: out[m] = sum_k(A[m,k] * x[k])
/// A is stored row-major in buffer A; x is input vector; out is output.
const WGSL_MATMUL_F32: &str = r#"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    M: u32,   // rows of A (output size)
    K: u32,   // cols of A (input size)
};

@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.M { return; }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k += 1u) {
        sum += A[row * params.K + k] * x[k];
    }
    out[row] = sum;
}
"#;

/// Matrix-vector multiply with f16 weight data (packed as u16).
/// A is [M × K] u16 elements (half-precision), x is [K] f32, out is [M] f32.
const WGSL_MATMUL_F16: &str = r#"
@group(0) @binding(0) var<storage, read> A: array<u32>;  // packed: two u16 per u32
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    M: u32,
    K: u32,
};

@group(0) @binding(3) var<uniform> params: Uniforms;

// Unpack IEEE-754 binary16 from a u32 word without requiring shader-f16.
fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits & 0x8000u) << 16u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x03FFu;

    if exp == 0u {
        if mant == 0u {
            return bitcast<f32>(sign);
        }
        let mag = f32(mant) * 5.960464477539063e-8; // 2^-24
        return select(mag, -mag, sign != 0u);
    }
    if exp == 31u {
        return bitcast<f32>(sign | 0x7F800000u | (mant << 13u));
    }

    return bitcast<f32>(sign | ((exp + 112u) << 23u) | (mant << 13u));
}

fn unpack16(word: u32, idx: u32) -> f32 {
    let bits = select(word & 0xFFFFu, word >> 16u, (idx & 1u) == 1u);
    return f16_bits_to_f32(bits);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.M { return; }
    var sum: f32 = 0.0;

    var flat_idx = row * params.K;
    for (var k: u32 = 0u; k < params.K; k += 1u) {
        let word = A[flat_idx >> 1u];
        sum += unpack16(word, flat_idx) * x[k];
        flat_idx += 1u;
    }
    out[row] = sum;
}
"#;

/// RMS normalization: out[i] = x[i] * rsqrt(mean(x²) + eps) * weight[i]
const WGSL_RMS_NORM: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    N: u32,
    eps: f32,
};

@group(0) @binding(3) var<uniform> params: Uniforms;

var<workgroup> partial_sq: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var local_sq: f32 = 0.0;
    var idx = lid.x;
    while idx < params.N {
        let v = x[idx];
        local_sq += v * v;
        idx = idx + 64u;
    }

    partial_sq[lid.x] = local_sq;
    workgroupBarrier();

    var step: u32 = 32u;
    while step > 0u {
        if lid.x < step {
            partial_sq[lid.x] += partial_sq[lid.x + step];
        }
        workgroupBarrier();
        step = step / 2u;
    }

    let mean_sq = partial_sq[0] / f32(params.N);
    let rsqrt = 1.0 / sqrt(mean_sq + params.eps);

    idx = lid.x;
    while idx < params.N {
        out[idx] = x[idx] * rsqrt * weight[idx];
        idx = idx + 64u;
    }
}
"#;

/// RoPE (half rotary): apply RoPE to last `rotary_dim` elements of each head.
/// x layout: [n_heads, head_dim] interleaved f32. RoPE on last `rotary_dim` dims.
const WGSL_ROPE_HALF: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;

struct Uniforms {
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    pos: u32,
    theta: f32,
};

@group(0) @binding(1) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x / params.rotary_dim;
    let i = gid.x % params.rotary_dim;
    if h >= params.n_heads || i >= params.rotary_dim / 2u { return; }

    // Index into this head's RoPE region
    let head_start = h * params.head_dim;
    let rope_start = head_start + params.head_dim - params.rotary_dim;

    let inv_freq = pow(params.theta, -f32(2u * i) / f32(params.rotary_dim));
    let angle = f32(params.pos) * inv_freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let x0 = x[rope_start + i];
    let x1 = x[rope_start + params.rotary_dim / 2u + i];

    // Python convention: R(-θ) = [[cos, sin], [-sin, cos]]
    x[rope_start + i] = x0 * cos_val + x1 * sin_val;
    x[rope_start + params.rotary_dim / 2u + i] = -x0 * sin_val + x1 * cos_val;
}
"#;

/// Softmax: x[i] = exp(x[i] - max) / sum(exp(x[i] - max))
const WGSL_SOFTMAX: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;

struct Uniforms {
    N: u32,
};

@group(0) @binding(1) var<uniform> params: Uniforms;

var<workgroup> wg_max: f32;
var<workgroup> wg_sum: f32;
var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var local = -1e30f;
    var idx = lid.x;
    while idx < params.N {
        local = max(local, x[idx]);
        idx = idx + 64u;
    }
    scratch[lid.x] = local;
    workgroupBarrier();

    var step: u32 = 32u;
    while step > 0u {
        if lid.x < step {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + step]);
        }
        workgroupBarrier();
        step = step / 2u;
    }

    if lid.x == 0u {
        wg_max = scratch[0];
    }
    workgroupBarrier();

    let mx = wg_max;

    var exp_sum: f32 = 0.0;
    idx = lid.x;
    while idx < params.N {
        let v = exp(x[idx] - mx);
        x[idx] = v;
        exp_sum += v;
        idx = idx + 64u;
    }
    scratch[lid.x] = exp_sum;
    workgroupBarrier();

    step = 32u;
    while step > 0u {
        if lid.x < step {
            scratch[lid.x] += scratch[lid.x + step];
        }
        workgroupBarrier();
        step = step / 2u;
    }

    if lid.x == 0u {
        wg_sum = scratch[0];
    }
    workgroupBarrier();

    idx = lid.x;
    while idx < params.N {
        x[idx] = x[idx] / wg_sum;
        idx = idx + 64u;
    }
}
"#;

/// Silu-mul: gate[i] = silu(gate[i]) * up[i]
const WGSL_SILU_MUL: &str = r#"
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;

struct Uniforms {
    N: u32,
};

@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.N { return; }
    let g = gate[i];
    let s = 1.0 / (1.0 + exp(-g));
    gate[i] = (g * s) * up[i];
}
"#;

/// Batch matrix-vector multiply: out[r, o] = sum_k weight[o, k] * x[r, k]
/// weight is [M × K] row-major, x is [R × K] row-major, out is [R × M] row-major.
/// Each workgroup handles one output row (one row in x maps to one row in out).
const WGSL_MATMUL_BATCH_F32: &str = r#"
@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    M: u32,   // output cols (weight rows)
    K: u32,   // input cols  (weight cols)
    R: u32,   // number of input rows
};

@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn batch_matmul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;      // output column (0..M)
    let row = gid.y;      // input row (0..R)
    if col >= params.M || row >= params.R { return; }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k += 1u) {
        sum += weight[col * params.K + k] * x[row * params.K + k];
    }
    out[row * params.M + col] = sum;
}
"#;

/// Batch matrix-vector multiply with f16 weights.
const WGSL_MATMUL_BATCH_F16: &str = r#"
@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    M: u32,   // output cols
    K: u32,   // input cols
    R: u32,   // number of input rows
};

@group(0) @binding(3) var<uniform> params: Uniforms;

fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits & 0x8000u) << 16u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x03FFu;
    if exp == 0u {
        if mant == 0u { return bitcast<f32>(sign); }
        return select(f32(mant) * 5.960464477539063e-8, -f32(mant) * 5.960464477539063e-8, sign != 0u);
    }
    if exp == 31u { return bitcast<f32>(sign | 0x7F800000u | (mant << 13u)); }
    return bitcast<f32>(sign | ((exp + 112u) << 23u) | (mant << 13u));
}

fn unpack16(word: u32, idx: u32) -> f32 {
    let bits = select(word & 0xFFFFu, word >> 16u, (idx & 1u) == 1u);
    return f16_bits_to_f32(bits);
}

@compute @workgroup_size(64)
fn batch_matmul_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;      // output column
    let row = gid.y;      // input row
    if col >= params.M || row >= params.R { return; }
    var sum: f32 = 0.0;
    let base_a = col * params.K;
    let base_x = row * params.K;
    for (var k: u32 = 0u; k < params.K; k += 1u) {
        let word = A[(base_a + k) >> 1u];
        sum += unpack16(word, base_a + k) * x[base_x + k];
    }
    out[row * params.M + col] = sum;
}
"#;

/// Batch Layer Normalization: for each row, compute mean/variance, then normalize.
/// out[r, i] = (x[r, i] - mean(r)) * rsqrt(var(r) + eps) * weight[i] + bias[i]
const WGSL_LAYER_NORM_BATCH: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Uniforms {
    cols: u32,
    rows: u32,
    eps: f32,
};

@group(0) @binding(4) var<uniform> params: Uniforms;

var<workgroup> wg_mean: f32;
var<workgroup> wg_var: f32;

@compute @workgroup_size(64)
fn batch_layer_norm(@builtin(global_invocation_id) gid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.cols || row >= params.rows { return; }

    let base = row * params.cols;

    // Compute mean (single-thread per row is fine for small cols via loop)
    // Use workgroup reduction when cols <= 64 for the first lane per row
    var mean: f32 = 0.0;
    var var_sum: f32 = 0.0;
    for (var c: u32 = 0u; c < params.cols; c += 1u) {
        let v = x[base + c];
        mean += v;
    }
    mean = mean / f32(params.cols);

    for (var c2: u32 = 0u; c2 < params.cols; c2 += 1u) {
        let d = x[base + c2] - mean;
        var_sum += d * d;
    }
    let inv = 1.0 / sqrt(var_sum / f32(params.cols) + params.eps);

    let v = x[base + col];
    out[base + col] = (v - mean) * inv * weight[col] + bias[col];
}
"#;

/// Batch RMS Normalization (per row).
/// out[r, i] = x[r, i] * rsqrt(mean(x[r]²) + eps) * weight[i]
const WGSL_RMS_NORM_BATCH: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    cols: u32,
    rows: u32,
    eps: f32,
};

@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn batch_rms_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.cols || row >= params.rows { return; }

    let base = row * params.cols;

    var ms: f32 = 0.0;
    for (var c: u32 = 0u; c < params.cols; c += 1u) {
        let v = x[base + c];
        ms += v * v;
    }
    ms = ms / f32(params.cols);
    let inv = 1.0 / sqrt(ms + params.eps);

    out[base + col] = x[base + col] * inv * weight[col];
}
"#;

// ---------------------------------------------------------------------------
// WebGPU backend state
// ---------------------------------------------------------------------------

/// Holds the WebGPU device, queue, and compiled shaders for inference.
pub struct WebGpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipelines: WebGpuPipelines,
    limits: wgpu::Limits,
}

struct WebGpuPipelines {
    matmul_f32: wgpu::ComputePipeline,
    matmul_f16: wgpu::ComputePipeline,
    matmul_batch_f32: wgpu::ComputePipeline,
    matmul_batch_f16: wgpu::ComputePipeline,
    rms_norm: wgpu::ComputePipeline,
    rms_norm_batch: wgpu::ComputePipeline,
    layer_norm_batch: wgpu::ComputePipeline,
    rope_half: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    silu_mul: wgpu::ComputePipeline,
}

impl WebGpuBackend {
    /// Create a WebGPU backend with pre-compiled shaders.
    /// Returns `None` if WebGPU is unavailable (falls back to CPU).
    pub async fn create() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("cellm WebGPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 256 * 1024 * 1024, // 256 MB
                        max_buffer_size: 256 * 1024 * 1024,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .ok()?;

        let limits = device.limits();

        let pipelines = Self::build_pipelines(&device);

        Some(Self {
            device,
            queue,
            pipelines,
            limits,
        })
    }

    fn build_pipelines(device: &wgpu::Device) -> WebGpuPipelines {
        let make = |label: &str, source: &str, entry: &str| -> wgpu::ComputePipeline {
            let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &sm,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        WebGpuPipelines {
            matmul_f32: make("matmul_f32", WGSL_MATMUL_F32, "main"),
            matmul_f16: make("matmul_f16", WGSL_MATMUL_F16, "main"),
            matmul_batch_f32: make("matmul_batch_f32", WGSL_MATMUL_BATCH_F32, "batch_matmul_f32"),
            matmul_batch_f16: make("matmul_batch_f16", WGSL_MATMUL_BATCH_F16, "batch_matmul_f16"),
            rms_norm: make("rms_norm", WGSL_RMS_NORM, "main"),
            rms_norm_batch: make("rms_norm_batch", WGSL_RMS_NORM_BATCH, "batch_rms_norm"),
            layer_norm_batch: make("layer_norm_batch", WGSL_LAYER_NORM_BATCH, "batch_layer_norm"),
            rope_half: make("rope_half", WGSL_ROPE_HALF, "main"),
            softmax: make("softmax", WGSL_SOFTMAX, "main"),
            silu_mul: make("silu_mul", WGSL_SILU_MUL, "main"),
        }
    }

    // -----------------------------------------------------------------------
    // Upload helpers
    // -----------------------------------------------------------------------

    /// Upload f32 data to a GPU buffer with copy_dst + storage usage.
    pub fn upload_f32(&self, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    /// Upload packed uniform words.
    pub fn upload_uniform_u32(&self, data: &[u32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create an f32 storage buffer (uninitialized, for output).
    pub fn create_f32_output(&self, count: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Upload u16 data to GPU (packed as u32: two u16 per u32 for WGSL).
    pub fn upload_f16(&self, data: &[u16]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let n = data.len();
        let mut packed = Vec::with_capacity(n.max(1) / 2);
        for chunk in data.chunks(2) {
            let lo = chunk[0] as u32;
            let hi = if chunk.len() > 1 { (chunk[1] as u32) << 16 } else { 0 };
            packed.push(lo | hi);
        }
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Read back f32 results from a GPU buffer (async).
    pub async fn download_f32(&self, buf: &wgpu::Buffer, count: usize, out: &mut [f32]) {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, (count * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        #[cfg(target_arch = "wasm32")]
        self.device.poll(wgpu::Maintain::Poll);
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let view = slice.get_mapped_range();
        out.copy_from_slice(bytemuck::cast_slice(&view));
        drop(view);
        staging.unmap();
    }

    // -----------------------------------------------------------------------
    // Kernel dispatch
    // -----------------------------------------------------------------------

    /// matmul f32: out[out_dim] = weight[out_dim × in_dim] · x[in_dim]
    /// Weight is on GPU (cached), x is uploaded fresh each call.
    pub async fn matmul_f32(
        &self,
        w_buf: &wgpu::Buffer,
        out_dim: u32,
        in_dim: u32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output(out_dim as usize);

        let uniforms = [out_dim, in_dim, 0u32, 0u32];
        let u_buf = self.upload_uniform_u32(&uniforms);

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.matmul_f32.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.matmul_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_count = (out_dim + 63) / 64;
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, out_dim as usize, out).await;
    }

    /// matmul f16: weight is stored as f16 (u16) on GPU.
    pub async fn matmul_f16(
        &self,
        w_buf: &wgpu::Buffer,
        out_dim: u32,
        in_dim: u32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output(out_dim as usize);

        let uniforms = [out_dim, in_dim, 0u32, 0u32];
        let u_buf = self.upload_uniform_u32(&uniforms);

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.matmul_f16.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.matmul_f16);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((out_dim + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, out_dim as usize, out).await;
    }

    /// RMS norm: out[i] = x[i] * rsqrt(mean(x²) + eps) * weight[i]
    pub async fn rms_norm(&self, weight_buf: &wgpu::Buffer, eps: f32, x: &[f32], out: &mut [f32]) {
        let n = x.len() as u32;
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output(n as usize);

        let uniforms = [n, eps.to_bits(), 0u32, 0u32];
        let u_buf = self.upload_uniform_u32(&uniforms);

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.rms_norm);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, n as usize, out).await;
    }

    /// RoPE half: apply RoPE to last `rotary_dim` elements of each head.
    pub async fn rope_half(
        &self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos: usize,
        theta: f32,
    ) {
        let x_buf = self.upload_f32(x);
        let uniforms = [
            n_heads as u32,
            head_dim as u32,
            rotary_dim as u32,
            pos as u32,
            theta.to_bits(),
            0u32,
            0u32,
            0u32,
        ];
        let u_buf = self.upload_uniform_u32(&uniforms);

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.rope_half.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.rope_half);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = n_heads as u32 * rotary_dim as u32;
            pass.dispatch_workgroups((total + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&x_buf, x.len(), x).await;
    }

    /// Softmax in-place.
    pub async fn softmax(&self, data: &mut [f32]) {
        let n = data.len() as u32;
        let buf = self.upload_f32(data);
        let uniforms = [n, 0u32, 0u32, 0u32];
        let u_buf = self.upload_uniform_u32(&uniforms);

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.softmax.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.softmax);
            pass.set_bind_group(0, &bind_group, &[]);
            // Single workgroup for small softmax (< 64 elements)
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&buf, n as usize, data).await;
    }

    /// Silu-mul in-place: gate[i] = silu(gate[i]) * up[i]
    pub async fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
        let n = gate.len() as u32;
        let gate_buf = self.upload_f32(gate);
        let up_buf = self.upload_f32(up);
        let uniforms = [n, 0u32, 0u32, 0u32];
        let u_buf = self.upload_uniform_u32(&uniforms);

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.silu_mul.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gate_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: up_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.silu_mul);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&gate_buf, n as usize, gate).await;
    }

    // -----------------------------------------------------------------------
    // Batch kernels for vision encoder (multiple input rows per dispatch)
    // -----------------------------------------------------------------------

    /// Batch matmul: out[r, o] = sum_k weight[o, k] * x[r, k]
    /// weight is [out_dim × in_dim] row-major (cached on GPU in w_buf).
    /// x is [rows × in_dim] row-major, out is [rows × out_dim] row-major.
    pub async fn matmul_batch_f32(
        &self,
        w_buf: &wgpu::Buffer,
        rows: u32,
        out_dim: u32,
        in_dim: u32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let total = rows * out_dim;
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output(total as usize);

        let uniforms = [out_dim, in_dim, rows, 0u32];
        let u_buf = self.upload_uniform_u32(&uniforms);

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_batch_f32"),
                layout: &self.pipelines.matmul_batch_f32.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.matmul_batch_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (out_dim + 63) / 64;
            pass.dispatch_workgroups(wg_x, rows, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, total as usize, out).await;
    }

    pub async fn matmul_batch_f16(
        &self,
        w_buf: &wgpu::Buffer,
        rows: u32,
        out_dim: u32,
        in_dim: u32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let total = rows * out_dim;
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output(total as usize);

        let uniforms = [out_dim, in_dim, rows, 0u32];
        let u_buf = self.upload_uniform_u32(&uniforms);

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_batch_f16"),
                layout: &self.pipelines.matmul_batch_f16.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.matmul_batch_f16);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (out_dim + 63) / 64;
            pass.dispatch_workgroups(wg_x, rows, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, total as usize, out).await;
    }

    /// Batch layer norm: out[r,i] = (x[r,i]-mean)*rsqrt(var+eps)*w[i]+b[i]
    pub async fn layer_norm_batch(
        &self,
        w_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        rows: u32,
        cols: u32,
        eps: f32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output((rows * cols) as usize);

        let uniforms = [cols, rows, eps.to_bits(), 0u32];
        let u_buf = self.upload_uniform_u32(&uniforms);

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("layer_norm_batch"),
                layout: &self.pipelines.layer_norm_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.layer_norm_batch);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (cols + 63) / 64;
            pass.dispatch_workgroups(wg_x, rows, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, (rows * cols) as usize, out).await;
    }

    /// Batch RMS norm: out[r,i] = x[r,i] * rsqrt(mean(x[r]²)+eps) * w[i]
    /// Batch RMS norm (per row).
    pub async fn rms_norm_batch(
        &self,
        w_buf: &wgpu::Buffer,
        rows: u32,
        cols: u32,
        eps: f32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output((rows * cols) as usize);

        let uniforms = [cols, rows, eps.to_bits(), 0u32];
        let u_buf = self.upload_uniform_u32(&uniforms);

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rms_norm_batch"),
                layout: &self.pipelines.rms_norm_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.rms_norm_batch);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (cols + 63) / 64;
            pass.dispatch_workgroups(wg_x, rows, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, (rows * cols) as usize, out).await;
    }
}

// ---------------------------------------------------------------------------
// VisionWebGpu: high-level wrapper for vision-encoder GPU acceleration
// ---------------------------------------------------------------------------

/// Manages WebGPU weight caching and dispatch for vision-encoder linear
/// projections and normalizations.  Created from a `WebGpuBackend` and
/// passed into `vlm::run_vision_cellm` via `LinearBackend::WebGpu`.
///
/// Call `into_gpu()` after use to recover the `WebGpuBackend` for text inference.
pub struct VisionWebGpu {
    pub gpu: WebGpuBackend,
    weight_cache: std::collections::HashMap<(usize, usize, usize), wgpu::Buffer>,
}

impl VisionWebGpu {
    pub fn new(gpu: WebGpuBackend) -> Self {
        Self {
            gpu,
            weight_cache: std::collections::HashMap::new(),
        }
    }

    /// Recover the WebGpuBackend after vision processing is complete.
    pub fn into_gpu(self) -> WebGpuBackend {
        self.gpu
    }

    /// Upload a weight matrix (or retrieve cached) and run batch matmul.
    /// weight is [out_dim × in_dim] row-major; x is [rows × in_dim]; out is [rows × out_dim].
    pub async fn linear_batch(
        &mut self,
        x: &[f32],
        rows: usize,
        in_dim: usize,
        weight: &[f32],
        out_dim: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        if weight.is_empty() || rows == 0 || in_dim == 0 || out_dim == 0 {
            return;
        }

        let key = (weight.as_ptr() as usize, in_dim, out_dim);
        let w_buf = self.weight_cache.entry(key).or_insert_with(|| {
            // Store weight transposed: WGSL reads weight[col * K + k]
            // where weight is [M × K] with M=out_dim, K=in_dim.
            // Our input weight is [out_dim × in_dim] row-major, which
            // matches exactly — weight[col * in_dim + k] is correct.
            self.gpu.upload_f32(weight)
        });

        self.gpu.matmul_batch_f32(
            w_buf,
            rows as u32,
            out_dim as u32,
            in_dim as u32,
            x,
            out,
        ).await;

        if let Some(b) = bias {
            for r in 0..rows {
                let row = &mut out[r * out_dim..(r + 1) * out_dim];
                for c in 0..out_dim {
                    row[c] += b[c];
                }
            }
        }
    }

    /// Upload weight_f16 (or retrieve cached) and run batch matmul with f16 weights.
    pub async fn linear_batch_f16(
        &mut self,
        x: &[f32],
        rows: usize,
        in_dim: usize,
        weight_f16: &[u16],
        out_dim: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        if weight_f16.is_empty() || rows == 0 || in_dim == 0 || out_dim == 0 {
            return;
        }

        let key = (weight_f16.as_ptr() as usize, in_dim, out_dim);
        let w_buf = self.weight_cache.entry(key).or_insert_with(|| {
            self.gpu.upload_f16(weight_f16)
        });

        self.gpu.matmul_batch_f16(
            w_buf,
            rows as u32,
            out_dim as u32,
            in_dim as u32,
            x,
            out,
        ).await;

        if let Some(b) = bias {
            for r in 0..rows {
                let row = &mut out[r * out_dim..(r + 1) * out_dim];
                for c in 0..out_dim {
                    row[c] += b[c];
                }
            }
        }
    }

    /// Upload weight+bias (or retrieve cached) and run batch layer norm.
    pub async fn layer_norm_batch(
        &mut self,
        x: &[f32],
        rows: usize,
        cols: usize,
        weight: &[f32],
        bias: &[f32],
        eps: f32,
        out: &mut [f32],
    ) {
        if cols == 0 || rows == 0 {
            return;
        }

        let w_key = (weight.as_ptr() as usize, cols, 0);
        let b_key = (bias.as_ptr() as usize, cols, 1);

        // Avoid double mutable borrow by checking existence first
        if !self.weight_cache.contains_key(&w_key) {
            self.weight_cache.insert(w_key, self.gpu.upload_f32(weight));
        }
        if !self.weight_cache.contains_key(&b_key) {
            self.weight_cache.insert(b_key, self.gpu.upload_f32(bias));
        }

        let w_buf = &self.weight_cache[&w_key];
        let b_buf = &self.weight_cache[&b_key];

        self.gpu.layer_norm_batch(
            w_buf, b_buf,
            rows as u32, cols as u32,
            eps, x, out,
        ).await;
    }

    /// Upload weight (or retrieve cached) and run batch RMS norm.
    pub async fn rms_norm_batch(
        &mut self,
        x: &[f32],
        rows: usize,
        cols: usize,
        weight: &[f32],
        eps: f32,
        out: &mut [f32],
    ) {
        if cols == 0 || rows == 0 {
            return;
        }

        let w_key = (weight.as_ptr() as usize, cols, 2);
        let w_buf = self.weight_cache.entry(w_key).or_insert_with(|| {
            self.gpu.upload_f32(weight)
        });

        self.gpu.rms_norm_batch(
            w_buf,
            rows as u32, cols as u32,
            eps, x, out,
        ).await;
    }
}
