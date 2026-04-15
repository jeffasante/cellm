use std::fs;
fn main() {
    let content = r###"#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize, ComputeCommandEncoderRef};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::rc::autoreleasepool;

pub struct MetalKernels;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub type MetalBuffer = Buffer;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalBuffer;

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalKernels {
    pub fn smoke_test_add_f32() -> anyhow::Result<()> {
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("Metal: no device"))?;
        let queue = device.new_command_queue();
        let src = "kernel void add(device const float* a, device const float* b, device float* out, uint id [[thread_position_in_grid]]) { out[id] = a[id] + b[id]; }";
        let options = metal::CompileOptions::new();
        let lib = device.new_library_with_source(src, &options).unwrap();
        let f = lib.get_function("add", None).unwrap();
        let pso = device.new_compute_pipeline_state_with_function(&f).unwrap();
        let n = 1024usize;
        let bytes = (n * 4) as u64;
        let a = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let b = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let out = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        autoreleasepool(|| {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&a), 0);
            enc.set_buffer(1, Some(&b), 0);
            enc.set_buffer(2, Some(&out), 0);
            enc.dispatch_threads(MTLSize{width: n as u64, height:1, depth:1}, MTLSize{width: 32, height:1, depth:1});
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });
        Ok(())
    }
    pub fn create_matmul() -> anyhow::Result<MetalMatmul> {
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("Metal: no device"))?;
        let queue = device.new_command_queue();
        let src = "kernel void mm(device const float* a, device const float* b, device float* out, constant uint& m, constant uint& n, constant uint& k, uint2 gid [[thread_position_in_grid]]) { uint row=gid.y; uint col=gid.x; if(row>=m||col>=n)return; float acc=0; for(uint i=0;i<k;++i)acc+=a[row*k+i]*b[i*n+col]; out[row*n+col]=acc; }";
        let options = metal::CompileOptions::new();
        let lib = device.new_library_with_source(src, &options).unwrap();
        let f = lib.get_function("mm", None).unwrap();
        let pso = device.new_compute_pipeline_state_with_function(&f).unwrap();
        Ok(MetalMatmul { queue, _lib: lib, pso })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalKernels {
    pub fn smoke_test_add_f32() -> anyhow::Result<()> { anyhow::bail!("No Metal") }
    pub fn create_matmul() -> anyhow::Result<MetalMatmul> { anyhow::bail!("No Metal") }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalMatmul { pub queue: CommandQueue, pub _lib: Library, pub pso: ComputePipelineState }
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalMatmul;

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
    pub pso_add_f32: ComputePipelineState,
    pub pso_silu_mul_f32: ComputePipelineState,
    x_buf: Option<Buffer>,
    w_buf: Option<Buffer>,
    out_buf: Option<Buffer>,
    pub tensor_cache: std::collections::HashMap<String, Buffer>,
}

const ELEM_OPS_SHADER: &str = r##"
#include <metal_stdlib>
using namespace metal;
kernel void add_f32_inplace(device float* a [[buffer(0)]], device const float* b [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < n) a[gid] += b[gid]; }
kernel void silu_mul_f32_inplace(device float* a [[buffer(0)]], device const float* b [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < n) { float x = a[gid]; a[gid] = (x / (1.0f + exp(-x))) * b[gid]; } }
kernel void rms_norm_f16w(device const float* x [[buffer(0)]], device const ushort* w [[buffer(1)]], device float* out [[buffer(2)]], constant uint& n [[buffer(3)]], constant float& eps [[buffer(4)]], constant uint& w_add_one[[buffer(5)]], uint tid [[thread_index_in_threadgroup]], uint tgsize [[threads_per_threadgroup]], threadgroup float* shared [[threadgroup(0)]]) {
    float local_sq = 0.0f;
    for (uint i = tid; i < n; i += tgsize) { float v = x[i]; local_sq += v * v; }
    shared[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1; stride > 0; stride >>= 1) { if (tid < stride) shared[tid] += shared[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv_rms = rsqrt(shared[0] / float(n) + eps);
    for (uint i = tid; i < n; i += tgsize) { float wf = float(as_type<half>(w[i])); if (w_add_one) wf += 1.0f; out[i] = x[i] * inv_rms * wf; }
}
kernel void rope_adj_f32(device float* x [[buffer(0)]], constant uint& nh [[buffer(1)]], constant uint& hd [[buffer(2)]], constant uint& p [[buffer(3)]], constant float& th [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    uint h = gid / (hd/2); uint d = gid % (hd/2); if (h >= nh) return;
    float ang = float(p) * pow(th, -(2.0f * float(d)) / float(hd));
    float ca = cos(ang), sa = sin(ang);
    uint idx = h * hd + d * 2;
    float x0 = x[idx], x1 = x[idx+1];
    x[idx] = x0 * ca - x1 * sa; x[idx+1] = x0 * sa + x1 * ca;
}
kernel void rope_half_f32(device float* x [[buffer(0)]], constant uint& nh [[buffer(1)]], constant uint& hd [[buffer(2)]], constant uint& rd [[buffer(3)]], constant uint& p [[buffer(4)]], constant float& th [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    uint h = gid / (rd/2); uint d = gid % (rd/2); if (h >= nh) return;
    float ang = float(p) * pow(th, -(2.0f * float(d)) / float(rd));
    float ca = cos(ang), sa = sin(ang);
    uint base = h * hd; float x0 = x[base + d], x1 = x[base + rd/2 + d];
    x[base + d] = x0 * ca - x1 * sa; x[base + rd/2 + d] = x1 * ca + x0 * sa;
}
kernel void mv_f16(device const ushort* A [[buffer(0)]], device const float* x [[buffer(1)]], device float* out [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    device const ushort* r = A + gid * cols;
    float acc = 0.0f; for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(r[i]));
    out[gid] = acc;
}
kernel void mv_i8(device const char* A [[buffer(0)]], device const ushort* S [[buffer(1)]], device const float* x [[buffer(2)]], device float* out [[buffer(3)]], constant uint& rows [[buffer(4)]], constant uint& cols [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    float s = float(as_type<half>(S[gid]));
    device const char* r = A + gid * cols;
    float acc = 0.0f; for (uint i = 0; i < cols; ++i) acc += x[i] * float(r[i]);
    out[gid] = acc * s;
}
kernel void mv_qkv_f16(device const ushort* Aq, device const ushort* Ak, device const ushort* Av, device const float* x, device float* oq, device float* ok, device float* ov, constant uint& rq, constant uint& rkv, constant uint& c, uint gid [[thread_position_in_grid]]) {
    uint total = rq + rkv + rkv; if (gid >= total) return;
    if (gid < rq) { float acc = 0.0f; device const ushort* r = Aq + gid * c; for (uint i=0; i<c; ++i) acc += x[i]*float(as_type<half>(r[i])); oq[gid]=acc; return; }
    uint off = gid - rq; if (off < rkv) { float acc = 0.0f; device const ushort* r = Ak + off * c; for (uint i=0; i<c; ++i) acc += x[i]*float(as_type<half>(r[i])); ok[off]=acc; return; }
    off -= rkv; float acc = 0.0f; device const ushort* r = Av + off * c; for (uint i=0; i<c; ++i) acc += x[i]*float(as_type<half>(r[i])); ov[off]=acc;
}
kernel void mv2_f16(device const ushort* A0, device const ushort* A1, device const float* x, device float* o0, device float* o1, constant uint& r0, constant uint& r1, constant uint& c, uint gid [[thread_position_in_grid]]) {
    uint total = r0 + r1; if (gid >= total) return;
    if (gid < r0) { float acc = 0.0f; device const ushort* r = A0 + gid * c; for (uint i=0; i<c; ++i) acc += x[i]*float(as_type<half>(r[i])); o0[gid]=acc; return; }
    uint off = gid - r0; float acc = 0.0f; device const ushort* r = A1 + off * c; for (uint i=0; i<c; ++i) acc += x[i]*float(as_type<half>(r[i])); o1[off]=acc;
}
"##;

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalOps {
    pub fn create() -> anyhow::Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal"))?;
        let queue = device.new_command_queue();
        let options = metal::CompileOptions::new();
        let lib = device.new_library_with_source(ELEM_OPS_SHADER, &options).unwrap();
        Ok(Self {
            pso_rms_norm: device.new_compute_pipeline_state_with_function(&lib.get_function("rms_norm_f16w", None).unwrap()).unwrap(),
            pso_rope_adj: device.new_compute_pipeline_state_with_function(&lib.get_function("rope_adj_f32", None).unwrap()).unwrap(),
            pso_rope_half: device.new_compute_pipeline_state_with_function(&lib.get_function("rope_half_f32", None).unwrap()).unwrap(),
            pso_mv_f16: device.new_compute_pipeline_state_with_function(&lib.get_function("mv_f16", None).unwrap()).unwrap(),
            pso_mv_i8: device.new_compute_pipeline_state_with_function(&lib.get_function("mv_i8", None).unwrap()).unwrap(),
            pso_mv_qkv_f16: device.new_compute_pipeline_state_with_function(&lib.get_function("mv_qkv_f16", None).unwrap()).unwrap(),
            pso_mv2_f16: device.new_compute_pipeline_state_with_function(&lib.get_function("mv2_f16", None).unwrap()).unwrap(),
            pso_add_f32: device.new_compute_pipeline_state_with_function(&lib.get_function("add_f32_inplace", None).unwrap()).unwrap(),
            pso_silu_mul_f32: device.new_compute_pipeline_state_with_function(&lib.get_function("silu_mul_f32_inplace", None).unwrap()).unwrap(),
            device, queue, _lib: lib,
            x_buf: None, w_buf: None, out_buf: None,
            tensor_cache: std::collections::HashMap::new(),
        })
    }
    pub fn begin_pass(&self) -> anyhow::Result<()> { Ok(()) }
    pub fn end_pass(&self) -> anyhow::Result<()> { Ok(()) }
    pub fn rms_norm_f16w(&mut self, x: &[f32], w: &[u16], eps: f32, add_one: bool, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        let n = x.len(); let w_key = format!("rmsnorm.w.{}", cache_key);
        if !self.tensor_cache.contains_key(&w_key) { self.tensor_cache.insert(w_key.clone(), upload_u16(&self.device, w)?); }
        let wb = self.tensor_cache.get(&w_key).unwrap();
        ensure_buf_f32(&self.device, &mut self.x_buf, n)?; ensure_buf_f32(&self.device, &mut self.out_buf, n)?;
        let xb = self.x_buf.as_ref().unwrap(); let ob = self.out_buf.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer(); let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso_rms_norm); enc.set_threadgroup_memory_length(0, 1024*4);
            enc.set_buffer(0, Some(xb), 0); enc.set_buffer(1, Some(wb), 0); enc.set_buffer(2, Some(ob), 0);
            let n32 = n as u32; let eps32 = eps; let add32 = if add_one {1u32} else {0u32};
            enc.set_bytes(3, 4, &n32 as *const _ as *const _); enc.set_bytes(4, 4, &eps32 as *const _ as *const _); enc.set_bytes(5, 4, &add32 as *const _ as *const _);
            enc.dispatch_thread_groups(MTLSize{width:1, height:1, depth:1}, MTLSize{width:1024.min(n as u64), height:1, depth:1});
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }
    pub fn rope_half_f32(&mut self, x: &mut [f32], nh: usize, hd: usize, rd: usize, p: usize, th: f32) -> anyhow::Result<()> {
        let n = x.len(); ensure_buf_f32(&self.device, &mut self.x_buf, n)?; let xb = self.x_buf.as_ref().unwrap(); write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer(); let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso_rope_half); enc.set_buffer(0, Some(xb), 0);
            let p32=p as u32; let nh32=nh as u32; let hd32=hd as u32; let rd32=rd as u32;
            enc.set_bytes(1, 4, &nh32 as *const _ as *const _); enc.set_bytes(2, 4, &hd32 as *const _ as *const _); enc.set_bytes(3, 4, &rd32 as *const _ as *const _); enc.set_bytes(4, 4, &p32 as *const _ as *const _); enc.set_bytes(5, 4, &th as *const _ as *const _);
            let threads = (nh * rd / 2) as u64; enc.dispatch_threads(MTLSize{width: threads, height:1, depth:1}, MTLSize{width: 32.min(threads), height:1, depth:1});
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(xb, x)
    }
    pub fn rope_adj_f32(&self, x: &mut [f32], nh: usize, hd: usize, p: usize, th: f32) -> anyhow::Result<()> { anyhow::bail!("Mutable required for scratch") }
    pub fn rope_adj_f32_mut(&mut self, x: &mut [f32], nh: usize, hd: usize, p: usize, th: f32) -> anyhow::Result<()> {
        let n = x.len(); ensure_buf_f32(&self.device, &mut self.x_buf, n)?; let xb = self.x_buf.as_ref().unwrap(); write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer(); let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso_rope_adj); enc.set_buffer(0, Some(xb), 0);
            let p32=p as u32; let nh32=nh as u32; let hd32=hd as u32;
            enc.set_bytes(1, 4, &nh32 as *const _ as *const _); enc.set_bytes(2, 4, &hd32 as *const _ as *const _); enc.set_bytes(3, 4, &p32 as *const _ as *const _); enc.set_bytes(4, 4, &th as *const _ as *const _);
            let threads = (nh * hd / 2) as u64; enc.dispatch_threads(MTLSize{width: threads, height:1, depth:1}, MTLSize{width: 32.min(threads), height:1, depth:1});
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(xb, x)
    }
    pub fn logits_f16(&mut self, x: &[f32], w: &[u16], v: usize, h: usize, k: &str, out: &mut [f32]) -> anyhow::Result<()> {
        let w_key = format!("logits.f16.w.{}", k); if !self.tensor_cache.contains_key(&w_key) { self.tensor_cache.insert(w_key.clone(), upload_u16(&self.device, w)?); }
        let wb = self.tensor_cache.get(&w_key).unwrap(); ensure_buf_f32(&self.device, &mut self.x_buf, h)?; ensure_buf_f32(&self.device, &mut self.out_buf, v)?;
        let xb = self.x_buf.as_ref().unwrap(); let ob = self.out_buf.as_ref().unwrap(); write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer(); let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso_mv_f16); enc.set_buffer(0, Some(wb), 0); enc.set_buffer(1, Some(xb), 0); enc.set_buffer(2, Some(ob), 0);
            let v32 = v as u32; let h32 = h as u32; enc.set_bytes(3, 4, &v32 as *const _ as *const _); enc.set_bytes(4, 4, &h32 as *const _ as *const _);
            enc.dispatch_threads(MTLSize{width: v as u64, height:1, depth:1}, MTLSize{width: 32, height:1, depth:1});
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }
    pub fn logits_i8(&mut self, x: &[f32], w: &[i8], s: &[u16], v: usize, h: usize, k: &str, out: &mut [f32]) -> anyhow::Result<()> {
        let w_key = format!("logits.i8.w.{}", k); let s_key = format!("logits.i8.s.{}", k);
        if !self.tensor_cache.contains_key(&w_key) { self.tensor_cache.insert(w_key.clone(), upload_i8(&self.device, w)?); self.tensor_cache.insert(s_key.clone(), upload_u16(&self.device, s)?); }
        let wb = self.tensor_cache.get(&w_key).unwrap(); let sb = self.tensor_cache.get(&s_key).unwrap();
        ensure_buf_f32(&self.device, &mut self.x_buf, h)?; ensure_buf_f32(&self.device, &mut self.out_buf, v)?;
        let xb = self.x_buf.as_ref().unwrap(); let ob = self.out_buf.as_ref().unwrap(); write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer(); let enc = cb.new_compute_command_encoder();
            self.encode_mv_i8(enc, wb, sb, xb, ob, v, h); enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }
    pub fn logits_qkv_f16(&mut self, x: &[f32], wq: &[u16], wk: &[u16], wv: &[u16], vq: usize, vk: usize, vv: usize, h: usize, kq: &str, kk: &str, kv: &str, oq: &mut [f32], ok: &mut [f32], ov: &mut [f32]) -> anyhow::Result<()> {
        let wqk = format!("logits.f16.wq.{}", kq); let wkk = format!("logits.f16.wk.{}", kk); let wvk = format!("logits.f16.wv.{}", kv);
        if !self.tensor_cache.contains_key(&wqk) { self.tensor_cache.insert(wqk.clone(), upload_u16(&self.device, wq)?); self.tensor_cache.insert(wkk.clone(), upload_u16(&self.device, wk)?); self.tensor_cache.insert(wvk.clone(), upload_u16(&self.device, wv)?); }
        let wbq = self.tensor_cache.get(&wqk).unwrap(); let wbk = self.tensor_cache.get(&wkk).unwrap(); let wbv = self.tensor_cache.get(&wvk).unwrap();
        ensure_buf_f32(&self.device, &mut self.x_buf, h)?;
        let xb = self.x_buf.as_ref().unwrap(); write_f32(xb, x)?;
        let q_buf = self.device.new_buffer((vq*4) as u64, MTLResourceOptions::StorageModeShared);
        let k_buf = self.device.new_buffer((vk*4) as u64, MTLResourceOptions::StorageModeShared);
        let v_buf = self.device.new_buffer((vv*4) as u64, MTLResourceOptions::StorageModeShared);
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer(); let enc = cb.new_compute_command_encoder();
            self.encode_qkv_f16(enc, wbq, wbk, wbv, xb, &q_buf, &k_buf, &v_buf, vq, vk, h);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(&q_buf, oq)?; read_f32(&k_buf, ok)?; read_f32(&v_buf, ov)?; Ok(())
    }
    pub fn encode_mv_i8(&self, enc: &ComputeCommandEncoderRef, w: &Buffer, s: &Buffer, x: &Buffer, o: &Buffer, r: usize, c: usize) {
        enc.set_compute_pipeline_state(&self.pso_mv_i8); enc.set_buffer(0, Some(w), 0); enc.set_buffer(1, Some(s), 0); enc.set_buffer(2, Some(x), 0); enc.set_buffer(3, Some(o), 0);
        let r32=r as u32; let c32=c as u32; enc.set_bytes(4, 4, &r32 as *const _ as *const _); enc.set_bytes(5, 4, &c32 as *const _ as *const _);
        enc.dispatch_threads(MTLSize{width: r as u64, height:1, depth:1}, MTLSize{width: 32, height:1, depth:1});
    }
    pub fn encode_mv_f16(&self, enc: &ComputeCommandEncoderRef, w: &Buffer, x: &Buffer, o: &Buffer, r: usize, c: usize) {
        enc.set_compute_pipeline_state(&self.pso_mv_f16); enc.set_buffer(0, Some(w), 0); enc.set_buffer(1, Some(x), 0); enc.set_buffer(2, Some(o), 0);
        let r32=r as u32; let c32=c as u32; enc.set_bytes(3, 4, &r32 as *const _ as *const _); enc.set_bytes(4, 4, &c32 as *const _ as *const _);
        enc.dispatch_threads(MTLSize{width: r as u64, height:1, depth:1}, MTLSize{width: 32, height:1, depth:1});
    }
    pub fn encode_qkv_f16(&self, enc: &ComputeCommandEncoderRef, wq: &Buffer, wk: &Buffer, wv: &Buffer, x: &Buffer, oq: &Buffer, ok: &Buffer, ov: &Buffer, rq: usize, rkv: usize, c: usize) {
        enc.set_compute_pipeline_state(&self.pso_mv_qkv_f16); enc.set_buffer(0, Some(wq), 0); enc.set_buffer(1, Some(wk), 0); enc.set_buffer(2, Some(wv), 0); enc.set_buffer(3, Some(x), 0); enc.set_buffer(4, Some(oq), 0); enc.set_buffer(5, Some(ok), 0); enc.set_buffer(6, Some(ov), 0);
        let rq32=rq as u32; let rkv32=rkv as u32; let c32=c as u32; enc.set_bytes(7, 4, &rq32 as *const _ as *const _); enc.set_bytes(8, 4, &rkv32 as *const _ as *const _); enc.set_bytes(9, 4, &c32 as *const _ as *const _);
        enc.dispatch_threads(MTLSize{width: (rq+rkv+rkv) as u64, height:1, depth:1}, MTLSize{width: 32, height:1, depth:1});
    }
    pub fn encode_mv2_f16(&self, enc: &ComputeCommandEncoderRef, w0: &Buffer, w1: &Buffer, x: &Buffer, o0: &Buffer, o1: &Buffer, r0: usize, r1: usize, c: usize) {
        enc.set_compute_pipeline_state(&self.pso_mv2_f16); enc.set_buffer(0, Some(w0), 0); enc.set_buffer(1, Some(w1), 0); enc.set_buffer(2, Some(x), 0); enc.set_buffer(3, Some(o0), 0); enc.set_buffer(4, Some(o1), 0);
        let r032=r0 as u32; let r132=r1 as u32; let c32=c as u32; enc.set_bytes(5, 4, &r032 as *const _ as *const _); enc.set_bytes(6, 4, &r132 as *const _ as *const _); enc.set_bytes(7, 4, &c32 as *const _ as *const _);
        enc.dispatch_threads(MTLSize{width: (r0+r1) as u64, height:1, depth:1}, MTLSize{width: 32, height:1, depth:1});
    }
    pub fn encode_add_f32_inplace(&self, enc: &ComputeCommandEncoderRef, a: &Buffer, b: &Buffer, n: usize) {
        enc.set_compute_pipeline_state(&self.pso_add_f32); enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(b), 0);
        let n32=n as u32; enc.set_bytes(2, 4, &n32 as *const _ as *const _); enc.dispatch_threads(MTLSize{width: n as u64, height:1, depth:1}, MTLSize{width: 32, height:1, depth:1});
    }
    pub fn encode_silu_mul_f32_inplace(&self, enc: &ComputeCommandEncoderRef, a: &Buffer, b: &Buffer, n: usize) {
        enc.set_compute_pipeline_state(&self.pso_silu_mul_f32); enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(b), 0);
        let n32=n as u32; enc.set_bytes(2, 4, &n32 as *const _ as *const _); enc.dispatch_threads(MTLSize{width: n as u64, height:1, depth:1}, MTLSize{width: 32, height:1, depth:1});
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn upload_u16(d: &Device, s: &[u16]) -> anyhow::Result<Buffer> { let b = d.new_buffer((s.len()*2) as u64, MTLResourceOptions::StorageModeShared); unsafe { std::slice::from_raw_parts_mut(b.contents() as *mut u16, s.len()).copy_from_slice(s); } Ok(b) }
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn upload_i8(d: &Device, s: &[i8]) -> anyhow::Result<Buffer> { let b = d.new_buffer(s.len() as u64, MTLResourceOptions::StorageModeShared); unsafe { std::slice::from_raw_parts_mut(b.contents() as *mut i8, s.len()).copy_from_slice(s); } Ok(b) }
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn write_f32(b: &Buffer, s: &[f32]) -> anyhow::Result<()> { unsafe { std::slice::from_raw_parts_mut(b.contents() as *mut f32, s.len()).copy_from_slice(s); } Ok(()) }
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn read_f32(b: &Buffer, d: &mut [f32]) -> anyhow::Result<()> { unsafe { d.copy_from_slice(std::slice::from_raw_parts(b.contents() as *const f32, d.len())); } Ok(()) }
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ensure_buf_f32(d: &Device, b: &mut Option<Buffer>, n: usize) -> anyhow::Result<()> { if b.as_ref().map(|x| x.length() < (n*4) as u64).unwrap_or(true) { *b = Some(d.new_buffer((n*4) as u64, MTLResourceOptions::StorageModeShared)); } Ok(()) }

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalOps;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalOps {
    pub fn create() -> anyhow::Result<Self> { anyhow::bail!("No") }
    pub fn begin_pass(&self) -> anyhow::Result<()> { Ok(()) }
    pub fn end_pass(&self) -> anyhow::Result<()> { Ok(()) }
    pub fn rms_norm_f16w(&mut self, _: &[f32], _: &[u16], _: f32, _: bool, _: &str, _: &mut [f32]) -> anyhow::Result<()> { anyhow::bail!("No") }
    pub fn rope_half_f32(&mut self, _: &mut [f32], _: usize, _: usize, _: usize, _: usize, _: f32) -> anyhow::Result<()> { anyhow::bail!("No") }
    pub fn rope_adj_f32(&self, _: &mut [f32], _: usize, _: usize, _: usize, _: f32) -> anyhow::Result<()> { anyhow::bail!("No") }
    pub fn logits_f16(&mut self, _: &[f32], _: &[u16], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> { anyhow::bail!("No") }
    pub fn logits_i8(&mut self, _: &[f32], _: &[i8], _: &[u16], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> { anyhow::bail!("No") }
    pub fn logits_qkv_f16(&mut self, _: &[f32], _: &[u16], _: &[u16], _: &[u16], _: usize, _: usize, _: usize, _: usize, _: &str, _: &str, _: &str, _: &mut [f32], _: &mut [f32], _: &mut [f32]) -> anyhow::Result<()> { anyhow::bail!("No") }
}
"###;
    fs::write("crates/cellm-kernels/src/metal.rs", content).unwrap();
}
