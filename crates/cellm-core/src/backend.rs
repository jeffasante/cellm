use crate::{KvCacheReadView, KvCacheView, TensorView, CoreError};
use half::f16;

/// The compute backend trait. CPU implements this for Phase 1.
/// Metal and Vulkan will implement the same interface.
///
/// All ops write into a pre-allocated `out` tensor view.
/// The backend resolves handles → byte slices through the storage registry.
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;

    /// Matrix multiply: out = a @ b
    /// a: [m, k], b: [k, n], out: [m, n]  (all f32 for phase 1)
    fn matmul(
        &self,
        a: &TensorView,
        b: &TensorView,
        out: &mut TensorView,
        arena: &[u8],
    ) -> Result<(), CoreError>;

    /// RMS normalization: out = x / rms(x) * weight
    fn rms_norm(
        &self,
        x: &TensorView,
        weight: &TensorView,
        out: &mut TensorView,
        eps: f32,
        arena: &[u8],
    ) -> Result<(), CoreError>;

    /// Rotary position embedding, applied in-place to q and k.
    fn rope_inplace(
        &self,
        q: &mut TensorView,
        k: &mut TensorView,
        positions: &[usize],
        theta: f32,
        arena: &mut [u8],
    ) -> Result<(), CoreError>;

    /// SiLU activation: out = x * sigmoid(x)
    fn silu(
        &self,
        x: &TensorView,
        out: &mut TensorView,
        arena: &[u8],
    ) -> Result<(), CoreError>;

    /// Element-wise add: out = a + b
    fn add(
        &self,
        a: &TensorView,
        b: &TensorView,
        out: &mut TensorView,
        arena: &[u8],
    ) -> Result<(), CoreError>;

    /// Element-wise multiply: out = a * b
    fn mul(
        &self,
        a: &TensorView,
        b: &TensorView,
        out: &mut TensorView,
        arena: &[u8],
    ) -> Result<(), CoreError>;

    /// Softmax in-place over last dimension.
    fn softmax_inplace(
        &self,
        x: &mut TensorView,
        arena: &mut [u8],
    ) -> Result<(), CoreError>;

    /// Causal scaled dot-product attention.
    /// q: [seq, n_heads, head_dim]
    /// k: [seq, n_kv_heads, head_dim]
    /// v: [seq, n_kv_heads, head_dim]
    /// out: [seq, n_heads, head_dim]
    fn attention(
        &self,
        q: &TensorView,
        k: &TensorView,
        v: &TensorView,
        n_heads: usize,
        n_kv_heads: usize,
        out: &mut TensorView,
        arena: &[u8],
        scratch: &mut Vec<f32>,
    ) -> Result<(), CoreError>;

    /// Write a single token's K/V into a paged KV cache (native f16).
    ///
    /// Default implementation is a CPU copy into the cache view. Backends can
    /// override this to do device-side writes with no conversion.
    fn kv_write_token_f16(
        &self,
        cache: &mut KvCacheView<'_>,
        block_id: u32,
        layer: usize,
        token_offset: usize,
        k: &[f16],
        v: &[f16],
    ) -> Result<(), CoreError> {
        cache.write_token_f16(block_id, layer, token_offset, k, v)
    }

    /// Read a single token's K/V from a paged KV cache (native f16).
    fn kv_read_token_f16(
        &self,
        cache: &KvCacheReadView<'_>,
        block_id: u32,
        layer: usize,
        token_offset: usize,
        k_out: &mut [f16],
        v_out: &mut [f16],
    ) -> Result<(), CoreError> {
        cache.read_token_f16(block_id, layer, token_offset, k_out, v_out)
    }

    /// Write a single token's K/V into a paged KV cache.
    ///
    /// Convenience API for f32 compute code. Storage is f16 under the hood.
    fn kv_write_token_f32(
        &self,
        cache: &mut KvCacheView<'_>,
        block_id: u32,
        layer: usize,
        token_offset: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), CoreError> {
        cache.write_token(block_id, layer, token_offset, k, v)
    }

    /// Read a single token's K/V from a paged KV cache.
    ///
    /// Convenience API for f32 compute code. Storage is f16 under the hood.
    fn kv_read_token_f32(
        &self,
        cache: &KvCacheReadView<'_>,
        block_id: u32,
        layer: usize,
        token_offset: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError> {
        cache.read_token(block_id, layer, token_offset, k_out, v_out)
    }
}
