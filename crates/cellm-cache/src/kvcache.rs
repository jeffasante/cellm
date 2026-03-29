use cellm_core::{CoreError, KvCacheLayout, KvCacheReadView, KvCacheView};
use half::f16;

use crate::BlockAllocator;

/// Physical paged KV cache storage.
///
/// Owns:
/// - a `BlockAllocator` (block id free-list)
/// - physical K and V buffers (f16)
///
/// Layout per block:
///   [layer][token_offset][kv_head][head_dim]
#[derive(Debug)]
pub struct KVCache {
    layout: KvCacheLayout,
    allocator: BlockAllocator,
    k: Vec<f16>,
    v: Vec<f16>,
}

impl KVCache {
    pub fn new(layout: KvCacheLayout) -> Result<Self, CoreError> {
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
        Ok(Self {
            allocator: BlockAllocator::new(layout.total_blocks),
            layout,
            k: vec![f16::from_f32(0.0); total_elems],
            v: vec![f16::from_f32(0.0); total_elems],
        })
    }

    pub fn layout(&self) -> KvCacheLayout {
        self.layout
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
            k: &mut self.k,
            v: &mut self.v,
        }
    }

    pub fn view(&self) -> KvCacheReadView<'_> {
        KvCacheReadView {
            layout: self.layout,
            k: &self.k,
            v: &self.v,
        }
    }
}
