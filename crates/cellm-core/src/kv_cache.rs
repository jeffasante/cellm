use crate::CoreError;
use half::f16;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct KvCacheLayout {
    pub total_blocks: usize,
    pub tokens_per_block: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCacheLayout {
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub fn elems_per_block_per_layer(&self) -> usize {
        self.tokens_per_block * self.kv_dim()
    }

    pub fn elems_per_block(&self) -> usize {
        self.num_layers * self.elems_per_block_per_layer()
    }

    pub fn total_elems(&self) -> usize {
        self.total_blocks * self.elems_per_block()
    }

    pub fn total_bytes_f16(&self) -> usize {
        self.total_elems() * 2 * 2 // (k + v) * sizeof(f16)
    }

    fn check_bounds(
        &self,
        block_id: u32,
        layer: usize,
        token_offset: usize,
    ) -> Result<(), CoreError> {
        if (block_id as usize) >= self.total_blocks {
            return Err(CoreError::Backend(format!(
                "kv cache: block_id {block_id} out of range (total_blocks={})",
                self.total_blocks
            )));
        }
        if layer >= self.num_layers {
            return Err(CoreError::Backend(format!(
                "kv cache: layer {layer} out of range (num_layers={})",
                self.num_layers
            )));
        }
        if token_offset >= self.tokens_per_block {
            return Err(CoreError::Backend(format!(
                "kv cache: token_offset {token_offset} out of range (tokens_per_block={})",
                self.tokens_per_block
            )));
        }
        Ok(())
    }

    pub fn token_base_elem(
        &self,
        block_id: u32,
        layer: usize,
        token_offset: usize,
    ) -> Result<usize, CoreError> {
        self.check_bounds(block_id, layer, token_offset)?;
        let block_base = (block_id as usize) * self.elems_per_block();
        let layer_base = layer * self.elems_per_block_per_layer();
        let tok_base = token_offset * self.kv_dim();
        Ok(block_base + layer_base + tok_base)
    }
}

pub struct KvCacheView<'a> {
    pub layout: KvCacheLayout,
    pub k: &'a mut [f16],
    pub v: &'a mut [f16],
}

impl<'a> KvCacheView<'a> {
    pub fn write_token_f16(
        &mut self,
        block_id: u32,
        layer: usize,
        token_offset: usize,
        k_src: &[f16],
        v_src: &[f16],
    ) -> Result<(), CoreError> {
        let kv_dim = self.layout.kv_dim();
        if k_src.len() != kv_dim || v_src.len() != kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache: expected k/v len {kv_dim}, got {}/{}",
                k_src.len(),
                v_src.len()
            )));
        }

        let base = self.layout.token_base_elem(block_id, layer, token_offset)?;
        self.k[base..base + kv_dim].copy_from_slice(k_src);
        self.v[base..base + kv_dim].copy_from_slice(v_src);
        Ok(())
    }

    pub fn write_token(
        &mut self,
        block_id: u32,
        layer: usize,
        token_offset: usize,
        k_src: &[f32],
        v_src: &[f32],
    ) -> Result<(), CoreError> {
        let kv_dim = self.layout.kv_dim();
        if k_src.len() != kv_dim || v_src.len() != kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache: expected k/v len {kv_dim}, got {}/{}",
                k_src.len(),
                v_src.len()
            )));
        }

        let base = self.layout.token_base_elem(block_id, layer, token_offset)?;
        for i in 0..kv_dim {
            self.k[base + i] = f16::from_f32(k_src[i]);
            self.v[base + i] = f16::from_f32(v_src[i]);
        }
        Ok(())
    }

    pub fn read_token(
        &self,
        block_id: u32,
        layer: usize,
        token_offset: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError> {
        let kv_dim = self.layout.kv_dim();
        if k_out.len() != kv_dim || v_out.len() != kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache: expected k/v len {kv_dim}, got {}/{}",
                k_out.len(),
                v_out.len()
            )));
        }

        let base = self.layout.token_base_elem(block_id, layer, token_offset)?;
        for i in 0..kv_dim {
            k_out[i] = self.k[base + i].to_f32();
            v_out[i] = self.v[base + i].to_f32();
        }
        Ok(())
    }

    pub fn read_token_f16(
        &self,
        block_id: u32,
        layer: usize,
        token_offset: usize,
        k_out: &mut [f16],
        v_out: &mut [f16],
    ) -> Result<(), CoreError> {
        let kv_dim = self.layout.kv_dim();
        if k_out.len() != kv_dim || v_out.len() != kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache: expected k/v len {kv_dim}, got {}/{}",
                k_out.len(),
                v_out.len()
            )));
        }

        let base = self.layout.token_base_elem(block_id, layer, token_offset)?;
        k_out.copy_from_slice(&self.k[base..base + kv_dim]);
        v_out.copy_from_slice(&self.v[base..base + kv_dim]);
        Ok(())
    }
}

pub struct KvCacheReadView<'a> {
    pub layout: KvCacheLayout,
    pub k: &'a [f16],
    pub v: &'a [f16],
}

impl<'a> KvCacheReadView<'a> {
    pub fn read_token_f16(
        &self,
        block_id: u32,
        layer: usize,
        token_offset: usize,
        k_out: &mut [f16],
        v_out: &mut [f16],
    ) -> Result<(), CoreError> {
        let kv_dim = self.layout.kv_dim();
        if k_out.len() != kv_dim || v_out.len() != kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache: expected k/v len {kv_dim}, got {}/{}",
                k_out.len(),
                v_out.len()
            )));
        }

        let base = self.layout.token_base_elem(block_id, layer, token_offset)?;
        k_out.copy_from_slice(&self.k[base..base + kv_dim]);
        v_out.copy_from_slice(&self.v[base..base + kv_dim]);
        Ok(())
    }

    pub fn read_token(
        &self,
        block_id: u32,
        layer: usize,
        token_offset: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError> {
        let kv_dim = self.layout.kv_dim();
        if k_out.len() != kv_dim || v_out.len() != kv_dim {
            return Err(CoreError::Backend(format!(
                "kv cache: expected k/v len {kv_dim}, got {}/{}",
                k_out.len(),
                v_out.len()
            )));
        }

        let base = self.layout.token_base_elem(block_id, layer, token_offset)?;
        for i in 0..kv_dim {
            k_out[i] = self.k[base + i].to_f32();
            v_out[i] = self.v[base + i].to_f32();
        }
        Ok(())
    }
}
