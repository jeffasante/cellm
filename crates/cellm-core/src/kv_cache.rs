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

pub trait DeviceKvStorage: Send + Sync + std::fmt::Debug {
    fn as_any(&self) -> &dyn std::any::Any;

    fn write_token_f16(
        &mut self,
        base: usize,
        k_src: &[f16],
        v_src: &[f16],
    ) -> Result<(), CoreError>;

    fn write_token_f32(
        &mut self,
        base: usize,
        k_src: &[f32],
        v_src: &[f32],
    ) -> Result<(), CoreError>;

    fn read_token_f16(
        &self,
        base: usize,
        k_out: &mut [f16],
        v_out: &mut [f16],
    ) -> Result<(), CoreError>;

    fn read_token_f32(
        &self,
        base: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError>;

    fn gather_tokens_f32(
        &self,
        bases: &[usize],
        kv_dim: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError>;

    fn attention_single_token_gqa_f32(
        &self,
        bases: &[usize],
        q: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        let seq = bases.len();
        let kv_dim = n_kv_heads.saturating_mul(head_dim);
        if q.len() != n_heads.saturating_mul(head_dim) {
            return Err(CoreError::Backend(format!(
                "kv cache: attention q len mismatch {} expected {}",
                q.len(),
                n_heads.saturating_mul(head_dim)
            )));
        }
        if out.len() != n_heads.saturating_mul(head_dim) {
            return Err(CoreError::Backend(format!(
                "kv cache: attention out len mismatch {} expected {}",
                out.len(),
                n_heads.saturating_mul(head_dim)
            )));
        }
        if seq == 0 {
            out.fill(0.0);
            return Ok(());
        }
        let mut k_seq = vec![0.0f32; seq.saturating_mul(kv_dim)];
        let mut v_seq = vec![0.0f32; seq.saturating_mul(kv_dim)];
        self.gather_tokens_f32(bases, kv_dim, &mut k_seq, &mut v_seq)?;
        attention_single_token_f32_gqa_local(
            q,
            &k_seq,
            &v_seq,
            seq,
            n_heads,
            n_kv_heads,
            head_dim,
            out,
        )
    }
}

pub struct KvCacheView<'a> {
    pub layout: KvCacheLayout,
    pub storage: &'a mut dyn DeviceKvStorage,
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
        self.storage.write_token_f16(base, k_src, v_src)
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
        self.storage.write_token_f32(base, k_src, v_src)
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
        self.storage.read_token_f32(base, k_out, v_out)
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
        self.storage.read_token_f16(base, k_out, v_out)
    }
}

pub struct KvCacheReadView<'a> {
    pub layout: KvCacheLayout,
    pub storage: &'a dyn DeviceKvStorage,
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
        self.storage.read_token_f16(base, k_out, v_out)
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
        self.storage.read_token_f32(base, k_out, v_out)
    }

    pub fn gather_by_bases_f32(
        &self,
        bases: &[usize],
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CoreError> {
        if k_out.len() != v_out.len() {
            return Err(CoreError::Backend(format!(
                "kv cache: gather output len mismatch k={} v={}",
                k_out.len(),
                v_out.len()
            )));
        }
        let kv_dim = self.layout.kv_dim();
        if bases.is_empty() {
            if !k_out.is_empty() || !v_out.is_empty() {
                return Err(CoreError::Backend(
                    "kv cache: gather with empty bases requires empty outputs".into(),
                ));
            }
            return Ok(());
        }
        let need = bases.len() * kv_dim;
        if k_out.len() != need {
            return Err(CoreError::Backend(format!(
                "kv cache: gather output len {} mismatch expected {} (bases={} kv_dim={})",
                k_out.len(),
                need,
                bases.len(),
                kv_dim
            )));
        }
        self.storage.gather_tokens_f32(bases, kv_dim, k_out, v_out)
    }

    pub fn attention_single_token_gqa_from_bases(
        &self,
        bases: &[usize],
        q: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        self.storage.attention_single_token_gqa_f32(
            bases,
            q,
            n_heads,
            n_kv_heads,
            head_dim,
            out,
        )
    }
}

fn softmax_f32_inplace_local(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

fn attention_single_token_f32_gqa_local(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<(), CoreError> {
    if q.len() != n_heads.saturating_mul(head_dim) {
        return Err(CoreError::Backend("kv cache: local attention q shape mismatch".into()));
    }
    if k.len() != seq.saturating_mul(n_kv_heads).saturating_mul(head_dim) {
        return Err(CoreError::Backend("kv cache: local attention k shape mismatch".into()));
    }
    if v.len() != seq.saturating_mul(n_kv_heads).saturating_mul(head_dim) {
        return Err(CoreError::Backend("kv cache: local attention v shape mismatch".into()));
    }
    if out.len() != n_heads.saturating_mul(head_dim) {
        return Err(CoreError::Backend("kv cache: local attention out shape mismatch".into()));
    }
    out.fill(0.0);
    if seq == 0 {
        return Ok(());
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let group_size = (n_heads / n_kv_heads).max(1);
    let mut scores = vec![0.0f32; seq];

    for h in 0..n_heads {
        let kv_h = (h / group_size).min(n_kv_heads.saturating_sub(1));
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        for t in 0..seq {
            let kt_base = (t * n_kv_heads + kv_h) * head_dim;
            let kt = &k[kt_base..kt_base + head_dim];
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += qh[i] * kt[i];
            }
            scores[t] = dot * scale;
        }
        softmax_f32_inplace_local(&mut scores);

        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for t in 0..seq {
            let vt_base = (t * n_kv_heads + kv_h) * head_dim;
            let vt = &v[vt_base..vt_base + head_dim];
            let w = scores[t];
            for i in 0..head_dim {
                out_h[i] += w * vt[i];
            }
        }
    }
    Ok(())
}
