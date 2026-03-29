use crate::{BlockAllocator, CacheError};

/// Maps a session's token positions → KV block ids.
///
/// Each block id in `blocks` represents storage for `tokens_per_block` tokens.
#[derive(Debug, Clone)]
pub struct PageTable {
    session_id: u64,
    tokens_per_block: usize,
    blocks: Vec<u32>,
    token_count: usize,
}

impl PageTable {
    pub fn new(session_id: u64, tokens_per_block: usize) -> Result<Self, CacheError> {
        if tokens_per_block == 0 {
            return Err(CacheError::InvalidConfig("tokens_per_block must be > 0"));
        }
        Ok(Self {
            session_id,
            tokens_per_block,
            blocks: Vec::new(),
            token_count: 0,
        })
    }

    pub fn session_id(&self) -> u64 {
        self.session_id
    }

    pub fn tokens_per_block(&self) -> usize {
        self.tokens_per_block
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn blocks(&self) -> &[u32] {
        &self.blocks
    }

    /// Append one token position, allocating a new block if needed.
    pub fn append_token(&mut self, alloc: &mut BlockAllocator) -> Result<(), CacheError> {
        let need_new_block = self.token_count == self.blocks.len() * self.tokens_per_block;
        if need_new_block {
            let id = alloc.alloc().ok_or(CacheError::OutOfBlocks {
                requested: 1,
                free: alloc.free_count(),
            })?;
            self.blocks.push(id);
        }
        self.token_count += 1;
        Ok(())
    }

    /// Append `n` tokens, allocating blocks as required.
    pub fn append_tokens(&mut self, alloc: &mut BlockAllocator, n: usize) -> Result<(), CacheError> {
        for _ in 0..n {
            self.append_token(alloc)?;
        }
        Ok(())
    }

    /// Return the block id that owns token position `pos`.
    pub fn block_for_token(&self, pos: usize) -> Result<u32, CacheError> {
        if pos >= self.token_count {
            return Err(CacheError::InvalidTokenPos {
                pos,
                token_count: self.token_count,
            });
        }
        Ok(self.blocks[pos / self.tokens_per_block])
    }

    /// Return the offset within the owning block for token position `pos`.
    pub fn offset_in_block(&self, pos: usize) -> Result<usize, CacheError> {
        if pos >= self.token_count {
            return Err(CacheError::InvalidTokenPos {
                pos,
                token_count: self.token_count,
            });
        }
        Ok(pos % self.tokens_per_block)
    }

    /// Free all blocks back to the allocator and reset counts.
    pub fn free_all(&mut self, alloc: &mut BlockAllocator) -> Result<(), CacheError> {
        for &id in &self.blocks {
            alloc.free(id)?;
        }
        self.blocks.clear();
        self.token_count = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_blocks_on_demand() {
        let mut alloc = BlockAllocator::new(10);
        let mut pt = PageTable::new(7, 4).unwrap();

        pt.append_tokens(&mut alloc, 1).unwrap();
        assert_eq!(pt.blocks().len(), 1);
        assert_eq!(pt.token_count(), 1);

        pt.append_tokens(&mut alloc, 3).unwrap();
        assert_eq!(pt.blocks().len(), 1);
        assert_eq!(pt.token_count(), 4);

        pt.append_tokens(&mut alloc, 1).unwrap();
        assert_eq!(pt.blocks().len(), 2);
        assert_eq!(pt.token_count(), 5);
    }

    #[test]
    fn token_mapping_is_correct() {
        let mut alloc = BlockAllocator::new(10);
        let mut pt = PageTable::new(1, 4).unwrap();
        pt.append_tokens(&mut alloc, 9).unwrap(); // 3 blocks

        let b0 = pt.blocks()[0];
        let b1 = pt.blocks()[1];
        let b2 = pt.blocks()[2];

        assert_eq!(pt.block_for_token(0).unwrap(), b0);
        assert_eq!(pt.offset_in_block(0).unwrap(), 0);
        assert_eq!(pt.block_for_token(3).unwrap(), b0);
        assert_eq!(pt.offset_in_block(3).unwrap(), 3);

        assert_eq!(pt.block_for_token(4).unwrap(), b1);
        assert_eq!(pt.offset_in_block(4).unwrap(), 0);
        assert_eq!(pt.block_for_token(7).unwrap(), b1);
        assert_eq!(pt.offset_in_block(7).unwrap(), 3);

        assert_eq!(pt.block_for_token(8).unwrap(), b2);
        assert_eq!(pt.offset_in_block(8).unwrap(), 0);
    }

    #[test]
    fn free_all_returns_blocks() {
        let mut alloc = BlockAllocator::new(3);
        let mut pt = PageTable::new(1, 2).unwrap();

        pt.append_tokens(&mut alloc, 5).unwrap(); // needs 3 blocks
        assert_eq!(alloc.free_count(), 0);

        pt.free_all(&mut alloc).unwrap();
        assert_eq!(alloc.free_count(), 3);
        assert_eq!(pt.blocks().len(), 0);
        assert_eq!(pt.token_count(), 0);
    }

    #[test]
    fn out_of_blocks_errors() {
        let mut alloc = BlockAllocator::new(1);
        let mut pt = PageTable::new(1, 2).unwrap();
        pt.append_tokens(&mut alloc, 2).unwrap(); // first block covers 2 tokens
        let err = pt.append_token(&mut alloc).unwrap_err(); // needs 2nd block
        assert_eq!(
            err,
            CacheError::OutOfBlocks {
                requested: 1,
                free: 0
            }
        );
    }
}
