// Author: Jeffrey Asante (https://jeffasante.github.io/)
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    OutOfBlocks { requested: usize, free: usize },
    InvalidBlockId(u32),
    DoubleFree(u32),
    InvalidTokenPos { pos: usize, token_count: usize },
    InvalidConfig(&'static str),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::OutOfBlocks { requested, free } => write!(
                f,
                "out of KV blocks: requested {requested}, free {free}"
            ),
            CacheError::InvalidBlockId(id) => write!(f, "invalid block id: {id}"),
            CacheError::DoubleFree(id) => write!(f, "double free: block id {id}"),
            CacheError::InvalidTokenPos { pos, token_count } => write!(
                f,
                "token position out of range: pos={pos}, token_count={token_count}"
            ),
            CacheError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for CacheError {}

