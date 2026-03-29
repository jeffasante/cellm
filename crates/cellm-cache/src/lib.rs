//! cellm-cache: KV block allocator, page table, and eviction policy.

pub mod allocator;
pub mod error;
pub mod kvcache;
pub mod pagetable;
pub mod eviction;

pub use allocator::BlockAllocator;
pub use error::CacheError;
pub use kvcache::KVCache;
pub use pagetable::PageTable;
pub use eviction::EvictionPolicy;
