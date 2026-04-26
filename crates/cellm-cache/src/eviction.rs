// Author: Jeffrey Asante (https://jeffasante.github.io/)
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Evict least-recently-used sessions first (requires recency tracking in scheduler).
    Lru,
    /// Evict lowest-priority sessions first (requires per-session priority).
    Priority,
    /// Do not evict; return out-of-memory when blocks run out.
    Disabled,
}
