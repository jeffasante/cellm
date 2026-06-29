//! N-gram speculative decoding with external corpus support.
//!
//! Allows providing a separate token sequence (`ngram_corpus`) as the lookup
//! source for n-gram draft candidates, instead of only matching against the
//! prompt itself. The corpus tokens are never passed through the model — they
//! are only used as a pattern-matching dictionary for draft generation.
//!
//! ## Strategy: ngram-simple
//!
//! 1. Look at the last `n` tokens in the generated sequence (the "key" n-gram).
//! 2. Find the same n-gram in the corpus.
//! 3. Take the next `m` tokens after the match as the "draft".
//! 4. Verify each draft token against the model output — accept matches,
//!    reject at the first mismatch and use the model's own token instead.

/// N-gram based speculative decoder that draws draft candidates from an
/// external corpus buffer.
#[derive(Debug)]
pub struct NgramSpeculator {
    /// The external token buffer to search for n-gram matches.
    corpus: Vec<u32>,
    /// Size of the n-gram key used for matching (n).
    n: usize,
    /// Maximum number of draft tokens to generate (m).
    m_max: usize,
}

impl NgramSpeculator {
    /// Create a new speculator with the given corpus and parameters.
    ///
    /// * `corpus` - External token sequence to search for n-gram matches.
    ///   Passed directly — no copy made if you don't need one.
    /// * `n` - N-gram size for the lookup key. Larger n = more specific matches.
    /// * `m_max` - Maximum number of draft tokens to propose.
    pub fn new(corpus: Vec<u32>, n: usize, m_max: usize) -> Self {
        Self { corpus, n: n.max(1), m_max: m_max.max(1) }
    }

    /// Update the corpus buffer.
    pub fn set_corpus(&mut self, corpus: Vec<u32>) {
        self.corpus = corpus;
    }

    /// Set the n-gram key size.
    pub fn set_n(&mut self, n: usize) {
        self.n = n.max(1);
    }

    /// Set the maximum draft length.
    pub fn set_m_max(&mut self, m_max: usize) {
        self.m_max = m_max.max(1);
    }

    /// Get a reference to the current corpus.
    pub fn corpus(&self) -> &[u32] {
        &self.corpus
    }

    /// Find the draft token sequence for the given generated token history.
    ///
    /// Uses **ngram-simple** strategy:
    /// - Take the last `self.n` tokens from `generated` as the key.
    /// - Search `self.corpus` for the first matching n-gram.
    /// - Return the next `self.m_max` tokens after that match.
    ///
    /// Returns an empty vec if no match is found or the key is too short.
    pub fn draft(&self, generated: &[u32]) -> Vec<u32> {
        if self.corpus.len() < self.n || generated.len() < self.n {
            return Vec::new();
        }

        let key = &generated[generated.len().saturating_sub(self.n)..];

        // Scan corpus for the matching n-gram.
        // We stop at corpus.len() - self.n to leave room for the match.
        let max_start = self.corpus.len().saturating_sub(self.n);
        for i in 0..max_start {
            if self.corpus[i..i + self.n] == *key {
                // Found a match. Take the next m_max tokens (or fewer if near end).
                let draft_start = i + self.n;
                let draft_end = (draft_start + self.m_max).min(self.corpus.len());
                return self.corpus[draft_start..draft_end].to_vec();
            }
        }

        Vec::new()
    }

    /// Find the draft token sequence, preferring the **last** (most recent)
    /// match in the corpus rather than the first. This can give better drafts
    /// when the corpus has repeated patterns.
    pub fn draft_last_match(&self, generated: &[u32]) -> Vec<u32> {
        if self.corpus.len() < self.n || generated.len() < self.n {
            return Vec::new();
        }

        let key = &generated[generated.len().saturating_sub(self.n)..];

        // Scan from the end for the last match.
        let max_start = self.corpus.len().saturating_sub(self.n);
        for i in (0..max_start).rev() {
            if self.corpus[i..i + self.n] == *key {
                let draft_start = i + self.n;
                let draft_end = (draft_start + self.m_max).min(self.corpus.len());
                return self.corpus[draft_start..draft_end].to_vec();
            }
        }

        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_draft() {
        let spec = NgramSpeculator::new(
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90],
            2,   // n=2
            3,   // m_max=3
        );
        // generated has ...50, 60 as last 2 tokens
        let gen = vec![0, 100, 50, 60];
        let d = spec.draft(&gen);
        assert_eq!(d, vec![70, 80, 90], "should match 50,60 → 70,80,90");
    }

    #[test]
    fn test_no_match() {
        let spec = NgramSpeculator::new(
            vec![10, 20, 30],
            2,
            3,
        );
        let gen = vec![99, 88];
        assert!(spec.draft(&gen).is_empty(), "no match in corpus");
    }

    #[test]
    fn test_key_too_short() {
        let spec = NgramSpeculator::new(
            vec![10, 20, 30],
            5,
            3,
        );
        let gen = vec![1, 2, 3, 4];
        assert!(spec.draft(&gen).is_empty(), "corpus too small for n=5");
    }

    #[test]
    fn test_generated_too_short() {
        let spec = NgramSpeculator::new(
            vec![10, 20, 30, 40, 50],
            3,
            2,
        );
        let gen = vec![1, 2];
        assert!(spec.draft(&gen).is_empty(), "generated too short for n=3");
    }

    #[test]
    fn test_draft_at_corpus_end() {
        let spec = NgramSpeculator::new(
            vec![10, 20, 30, 40],
            2,
            5, // m_max bigger than remaining tokens
        );
        let gen = vec![10, 20];
        let d = spec.draft(&gen);
        assert_eq!(d, vec![30, 40], "should truncate at corpus end");
    }

    #[test]
    fn test_empty_corpus() {
        let spec = NgramSpeculator::new(
            vec![],
            2,
            3,
        );
        let gen = vec![1, 2, 3, 4];
        assert!(spec.draft(&gen).is_empty());
    }

    #[test]
    fn test_last_match_preferred() {
        let spec = NgramSpeculator::new(
            vec![10, 20, 99, 10, 20, 88],
            2,
            2,
        );
        // ngram-simple (first match) picks 10,20 at pos 0 → tokens [99, 10]
        let gen = vec![5, 5, 10, 20];
        let d_first = spec.draft(&gen);
        assert_eq!(d_first, vec![99, 10], "first match: 10,20 at pos 0 → 99, 10");

        // last match picks 10,20 at pos 3 → tokens [88]
        let d_last = spec.draft_last_match(&gen);
        assert_eq!(d_last, vec![88], "last match: 10,20 at pos 3 → 88");
    }
}
