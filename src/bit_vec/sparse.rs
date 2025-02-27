use crate::EliasFanoVec;

/// A succinct representation of a sparse vector with rank and select support.
/// The vector is a compressed sequence of indices of set bits.
struct SparseRSVec {
    vec: EliasFanoVec,
}

impl SparseRSVec {
    /// Creates a new `SparseRSVec` from a sequence of set bits represented as indices.
    /// The input must be sorted in ascending order and free of duplicates.
    fn new(input: &[u64]) -> Self {
        Self { vec: EliasFanoVec::from_slice(input) }
    }

    /// Returns true if the bit at position `i` is set.
    fn is_set(&self, i: u64) -> Option<bool> {
        // TODO unchecked version
        self.vec.predecessor(i).map(|p| p == i)
    }

    /// Gets the bit at position `i`.
    /// Returns `Some(1)` if the bit is set, `Some(0)` if it is not set, and `None` if `i` is out of bounds.
    fn get(&self, i: u64) -> Option<u64> {
        // todo unchecked version
        self.is_set(i).map(|b| b as u64)
    }

    fn select1(&self, i: usize) -> Option<u64> {
        self.vec.get(i)
    }

    fn rank1(&self, i: u64) -> Option<u64> {
        todo!("implement rank for elias fano")
    }
}