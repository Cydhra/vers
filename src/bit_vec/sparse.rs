use crate::{BitVec, EliasFanoVec};

/// A succinct representation of a sparse vector with rank and select support.
/// The vector is a compressed sequence of indices of set bits.
struct SparseRSVec {
    vec: EliasFanoVec,
}

impl SparseRSVec {
    /// Creates a new `SparseRSVec` from a sequence of set bits represented as indices.
    /// The input must be sorted in ascending order and free of duplicates.
    pub fn new(input: &[u64]) -> Self {
        debug_assert!(input.is_sorted(), "input must be sorted");
        debug_assert!(
            input.windows(2).all(|w| w[0] != w[1]),
            "input must be free of duplicates"
        );

        Self {
            vec: EliasFanoVec::from_slice(input),
        }
    }

    /// Creates a new `SparseRSVec` from a `BitVec`, by compressing the indices of 0-bits if `zero` is true,
    /// or the indices of 1-bits if `zero` is false.
    pub fn from_bitvec(input: &BitVec, zero: bool) -> Self {
        if zero {
            Self::new(
                input
                    .iter()
                    .enumerate()
                    .filter(|&(_, bit)| bit == 0)
                    .map(|(i, _)| i as u64)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        } else {
            Self::new(
                input
                    .iter()
                    .enumerate()
                    .filter(|&(_, bit)| bit == 1)
                    .map(|(i, _)| i as u64)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        }
    }

    /// Returns true if the bit at position `i` is set.
    pub fn is_set(&self, i: u64) -> Option<bool> {
        // TODO unchecked version
        self.vec.predecessor(i).map(|p| p == i)
    }

    /// Gets the bit at position `i`.
    /// Returns `Some(1)` if the bit is set, `Some(0)` if it is not set, and `None` if `i` is out of bounds.
    pub fn get(&self, i: u64) -> Option<u64> {
        // todo unchecked version
        self.is_set(i).map(|b| b as u64)
    }

    pub fn select1(&self, i: usize) -> Option<u64> {
        self.vec.get(i)
    }

    pub fn rank1(&self, i: u64) -> Option<u64> {
        todo!("implement rank for elias fano")
    }
}
