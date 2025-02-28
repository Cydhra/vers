use crate::{BitVec, EliasFanoVec};

/// A succinct representation of a sparse vector with rank and select support.
/// The vector is a compressed sequence of indices of set bits.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseRSVec {
    vec: EliasFanoVec,
    len: u64,
}

impl SparseRSVec {
    /// Creates a new `SparseRSVec` from a sequence of set bits represented as indices.
    /// The input must be sorted in ascending order and free of duplicates.
    ///
    /// The length of the vector must be passed as well, as it cannot be inferred from the input,
    /// if the last bit in the vector is not set.
    ///
    /// # Parameters
    /// - `input`: The positions of set bits, or unset bits if the sparse vector should compress
    /// zeros.
    /// - `len`: The length of the vector, which is needed if the last bit is not in the input slice.
    pub fn new(input: &[u64], len: u64) -> Self {
        debug_assert!(input.is_sorted(), "input must be sorted");
        debug_assert!(
            input.windows(2).all(|w| w[0] != w[1]),
            "input must be free of duplicates"
        );

        Self {
            vec: EliasFanoVec::from_slice(input),
            len,
        }
    }

    /// Creates a new `SparseRSVec` from a `BitVec`, by compressing the indices of 0-bits if `zero` is true,
    /// or the indices of 1-bits if `zero` is false.
    ///
    /// # Parameters
    /// - `input`: The input `BitVec` to compress.
    /// - `zero`: If true, compress the indices of 0-bits, otherwise compress the indices of 1-bits.
    pub fn from_bitvec(input: &BitVec, zero: bool) -> Self {
        let len = input.len() as u64;
        if zero {
            Self::new(
                input
                    .iter()
                    .enumerate()
                    .filter(|&(_, bit)| bit == 0)
                    .map(|(i, _)| i as u64)
                    .collect::<Vec<_>>()
                    .as_slice(),
                len,
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
                len,
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

impl From<&[u64]> for SparseRSVec {
    fn from(input: &[u64]) -> Self {
        Self::new(input)
    }
}
