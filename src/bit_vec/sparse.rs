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

    /// Creates a new `SparseRSVec` from a `BitVec`, by compressing the sparse 1-bits.
    ///
    /// # Parameters
    /// - `input`: The input `BitVec` to compress.
    pub fn from_bitvec(input: &BitVec) -> Self {
        let len = input.len() as u64;
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

    /// Returns true if the bit at position `i` is set.
    ///
    /// # Panics
    /// If `i` is out of bounds the function might panic or produce incorrect results.
    /// Use [`is_set`] for a checked version.
    ///
    /// [`is_set`]: #method.is_set
    pub fn is_set_unchecked(&self, i: u64) -> bool {
        self.vec.predecessor_unchecked(i) == i
    }

    /// Returns true if the bit at position `i` is set.
    ///
    /// Returns `None` if `i` is out of bounds.
    pub fn is_set(&self, i: u64) -> Option<bool> {
        self.vec.predecessor(i).map(|p| p == i)
    }

    /// Gets the bit at position `i`.
    /// Returns `Some(1)` if the bit is set, `Some(0)` if it is not set, and `None` if `i` is out of bounds.
    pub fn get(&self, i: u64) -> Option<u64> {
        // todo unchecked version
        self.is_set(i).map(|b| b as u64)
    }

    /// Return the position of the 1-bit with the given rank.
    /// The following holds for all `pos` with 1-bits:
    /// ``select1(rank1(pos)) == pos``
    ///
    /// If the rank is larger than the number of sparse bits in the vector, the vector length is returned.
    pub fn select1(&self, i: usize) -> u64 {
        self.vec.get(i).unwrap_or(self.len)
    }

    /// Returns the number of 1-bits in the vector up to position `i`.
    ///
    /// If `i` is out of bounds, the number of 1-bits in the vector is returned.
    pub fn rank1(&self, i: u64) -> u64 {
        self.vec.rank(i)
    }

    /// Returns the number of non-sparse bits in the vector up to position `i`.
    /// The non-sparse bits are the ones this vector is not built from, meaning they can be either 1 or 0,
    /// depending on the input to the constructor.
    ///
    /// If `i` is out of bounds, the number of non-sparse bits in the vector is returned.
    pub fn rank0(&self, i: u64) -> u64 {
        if i >= self.len {
            self.len - self.vec.rank(self.len)
        } else {
            i - self.vec.rank(i)
        }
    }
}
