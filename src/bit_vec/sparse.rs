//! A sparse bit vector with `rank1`, `rank0` and `select1` support.
//! The vector requires `O(n log u/n) + 2n + o(n)` bits of space, where `n` is the number of bits in the vector
//! and `u` is the number of 1-bits.
//! The vector is constructed from a sorted list of indices of 1-bits, or from an existing
//! [`BitVec`](crate::BitVec).

use crate::{BitVec, EliasFanoVec};

/// A succinct representation of a sparse vector with rank and select support.
/// It is a thin wrapper around an [`EliasFanoVec`] that compresses the indices of 1-bits.
///
/// Therefore, no `select0` function is provided.
/// However, the constructor [`from_bitvec_inverted`] can be used to cheaply invert the input `BitVec`,
/// reversing the roles of 1-bits and 0-bits.
///
/// # Examples
/// ```
/// use vers_vecs::SparseRSVec;
///
/// let sparse = SparseRSVec::new(&[1, 3, 5, 7, 9], 12);
/// assert_eq!(sparse.get(5), Some(1));
/// assert_eq!(sparse.get(11), Some(0));
/// assert_eq!(sparse.get(12), None);
///
/// assert_eq!(sparse.rank1(5), 2);
/// assert_eq!(sparse.select1(2), 5);
/// ```
///
/// It cn also be constructed from a `BitVec` directly:
/// ```
/// use vers_vecs::SparseRSVec;
/// use vers_vecs::BitVec;
///
/// let mut bv = BitVec::from_zeros(12);
/// bv.flip_bit(6);
/// bv.flip_bit(7);
///
/// let sparse = SparseRSVec::from_bitvec(&bv);
/// assert_eq!(sparse.rank1(5), 0);
/// assert_eq!(sparse.select1(0), 6);
/// ```
///
/// [`EliasFanoVec`]: struct.EliasFanoVec.html
/// [`from_bitvec_inverted`]: #method.from_bitvec_inverted
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
    ///   zeros.
    /// - `len`: The length of the vector, which is needed if the last bit is not in the input slice.
    #[must_use]
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
    #[must_use]
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

    /// Creates a new `SparseRSVec` from a `BitVec`.
    /// However, before compressing the 1-bits, the input is inverted.
    /// This means that the sparse vector will compress the 0-bits instead of the 1-bits,
    /// and the [`rank1`] and [`select1`] functions will return the number of 0-bits and the position of 0-bits.
    ///
    /// This is a convenience function to allow for easy creation of sparse vectors that compress
    /// zeros, despite the lack of a `select0` function.
    ///
    /// However, do note that [`get`] will return the inverted value of the bit at position `i` from
    /// the original `BitVec`.
    ///
    /// # Parameters
    /// - `input`: The input `BitVec` to compress.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::SparseRSVec;
    /// use vers_vecs::BitVec;
    ///
    /// let mut bv = BitVec::from_ones(12);
    /// // set 6 and 7 to 0
    /// bv.flip_bit(6);
    /// bv.flip_bit(7);
    ///
    /// let sparse = SparseRSVec::from_bitvec_inverted(&bv);
    /// // now select1 gives the position of 0-bits
    /// assert_eq!(sparse.select1(1), 7);
    /// ```
    ///
    /// [`rank1`]: #method.rank1
    /// [`select1`]: #method.select1
    /// [`get`]: #method.get
    #[must_use]
    pub fn from_bitvec_inverted(input: &BitVec) -> Self {
        let len = input.len() as u64;
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
    }

    /// Returns true if the bit at position `i` is set.
    ///
    /// If `i` is out of bounds the function produces incorrect results.
    /// Use [`is_set`] for a checked version.
    ///
    /// [`is_set`]: #method.is_set
    #[must_use]
    pub fn is_set_unchecked(&self, i: u64) -> bool {
        self.vec.predecessor_unchecked(i) == i
    }

    /// Returns true if the bit at position `i` is set.
    ///
    /// Returns `None` if `i` is out of bounds.
    #[must_use]
    pub fn is_set(&self, i: u64) -> Option<bool> {
        if i >= self.len {
            None
        } else {
            // if the predecessor is None, the bit is left of the first 1-bit
            Some(self.vec.predecessor(i).is_some_and(|p| p == i))
        }
    }

    /// Gets the bit at position `i`.
    /// Returns 1 if the bit is set, 0 if it is not set.
    ///
    /// # Panics
    /// If `i` is out of bounds the function might panic or produce incorrect results.
    /// Use [`get`] for a checked version.
    #[must_use]
    pub fn get_unchecked(&self, i: u64) -> u64 {
        self.is_set_unchecked(i).into()
    }

    /// Gets the bit at position `i`.
    /// Returns `Some(1)` if the bit is set, `Some(0)` if it is not set, and `None` if `i` is out of bounds.
    #[must_use]
    pub fn get(&self, i: u64) -> Option<u64> {
        self.is_set(i).map(std::convert::Into::into)
    }

    /// Return the position of the 1-bit with the given rank.
    /// The following holds for all `pos` with 1-bits:
    /// ``select1(rank1(pos)) == pos``
    ///
    /// If the rank is larger than the number of sparse bits in the vector, the vector length is returned.
    #[must_use]
    pub fn select1(&self, i: usize) -> u64 {
        self.vec.get(i).unwrap_or(self.len)
    }

    /// Returns the number of 1-bits in the vector up to position `i`.
    ///
    /// If `i` is out of bounds, the number of 1-bits in the vector is returned.
    #[must_use]
    pub fn rank1(&self, i: u64) -> u64 {
        self.vec.rank(i)
    }

    /// Returns the number of 0-bits in the vector up to position `i`.
    ///
    /// If `i` is out of bounds, the number of 0-bits in the vector is returned.
    #[must_use]
    pub fn rank0(&self, i: u64) -> u64 {
        if i >= self.len {
            self.len - self.vec.rank(self.len)
        } else {
            i - self.vec.rank(i)
        }
    }

    /// Returns an iterator over the 1-bits in the vector.
    /// The iterator yields the positions of the 1-bits in ascending order.
    pub fn iter1(&self) -> impl Iterator<Item = u64> + '_ {
        self.vec.iter()
    }

    /// Returns the length of the bit vector if it was uncompressed.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Returns true if the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bytes used by the vector on the heap.
    ///  Does not include allocated memory that isn't used.
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.vec.heap_size()
    }
}

impl From<BitVec> for SparseRSVec {
    fn from(input: BitVec) -> Self {
        Self::from_bitvec_inverted(&input)
    }
}

impl<'a> From<&'a BitVec> for SparseRSVec {
    fn from(input: &'a BitVec) -> Self {
        Self::from_bitvec_inverted(input)
    }
}

#[cfg(test)]
mod tests {
    use super::SparseRSVec;
    use crate::BitVec;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_sparse_rank() {
        let sparse = SparseRSVec::new(&[1, 3, 5, 7, 9], 12);
        assert_eq!(sparse.rank1(0), 0);
        assert_eq!(sparse.rank1(1), 0);
        assert_eq!(sparse.rank1(2), 1);
        assert_eq!(sparse.rank1(3), 1);
        assert_eq!(sparse.rank1(4), 2);
        assert_eq!(sparse.rank1(5), 2);
        assert_eq!(sparse.rank1(6), 3);
        assert_eq!(sparse.rank1(7), 3);
        assert_eq!(sparse.rank1(8), 4);
        assert_eq!(sparse.rank1(9), 4);
        assert_eq!(sparse.rank1(10), 5);
        assert_eq!(sparse.rank1(11), 5);
        assert_eq!(sparse.rank1(12), 5);
        assert_eq!(sparse.rank1(999), 5);
    }

    #[test]
    fn test_sparse_select() {
        let sparse = SparseRSVec::new(&[1, 3, 5, 7, 9], 12);
        assert_eq!(sparse.select1(0), 1);
        assert_eq!(sparse.select1(1), 3);
        assert_eq!(sparse.select1(2), 5);
        assert_eq!(sparse.select1(3), 7);
        assert_eq!(sparse.select1(4), 9);
        assert_eq!(sparse.select1(5), 12);
        assert_eq!(sparse.select1(6), 12);
    }

    #[test]
    fn test_sparse_rank0() {
        let sparse = SparseRSVec::new(&[1, 3, 5, 7, 9], 12);
        assert_eq!(sparse.rank0(0), 0);
        assert_eq!(sparse.rank0(1), 1);
        assert_eq!(sparse.rank0(2), 1);
        assert_eq!(sparse.rank0(3), 2);
        assert_eq!(sparse.rank0(4), 2);
        assert_eq!(sparse.rank0(5), 3);
        assert_eq!(sparse.rank0(6), 3);
        assert_eq!(sparse.rank0(7), 4);
        assert_eq!(sparse.rank0(8), 4);
        assert_eq!(sparse.rank0(9), 5);
        assert_eq!(sparse.rank0(10), 5);
        assert_eq!(sparse.rank0(11), 6);
        assert_eq!(sparse.rank0(12), 7);
        assert_eq!(sparse.rank0(999), 7);
    }

    #[test]
    fn test_empty_sparse() {
        let sparse = SparseRSVec::new(&[], 0);
        assert_eq!(sparse.rank1(0), 0);
        assert_eq!(sparse.rank1(1), 0);
        assert_eq!(sparse.rank1(999), 0);
        assert_eq!(sparse.select1(0), 0);
        assert_eq!(sparse.select1(1), 0);
        assert_eq!(sparse.select1(999), 0);
        assert_eq!(sparse.rank0(0), 0);
        assert_eq!(sparse.rank0(1), 0);
        assert_eq!(sparse.rank0(999), 0);
        assert!(sparse.is_empty());
        assert_eq!(sparse.len(), 0);
    }

    #[test]
    fn test_sparse_get() {
        let sparse = SparseRSVec::new(&[1, 3, 5, 7, 9], 12);
        assert_eq!(sparse.get(0), Some(0));
        assert_eq!(sparse.get(1), Some(1));
        assert_eq!(sparse.get(2), Some(0));
        assert_eq!(sparse.get(3), Some(1));
        assert_eq!(sparse.get(4), Some(0));
        assert_eq!(sparse.get(5), Some(1));
        assert_eq!(sparse.get(6), Some(0));
        assert_eq!(sparse.get(7), Some(1));
        assert_eq!(sparse.get(8), Some(0));
        assert_eq!(sparse.get(9), Some(1));
        assert_eq!(sparse.get(10), Some(0));
        assert_eq!(sparse.get(11), Some(0));
        assert_eq!(sparse.get(12), None);
        assert_eq!(sparse.get(999), None);
    }

    #[test]
    fn test_from_bitvector() {
        let mut bv = BitVec::from_ones(12);
        bv.flip_bit(6);
        bv.flip_bit(7);

        let sparse = SparseRSVec::from_bitvec(&bv);
        assert_eq!(sparse.rank1(0), 0);
        assert_eq!(sparse.rank1(1), 1);
        assert_eq!(sparse.rank1(2), 2);
        assert_eq!(sparse.rank1(7), 6);
        assert_eq!(sparse.rank1(8), 6);
        assert_eq!(sparse.rank1(9), 7);
        assert_eq!(sparse.rank1(12), 10);

        let sparse = SparseRSVec::from_bitvec_inverted(&bv);
        assert_eq!(sparse.rank1(0), 0);
        assert_eq!(sparse.rank1(1), 0);
        assert_eq!(sparse.rank1(2), 0);
        assert_eq!(sparse.rank1(7), 1);
        assert_eq!(sparse.rank1(8), 2);
        assert_eq!(sparse.rank1(9), 2);
        assert_eq!(sparse.rank1(12), 2);
    }

    #[test]
    fn test_large_block() {
        // test that the implementation works correctly if the search triggers a binary search
        let sparse = SparseRSVec::new(
            &[
                1, 100_000, 100_001, 100_002, 100_003, 100_004, 100_005, 100_006, 100_007, 100_008,
                100_009, 100_010, 1_000_000,
            ],
            2_000_000,
        );
        assert_eq!(sparse.rank1(100_008), 9);
        assert_eq!(sparse.rank1(100_012), 12);
    }

    #[test]
    fn test_fuzzy() {
        const L: usize = 100_000;
        let mut bv = BitVec::from_zeros(L);
        let mut rng = StdRng::from_seed([0; 32]);

        for _ in 0..L / 4 {
            bv.flip_bit(rng.gen_range(0..L));
        }

        let sparse = SparseRSVec::from_bitvec(&bv);

        let mut ones = 0;
        for i in 0..L {
            assert_eq!(bv.get(i), sparse.get(i as u64));
            assert_eq!(ones, sparse.rank1(i as u64));
            assert_eq!(i as u64 - ones, sparse.rank0(i as u64));
            if bv.get(i) == Some(1) {
                assert_eq!(i, sparse.select1(ones as usize).try_into().unwrap());
                ones += 1;
            }
        }
    }

    #[test]
    fn test_from_padded_bitvec() {
        // test no garbage is added to the sparse vec when the bit vector contains trailing data
        let mut bv = BitVec::new();
        bv.append_bit(1);
        bv.append_bit(0);
        bv.append_bits(u64::MAX, 10);
        bv.drop_last(10);
        bv.append_bit(0);
        bv.drop_last(1);

        let sparse = SparseRSVec::from_bitvec(&bv);
        assert_eq!(sparse.len(), 2);
        assert_eq!(sparse.get(0), Some(1));
        assert_eq!(sparse.get(1), Some(0));
        assert_eq!(sparse.iter1().collect::<Vec<_>>(), vec![0]);
    }
}
