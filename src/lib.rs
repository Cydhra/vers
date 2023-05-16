use std::marker::PhantomData;
use std::ops::Rem;

pub use fast_bit_vec::FastBitVector;

mod fast_bit_vec;

/// Size of a word in bitvectors. All vectors operate on 64-bit words.
const WORD_SIZE: usize = 64;

/// A common trait for all bit vectors for applications that want to use them
/// interchangeably. Offers all common functionality like rank, select, and collection accessors.
pub trait BitVector {
    /// Return the 0-rank of the bit at the given position. The 0-rank is the number of
    /// 0-bits in the vector up to but excluding the bit at the given position.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    fn rank0(&self, pos: usize) -> usize;

    /// Return the 1-rank of the bit at the given position. The 1-rank is the number of
    /// 1-bits in the vector up to but excluding the bit at the given position.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    fn rank1(&self, pos: usize) -> usize;

    /// Return the position of the 0-bit with the given rank. See `rank0`.
    fn select0(&self, rank: usize) -> usize;

    /// Return the position of the 1-bit with the given rank. See `rank1`.
    fn select1(&self, rank: usize) -> usize;

    /// Return the length of the vector, i.e. the number of bits it contains.
    fn len(&self) -> usize;

    /// Return whether the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A builder for `BitVector`s. This is used to efficiently construct a `BitVector` by appending
/// bits to it. Once all bits have been appended, the `BitVector` can be built using the `build`
/// method. If the number of bits to be appended is known in advance, it is recommended to use
/// `with_capacity` to avoid re-allocations. If bits are already available in little endian u64
/// words, those words can be appended using `append_word`.
#[derive(Clone, Debug)]
pub struct BitVectorBuilder<S: BuildingStrategy> {
    phantom: PhantomData<S>,
    words: Vec<u64>,
    len: usize,
}

impl<S: BuildingStrategy> Default for BitVectorBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: BuildingStrategy> BitVectorBuilder<S> {
    /// Create a new empty `BitVectorBuilder`.
    #[must_use]
    pub fn new() -> BitVectorBuilder<S> {
        BitVectorBuilder {
            phantom: PhantomData,
            words: Vec::new(),
            len: 0,
        }
    }

    /// Create a new empty `BitVectorBuilder` with the specified initial capacity to avoid
    /// re-allocations.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> BitVectorBuilder<S> {
        BitVectorBuilder {
            phantom: PhantomData,
            words: Vec::with_capacity(capacity),
            len: 0,
        }
    }

    /// Append a bit to the vector.
    pub fn append_bit<T: Rem + From<u8>>(&mut self, bit: T)
    where
        T::Output: Into<u64>,
    {
        let bit: u64 = (bit % T::from(2u8)).into();

        if self.len % WORD_SIZE == 0 {
            self.words.push(0);
        }

        self.words[self.len / WORD_SIZE] |= bit << (self.len % WORD_SIZE);
        self.len += 1;
    }

    /// Append a word to the vector. The word is assumed to be in little endian, i.e. the least
    /// significant bit is the first bit. It is a logical error to append a word if the vector is
    /// not 64-bit aligned (i.e. has not a length that is a multiple of 64). If the vector is not
    /// 64-bit aligned, the last word already present will be padded with zeros,without
    /// affecting the length, meaning the bit-vector is corrupted afterwards.
    pub fn append_word(&mut self, word: u64) {
        debug_assert!(self.len % WORD_SIZE == 0);
        self.words.push(word);
        self.len += WORD_SIZE;
    }

    /// Build the `BitVector` from all bits that have been appended so far. This will consume the
    /// `BitVectorBuilder`.
    #[must_use]
    pub fn build(self) -> S::Vector {
        S::build(self)
    }
}

/// A trait to be implemented for each bit vector type to specify how it is built from a
/// `BitVectorBuilder`.
pub trait BuildingStrategy {
    /// The type of the bit vector that is built.
    type Vector: BitVector;

    /// Build the `BitVector` from all bits that have been appended so far. This will consume the
    /// `BitVectorBuilder`.
    fn build(builder: BitVectorBuilder<Self>) -> Self::Vector
    where
        Self: Sized;
}

#[cfg(test)]
mod common_tests {
    use crate::{BitVector, BitVectorBuilder, BuildingStrategy};

    pub(crate) fn test_append_bit_long<B: BuildingStrategy>(
        mut bv: BitVectorBuilder<B>,
        super_block_size: usize,
    ) {
        let len = super_block_size + 1;
        for _ in 0..len {
            bv.append_bit(0u8);
            bv.append_bit(1u8);
        }

        let bv = bv.build();

        assert_eq!(bv.len(), len * 2);
        assert_eq!(bv.rank0(2 * len - 1), len);
        assert_eq!(bv.rank1(2 * len - 1), len - 1);
    }

    pub(crate) fn test_rank<B: BuildingStrategy>(mut bv: BitVectorBuilder<B>) {
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        let bv = bv.build();

        // first bit must always have rank 0
        assert_eq!(bv.rank0(0), 0);
        assert_eq!(bv.rank1(0), 0);

        assert_eq!(bv.rank1(2), 1);
        assert_eq!(bv.rank1(3), 2);
        assert_eq!(bv.rank1(4), 2);
        assert_eq!(bv.rank0(3), 1);
    }

    pub(crate) fn test_multi_words_rank<B: BuildingStrategy>(mut bv: BitVectorBuilder<B>) {
        bv.append_word(0);
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        let bv = bv.build();

        assert_eq!(bv.rank0(63), 63);
        assert_eq!(bv.rank0(64), 64);
        assert_eq!(bv.rank0(65), 65);
        assert_eq!(bv.rank0(66), 65);
    }

    pub(crate) fn test_only_zeros_rank<B: BuildingStrategy>(
        mut bv: BitVectorBuilder<B>,
        super_block_size: usize,
        word_size: usize,
    ) {
        for _ in 0..2 * (super_block_size / word_size) {
            bv.append_word(0);
        }
        bv.append_bit(0u8);
        let bv = bv.build();

        assert_eq!(bv.len(), 2 * super_block_size + 1);

        for i in 0..bv.len() {
            assert_eq!(bv.rank0(i), i);
            assert_eq!(bv.rank1(i), 0);
        }
    }

    pub(crate) fn test_only_ones_rank<B: BuildingStrategy>(
        mut bv: BitVectorBuilder<B>,
        super_block_size: usize,
        word_size: usize,
    ) {
        for _ in 0..2 * (super_block_size / word_size) {
            bv.append_word(u64::MAX);
        }
        bv.append_bit(1u8);
        let bv = bv.build();

        assert_eq!(bv.len(), 2 * super_block_size + 1);

        for i in 0..bv.len() {
            assert_eq!(bv.rank0(i), 0);
            assert_eq!(bv.rank1(i), i);
        }
    }

    pub(crate) fn test_simple_select<B: BuildingStrategy>(mut bv: BitVectorBuilder<B>) {
        bv.append_word(0b10110);
        let bv = bv.build();
        assert_eq!(bv.select0(0), 0);
        assert_eq!(bv.select0(1), 1);
        assert_eq!(bv.select0(2), 4);
    }

    pub(crate) fn test_multi_words_select<B: BuildingStrategy>(mut bv: BitVectorBuilder<B>) {
        bv.append_word(0);
        bv.append_word(0);
        bv.append_word(0b10110);
        let bv = bv.build();
        assert_eq!(bv.select0(32), 32);
        assert_eq!(bv.select0(128), 128);
        assert_eq!(bv.select0(129), 129);
        assert_eq!(bv.select0(130), 132);
    }

    pub(crate) fn test_only_zeros_select<B: BuildingStrategy>(
        mut bv: BitVectorBuilder<B>,
        super_block_size: usize,
        word_size: usize,
    ) {
        for _ in 0..2 * (super_block_size / word_size) {
            bv.append_word(0);
        }
        bv.append_bit(0u8);
        let bv = bv.build();

        assert_eq!(bv.len(), 2 * super_block_size + 1);

        for i in 0..bv.len() {
            assert_eq!(bv.select0(i), i);
        }
    }
}
