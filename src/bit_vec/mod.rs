use std::marker::PhantomData;

pub mod fast_rs_vec;

/// Size of a word in bitvectors. All vectors operate on 64-bit words.
const WORD_SIZE: usize = 64;

/// A simple bit vector that does not support rank and select queries. It has a constant memory
/// overhead of 32 bytes.
#[derive(Clone, Debug)]
pub struct BitVec {
    data: Vec<u64>,
    len: usize,
}

impl BitVec {
    /// Create a new empty bit vector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
        }
    }

    /// Create a new empty bit vector with the given capacity. The capacity is measured in bits.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity / WORD_SIZE + 1),
            len: 0,
        }
    }

    /// Create a new bit vector with all zeros and the given length. The length is measured in bits.
    #[must_use]
    pub fn from_zeros(len: usize) -> Self {
        let mut data = Vec::with_capacity(len / WORD_SIZE + 1);
        for _ in 0..len / WORD_SIZE {
            data.push(0);
        }
        if len % WORD_SIZE != 0 {
            data.push(0);
        }
        Self {
            data,
            len,
        }
    }

    /// Append a bit to the bit vector.
    pub fn append(&mut self, bit: bool) {
        if self.len % WORD_SIZE == 0 {
            self.data.push(0);
        }
        if bit {
            self.data[self.len / WORD_SIZE] |= 1 << (self.len % WORD_SIZE);
        }
        self.len += 1;
    }

    /// Append a bit from a quad-word. The least significant bit is appended to the bit vector.
    /// All other bits are ignored.
    pub fn append_bit(&mut self, bit: u64) {
        if self.len % WORD_SIZE == 0 {
            self.data.push(0);
        }
        self.data[self.len / WORD_SIZE] |= (bit % 2) << (self.len % WORD_SIZE);
        self.len += 1;
    }

    /// Append a word to the bit vector. The least significant bit is appended first.
    pub fn append_word(&mut self, word: u64) {
        if self.len % WORD_SIZE == 0 {
            self.data.push(word);
        } else {
            self.data[self.len / WORD_SIZE] |= word << (self.len % WORD_SIZE);
            self.data.push(word >> (WORD_SIZE - self.len % WORD_SIZE));
        }
        self.len += WORD_SIZE;
    }

    /// Flip the bit at the given position.
    pub fn flip_bit(&mut self, pos: usize) {
        self.data[pos / WORD_SIZE] ^= 1 << (pos % WORD_SIZE);
    }

    /// Return the bit at the given position.
    pub fn get(&self, pos: usize) -> bool {
        self.data[pos / WORD_SIZE] & (1 << (pos % WORD_SIZE)) != 0
    }
}

/// A common trait for all bit vectors for applications that want to use them
/// interchangeably. Offers all common functionality like rank, select, and collection accessors.
pub trait RsVector {
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

    /// Return the bit at the given position within a u64 word. The bit takes the least significant
    /// bit of the returned u64 word.
    fn get(&self, pos: usize) -> u64;
}

/// A builder for `BitVector`s. This is used to efficiently construct a `BitVector` by appending
/// bits to it. Once all bits have been appended, the `BitVector` can be built using the `build`
/// method. If the number of bits to be appended is known in advance, it is recommended to use
/// `with_capacity` to avoid re-allocations. If bits are already available in little endian u64
/// words, those words can be appended using `append_word`.
#[derive(Clone, Debug)]
pub struct RsVectorBuilder<S: BuildingStrategy> {
    phantom: PhantomData<S>,
    vec: BitVec,
}

impl<S: BuildingStrategy> Default for RsVectorBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: BuildingStrategy> RsVectorBuilder<S> {
    /// Create a new empty `BitVectorBuilder`.
    #[must_use]
    pub fn new() -> RsVectorBuilder<S> {
        RsVectorBuilder {
            phantom: PhantomData,
            vec: BitVec::new(),
        }
    }

    /// Create a new empty `BitVectorBuilder` with the specified initial capacity to avoid
    /// re-allocations. The capacity is measured in bits
    #[must_use]
    pub fn with_capacity(capacity: usize) -> RsVectorBuilder<S> {
        RsVectorBuilder {
            phantom: PhantomData,
            vec: BitVec::with_capacity(capacity),
        }
    }

    /// Append a bit to the vector.
    pub fn append_bit<T>(&mut self, bit: T)
        where
            T: Into<u64>,
    {
        self.vec.append_bit(bit.into())
    }

    /// Append a word to the vector. The word is assumed to be in little endian, i.e. the least
    /// significant bit is the first bit.
    pub fn append_word(&mut self, word: u64) {
        self.vec.append_word(word)
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
    type Vector: RsVector;

    /// Build the `BitVector` from all bits that have been appended so far. This will consume the
    /// `BitVectorBuilder`.
    fn build(builder: RsVectorBuilder<Self>) -> Self::Vector
        where
            Self: Sized;
}

#[cfg(test)]
mod common_tests {
    use super::{RsVectorBuilder, BuildingStrategy, RsVector};

    pub(crate) fn test_append_bit_long<B: BuildingStrategy>(
        mut bv: RsVectorBuilder<B>,
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

    pub(crate) fn test_rank<B: BuildingStrategy>(mut bv: RsVectorBuilder<B>) {
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

    pub(crate) fn test_multi_words_rank<B: BuildingStrategy>(mut bv: RsVectorBuilder<B>) {
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
        mut bv: RsVectorBuilder<B>,
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
        mut bv: RsVectorBuilder<B>,
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

    pub(crate) fn test_simple_select<B: BuildingStrategy>(mut bv: RsVectorBuilder<B>) {
        bv.append_word(0b10110);
        let bv = bv.build();
        assert_eq!(bv.select0(0), 0);
        assert_eq!(bv.select1(1), 2);
        assert_eq!(bv.select0(1), 1);
        assert_eq!(bv.select0(2), 4);
    }

    pub(crate) fn test_multi_words_select<B: BuildingStrategy>(mut bv: RsVectorBuilder<B>) {
        bv.append_word(0);
        bv.append_word(0);
        bv.append_word(0b10110);
        let bv = bv.build();
        assert_eq!(bv.select1(0), 0);
        assert_eq!(bv.select1(1), 130);
        assert_eq!(bv.select0(32), 32);
        assert_eq!(bv.select0(128), 128);
        assert_eq!(bv.select0(129), 129);
        assert_eq!(bv.select0(130), 132);
    }

    pub(crate) fn test_only_zeros_select<B: BuildingStrategy>(
        mut bv: RsVectorBuilder<B>,
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

        assert_eq!(bv.select1(0), 0);
    }

    pub(crate) fn test_only_ones_select<B: BuildingStrategy>(
        mut bv: RsVectorBuilder<B>,
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
            assert_eq!(bv.select1(i), i);
        }

        assert_eq!(bv.select0(0), 0);
    }
}
