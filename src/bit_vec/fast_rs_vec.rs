use super::WORD_SIZE;
use crate::util::unroll;
use crate::BitVec;
use core::arch::x86_64::_pdep_u64;
use std::cmp::min;
use std::mem::size_of;

/// Size of a block in the bitvector. The size is deliberately chosen to fit one block into a
/// AVX256 register, so that we can use SIMD instructions to speed up rank and select queries.
const BLOCK_SIZE: usize = 512;

/// Size of a super block in the bitvector. Super-blocks exist to decrease the memory overhead
/// of block descriptors.
/// Increasing or decreasing the super block size has negligible effect on performance of rank
/// instruction. This means we want to make the super block size as large as possible, as long as
/// the zero-counter in normal blocks still fits in a reasonable amount of bits. However, this has
/// impact on the performance of select queries. The larger the super block size, the deeper will
/// a binary search be. We found 4096 to be a good compromise between memory overhead and
/// performance.
const SUPER_BLOCK_SIZE: usize = 1 << 12;

/// Size of a select block. The select block is used to speed up select queries. The select block
/// contains the indices of every `SELECT_BLOCK_SIZE`'th 1-bit and 0-bit in the bitvector.
/// The smaller this block-size, the faster are select queries, but the more memory is used.
const SELECT_BLOCK_SIZE: usize = 1 << 13;

/// Meta-data for a block. The `zeros` field stores the number of zeros up to the block,
/// beginning from the last super-block boundary. This means the first block in a super-block
/// always stores the number zero, which serves as a sentinel value to avoid special-casing the
/// first block in a super-block (which would be a performance hit due branch prediction failures).
#[derive(Clone, Copy, Debug)]
struct BlockDescriptor {
    zeros: u16,
}

/// Meta-data for a super-block. The `zeros` field stores the number of zeros up to this super-block.
/// This allows the `BlockDescriptor` to store the number of zeros in a much smaller
/// space. The `zeros` field is the number of zeros up to the super-block.
#[derive(Clone, Copy, Debug)]
struct SuperBlockDescriptor {
    zeros: usize,
}

/// Meta-data for the select query. Each entry i in the select vector contains the indices to find
/// the i * `SELECT_BLOCK_SIZE`'th 0- and 1-bit in the bitvector. Those indices may be very far apart.
#[derive(Clone, Copy, Debug)]
struct SelectSuperBlockDescriptor {
    index_0: usize,
    index_1: usize,
}

/// A bitvector that supports constant-time rank and select queries and is optimized for fast queries.
/// The bitvector is stored as a vector of `u64`s. The bit-vector stores meta-data for constant-time
/// rank and select queries, which takes sub-linear additional space. The space overhead is
/// 32 bytes per 512 bytes of user data (6.25%).
#[derive(Clone, Debug)]
pub struct RsVec {
    data: Vec<u64>,
    len: usize,
    blocks: Vec<BlockDescriptor>,
    super_blocks: Vec<SuperBlockDescriptor>,
    select_blocks: Vec<SelectSuperBlockDescriptor>,
}

impl RsVec {
    #[target_feature(enable = "popcnt")]
    unsafe fn naive_rank0(&self, pos: usize) -> usize {
        self.rank(true, pos)
    }

    #[target_feature(enable = "popcnt")]
    unsafe fn naive_rank1(&self, pos: usize) -> usize {
        self.rank(false, pos)
    }

    #[target_feature(enable = "bmi2")]
    unsafe fn bmi_select0(&self, rank: usize) -> usize {
        self.impl_select0(rank)
    }

    #[target_feature(enable = "bmi2")]
    unsafe fn bmi_select1(&self, rank: usize) -> usize {
        self.impl_select1(rank)
    }

    // I measured 5-10% improvement with this. I don't know why it's not inlined by default, the
    // branch elimination profits alone should make it worth it.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn rank(&self, zero: bool, pos: usize) -> usize {
        #[allow(unused_variables)]
        let index = pos / WORD_SIZE;
        let block_index = pos / BLOCK_SIZE;
        let super_block_index = pos / SUPER_BLOCK_SIZE;
        let mut rank = 0;

        // at first add the number of zeros/ones before the current super block
        rank += if zero {
            self.super_blocks[super_block_index].zeros
        } else {
            (super_block_index * SUPER_BLOCK_SIZE) - self.super_blocks[super_block_index].zeros
        };

        // then add the number of zeros/ones before the current block
        rank += if zero {
            self.blocks[block_index].zeros as usize
        } else {
            ((block_index % (SUPER_BLOCK_SIZE / BLOCK_SIZE)) * BLOCK_SIZE)
                - self.blocks[block_index].zeros as usize
        };

        // naive popcount of blocks
        for &i in &self.data[(block_index * BLOCK_SIZE) / WORD_SIZE..index] {
            rank += if zero {
                i.count_zeros() as usize
            } else {
                i.count_ones() as usize
            };
        }

        rank += if zero {
            (!self.data[index] & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as usize
        } else {
            (self.data[index] & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as usize
        };

        rank
    }

    #[allow(clippy::inline_always)]
    #[allow(clippy::assertions_on_constants)]
    #[inline(always)]
    unsafe fn impl_select0(&self, mut rank: usize) -> usize {
        let mut super_block = self.select_blocks[rank / SELECT_BLOCK_SIZE].index_0;

        if rank / SELECT_BLOCK_SIZE + 1 < self.select_blocks.len()
            && self.select_blocks[rank / SELECT_BLOCK_SIZE + 1].index_0 - super_block > 8
        {
            let mut upper_bound = self.select_blocks[rank / SELECT_BLOCK_SIZE + 1].index_0;

            while super_block < upper_bound - 1 {
                let middle = super_block + ((upper_bound - super_block) >> 1);
                if self.super_blocks[middle].zeros <= rank {
                    super_block = middle;
                } else {
                    upper_bound = middle;
                }
            }

            if self.super_blocks[super_block].zeros <= rank {
                super_block += 1;
            }
        } else {
            // linear search for super block that contains the rank
            while self.super_blocks.len() > (super_block + 1)
                && self.super_blocks[super_block + 1].zeros <= rank
            {
                super_block += 1;
            }
        }

        rank -= self.super_blocks[super_block].zeros;

        // full binary search for block that contains the rank, manually loop-unrolled, because
        // LLVM doesn't do it for us, but it gains just under 20% performance
        let mut block_index = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
        debug_assert!(SUPER_BLOCK_SIZE / BLOCK_SIZE == 8, "change unroll constant");
        unroll!(3,
            |boundary = { min((SUPER_BLOCK_SIZE / BLOCK_SIZE) / 2, (self.blocks.len() - block_index) / 2)}|
                if rank >= self.blocks[block_index + boundary].zeros as usize {
                    block_index += boundary;
                },
            boundary /= 2);

        rank -= self.blocks[block_index].zeros as usize;

        // todo non-bmi2 implementation as opt-in feature
        // linear search for word that contains the rank. Binary search is not possible here,
        // because we don't have accumulated popcounts for the words. We use pdep to find the
        // position of the rank-th zero bit in the word, if the word contains enough zeros, otherwise
        // we subtract the number of ones in the word from the rank and continue with the next word.
        let mut index_counter = 0;
        debug_assert!(BLOCK_SIZE / WORD_SIZE == 8, "change unroll constant");
        unroll!(7, |n = {0}| {
            let word = self.data[block_index * BLOCK_SIZE / WORD_SIZE + n];
            if (word.count_zeros() as usize) <= rank {
                rank -= word.count_zeros() as usize;
                index_counter += WORD_SIZE;
            } else {
                return block_index * BLOCK_SIZE
                    + index_counter
                    + _pdep_u64(1 << rank, !word).trailing_zeros() as usize;
            }
        }, n += 1);

        // the last word must contain the rank-th zero bit, otherwise the rank is outside of the
        // block, and thus outside of the bitvector
        block_index * BLOCK_SIZE
            + index_counter
            + _pdep_u64(
                1 << rank,
                !self.data[block_index * BLOCK_SIZE / WORD_SIZE + 7],
            )
            .trailing_zeros() as usize
    }

    #[allow(clippy::inline_always)]
    #[allow(clippy::assertions_on_constants)]
    #[inline(always)]
    unsafe fn impl_select1(&self, mut rank: usize) -> usize {
        let mut super_block = self.select_blocks[rank / SELECT_BLOCK_SIZE].index_1;

        if rank / SELECT_BLOCK_SIZE + 1 < self.select_blocks.len()
            && self.select_blocks[rank / SELECT_BLOCK_SIZE + 1].index_1 - super_block > 8
        {
            let mut upper_bound = self.select_blocks[rank / SELECT_BLOCK_SIZE + 1].index_1;

            // binary search for super block that contains the rank
            while super_block < upper_bound - 1 {
                let middle = super_block + ((upper_bound - super_block) >> 1);
                if ((middle + 1) * SUPER_BLOCK_SIZE - self.super_blocks[middle].zeros) <= rank
                {
                    super_block = middle;
                } else {
                    upper_bound = middle;
                }
            }

            if ((super_block + 1) * SUPER_BLOCK_SIZE - self.super_blocks[super_block].zeros) <= rank {
                super_block += 1;
            }
        } else {
            // linear search for super block that contains the rank
            while self.super_blocks.len() > (super_block + 1)
                && ((super_block + 1) * SUPER_BLOCK_SIZE - self.super_blocks[super_block + 1].zeros)
                    <= rank
            {
                super_block += 1;
            }
        }

        rank -= (super_block) * SUPER_BLOCK_SIZE - self.super_blocks[super_block].zeros;

        // full binary search for block that contains the rank, manually loop-unrolled, because
        // LLVM doesn't do it for us, but it gains just under 20% performance
        let block_at_super_block = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
        let mut block_index = block_at_super_block;
        debug_assert!(SUPER_BLOCK_SIZE / BLOCK_SIZE == 8, "change unroll constant");
        unroll!(3,
            |boundary = { min((SUPER_BLOCK_SIZE / BLOCK_SIZE) / 2, (self.blocks.len() - block_index) / 2)}|
                if rank >= (block_index + boundary - block_at_super_block) * BLOCK_SIZE - self.blocks[block_index + boundary].zeros as usize {
                    block_index += boundary;
                }
            , boundary /= 2);

        rank -= (block_index - block_at_super_block) * BLOCK_SIZE
            - self.blocks[block_index].zeros as usize;

        // todo non-bmi2 implementation as opt-in feature
        // linear search for word that contains the rank. Binary search is not possible here,
        // because we don't have accumulated popcounts for the words. We use pdep to find the
        // position of the rank-th zero bit in the word, if the word contains enough zeros, otherwise
        // we subtract the number of ones in the word from the rank and continue with the next word.
        let mut index_counter = 0;
        debug_assert!(BLOCK_SIZE / WORD_SIZE == 8, "change unroll constant");
        unroll!(7, |n = {0}| {
            let word = self.data[block_index * BLOCK_SIZE / WORD_SIZE + n];
            if (word.count_ones() as usize) <= rank {
                rank -= word.count_ones() as usize;
                index_counter += WORD_SIZE;
            } else {
                return block_index * BLOCK_SIZE
                    + index_counter
                    + _pdep_u64(1 << rank, word).trailing_zeros() as usize;
            }
        }, n += 1);

        // the last word must contain the rank-th zero bit, otherwise the rank is outside of the
        // block, and thus outside of the bitvector
        block_index * BLOCK_SIZE
            + index_counter
            + _pdep_u64(
                1 << rank,
                self.data[block_index * BLOCK_SIZE / WORD_SIZE + 7],
            )
            .trailing_zeros() as usize
    }

    /// Return the 0-rank of the bit at the given position. The 0-rank is the number of
    /// 0-bits in the vector up to but excluding the bit at the given position.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    pub fn rank0(&self, pos: usize) -> usize {
        unsafe { self.naive_rank0(pos) }
    }

    /// Return the 1-rank of the bit at the given position. The 1-rank is the number of
    /// 1-bits in the vector up to but excluding the bit at the given position.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    pub fn rank1(&self, pos: usize) -> usize {
        unsafe { self.naive_rank1(pos) }
    }

    /// Return the position of the 0-bit with the given rank. See `rank0`.
    pub fn select0(&self, rank: usize) -> usize {
        unsafe { self.bmi_select0(rank) }
    }

    /// Return the position of the 1-bit with the given rank. See `rank1`.
    pub fn select1(&self, rank: usize) -> usize {
        unsafe { self.bmi_select1(rank) }
    }

    /// Return the length of the vector, i.e. the number of bits it contains.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the bit at the given position within a u64 word. The bit takes the least significant
    /// bit of the returned u64 word.
    pub fn get(&self, pos: usize) -> u64 {
        (self.data[pos / WORD_SIZE] >> (pos % WORD_SIZE)) & 1
    }

    /// Returns the number of bytes on the heap for this vector.
    pub fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>()
            + self.blocks.len() * size_of::<BlockDescriptor>()
            + self.super_blocks.len() * size_of::<SuperBlockDescriptor>()
            + self.select_blocks.len() * size_of::<SelectSuperBlockDescriptor>()
    }
}

/// A builder for `FastBitVector`. This is used to efficiently construct a `BitVector` by appending
/// bits to it. Once all bits have been appended, the `BitVector` can be built using the `build`
/// method. If the number of bits to be appended is known in advance, it is recommended to use
/// `with_capacity` to avoid re-allocations. If bits are already available in little endian u64
/// words, those words can be appended using `append_word`.
#[derive(Clone, Debug)]
pub struct RsVectorBuilder {
    vec: BitVec,
}

impl Default for RsVectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RsVectorBuilder {
    /// Create a new empty `BitVectorBuilder`.
    #[must_use]
    pub fn new() -> RsVectorBuilder {
        RsVectorBuilder { vec: BitVec::new() }
    }

    /// Create a new empty `BitVectorBuilder` with the specified initial capacity to avoid
    /// re-allocations. The capacity is measured in bits
    #[must_use]
    pub fn with_capacity(capacity: usize) -> RsVectorBuilder {
        RsVectorBuilder {
            vec: BitVec::with_capacity(capacity),
        }
    }

    /// Append a bit to the vector.
    pub fn append_bit<T>(&mut self, bit: T)
    where
        T: Into<u64>,
    {
        self.vec.append_bit(bit.into());
    }

    /// Append a word to the vector. The word is assumed to be in little endian, i.e. the least
    /// significant bit is the first bit.
    pub fn append_word(&mut self, word: u64) {
        self.vec.append_word(word);
    }
}

/// A trait to be implemented for each bit vector type to specify how it is built from a
/// `BitVectorBuilder`.
impl RsVectorBuilder {
    /// Build the `BitVector` from all bits that have been appended so far. This will consume the
    /// `BitVectorBuilder`.
    #[must_use]
    pub fn build(self) -> RsVec {
        Self::from_bit_vec(self.vec)
    }

    /// Build a `BitVector` from a `BitVec`. This will consume the `BitVec`.
    #[must_use]
    pub fn from_bit_vec(mut vec: BitVec) -> RsVec {
        // Construct the block descriptor meta data. Each block descriptor contains the number of
        // zeros in the super-block, up to but excluding the block.
        let mut blocks = Vec::with_capacity(vec.len() / BLOCK_SIZE + 1);
        let mut super_blocks = Vec::with_capacity(vec.len() / SUPER_BLOCK_SIZE + 1);
        let mut select_blocks = Vec::new();

        // sentinel value
        select_blocks.push(SelectSuperBlockDescriptor {
            index_0: 0,
            index_1: 0,
        });

        let mut total_zeros: usize = 0;
        let mut current_zeros: usize = 0;
        for (idx, &word) in vec.data.iter().enumerate() {
            // if we moved past a block boundary, append the block information for the previous
            // block and reset the counter if we moved past a super-block boundary.
            if idx % (BLOCK_SIZE / WORD_SIZE) == 0 {
                if idx % (SUPER_BLOCK_SIZE / WORD_SIZE) == 0 {
                    total_zeros += current_zeros;
                    current_zeros = 0;
                    super_blocks.push(SuperBlockDescriptor { zeros: total_zeros });
                }

                // this cannot overflow because the only block where it could (the last in a super-
                // block) is not added to the list of blocks
                #[allow(clippy::cast_possible_truncation)]
                blocks.push(BlockDescriptor {
                    zeros: current_zeros as u16,
                });
            }

            // count the zeros in the current word and add them to the counter
            // the last word may contain padding zeros, which should not be counted,
            // but since we do not append the last block descriptor, this is not a problem
            let new_zeros = word.count_zeros() as usize;
            let all_zeros = total_zeros + current_zeros + new_zeros;
            if all_zeros / SELECT_BLOCK_SIZE > (total_zeros + current_zeros) / SELECT_BLOCK_SIZE {
                if all_zeros / SELECT_BLOCK_SIZE == select_blocks.len() {
                    select_blocks.push(SelectSuperBlockDescriptor {
                        index_0: super_blocks.len() - 1,
                        index_1: 0,
                    });
                } else {
                    select_blocks[all_zeros / SELECT_BLOCK_SIZE].index_0 = super_blocks.len() - 1;
                }
            }

            let total_bits = (idx + 1) * WORD_SIZE;
            let all_ones = total_bits - all_zeros;
            if all_ones / SELECT_BLOCK_SIZE
                > (idx * WORD_SIZE - total_zeros - current_zeros) / SELECT_BLOCK_SIZE
            {
                if all_ones / SELECT_BLOCK_SIZE == select_blocks.len() {
                    select_blocks.push(SelectSuperBlockDescriptor {
                        index_0: 0,
                        index_1: super_blocks.len() - 1,
                    });
                } else {
                    select_blocks[all_ones / SELECT_BLOCK_SIZE].index_1 = super_blocks.len() - 1;
                }
            }

            current_zeros += new_zeros;
        }

        // pad the internal vector to be block-aligned, so SIMD operations don't try to read
        // past the end of the vector. Note that this does not affect the content of the vector,
        // because those bits are not considered part of the vector.
        while vec.data.len() % (BLOCK_SIZE / WORD_SIZE) != 0 {
            vec.data.push(0);
        }

        RsVec {
            data: vec.data,
            len: vec.len,
            blocks,
            super_blocks,
            select_blocks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RsVectorBuilder;
    use rand::distributions::{Distribution, Uniform};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_append_bit() {
        let mut bv = RsVectorBuilder::new();
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        let bv = bv.build();

        assert_eq!(bv.data[..1], vec![0b110]);
    }

    #[test]
    fn test_random_data_rank() {
        let mut bv = RsVectorBuilder::with_capacity(LENGTH);
        let mut rng = StdRng::from_seed([
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4,
            5, 6, 7,
        ]);
        let sample = Uniform::new(0, 2);
        static LENGTH: usize = 4 * SUPER_BLOCK_SIZE;

        for _ in 0..LENGTH {
            bv.append_bit(sample.sample(&mut rng) as u8);
        }

        let bv = bv.build();

        assert_eq!(bv.len(), LENGTH);

        for _ in 0..100 {
            let rnd_index = rng.gen_range(0..LENGTH);
            let actual_rank1 = bv.rank1(rnd_index);
            let actual_rank0 = bv.rank0(rnd_index);

            let data = &bv.data;
            let mut expected_rank1 = 0;
            let mut expected_rank0 = 0;

            let data_index = rnd_index / WORD_SIZE;
            let bit_index = rnd_index % WORD_SIZE;

            for i in 0..data_index {
                expected_rank1 += data[i].count_ones() as usize;
                expected_rank0 += data[i].count_zeros() as usize;
            }

            if bit_index > 0 {
                expected_rank1 += (data[data_index] & (1 << bit_index) - 1).count_ones() as usize;
                expected_rank0 += (!data[data_index] & (1 << bit_index) - 1).count_ones() as usize;
            }

            assert_eq!(actual_rank1, expected_rank1);
            assert_eq!(actual_rank0, expected_rank0);
        }
    }

    #[test]
    fn test_append_bit_long() {
        let mut bv = RsVectorBuilder::new();
        let len = SUPER_BLOCK_SIZE + 1;
        for _ in 0..len {
            bv.append_bit(0u8);
            bv.append_bit(1u8);
        }

        let bv = bv.build();

        assert_eq!(bv.len(), len * 2);
        assert_eq!(bv.rank0(2 * len - 1), len);
        assert_eq!(bv.rank1(2 * len - 1), len - 1);
    }

    #[test]
    fn test_rank() {
        let mut bv = RsVectorBuilder::new();
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

    #[test]
    fn test_multi_words_rank() {
        let mut bv = RsVectorBuilder::new();
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

    #[test]
    fn test_only_zeros_rank() {
        let mut bv = RsVectorBuilder::new();
        for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
            bv.append_word(0);
        }
        bv.append_bit(0u8);
        let bv = bv.build();

        assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

        for i in 0..bv.len() {
            assert_eq!(bv.rank0(i), i);
            assert_eq!(bv.rank1(i), 0);
        }
    }

    #[test]
    fn test_only_ones_rank() {
        let mut bv = RsVectorBuilder::new();
        for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
            bv.append_word(u64::MAX);
        }
        bv.append_bit(1u8);
        let bv = bv.build();

        assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

        for i in 0..bv.len() {
            assert_eq!(bv.rank0(i), 0);
            assert_eq!(bv.rank1(i), i);
        }
    }

    #[test]
    fn test_simple_select() {
        let mut bv = RsVectorBuilder::new();
        bv.append_word(0b10110);
        let bv = bv.build();
        assert_eq!(bv.select0(0), 0);
        assert_eq!(bv.select1(1), 2);
        assert_eq!(bv.select0(1), 3);
    }

    #[test]
    fn test_multi_words_select() {
        let mut bv = RsVectorBuilder::new();
        bv.append_word(0);
        bv.append_word(0);
        bv.append_word(0b10110);
        let bv = bv.build();
        assert_eq!(bv.select1(0), 129);
        assert_eq!(bv.select1(1), 130);
        assert_eq!(bv.select0(32), 32);
        assert_eq!(bv.select0(128), 128);
        assert_eq!(bv.select0(129), 131);
    }

    #[test]
    fn test_only_zeros_select() {
        let mut bv = RsVectorBuilder::new();
        for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
            bv.append_word(0);
        }
        bv.append_bit(0u8);
        let bv = bv.build();

        assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

        for i in 0..bv.len() {
            assert_eq!(bv.select0(i), i);
        }
    }

    #[test]
    fn test_only_ones_select() {
        let mut bv = RsVectorBuilder::new();

        for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
            bv.append_word(u64::MAX);
        }
        bv.append_bit(1u8);
        let bv = bv.build();

        assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

        for i in 0..bv.len() {
            assert_eq!(bv.select1(i), i);
        }
    }

    #[test]
    fn random_data_select() {
        let mut bv = RsVectorBuilder::with_capacity(LENGTH);
        let mut rng = StdRng::from_seed([
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4,
            5, 6, 7,
        ]);
        let sample = Uniform::new(0, 2);
        static LENGTH: usize = 4 * SUPER_BLOCK_SIZE;

        for _ in 0..LENGTH {
            bv.append_bit(sample.sample(&mut rng) as u8);
        }

        let bv = bv.build();

        assert_eq!(bv.len(), LENGTH);

        for _ in 0..100 {
            // since we need a random rank, do not generate a number within the full length of
            // the vector, as only approximately half of the bits are set.
            let rnd_rank = rng.gen_range(0..LENGTH / 2 - BLOCK_SIZE);
            let actual_index0 = bv.select0(rnd_rank);

            let data = &bv.data;
            let mut rank_counter = 0;
            let mut expected_index0 = 0;

            let mut index = 0;
            loop {
                let zeros = data[index].count_zeros() as usize;
                if rank_counter + zeros > rnd_rank {
                    break;
                } else {
                    rank_counter += zeros;
                    expected_index0 += WORD_SIZE;
                    index += 1;
                }
            }

            let mut bit_index = 0;
            loop {
                if data[index] & (1 << bit_index) == 0 {
                    if rank_counter == rnd_rank {
                        break;
                    } else {
                        rank_counter += 1;
                    }
                }
                expected_index0 += 1;
                bit_index += 1;
            }

            assert_eq!(actual_index0, expected_index0);
        }
    }
}
