#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use std::ops::Rem;

/// Size of a word in the bitvector.
const WORD_SIZE: usize = 64;

/// Size of a block in the bitvector. The size is deliberately chosen to fit one block into a
/// AVX256 register, so that we can use SIMD instructions to speed up rank and select queries.
const BLOCK_SIZE: usize = 256;

/// Size of a super block in the bitvector. Super-blocks exist to decrease the memory overhead
/// of block descriptors.
/// Increasing or decreasing the super block size has negligible effect on performance. This
/// means we want to make the super block size as large as possible, as long as the zero-counter
/// in normal blocks still fits in a reasonable amount of bits. So we chose the super block size
/// to be the largest power of two that still fits into a `u16`. Note that blocks store the number
/// of zeros up to the block, so they will never overflow even if the entire vector contains only
/// zeros.
const SUPER_BLOCK_SIZE: usize = 1 << 16;

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

/// A bitvector that supports constant-time rank and select queries. The bitvector is stored as
/// a vector of `u64`s. The last word is not necessarily full, in which case the remaining bits
/// are set to 0. The bit-vector stores meta-data for constant-time rank and select queries, which
/// takes sub-linear additional space.
#[derive(Clone, Debug)]
pub struct BitVector {
    data: Vec<u64>,
    len: usize,
    blocks: Vec<BlockDescriptor>,
    super_blocks: Vec<SuperBlockDescriptor>,
}

impl BitVector {
    /// Return the 0-rank of the bit at the given position. The 0-rank is the number of
    /// 0-bits in the vector up to but excluding the bit at the given position.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    pub fn rank0(&self, pos: usize) -> usize {
        unsafe {
            if cfg!(all(feature = "simd", target_arch = "x86_64")) {
                self.avx_rank0(pos)
            } else {
                self.naive_rank0(pos)
            }
        }
    }

    /// Return the 1-rank of the bit at the given position. The 1-rank is the number of
    /// 1-bits in the vector up to but excluding the bit at the given position.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    pub fn rank1(&self, pos: usize) -> usize {
        unsafe {
            if cfg!(all(feature = "simd", target_arch = "x86_64")) {
                self.avx_rank1(pos)
            } else {
                self.naive_rank1(pos)
            }
        }
    }

    #[target_feature(enable = "popcnt")]
    unsafe fn naive_rank0(&self, pos: usize) -> usize {
        self.rank(true, pos)
    }

    #[target_feature(enable = "popcnt")]
    unsafe fn naive_rank1(&self, pos: usize) -> usize {
        self.rank(false, pos)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn avx_rank0(&self, pos: usize) -> usize {
        self.rank(true, pos)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn avx_rank1(&self, pos: usize) -> usize {
        self.rank(false, pos)
    }

    // I measured 5-10% improvement with this. I don't know why it's not inlined by default, the
    // branch elimination profits alone should make it worth it.
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
            (super_block_index * SUPER_BLOCK_SIZE)
                - self.super_blocks[super_block_index].zeros
        };

        // then add the number of zeros/ones before the current block
        rank += if zero {
            self.blocks[block_index].zeros as usize
        } else {
            ((block_index % (SUPER_BLOCK_SIZE / BLOCK_SIZE)) * BLOCK_SIZE)
                - self.blocks[block_index].zeros as usize
        };

        // then add the number of zeros/ones in the current block up to the current position
        #[cfg(any(
            not(feature = "simd"),
            not(target_arch = "x86_64"),
            not(target_feature = "avx2")
        ))]
        {
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
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
        {
            // Wojciech Muła algorithm for SIMD popcount on AVX2.
            rank += unsafe {
                let full_block_boundary = (super_block_index * SUPER_BLOCK_SIZE
                    + (block_index % (SUPER_BLOCK_SIZE / BLOCK_SIZE)) * BLOCK_SIZE)
                    / WORD_SIZE;
                // if BLOCK_SIZE is changed, this method must be updated
                debug_assert!(BLOCK_SIZE / WORD_SIZE == 4);
                let block = if zero {
                    _mm256_set_epi64x(
                        !self.data[full_block_boundary] as i64,
                        !self.data[full_block_boundary + 1] as i64,
                        !self.data[full_block_boundary + 2] as i64,
                        !self.data[full_block_boundary + 3] as i64,
                    )
                } else {
                    _mm256_set_epi64x(
                        self.data[full_block_boundary] as i64,
                        self.data[full_block_boundary + 1] as i64,
                        self.data[full_block_boundary + 2] as i64,
                        self.data[full_block_boundary + 3] as i64,
                    )
                };

                // calculate a mask
                let mask1 = !u64::MAX.checked_shl((pos % BLOCK_SIZE) as u32).unwrap_or(0);
                let mask2 = !u64::MAX
                    .checked_shl(((pos % BLOCK_SIZE).saturating_sub(WORD_SIZE)) as u32)
                    .unwrap_or(0);
                let mask3 = !u64::MAX
                    .checked_shl(((pos % BLOCK_SIZE).saturating_sub(WORD_SIZE * 2)) as u32)
                    .unwrap_or(0);
                let mask4 = !u64::MAX
                    .checked_shl(((pos % BLOCK_SIZE).saturating_sub(WORD_SIZE * 3)) as u32)
                    .unwrap_or(0);
                let enable_mask =
                    _mm256_set_epi64x(mask1 as i64, mask2 as i64, mask3 as i64, mask4 as i64);
                let masked_block = _mm256_and_si256(block, enable_mask);

                // mask lower then higher nibbles
                let mask = _mm256_set1_epi8(0x0f);
                let rank_lookup = _mm256_set_epi8(
                    4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2,
                    2, 1, 2, 1, 1, 0,
                );

                // calculate popcount for lower and higher nibbles by using lookup table
                let low = _mm256_shuffle_epi8(rank_lookup, _mm256_and_si256(masked_block, mask));
                let high = _mm256_shuffle_epi8(
                    rank_lookup,
                    _mm256_and_si256(_mm256_srli_epi64::<4>(masked_block), mask),
                );
                (Self::hsum(low) + Self::hsum(high)) as usize
            };
        }

        rank
    }

    /// Horizontal sum (popcount) of a 256 bit vector.
    #[inline(always)]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    unsafe fn hsum(v: __m256i) -> u64 {
        let vlow = _mm256_castsi256_si128(v);
        let vhigh = _mm256_extracti128_si256::<1>(v);
        let vlow = _mm_add_epi64(vlow, vhigh);
        let conv = _mm_cvtsi128_si64(_mm_add_epi64(vlow, _mm_unpackhi_epi64(vlow, vlow))) as u64;
        (conv & 0xFF)
            + (conv >> 8 & 0xFF)
            + (conv >> 16 & 0xFF)
            + (conv >> 24 & 0xFF)
            + (conv >> 32 & 0xFF)
            + (conv >> 40 & 0xFF)
            + (conv >> 48 & 0xFF)
            + (conv >> 56 & 0xFF)
    }

    /// Return the length of the vector, i.e. the number of bits it contains.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A builder for `BitVector`s. This is used to efficiently construct a `BitVector` by appending
/// bits to it. Once all bits have been appended, the `BitVector` can be built using the `build`
/// method. If the number of bits to be appended is known in advance, it is recommended to use
/// `with_capacity` to avoid re-allocations. If bits are already available in little endian u64
/// words, those words can be appended using `append_word`.
#[derive(Clone, Debug)]
pub struct BitVectorBuilder {
    words: Vec<u64>,
    len: usize,
}

impl BitVectorBuilder {
    /// Create a new empty `BitVectorBuilder`.
    pub fn new() -> BitVectorBuilder {
        BitVectorBuilder {
            words: Vec::new(),
            len: 0,
        }
    }

    /// Create a new empty `BitVectorBuilder` with the specified initial capacity to avoid
    /// re-allocations.
    pub fn with_capacity(capacity: usize) -> BitVectorBuilder {
        BitVectorBuilder {
            words: Vec::with_capacity(capacity),
            len: 0,
        }
    }

    /// Append a bit to the vector.
    pub fn append_bit<T: Rem + From<u8>>(&mut self, bit: T)
    where
        T::Output: Into<u64>,
    {
        let bit = (bit % T::from(2u8)).into();

        if self.len % WORD_SIZE == 0 {
            self.words.push(0);
        }

        self.words[self.len / WORD_SIZE] |= (bit as u64) << (self.len % WORD_SIZE);
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
    pub fn build(mut self) -> BitVector {
        // Construct the block descriptor meta data. Each block descriptor contains the number of
        // zeros in the super-block, up to but excluding the block.
        let mut blocks = Vec::with_capacity(self.len / BLOCK_SIZE + 1);
        let mut super_blocks = Vec::with_capacity(self.len / SUPER_BLOCK_SIZE + 1);

        let mut total_zeros: usize = 0;
        let mut current_zeros: u32 = 0;
        for (idx, &word) in self.words.iter().enumerate() {
            // if we moved past a block boundary, append the block information for the previous
            // block and reset the counter if we moved past a super-block boundary.
            if idx % (BLOCK_SIZE / WORD_SIZE) == 0 {
                if idx % (SUPER_BLOCK_SIZE / WORD_SIZE) == 0 {
                    total_zeros += current_zeros as usize;
                    current_zeros = 0;
                    super_blocks.push(SuperBlockDescriptor { zeros: total_zeros });
                }

                blocks.push(BlockDescriptor { zeros: current_zeros as u16, });
            }

            // count the zeros in the current word and add them to the counter
            // the last word may contain padding zeros, which should not be counted,
            // but since we do not append the last block descriptor, this is not a problem
            current_zeros += word.count_zeros();
        }

        // pad the internal vector to be block-aligned, so SIMD operations don't try to read
        // past the end of the vector. Note that this does not affect the content of the vector,
        // because those bits are not considered part of the vector.
        while self.words.len() % (BLOCK_SIZE / WORD_SIZE) != 0 {
            self.words.push(0);
        }

        BitVector {
            data: self.words,
            len: self.len,
            blocks,
            super_blocks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Distribution;
    use rand::distributions::Uniform;
    use rand::Rng;

    #[test]
    fn test_append_bit() {
        let mut bv = BitVectorBuilder::new();
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        let bv = bv.build();

        assert_eq!(bv.data[..1], vec![0b110]);
    }

    #[test]
    fn test_append_bit_long() {
        let mut bv = BitVectorBuilder::new();

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
        let mut bv = BitVectorBuilder::new();
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
    fn test_multi_words() {
        let mut bv = BitVectorBuilder::new();
        bv.append_word(0);
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        let bv = bv.build();

        // if BLOCK_SIZE is changed, we need to update this test case
        assert_eq!(bv.data.len(), BLOCK_SIZE / WORD_SIZE);

        assert_eq!(bv.rank0(63), 63);
        assert_eq!(bv.rank0(64), 64);
        assert_eq!(bv.rank0(65), 65);
        assert_eq!(bv.rank0(66), 65);
    }

    #[test]
    fn test_super_block() {
        let mut bv = BitVectorBuilder::with_capacity(LENGTH);
        let mut rng = rand::thread_rng();
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
    fn test_only_zeros() {
        let mut bv = BitVectorBuilder::new();
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
    fn test_only_ones() {
        let mut bv = BitVectorBuilder::new();
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
}
