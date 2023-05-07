#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use std::ops::Rem;

const WORD_SIZE: usize = 64;
const BLOCK_SIZE: usize = 256;
const SUPER_BLOCK_SIZE: usize = 4096;

#[derive(Clone, Copy, Debug)]
struct BlockDescriptor {
    zeros: usize,
}

#[derive(Clone, Debug)]
pub struct BitVector {
    data: Vec<u64>,
    len: usize,
    blocks: Vec<BlockDescriptor>,
    super_blocks: Vec<BlockDescriptor>,
}

impl BitVector {
    /// Create a new empty bitvector.
    pub fn new() -> BitVector {
        BitVector {
            data: Vec::new(),
            len: 0,
            blocks: vec![BlockDescriptor { zeros: 0 }],
            super_blocks: vec![BlockDescriptor { zeros: 0 }],
        }
    }

    /// Appends a new 0 to the end of `data`. If the last block is full, a new block is
    /// created. If the last super block is full, a new super block is created.
    fn append_new_block(&mut self) {
        // append a full new block
        for _ in 0..BLOCK_SIZE / WORD_SIZE {
            self.data.push(0);
        }

        if self.data.len() > (self.super_blocks.len() * SUPER_BLOCK_SIZE) / WORD_SIZE {
            let new_super_block = BlockDescriptor {
                zeros: self.super_blocks[self.super_blocks.len() - 1].zeros,
            };
            self.super_blocks.push(new_super_block);

            self.blocks.push(BlockDescriptor { zeros: 0 });
        } else {
            let new_block = BlockDescriptor {
                zeros: self.blocks[self.blocks.len() - 1].zeros,
            };
            self.blocks.push(new_block);
        }
    }

    /// Appends a bit to the end of the vector. Accepts any numerical type that implements
    /// the `Rem` trait, and will only use the least significant bit (i.e. will calculate
    /// `bit % 2`). Any other bit of the input will be ignored.
    pub fn append_bit<T: Rem + From<u8>>(&mut self, bit: T)
    where
        T::Output: Into<u64>,
    {
        let bit = (bit % T::from(2u8)).into();

        if self.len % BLOCK_SIZE == 0 {
            self.append_new_block();
        }

        let pos = self.len % WORD_SIZE;
        self.data[self.len / WORD_SIZE] |= bit << pos;
        self.len += 1;

        if bit == 0 {
            self.blocks.last_mut().unwrap().zeros += 1;
            self.super_blocks.last_mut().unwrap().zeros += 1;
        }
    }

    /// Appends a word to the end of the vector. The word is interpreted as a sequence of
    /// bits, with the least significant bit being the first one (i.e. the word is
    /// interpreted as little-endian). The vector will be extended by the number of bits
    /// in the word.
    ///
    /// It is a logic error if the word has a partial word at its end before this operation
    /// (i.e. the length of the vector must be a multiple of the word size).
    pub fn append_word(&mut self, word: u64) {
        debug_assert!(self.len % WORD_SIZE == 0);

        if self.len % BLOCK_SIZE == 0 {
            self.append_new_block();
        }

        self.data[(self.len + 1) / WORD_SIZE] = word;
        self.len += WORD_SIZE;

        self.blocks.last_mut().unwrap().zeros += WORD_SIZE - word.count_ones() as usize;
        self.super_blocks.last_mut().unwrap().zeros += WORD_SIZE - word.count_ones() as usize;
    }

    /// Return the 0-rank of the bit at the given position. The 0-rank is the number of
    /// 0-bits in the vector up to but excluding the bit at the given position.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    pub fn rank0(&self, pos: usize) -> usize {
        self.rank(true, pos)
    }

    /// Return the 1-rank of the bit at the given position. The 1-rank is the number of
    /// 1-bits in the vector up to but excluding the bit at the given position.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    pub fn rank1(&self, pos: usize) -> usize {
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

        if super_block_index > 0 {
            rank += if zero {
                self.super_blocks[super_block_index - 1].zeros
            } else {
                (super_block_index * SUPER_BLOCK_SIZE)
                    - self.super_blocks[super_block_index - 1].zeros
            };
        }

        if block_index % (SUPER_BLOCK_SIZE / BLOCK_SIZE) > 0 {
            rank += if zero {
                self.blocks[block_index - 1].zeros
            } else {
                (block_index * BLOCK_SIZE) - self.blocks[block_index - 1].zeros
            };
        }

        #[cfg(any(not(feature = "simd"), not(target_arch = "x86_64")))]
        {
            // naive popcount of blocks

            for i in ((super_block_index * SUPER_BLOCK_SIZE) + (block_index * BLOCK_SIZE))
                / WORD_SIZE..index
            {
                rank += if zero {
                    self.data[i].count_zeros() as usize
                } else {
                    self.data[i].count_ones() as usize
                };
            }

            rank += if zero {
                (!self.data[index] & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as usize
            } else {
                (self.data[index] & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as usize
            };
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            // Wojciech Muła algorithm for SIMD popcount on SSSE3.
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
                    .checked_shl((pos.saturating_sub(WORD_SIZE) % BLOCK_SIZE) as u32)
                    .unwrap_or(0);
                let mask3 = !u64::MAX
                    .checked_shl((pos.saturating_sub(WORD_SIZE * 2) % BLOCK_SIZE) as u32)
                    .unwrap_or(0);
                let mask4 = !u64::MAX
                    .checked_shl((pos.saturating_sub(WORD_SIZE * 3) % BLOCK_SIZE) as u32)
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

    #[inline(always)]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    unsafe fn hsum(v: __m256i) -> u64 {
        let vlow = _mm256_castsi256_si128(v);
        let vhigh = _mm256_extracti128_si256::<1>(v);
        let vlow = _mm_add_epi64(vlow, vhigh);
        let conv = _mm_cvtsi128_si64(_mm_add_epi64(vlow, _mm_unpackhi_epi64(vlow, vlow))) as u64;
        (conv & 0x0F)
            + (conv >> 8 & 0x0F)
            + (conv >> 16 & 0x0F)
            + (conv >> 24 & 0x0F)
            + (conv >> 32 & 0x0F)
            + (conv >> 40 & 0x0F)
            + (conv >> 48 & 0x0F)
            + (conv >> 56 & 0x0F)
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

impl Default for BitVector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_bit() {
        let mut bv = BitVector::new();

        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        assert_eq!(bv.data[..1], vec![0b110]);
    }

    #[test]
    fn test_append_bit_long() {
        let mut bv = BitVector::new();

        let len = SUPER_BLOCK_SIZE + 1;
        for _ in 0..len {
            bv.append_bit(0u8);
            bv.append_bit(1u8);
        }

        assert_eq!(bv.len(), len * 2);
        assert_eq!(bv.rank0(2 * len - 1), len);
        assert_eq!(bv.rank1(2 * len - 1), len - 1);
    }

    #[test]
    fn test_rank() {
        let mut bv = BitVector::new();

        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
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
        let bv = BitVector {
            data: vec![0, 0b110, 0, 0],
            len: 67,
            blocks: vec![BlockDescriptor { zeros: 0 }],
            super_blocks: vec![BlockDescriptor { zeros: 0 }],
        };

        // if BLOCK_SIZE is changed, we need to update this test case
        assert_eq!(bv.data.len(), BLOCK_SIZE / WORD_SIZE);

        assert_eq!(bv.rank0(63), 63);
        assert_eq!(bv.rank0(64), 64);
        assert_eq!(bv.rank0(65), 65);
        assert_eq!(bv.rank0(66), 65);
    }
}
