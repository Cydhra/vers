// Select code is in here to keep it more organized.

use crate::bit_vec::fast_rs_vec::select::sealed::SealedSelectUtils;
use crate::bit_vec::fast_rs_vec::{BLOCK_SIZE, SUPER_BLOCK_SIZE};
use crate::bit_vec::WORD_SIZE;
use crate::util::pdep::Pdep;
use crate::util::unroll;

/// A safety constant for assertions to make sure that the block size doesn't change without
/// adjusting the code.
const BLOCKS_PER_SUPERBLOCK: usize = 16;

pub(crate) mod sealed {
    use crate::bit_vec::fast_rs_vec::select::BLOCKS_PER_SUPERBLOCK;
    use crate::bit_vec::fast_rs_vec::{BLOCK_SIZE, SUPER_BLOCK_SIZE};
    use crate::util::unroll;

    pub trait SealedSelectUtils: super::super::SealedRankSelect {
        /// Search for the block in a superblock that contains the rank. This function is only used
        /// internally and is not part of the public API.
        /// The function uses SIMD instructions if available, otherwise it falls back to a naive
        /// implementation.
        ///
        /// It loads the entire block into a SIMD register and compares the rank to the number of zeros
        /// in the block. The resulting mask is popcounted to find how many blocks from the block boundary
        /// the rank is.
        // #[cfg(all(
        //     feature = "simd",
        //     target_arch = "x86_64",
        //     target_feature = "avx",
        //     target_feature = "avx2",
        //     target_feature = "avx512f",
        //     target_feature = "avx512bw",
        // ))]
        // #[inline(always)]
        // fn search_block0(&self, rank: usize, block_index: &mut usize) {
        //     use std::arch::x86_64::{_mm256_cmpgt_epu16_mask, _mm256_loadu_epi16, _mm256_set1_epi16};
        //
        //     if self.get_block_count() > *block_index + (SUPER_BLOCK_SIZE / BLOCK_SIZE) {
        //         debug_assert!(
        //             SUPER_BLOCK_SIZE / BLOCK_SIZE == BLOCKS_PER_SUPERBLOCK,
        //             "change unroll constant to {}",
        //             64 - (SUPER_BLOCK_SIZE / BLOCK_SIZE).leading_zeros() - 1
        //         );
        //
        //         unsafe {
        //             let blocks = _mm256_loadu_epi16(self.blocks[*block_index..].as_ptr() as *const i16);
        //             let ranks = _mm256_set1_epi16(rank as i16);
        //             let mask = _mm256_cmpgt_epu16_mask(blocks, ranks);
        //
        //             debug_assert!(
        //                 mask.count_zeros() > 0,
        //                 "first block should always be zero, but still claims to be greater than rank"
        //             );
        //             *block_index += mask.count_zeros() as usize - 1;
        //         }
        //     } else {
        //         self.search_block0_naive(rank, block_index)
        //     }
        // }

        /// Search for the block in a superblock that contains the rank. This function is only used
        /// internally and is not part of the public API.
        /// It compares blocks in a loop-unrolled binary search to find the block that contains the rank.
        // #[cfg(not(all(
        //     feature = "simd",
        //     target_arch = "x86_64",
        //     target_feature = "avx",
        //     target_feature = "avx2",
        //     target_feature = "avx512f",
        //     target_feature = "avx512bw",
        // )))]
        #[inline(always)]
        fn search_block0(&self, rank: usize, block_index: &mut usize) {
            self.search_block0_naive(rank, block_index)
        }

        #[inline(always)]
        fn search_block0_naive(&self, rank: usize, block_index: &mut usize) {
            // full binary search for block that contains the rank, manually loop-unrolled, because
            // LLVM doesn't do it for us, but it gains just under 20% performance

            // this code relies on the fact that BLOCKS_PER_SUPERBLOCK blocks are in one superblock
            debug_assert!(
                SUPER_BLOCK_SIZE / BLOCK_SIZE == BLOCKS_PER_SUPERBLOCK,
                "change unroll constant to {}",
                64 - (SUPER_BLOCK_SIZE / BLOCK_SIZE).leading_zeros() - 1
            );
            unroll!(4,
            |boundary = { (SUPER_BLOCK_SIZE / BLOCK_SIZE) / 2}|
                if self.get_block_count() > *block_index + boundary && rank >= self.get_block_zeros(*block_index + boundary) as usize {
                    *block_index += boundary;
                },
            boundary /= 2);
        }

        /// Search for the block in a superblock that contains the rank. This function is only used
        /// internally and is not part of the public API.
        /// The function uses SIMD instructions if available, otherwise it falls back to a naive
        /// implementation.
        ///
        /// It loads the entire block into a SIMD register and compares the rank to the number of ones
        /// in the block. The resulting mask is popcounted to find how many blocks from the block boundary
        /// the rank is.
        // #[cfg(all(
        //     feature = "simd",
        //     target_arch = "x86_64",
        //     target_feature = "avx",
        //     target_feature = "avx2",
        //     target_feature = "avx512f",
        //     target_feature = "avx512bw",
        // ))]
        // #[inline(always)]
        // fn search_block1(
        //     &self,
        //     rank: usize,
        //     block_at_super_block: usize,
        //     block_index: &mut usize,
        // ) {
        //     use std::arch::x86_64::{
        //         _mm256_cmpgt_epu16_mask, _mm256_loadu_epi16, _mm256_set1_epi16, _mm256_set_epi16,
        //         _mm256_sub_epi16,
        //     };
        //
        //     if self.get_block_count() > *block_index + BLOCKS_PER_SUPERBLOCK {
        //         debug_assert!(
        //             SUPER_BLOCK_SIZE / BLOCK_SIZE == BLOCKS_PER_SUPERBLOCK,
        //             "change unroll constant to {}",
        //             64 - (SUPER_BLOCK_SIZE / BLOCK_SIZE).leading_zeros() - 1
        //         );
        //
        //         unsafe {
        //             let bit_nums = _mm256_set_epi16(
        //                 (15 * BLOCK_SIZE) as i16,
        //                 (14 * BLOCK_SIZE) as i16,
        //                 (13 * BLOCK_SIZE) as i16,
        //                 (12 * BLOCK_SIZE) as i16,
        //                 (11 * BLOCK_SIZE) as i16,
        //                 (10 * BLOCK_SIZE) as i16,
        //                 (9 * BLOCK_SIZE) as i16,
        //                 (8 * BLOCK_SIZE) as i16,
        //                 (7 * BLOCK_SIZE) as i16,
        //                 (6 * BLOCK_SIZE) as i16,
        //                 (5 * BLOCK_SIZE) as i16,
        //                 (4 * BLOCK_SIZE) as i16,
        //                 (3 * BLOCK_SIZE) as i16,
        //                 (2 * BLOCK_SIZE) as i16,
        //                 (1 * BLOCK_SIZE) as i16,
        //                 (0 * BLOCK_SIZE) as i16,
        //             );
        //
        //             let blocks = _mm256_loadu_epi16(self.blocks[*block_index..].as_ptr() as *const i16);
        //             let ones = _mm256_sub_epi16(bit_nums, blocks);
        //
        //             let ranks = _mm256_set1_epi16(rank as i16);
        //             let mask = _mm256_cmpgt_epu16_mask(ones, ranks);
        //
        //             debug_assert!(
        //                 mask.count_zeros() > 0,
        //                 "first block should always be zero, but still claims to be greater than rank"
        //             );
        //             *block_index += mask.count_zeros() as usize - 1;
        //         }
        //     } else {
        //         self.search_block1_naive(rank, block_at_super_block, block_index)
        //     }
        // }

        /// Search for the block in a superblock that contains the rank. This function is only used
        /// internally and is not part of the public API.
        /// It compares blocks in a loop-unrolled binary search to find the block that contains the rank.
        // #[cfg(not(all(
        //     feature = "simd",
        //     target_arch = "x86_64",
        //     target_feature = "avx",
        //     target_feature = "avx2",
        //     target_feature = "avx512f",
        //     target_feature = "avx512bw",
        // )))]
        #[inline(always)]
        fn search_block1(&self, rank: usize, block_at_super_block: usize, block_index: &mut usize) {
            self.search_block1_naive(rank, block_at_super_block, block_index)
        }

        #[inline(always)]
        fn search_block1_naive(
            &self,
            rank: usize,
            block_at_super_block: usize,
            block_index: &mut usize,
        ) {
            // full binary search for block that contains the rank, manually loop-unrolled, because
            // LLVM doesn't do it for us, but it gains just under 20% performance

            // this code relies on the fact that BLOCKS_PER_SUPERBLOCK blocks are in one superblock
            debug_assert!(
                SUPER_BLOCK_SIZE / BLOCK_SIZE == BLOCKS_PER_SUPERBLOCK,
                "change unroll constant to {}",
                64 - (SUPER_BLOCK_SIZE / BLOCK_SIZE).leading_zeros() - 1
            );
            unroll!(4,
            |boundary = { (SUPER_BLOCK_SIZE / BLOCK_SIZE) / 2}|
                if self.get_block_count() > *block_index + boundary && rank >= (*block_index + boundary - block_at_super_block) * BLOCK_SIZE - self.get_block_zeros(*block_index + boundary) as usize {
                    *block_index += boundary;
                }
            , boundary /= 2);
        }
    }
}

/// Defines the [`select0`] and [`select1`] methods for rank and select bitvector structs.
/// This trait is sealed and cannot be implemented outside of this crate.
/// It exists to deduplicate code between the `RsVec` struct and its archived form, and can be used
/// to abstract over the different types.
///
/// [`select0`]: SelectSupport::select0
/// [`select1`]: SelectSupport::select1
pub trait SelectSupport: SealedSelectUtils + super::RankSupport {
    /// Return the position of the 0-bit with the given rank. See `rank0`.
    /// The following holds:
    /// ``select0(rank0(pos)) == pos``
    ///
    /// If the rank is larger than the number of 0-bits in the vector, the vector length is returned.
    #[must_use]
    #[allow(clippy::assertions_on_constants)]
    #[inline(never)]
    fn select0(&self, mut rank: usize) -> usize {
        if rank >= self.total_rank0() {
            return self.bit_len();
        }

        let mut super_block = self
            .get_select_block(rank / crate::bit_vec::fast_rs_vec::SELECT_BLOCK_SIZE)
            .0;

        if self.get_super_block_count() > (super_block + 1)
            && self.get_super_block_zeros(super_block + 1) <= rank
        {
            let mut upper_bound = self
                .get_select_block(rank / crate::bit_vec::fast_rs_vec::SELECT_BLOCK_SIZE + 1)
                .0;

            while upper_bound - super_block > 8 {
                let middle = super_block + ((upper_bound - super_block) >> 1);
                if self.get_super_block_zeros(middle) <= rank {
                    super_block = middle;
                } else {
                    upper_bound = middle;
                }
            }

            // linear search for super block that contains the rank
            while self.get_super_block_count() > (super_block + 1)
                && self.get_super_block_zeros(super_block + 1) <= rank
            {
                super_block += 1;
            }
        }

        rank -= self.get_super_block_zeros(super_block);

        let mut block_index = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
        self.search_block0(rank, &mut block_index);

        rank -= self.get_block_zeros(block_index) as usize;

        // linear search for word that contains the rank. Binary search is not possible here,
        // because we don't have accumulated popcounts for the words. We use pdep to find the
        // position of the rank-th zero bit in the word, if the word contains enough zeros, otherwise
        // we subtract the number of ones in the word from the rank and continue with the next word.
        let mut index_counter = 0;
        debug_assert!(BLOCK_SIZE / WORD_SIZE == 8, "change unroll constant");
        unroll!(7, |n = {0}| {
            let word = self.get_data_word(block_index * BLOCK_SIZE / WORD_SIZE + n);
            if (word.count_zeros() as usize) <= rank {
                rank -= word.count_zeros() as usize;
                index_counter += WORD_SIZE;
            } else {
                return block_index * BLOCK_SIZE
                    + index_counter
                    + (1 << rank).pdep(!word).trailing_zeros() as usize;
            }
        }, n += 1);

        // the last word must contain the rank-th zero bit, otherwise the rank is outside of the
        // block, and thus outside of the bitvector
        block_index * BLOCK_SIZE
            + index_counter
            + (1 << rank)
                .pdep(!self.get_data_word(block_index * BLOCK_SIZE / WORD_SIZE + 7))
                .trailing_zeros() as usize
    }

    /// Return the position of the 1-bit with the given rank. See `rank1`.
    /// The following holds:
    /// ``select1(rank1(pos)) == pos``
    ///
    /// If the rank is larger than the number of 1-bits in the bit-vector, the vector length is returned.
    #[must_use]
    #[allow(clippy::assertions_on_constants)]
    fn select1(&self, mut rank: usize) -> usize {
        if rank >= self.total_rank1() {
            return self.bit_len();
        }

        let mut super_block = self
            .get_select_block(rank / crate::bit_vec::fast_rs_vec::SELECT_BLOCK_SIZE)
            .1;

        if self.get_super_block_count() > (super_block + 1)
            && ((super_block + 1) * SUPER_BLOCK_SIZE - self.get_super_block_zeros(super_block + 1))
                <= rank
        {
            let mut upper_bound = self
                .get_select_block(rank / crate::bit_vec::fast_rs_vec::SELECT_BLOCK_SIZE + 1)
                .1;

            // binary search for super block that contains the rank
            while upper_bound - super_block > 8 {
                let middle = super_block + ((upper_bound - super_block) >> 1);
                if ((middle + 1) * SUPER_BLOCK_SIZE - self.get_super_block_zeros(middle)) <= rank {
                    super_block = middle;
                } else {
                    upper_bound = middle;
                }
            }
            // linear search for super block that contains the rank
            while self.get_super_block_count() > (super_block + 1)
                && ((super_block + 1) * SUPER_BLOCK_SIZE
                    - self.get_super_block_zeros(super_block + 1))
                    <= rank
            {
                super_block += 1;
            }
        }

        rank -= (super_block) * SUPER_BLOCK_SIZE - self.get_super_block_zeros(super_block);

        // full binary search for block that contains the rank, manually loop-unrolled, because
        // LLVM doesn't do it for us, but it gains just under 20% performance
        let block_at_super_block = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
        let mut block_index = block_at_super_block;
        self.search_block1(rank, block_at_super_block, &mut block_index);

        rank -= (block_index - block_at_super_block) * BLOCK_SIZE
            - self.get_block_zeros(block_index) as usize;

        // linear search for word that contains the rank. Binary search is not possible here,
        // because we don't have accumulated popcounts for the words. We use pdep to find the
        // position of the rank-th zero bit in the word, if the word contains enough zeros, otherwise
        // we subtract the number of ones in the word from the rank and continue with the next word.
        let mut index_counter = 0;
        debug_assert!(BLOCK_SIZE / WORD_SIZE == 8, "change unroll constant");
        unroll!(7, |n = {0}| {
            let word = self.get_data_word(block_index * BLOCK_SIZE / WORD_SIZE + n);
            if (word.count_ones() as usize) <= rank {
                rank -= word.count_ones() as usize;
                index_counter += WORD_SIZE;
            } else {
                return block_index * BLOCK_SIZE
                    + index_counter
                    + (1 << rank).pdep(word).trailing_zeros() as usize;
            }
        }, n += 1);

        // the last word must contain the rank-th zero bit, otherwise the rank is outside of the
        // block, and thus outside of the bitvector
        block_index * BLOCK_SIZE
            + index_counter
            + (1 << rank)
                .pdep(self.get_data_word(block_index * BLOCK_SIZE / WORD_SIZE + 7))
                .trailing_zeros() as usize
    }
}

impl SealedSelectUtils for super::RsVec {}

impl SelectSupport for super::RsVec {}
