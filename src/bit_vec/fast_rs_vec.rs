use super::{BuildingStrategy, RsVector, WORD_SIZE};
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
/// 32 bytes per 512 bytes of user data (6.25%), plus 36 bytes constant overhead.
#[derive(Clone, Debug)]
pub struct FastBitVector {
    data: Vec<u64>,
    len: usize,
    blocks: Vec<BlockDescriptor>,
    super_blocks: Vec<SuperBlockDescriptor>,
    select_blocks: Vec<SelectSuperBlockDescriptor>,
}

impl FastBitVector {
    // todo non-popcount implementation as opt-in feature
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

        // linear search for super block that contains the rank
        while self.super_blocks.len() > (super_block + 1)
            && self.super_blocks[super_block + 1].zeros <= rank
        {
            super_block += 1;
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

        // linear search for super block that contains the rank
        while self.super_blocks.len() > (super_block + 1)
            && ((super_block + 1) * SUPER_BLOCK_SIZE - self.super_blocks[super_block + 1].zeros)
                <= rank
        {
            super_block += 1;
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
}

impl RsVector for FastBitVector {
    fn rank0(&self, pos: usize) -> usize {
        unsafe { self.naive_rank0(pos) }
    }

    fn rank1(&self, pos: usize) -> usize {
        unsafe { self.naive_rank1(pos) }
    }

    fn select0(&self, rank: usize) -> usize {
        unsafe { self.bmi_select0(rank) }
    }

    fn select1(&self, rank: usize) -> usize {
        unsafe { self.bmi_select1(rank) }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, pos: usize) -> u64 {
        (self.data[pos / WORD_SIZE] >> (pos % WORD_SIZE)) & 1
    }

    fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>()
            + self.blocks.len() * size_of::<BlockDescriptor>()
            + self.super_blocks.len() * size_of::<SuperBlockDescriptor>()
            + self.select_blocks.len() * size_of::<SelectSuperBlockDescriptor>()
    }
}

impl BuildingStrategy for FastBitVector {
    type Vector = FastBitVector;

    fn from_bit_vec(mut vec: BitVec) -> FastBitVector {
        // Construct the block descriptor meta data. Each block descriptor contains the number of
        // zeros in the super-block, up to but excluding the block.
        let mut blocks = Vec::with_capacity(vec.len / BLOCK_SIZE + 1);
        let mut super_blocks = Vec::with_capacity(vec.len / SUPER_BLOCK_SIZE + 1);
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

        FastBitVector {
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
        let mut bv = RsVectorBuilder::<FastBitVector>::new();
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        let bv = bv.build();

        assert_eq!(bv.data[..1], vec![0b110]);
    }

    #[test]
    fn test_random_data_rank() {
        let mut bv = RsVectorBuilder::<FastBitVector>::with_capacity(LENGTH);
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
        let bv = RsVectorBuilder::<FastBitVector>::new();
        crate::bit_vec::common_tests::test_append_bit_long(bv, SUPER_BLOCK_SIZE);
    }

    #[test]
    fn test_rank() {
        let bv = RsVectorBuilder::<FastBitVector>::new();
        crate::bit_vec::common_tests::test_rank(bv);
    }

    #[test]
    fn test_multi_words_rank() {
        let bv = RsVectorBuilder::<FastBitVector>::new();
        crate::bit_vec::common_tests::test_multi_words_rank(bv);
    }

    #[test]
    fn test_only_zeros_rank() {
        let bv = RsVectorBuilder::<FastBitVector>::new();
        crate::bit_vec::common_tests::test_only_zeros_rank(bv, SUPER_BLOCK_SIZE, WORD_SIZE);
    }

    #[test]
    fn test_only_ones_rank() {
        let bv = RsVectorBuilder::<FastBitVector>::new();
        crate::bit_vec::common_tests::test_only_ones_rank(bv, SUPER_BLOCK_SIZE, WORD_SIZE);
    }

    #[test]
    fn test_simple_select() {
        let bv = RsVectorBuilder::<FastBitVector>::new();
        crate::bit_vec::common_tests::test_simple_select(bv);
    }

    #[test]
    fn test_multi_words_select() {
        let bv = RsVectorBuilder::<FastBitVector>::new();
        crate::bit_vec::common_tests::test_multi_words_select(bv);
    }

    #[test]
    fn test_only_zeros_select() {
        let bv = RsVectorBuilder::<FastBitVector>::new();
        crate::bit_vec::common_tests::test_only_zeros_select(bv, SUPER_BLOCK_SIZE, WORD_SIZE);
    }

    #[test]
    fn test_only_ones_select() {
        let bv = RsVectorBuilder::<FastBitVector>::new();
        crate::bit_vec::common_tests::test_only_ones_select(bv, SUPER_BLOCK_SIZE, WORD_SIZE);
    }

    #[test]
    fn random_data_select() {
        let mut bv = RsVectorBuilder::<FastBitVector>::with_capacity(LENGTH);
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