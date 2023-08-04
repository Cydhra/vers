//! A fast succinct bit vector implementation with rank and select queries. Rank computes in
//! constant time, select on average in constant time, with a logarithmic worst case.

use super::WORD_SIZE;
use crate::util::unroll;
use crate::BitVec;
use core::arch::x86_64::_pdep_u64;
use std::mem::size_of;

/// Size of a block in the bitvector.
const BLOCK_SIZE: usize = 512;

/// Size of a super block in the bitvector. Super-blocks exist to decrease the memory overhead
/// of block descriptors.
/// Increasing or decreasing the super block size has negligible effect on performance of rank
/// instruction. This means we want to make the super block size as large as possible, as long as
/// the zero-counter in normal blocks still fits in a reasonable amount of bits. However, this has
/// impact on the performance of select queries. The larger the super block size, the deeper will
/// a binary search be. We found 2^13 to be a good compromise between memory overhead and
/// performance.
const SUPER_BLOCK_SIZE: usize = 1 << 13;

/// Size of a select block. The select block is used to speed up select queries. The select block
/// contains the indices of every `SELECT_BLOCK_SIZE`'th 1-bit and 0-bit in the bitvector.
/// The smaller this block-size, the faster are select queries, but the more memory is used.
const SELECT_BLOCK_SIZE: usize = 1 << 13;

/// Meta-data for a block. The `zeros` field stores the number of zeros up to the block,
/// beginning from the last super-block boundary. This means the first block in a super-block
/// always stores the number zero, which serves as a sentinel value to avoid special-casing the
/// first block in a super-block (which would be a performance hit due branch prediction failures).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct BlockDescriptor {
    zeros: u16,
}

/// Meta-data for a super-block. The `zeros` field stores the number of zeros up to this super-block.
/// This allows the `BlockDescriptor` to store the number of zeros in a much smaller
/// space. The `zeros` field is the number of zeros up to the super-block.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct SuperBlockDescriptor {
    zeros: usize,
}

/// Meta-data for the select query. Each entry i in the select vector contains the indices to find
/// the i * `SELECT_BLOCK_SIZE`'th 0- and 1-bit in the bitvector. Those indices may be very far apart.
/// The indices do not point into the bit-vector, but into the select-block vector.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct SelectSuperBlockDescriptor {
    index_0: usize,
    index_1: usize,
}

/// A bitvector that supports constant-time rank and select queries and is optimized for fast queries.
/// The bitvector is stored as a vector of `u64`s. The bit-vector stores meta-data for constant-time
/// rank and select queries, which takes sub-linear additional space. The space overhead is
/// 28 bits per 512 bits of user data (~5.47%).
///
/// # Example
/// ```rust
/// use vers_vecs::{BitVec, RsVec};
///
/// let mut bit_vec = BitVec::new();
/// bit_vec.append_word(u64::MAX);
///
/// let rs_vec = RsVec::from_bit_vec(bit_vec);
/// assert_eq!(rs_vec.rank1(64), 64);
/// assert_eq!(rs_vec.select1(64), 64);
///```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RsVec {
    data: Vec<u64>,
    len: usize,
    blocks: Vec<BlockDescriptor>,
    super_blocks: Vec<SuperBlockDescriptor>,
    select_blocks: Vec<SelectSuperBlockDescriptor>,
    rank0: usize,
    rank1: usize,
}

impl RsVec {
    /// Build an [`RsVec`] from a [`BitVec`]. This will consume the [`BitVec`]. Since [`RsVec`]s are
    /// immutable, this is the only way to construct an [`RsVec`].
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
        let mut last_zero_select_block: usize = 0;
        let mut last_one_select_block: usize = 0;

        for (idx, &word) in vec.data.iter().enumerate() {
            // if we moved past a block boundary, append the block information for the previous
            // block and reset the counter if we moved past a super-block boundary.
            if idx % (BLOCK_SIZE / WORD_SIZE) == 0 {
                if idx % (SUPER_BLOCK_SIZE / WORD_SIZE) == 0 {
                    total_zeros += current_zeros;
                    current_zeros = 0;
                    super_blocks.push(SuperBlockDescriptor { zeros: total_zeros });
                }

                // this cannot overflow because a super block isn't 2^16 bits long
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

                last_zero_select_block += 1;
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

                last_one_select_block += 1;
            }

            current_zeros += new_zeros;
        }

        // insert dummy select blocks at the end that just report the same index like the last real
        // block, so the bound check for binary search doesn't overflow
        // this is technically the incorrect value, but since all valid queries will be smaller,
        // this will only tell select to stay in the current super block, which is correct.
        // we cannot use a real value here, because this would change the size of the super-block
        if last_zero_select_block == select_blocks.len() - 1 {
            select_blocks.push(SelectSuperBlockDescriptor {
                index_0: select_blocks[last_zero_select_block].index_0,
                index_1: 0,
            });
        } else {
            debug_assert!(select_blocks[last_zero_select_block + 1].index_0 == 0);
            select_blocks[last_zero_select_block + 1].index_0 =
                select_blocks[last_zero_select_block].index_0;
        }
        if last_one_select_block == select_blocks.len() - 1 {
            select_blocks.push(SelectSuperBlockDescriptor {
                index_0: 0,
                index_1: select_blocks[last_one_select_block].index_1,
            });
        } else {
            debug_assert!(select_blocks[last_one_select_block + 1].index_1 == 0);
            select_blocks[last_one_select_block + 1].index_1 =
                select_blocks[last_one_select_block].index_1;
        }

        // pad the internal vector to be block-aligned, so SIMD operations don't try to read
        // past the end of the vector. Note that this does not affect the content of the vector,
        // because those bits are not considered part of the vector.
        // Note further, that currently no SIMD implementation exists.
        while vec.data.len() % (BLOCK_SIZE / WORD_SIZE) != 0 {
            vec.data.push(0);
        }

        RsVec {
            data: vec.data,
            len: vec.len,
            blocks,
            super_blocks,
            select_blocks,
            // the last block may contain padding zeros, which should not be counted
            rank0: total_zeros + current_zeros - ((WORD_SIZE - (vec.len % WORD_SIZE)) % WORD_SIZE),
            rank1: vec.len
                - (total_zeros + current_zeros - ((WORD_SIZE - (vec.len % WORD_SIZE)) % WORD_SIZE)),
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

    #[target_feature(enable = "bmi2")]
    #[allow(clippy::assertions_on_constants)]
    unsafe fn bmi_select0(&self, mut rank: usize) -> usize {
        if rank >= self.rank0 {
            return self.len;
        }

        let mut super_block = self.select_blocks[rank / SELECT_BLOCK_SIZE].index_0;

        if self.super_blocks.len() > (super_block + 1)
            && self.super_blocks[super_block + 1].zeros <= rank
        {
            let mut upper_bound = self.select_blocks[rank / SELECT_BLOCK_SIZE + 1].index_0;

            while upper_bound - super_block > 8 {
                let middle = super_block + ((upper_bound - super_block) >> 1);
                if self.super_blocks[middle].zeros <= rank {
                    super_block = middle;
                } else {
                    upper_bound = middle;
                }
            }

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
        debug_assert!(
            SUPER_BLOCK_SIZE / BLOCK_SIZE == 16,
            "change unroll constant to {}",
            64 - (SUPER_BLOCK_SIZE / BLOCK_SIZE).leading_zeros() - 1
        );
        unroll!(4,
            |boundary = { (SUPER_BLOCK_SIZE / BLOCK_SIZE) / 2}|
                if self.blocks.len() > block_index + boundary && rank >= self.blocks[block_index + boundary].zeros as usize {
                    block_index += boundary;
                },
            boundary /= 2);

        rank -= self.blocks[block_index].zeros as usize;

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

    #[target_feature(enable = "bmi2")]
    #[allow(clippy::assertions_on_constants)]
    unsafe fn bmi_select1(&self, mut rank: usize) -> usize {
        if rank >= self.rank1 {
            return self.len;
        }

        let mut super_block = self.select_blocks[rank / SELECT_BLOCK_SIZE].index_1;

        if self.super_blocks.len() > (super_block + 1)
            && ((super_block + 1) * SUPER_BLOCK_SIZE - self.super_blocks[super_block + 1].zeros)
            <= rank
        {
            let mut upper_bound = self.select_blocks[rank / SELECT_BLOCK_SIZE + 1].index_1;

            // binary search for super block that contains the rank
            while upper_bound - super_block > 8 {
                let middle = super_block + ((upper_bound - super_block) >> 1);
                if ((middle + 1) * SUPER_BLOCK_SIZE - self.super_blocks[middle].zeros) <= rank {
                    super_block = middle;
                } else {
                    upper_bound = middle;
                }
            }
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
        debug_assert!(SUPER_BLOCK_SIZE / BLOCK_SIZE == 16, "change unroll constant to {}", 64 - (SUPER_BLOCK_SIZE / BLOCK_SIZE).leading_zeros() - 1);
        unroll!(4,
            |boundary = { (SUPER_BLOCK_SIZE / BLOCK_SIZE) / 2}|
                if self.blocks.len() > block_index + boundary && rank >= (block_index + boundary - block_at_super_block) * BLOCK_SIZE - self.blocks[block_index + boundary].zeros as usize {
                    block_index += boundary;
                }
            , boundary /= 2);

        rank -= (block_index - block_at_super_block) * BLOCK_SIZE
            - self.blocks[block_index].zeros as usize;

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

    // I measured 5-10% improvement with this. I don't know why it's not inlined by default, the
    // branch elimination profits alone should make it worth it.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn rank(&self, zero: bool, pos: usize) -> usize {
        #[allow(clippy::collapsible_else_if)]
        // readability and more obvious where dead branch elimination happens
        if zero {
            if pos >= self.len() {
                return self.rank0;
            }
        } else {
            if pos >= self.len() {
                return self.rank1;
            }
        }

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

    /// Return the 0-rank of the bit at the given position. The 0-rank is the number of
    /// 0-bits in the vector up to but excluding the bit at the given position. Calling this
    /// function with an index larger than the length of the bit-vector will report the total
    /// number of 0-bits in the bit-vector.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    ///
    /// # Compatibility
    /// This function forcibly enables the `popcnt` x86 CPU feature.
    #[must_use]
    pub fn rank0(&self, pos: usize) -> usize {
        unsafe { self.naive_rank0(pos) }
    }

    /// Return the 1-rank of the bit at the given position. The 1-rank is the number of
    /// 1-bits in the vector up to but excluding the bit at the given position. Calling this
    /// function with an index larger than the length of the bit-vector will report the total
    /// number of 1-bits in the bit-vector.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    ///
    /// # Compatibility
    /// This function forcibly enables the `popcnt` x86 CPU feature.
    #[must_use]
    pub fn rank1(&self, pos: usize) -> usize {
        unsafe { self.naive_rank1(pos) }
    }

    /// Return the position of the 0-bit with the given rank. See `rank0`.
    /// The following holds:
    /// ``select0(rank0(pos)) == pos``
    ///
    /// If the rank is larger than the number of 0-bits in the vector, the vector length is returned.
    ///
    /// # Compatibility
    /// This function forcibly enables the `bmi2` x86 CPU feature. If this feature is not available
    /// on the CPU, this function will not work.
    #[must_use]
    pub fn select0(&self, rank: usize) -> usize {
        unsafe { self.bmi_select0(rank) }
    }

    /// Return the position of the 1-bit with the given rank. See `rank1`.
    /// The following holds:
    /// ``select1(rank1(pos)) == pos``
    ///
    /// If the rank is larger than the number of 1-bits in the bit-vector, the vector length is returned.
    ///
    /// # Compatibility
    /// This function forcibly enables the `bmi2` x86 CPU feature. If this feature is not available
    /// on the CPU, this function will not work.
    #[must_use]
    pub fn select1(&self, rank: usize) -> usize {
        unsafe { self.bmi_select1(rank) }
    }

    /// Return the length of the vector, i.e. the number of bits it contains.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the bit at the given position. The bit takes the least significant
    /// bit of the returned u64 word.
    /// If the position is larger than the length of the vector, `None` is returned.
    #[must_use]
    pub fn get(&self, pos: usize) -> Option<u64> {
        if pos >= self.len() {
            None
        } else {
            Some(self.get_unchecked(pos))
        }
    }

    /// Return the bit at the given position. The bit takes the least significant
    /// bit of the returned u64 word.
    ///
    /// # Panics
    /// This function may panic if `pos >= self.len()` (alternatively, it may return garbage).
    #[must_use]
    pub fn get_unchecked(&self, pos: usize) -> u64 {
        (self.data[pos / WORD_SIZE] >> (pos % WORD_SIZE)) & 1
    }

    /// Returns the number of bytes used on the heap for this vector. This does not include
    /// allocated space that is not used (e.g. by the allocation behavior of `Vec`).
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>()
            + self.blocks.len() * size_of::<BlockDescriptor>()
            + self.super_blocks.len() * size_of::<SuperBlockDescriptor>()
            + self.select_blocks.len() * size_of::<SelectSuperBlockDescriptor>()
    }
}

#[cfg(test)]
mod tests;
