//! A fast succinct bit vector implementation with rank and select queries. Rank computes in
//! constant-time, select on average in constant-time, with a logarithmic worst case.

use std::mem::size_of;

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512bw",
))]
pub use bitset::*;
pub use iter::*;

use crate::util::impl_vector_iterator;
use crate::BitVec;

use super::WORD_SIZE;

/// Size of a block in the bitvector.
const BLOCK_SIZE: u64 = 512;

/// Size of a super block in the bitvector. Super-blocks exist to decrease the memory overhead
/// of block descriptors.
/// Increasing or decreasing the super block size has negligible effect on performance of rank
/// instruction. This means we want to make the super block size as large as possible, as long as
/// the zero-counter in normal blocks still fits in a reasonable amount of bits. However, this has
/// impact on the performance of select queries. The larger the super block size, the deeper will
/// a binary search be. We found 2^13 to be a good compromise between memory overhead and
/// performance.
const SUPER_BLOCK_SIZE: u64 = 1 << 13;

/// Size of a select block. The select block is used to speed up select queries. The select block
/// contains the indices of every `SELECT_BLOCK_SIZE`'th 1-bit and 0-bit in the bitvector.
/// The smaller this block-size, the faster are select queries, but the more memory is used.
const SELECT_BLOCK_SIZE: u64 = 1 << 13;

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
    zeros: u64,
}

/// Meta-data for the select query. Each entry i in the select vector contains the indices to find
/// the i * `SELECT_BLOCK_SIZE`'th 0- and 1-bit in the bitvector. Those indices may be very far apart.
/// The indices do not point into the bit-vector, but into the super-block vector.
#[derive(Clone, Debug)]
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
    len: u64,
    blocks: Vec<BlockDescriptor>,
    super_blocks: Vec<SuperBlockDescriptor>,
    select_blocks: Vec<SelectSuperBlockDescriptor>,
    pub(crate) rank0: u64,
    pub(crate) rank1: u64,
}

impl RsVec {
    /// Build an `RsVec` from a [`BitVec`]. This will consume the `BitVec`. Since `RsVec`s are
    /// immutable, this is the only way to construct an `RsVec`.
    ///
    /// # Example
    /// See the example for `RsVec`.
    ///
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn from_bit_vec(vec: BitVec) -> RsVec {
        // Construct the block descriptor meta data. Each block descriptor contains the number of
        // zeros in the super-block, up to but excluding the block.
        let mut blocks = Vec::with_capacity((vec.len() / BLOCK_SIZE) as usize + 1);
        let mut super_blocks = Vec::with_capacity((vec.len() / SUPER_BLOCK_SIZE) as usize + 1);
        let mut select_blocks = Vec::new();

        // sentinel value
        select_blocks.push(SelectSuperBlockDescriptor {
            index_0: 0,
            index_1: 0,
        });

        let mut total_zeros: u64 = 0;
        let mut current_zeros: u64 = 0;
        let mut last_zero_select_block: usize = 0;
        let mut last_one_select_block: usize = 0;

        for (word_idx, &word) in vec.data.iter().enumerate() {
            // if we moved past a block boundary, append the block information for the previous
            // block and reset the counter if we moved past a super-block boundary.
            if (word_idx as u64).is_multiple_of(BLOCK_SIZE / WORD_SIZE) {
                if (word_idx as u64).is_multiple_of(SUPER_BLOCK_SIZE / WORD_SIZE) {
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
            let mut new_zeros = word.count_zeros() as u64;

            // in the last block, remove remaining zeros of limb that aren't part of the vector
            if word_idx == vec.data.len() - 1 && !vec.len.is_multiple_of(WORD_SIZE) {
                let mask = (1 << (vec.len % WORD_SIZE)) - 1;
                new_zeros -= (word | mask).count_zeros() as u64;
            }

            let all_zeros = total_zeros + current_zeros + new_zeros;
            if all_zeros / SELECT_BLOCK_SIZE > (total_zeros + current_zeros) / SELECT_BLOCK_SIZE {
                if (all_zeros / SELECT_BLOCK_SIZE) as usize == select_blocks.len() {
                    select_blocks.push(SelectSuperBlockDescriptor {
                        index_0: super_blocks.len() - 1,
                        index_1: 0,
                    });
                } else {
                    select_blocks[(all_zeros / SELECT_BLOCK_SIZE) as usize].index_0 =
                        super_blocks.len() - 1;
                }

                last_zero_select_block += 1;
            }

            let total_bits = (word_idx as u64 + 1) * WORD_SIZE;
            let all_ones = total_bits - all_zeros;
            if all_ones / SELECT_BLOCK_SIZE
                > (word_idx as u64 * WORD_SIZE - total_zeros - current_zeros) / SELECT_BLOCK_SIZE
            {
                if (all_ones / SELECT_BLOCK_SIZE) as usize == select_blocks.len() {
                    select_blocks.push(SelectSuperBlockDescriptor {
                        index_0: 0,
                        index_1: super_blocks.len() - 1,
                    });
                } else {
                    select_blocks[(all_ones / SELECT_BLOCK_SIZE) as usize].index_1 =
                        super_blocks.len() - 1;
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

        total_zeros += current_zeros;

        RsVec {
            data: vec.data,
            len: vec.len,
            blocks,
            super_blocks,
            select_blocks,
            rank0: total_zeros,
            rank1: vec.len - total_zeros,
        }
    }

    /// Return the 0-rank of the bit at the given position. The 0-rank is the number of
    /// 0-bits in the vector up to but excluding the bit at the given position. Calling this
    /// function with an index larger than the length of the bit-vector will report the total
    /// number of 0-bits in the bit-vector.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    #[must_use]
    pub fn rank0(&self, pos: u64) -> u64 {
        self.rank(true, pos)
    }

    /// Return the 1-rank of the bit at the given position. The 1-rank is the number of
    /// 1-bits in the vector up to but excluding the bit at the given position. Calling this
    /// function with an index larger than the length of the bit-vector will report the total
    /// number of 1-bits in the bit-vector.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    #[must_use]
    pub fn rank1(&self, pos: u64) -> u64 {
        self.rank(false, pos)
    }

    // I measured 5-10% improvement with this. I don't know why it's not inlined by default, the
    // branch elimination profits alone should make it worth it.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn rank(&self, zero: bool, pos: u64) -> u64 {
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

        let index = (pos / WORD_SIZE) as usize;
        let block_index = (pos / BLOCK_SIZE) as usize;
        let super_block_index = (pos / SUPER_BLOCK_SIZE) as usize;
        let mut rank = 0;

        // at first add the number of zeros/ones before the current super block
        rank += if zero {
            self.super_blocks[super_block_index].zeros
        } else {
            (super_block_index as u64 * SUPER_BLOCK_SIZE)
                - self.super_blocks[super_block_index].zeros
        };

        // then add the number of zeros/ones before the current block
        rank += if zero {
            self.blocks[block_index].zeros as u64
        } else {
            ((block_index as u64 % (SUPER_BLOCK_SIZE / BLOCK_SIZE)) * BLOCK_SIZE)
                - self.blocks[block_index].zeros as u64
        };

        // naive popcount of blocks
        for &i in &self.data[((block_index as u64 * BLOCK_SIZE) / WORD_SIZE) as usize..index] {
            rank += if zero {
                i.count_zeros() as u64
            } else {
                i.count_ones() as u64
            };
        }

        rank += if zero {
            (!self.data[index] & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as u64
        } else {
            (self.data[index] & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as u64
        };

        rank
    }

    /// Return the length of the vector, i.e. the number of bits it contains.
    #[must_use]
    pub fn len(&self) -> u64 {
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
    pub fn get(&self, pos: u64) -> Option<u64> {
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
    pub fn get_unchecked(&self, pos: u64) -> u64 {
        (self.data[(pos / WORD_SIZE) as usize] >> (pos % WORD_SIZE)) & 1
    }

    /// Return multiple bits at the given position. The number of bits to return is given by `len`.
    /// At most 64 bits can be returned.
    /// If the position at the end of the query is larger than the length of the vector,
    /// None is returned (even if the query partially overlaps with the vector).
    /// If the length of the query is larger than 64, None is returned.
    #[must_use]
    pub fn get_bits(&self, pos: u64, len: u64) -> Option<u64> {
        if len > WORD_SIZE {
            return None;
        }
        if pos + len > self.len {
            None
        } else {
            Some(self.get_bits_unchecked(pos, len))
        }
    }

    /// Return multiple bits at the given position. The number of bits to return is given by `len`.
    /// At most 64 bits can be returned.
    ///
    /// This function is always inlined, because it gains a lot from loop optimization and
    /// can utilize the processor pre-fetcher better if it is.
    ///
    /// # Errors
    /// If the length of the query is larger than 64, unpredictable data will be returned.
    /// Use [`get_bits`] to properly handle this case with an `Option`.
    ///
    /// # Panics
    /// If the position or interval is larger than the length of the vector,
    /// the function will either return unpredictable data, or panic.
    ///
    /// [`get_bits`]: #method.get_bits
    #[must_use]
    #[allow(clippy::comparison_chain)] // readability
    #[allow(clippy::cast_possible_truncation)] // parameter must be out of scope for this to happen
    pub fn get_bits_unchecked(&self, pos: u64, len: u64) -> u64 {
        debug_assert!(len <= WORD_SIZE);
        let partial_word = self.data[(pos / WORD_SIZE) as usize] >> (pos % WORD_SIZE);
        if pos % WORD_SIZE + len <= WORD_SIZE {
            partial_word & 1u64.checked_shl(len as u32).unwrap_or(0).wrapping_sub(1)
        } else {
            (partial_word
                | (self.data[(pos / WORD_SIZE + 1) as usize] << (WORD_SIZE - pos % WORD_SIZE)))
                & 1u64.checked_shl(len as u32).unwrap_or(0).wrapping_sub(1)
        }
    }

    /// Convert the `RsVec` into a [`BitVec`].
    /// This consumes the `RsVec`, and discards all meta-data.
    /// Since [`RsVec`]s are innately immutable, this conversion is the only way to modify the
    /// underlying data.
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
    ///
    /// let mut bit_vec = rs_vec.into_bit_vec();
    /// bit_vec.flip_bit(32);
    /// let rs_vec = RsVec::from_bit_vec(bit_vec);
    /// assert_eq!(rs_vec.rank1(64), 63);
    /// assert_eq!(rs_vec.select0(0), 32);
    /// ```
    #[must_use]
    pub fn into_bit_vec(self) -> BitVec {
        BitVec {
            data: self.data,
            len: self.len,
        }
    }

    /// Check if two `RsVec`s are equal. For sparse vectors (either sparsely filled with 1-bits or
    /// 0-bits), this is faster than comparing the vectors bit by bit.
    /// Choose the value of `ZERO` depending on which bits are more sparse.
    ///
    /// This method is faster than [`full_equals`] for sparse vectors beginning at roughly 1
    /// million bits. Above 4 million bits, this method becomes faster than full equality in general.
    ///
    /// # Parameters
    /// - `other`: The other `RsVec` to compare to.
    /// - `ZERO`: Whether to compare the sparse 0-bits (true) or the sparse 1-bits (false).
    ///
    /// # Returns
    /// `true` if the vectors' contents are equal, `false` otherwise.
    ///
    /// [`full_equals`]: RsVec::full_equals
    #[must_use]
    pub fn sparse_equals<const ZERO: bool>(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        if self.rank0 != other.rank0 || self.rank1 != other.rank1 {
            return false;
        }

        let iter: SelectIter<ZERO> = self.select_iter();

        let len = if ZERO { self.rank0 } else { self.rank1 };

        // we need to manually enumerate() the iter, because the number of set bits could exceed
        // the size of usize.
        for (rank, bit_index) in (0..len).zip(iter) {
            // since rank is inlined, we get dead code elimination depending on ZERO
            if (other.get_unchecked(bit_index) == 0) != ZERO || other.rank(ZERO, bit_index) != rank
            {
                return false;
            }
        }

        true
    }

    /// Check if two `RsVec`s are equal. This compares limb by limb. This is usually faster than a
    /// [`sparse_equals`] call for small vectors.
    ///
    /// # Parameters
    /// - `other`: The other `RsVec` to compare to.
    ///
    /// # Returns
    /// `true` if the vectors' contents are equal, `false` otherwise.
    ///
    /// [`sparse_equals`]: RsVec::sparse_equals
    #[must_use]
    pub fn full_equals(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        if self.rank0 != other.rank0 || self.rank1 != other.rank1 {
            return false;
        }

        if self.data[..(self.len / WORD_SIZE) as usize]
            .iter()
            .zip(other.data[..(other.len / 64) as usize].iter())
            .any(|(a, b)| a != b)
        {
            return false;
        }

        // if last incomplete block exists, test it without junk data
        if !self.len.is_multiple_of(WORD_SIZE)
            && self.data[(self.len / WORD_SIZE) as usize] & ((1 << (self.len % WORD_SIZE)) - 1)
                != other.data[(self.len / WORD_SIZE) as usize]
                    & ((1 << (other.len % WORD_SIZE)) - 1)
        {
            return false;
        }

        true
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

impl_vector_iterator! { RsVec, RsVecIter, RsVecRefIter }

impl PartialEq for RsVec {
    /// Check if two `RsVec`s are equal. This method calls [`sparse_equals`] if the vector has more
    /// than 4'000'000 bits, and [`full_equals`] otherwise.
    ///
    /// This was determined with benchmarks on an `x86_64` machine,
    /// on which [`sparse_equals`] outperforms [`full_equals`] consistently above this threshold.
    ///
    /// # Parameters
    /// - `other`: The other `RsVec` to compare to.
    ///
    /// # Returns
    /// `true` if the vectors' contents are equal, `false` otherwise.
    ///
    /// [`sparse_equals`]: RsVec::sparse_equals
    /// [`full_equals`]: RsVec::full_equals
    fn eq(&self, other: &Self) -> bool {
        if self.len > 4_000_000 {
            if self.rank1 > self.rank0 {
                self.sparse_equals::<true>(other)
            } else {
                self.sparse_equals::<false>(other)
            }
        } else {
            self.full_equals(other)
        }
    }
}

impl From<BitVec> for RsVec {
    /// Build an [`RsVec`] from a [`BitVec`]. This will consume the [`BitVec`]. Since [`RsVec`]s are
    /// immutable, this is the only way to construct an [`RsVec`].
    ///
    /// # Example
    /// See the example for [`RsVec`].
    ///
    /// [`BitVec`]: BitVec
    /// [`RsVec`]: RsVec
    fn from(vec: BitVec) -> Self {
        RsVec::from_bit_vec(vec)
    }
}

impl From<RsVec> for BitVec {
    fn from(value: RsVec) -> Self {
        value.into_bit_vec()
    }
}

// iter code in here to keep it more organized
mod iter;
// select code in here to keep it more organized
mod select;

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512bw",
))]
mod bitset;

#[cfg(test)]
mod tests;
