//! A fast and quasi-succinct range minimum query data structure.
//! It is based on a linear-space RMQ data structure
//! but uses constant-sized structures in place of logarithmic ones,
//! which makes it faster at the cost of increasing the space bound to O(n log n).

use std::cmp::min_by;
use std::mem::size_of;
use std::ops::{Bound, Deref, RangeBounds};

use crate::rmq::binary_rmq::BinaryRmq;
use crate::util::pdep::Pdep;

/// Size of the blocks the data is split into. One block is indexable with a u8, hence its size.
const BLOCK_SIZE: usize = 128;

/// A constant size small bitvector that supports rank0 and select0 specifically for the RMQ
/// structure
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct SmallBitVector(u128);

impl SmallBitVector {
    /// Calculates the rank0 of the bitvector up to the i-th bit by masking out the bits after i
    /// and counting the ones of the bitwise-inverted bitvector.
    #[allow(clippy::cast_possible_truncation)] // parameter must be out of scope for this to happen
    fn rank0(&self, i: usize) -> usize {
        debug_assert!(i <= 128);
        let mask = 1u128.checked_shl(i as u32).unwrap_or(0).wrapping_sub(1);
        (!self.0 & mask).count_ones() as usize
    }

    fn select0(&self, mut rank: usize) -> usize {
        let word = (self.0 & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        if (word.count_zeros() as usize) <= rank {
            rank -= word.count_zeros() as usize;
        } else {
            return (1 << rank).pdep(!word).trailing_zeros() as usize;
        }
        let word = (self.0 >> 64) as u64;
        64 + (1 << (rank % 64)).pdep(!word).trailing_zeros() as usize
    }

    fn set_bit(&mut self, i: usize) {
        debug_assert!(i <= 128);
        let mask = 1u128 << i;
        self.0 |= mask;
    }
}

/// A block has a bit vector indicating the minimum element in the prefix (suffix) of the
/// block up to each bit's index. This way a simple select(rank(k)) query can be used to find the
/// minimum element in the block prefix (suffix) of length k.
/// The space requirement for this structure is (sub-)linear in the block size.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Block {
    prefix_minima: SmallBitVector,
    suffix_minima: SmallBitVector,
}

/// A data structure for fast range minimum queries based on a structure with theoretically linear space overhead.
/// In practice, the space overhead is O(n log n), because of real-machine considerations.
/// However, this increases speed and will only be a problem for incredibly large data sets.
/// The data structure can handle up to 2^40 elements, after which some queries may cause
/// panics.
///
/// # Example
/// ```rust
/// use vers_vecs::FastRmq;
///
/// let data = vec![4, 10, 3, 11, 2, 12];
/// let rmq = FastRmq::from_vec(data);
///
/// assert_eq!(rmq.range_min(0, 1), 0);
/// assert_eq!(rmq.range_min(0, 2), 2);
/// assert_eq!(rmq.range_min(0, 3), 2);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FastRmq {
    data: Vec<u64>,
    block_minima: BinaryRmq,
    block_min_indices: Vec<u8>,
    blocks: Vec<Block>,
}

impl FastRmq {
    /// Creates a new range minimum query data structure from the given data. Creation time is
    /// O(n log n) and space overhead is O(n log n) with a fractional constant factor
    /// (see [`FastRmq`])
    ///
    /// # Panics
    /// This function will panic if the input is larger than 2^40 elements.
    #[must_use]
    pub fn from_vec(data: Vec<u64>) -> Self {
        assert!(data.len() < 1 << 40, "input too large for fast rmq");

        let mut block_minima = Vec::with_capacity(data.len() / BLOCK_SIZE + 1);
        let mut block_min_indices = Vec::with_capacity(data.len() / BLOCK_SIZE + 1);
        let mut blocks = Vec::with_capacity(data.len() / BLOCK_SIZE + 1);

        data.chunks(BLOCK_SIZE).for_each(|block| {
            let mut prefix_minima = SmallBitVector::default();
            let mut suffix_minima = SmallBitVector::default();

            let mut prefix_minimum = block[0];
            let mut block_minimum = block[0];
            let mut block_minimum_index = 0u8;

            for (i, elem) in block.iter().enumerate().skip(1) {
                if *elem < prefix_minimum {
                    prefix_minimum = *elem;
                } else {
                    prefix_minima.set_bit(i);
                }

                // This is safe because the block size is constant and smaller than 256
                #[allow(clippy::cast_possible_truncation)]
                if *elem < block_minimum {
                    block_minimum = *elem;
                    block_minimum_index = i as u8;
                }
            }

            let mut suffix_minimum = block[block.len() - 1];

            for i in 2..=block.len() {
                if block[block.len() - i] < suffix_minimum {
                    suffix_minimum = block[block.len() - i];
                } else {
                    suffix_minima.set_bit(i - 1);
                }
            }

            block_minima.push(block_minimum);
            block_min_indices.push(block_minimum_index);
            blocks.push(Block {
                prefix_minima,
                suffix_minima,
            });
        });

        Self {
            data,
            block_minima: BinaryRmq::from_vec(block_minima),
            block_min_indices,
            blocks,
        }
    }

    /// Convenience function for [`FastRmq::range_min`] for using range operators.
    /// The range is clamped to the length of the data structure, sso this function will not panic,
    /// unless called on an empty data structure, because that does not have a valid index.
    ///
    /// # Example
    /// ```rust
    /// use vers_vecs::FastRmq;
    /// let rmq = FastRmq::from_vec(vec![5, 4, 3, 2, 1]);
    /// assert_eq!(rmq.range_min_with_range(0..3), 2);
    /// assert_eq!(rmq.range_min_with_range(0..=3), 3);
    /// ```
    ///
    /// # Panics
    /// This function will panic if the data structure is empty.
    #[must_use]
    pub fn range_min_with_range<T: RangeBounds<usize>>(&self, range: T) -> usize {
        let start = match range.start_bound() {
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i + 1,
            Bound::Unbounded => 0,
        }
        .clamp(0, self.len() - 1);

        let end = match range.end_bound() {
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i - 1,
            Bound::Unbounded => self.len() - 1,
        }
        .clamp(0, self.len() - 1);
        self.range_min(start, end)
    }

    /// Returns the index of the minimum element in the range [i, j] in O(1) time.
    /// Runtime may still vary for different ranges,
    /// but is independent of the size of the data structure and bounded by a constant for all
    /// possible ranges. The range is inclusive.
    ///
    /// # Panics
    /// Calling this function with i > j will produce either a panic or an incorrect result.
    /// Calling this function where one of the indices is out of bounds will produce a panic or an
    /// incorrect result.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn range_min(&self, i: usize, j: usize) -> usize {
        let block_i = i / BLOCK_SIZE;
        let block_j = j / BLOCK_SIZE;

        // if the range is contained in a single block, we just search it
        if block_i == block_j {
            let rank_i_prefix = self.blocks[block_i].prefix_minima.rank0(i % BLOCK_SIZE + 1);
            let rank_j_prefix = self.blocks[block_i].prefix_minima.rank0(j % BLOCK_SIZE + 1);

            if rank_j_prefix > rank_i_prefix {
                return block_i * BLOCK_SIZE
                    + self.blocks[block_i]
                        .prefix_minima
                        .select0(rank_j_prefix - 1);
            }

            let rank_i_suffix = self.blocks[block_i]
                .suffix_minima
                .rank0(BLOCK_SIZE - (i % BLOCK_SIZE));
            let rank_j_suffix = self.blocks[block_i]
                .suffix_minima
                .rank0(BLOCK_SIZE - (j % BLOCK_SIZE));

            if rank_j_suffix > rank_i_suffix {
                return (block_i + 1) * BLOCK_SIZE
                    - self.blocks[block_i]
                        .suffix_minima
                        .select0(rank_j_suffix - 1);
            }

            return i + self.data[i..=j]
                .iter()
                .enumerate()
                .min_by_key(|(_, &x)| x)
                .unwrap()
                .0;
        }

        let partial_block_i_min = (block_i + 1) * BLOCK_SIZE
            - self.blocks[block_i].suffix_minima.select0(
                self.blocks[block_i]
                    .suffix_minima
                    .rank0(BLOCK_SIZE - (i % BLOCK_SIZE))
                    - 1,
            )
            - 1;

        let partial_block_j_min = block_j * BLOCK_SIZE
            + self.blocks[block_j]
                .prefix_minima
                .select0(self.blocks[block_j].prefix_minima.rank0(j % BLOCK_SIZE + 1) - 1);

        // if there are full blocks between the two partial blocks, we can use the block minima
        // to find the minimum in the range [block_i + 1, block_j - 1]
        if block_i + 1 < block_j {
            let intermediate_min_block = self.block_minima.range_min(block_i + 1, block_j - 1);
            let min_block_index = intermediate_min_block * BLOCK_SIZE
                + self.block_min_indices[intermediate_min_block] as usize;

            min_by(
                min_by(partial_block_i_min, partial_block_j_min, |&a, &b| {
                    self.data[a].cmp(&self.data[b])
                }),
                min_block_index,
                |&a, &b| self.data[a].cmp(&self.data[b]),
            )
        } else {
            min_by(partial_block_i_min, partial_block_j_min, |&a, &b| {
                self.data[a].cmp(&self.data[b])
            })
        }
    }

    /// Returns the length of the RMQ data structure (i.e. the number of elements)
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the RMQ data structure is empty (i.e. contains no elements)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the amount of memory used by the RMQ data structure in bytes. Does not include
    /// space allocated but not in use (e.g. unused capacity of vectors).
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>()
            + self.block_minima.heap_size()
            + self.block_min_indices.len()
            + self.blocks.len() * size_of::<Block>()
    }
}

/// Implements Deref to delegate to the underlying data structure. This allows the user to use
/// indexing syntax on the RMQ data structure to access the underlying data, as well as iterators,
/// etc.
impl Deref for FastRmq {
    type Target = Vec<u64>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl From<Vec<u64>> for FastRmq {
    fn from(data: Vec<u64>) -> Self {
        Self::from_vec(data)
    }
}

/// Creates a new range minimum query data structure from the given data.
/// The iterator is consumed and the data is stored in a vector.
///
/// See [`FastRmq::from_vec`] for more information.
///
/// [`FastRmq::from_vec`]: FastRmq::from_vec
impl FromIterator<u64> for FastRmq {
    fn from_iter<T: IntoIterator<Item = u64>>(iter: T) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests;
