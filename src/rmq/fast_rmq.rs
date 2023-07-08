use std::arch::x86_64::_pdep_u64;
use std::cmp::min_by;
use std::mem::size_of;

use crate::rmq::small_naive::SmallNaiveRmq;

/// Size of the blocks the data is split into. One block is indexable with a u8, hence its size.
const BLOCK_SIZE: usize = 128;

/// A constant size small bitvector that supports rank0 and select0 specifically for the RMQ
/// structure
struct SmallBitVector(u128);

impl SmallBitVector {
    /// Calculates the rank0 of the bitvector up to the i-th bit by masking out the bits after i
    /// and counting the ones of the bitwise-inverted bitvector.
    fn rank0(&self, i: usize) -> usize {
        debug_assert!(i <= 128);
        let mut mask = ![(-1i128 << (i & 127)), 0][(i == 128) as usize] as u128;
        (!self.0 & mask).count_ones() as usize
    }

    fn select0(&self, rank: usize) -> usize {
        unsafe { self.select0_impl(rank) }
    }

    #[target_feature(enable = "bmi2")]
    unsafe fn select0_impl(&self, mut rank: usize) -> usize {
        let word = (self.0 & 0xFFFFFFFFFFFFFFFF) as u64;
        if (word.count_zeros() as usize) <= rank {
            rank -= word.count_zeros() as usize;
        } else {
            return _pdep_u64(1 << rank, !word).trailing_zeros() as usize;
        }
        let word = (self.0 >> 64) as u64;
        return 64 + _pdep_u64(1 << (rank % 64), !word).trailing_zeros() as usize;
    }

    fn set_bit(&mut self, i: usize) {
        debug_assert!(i <= 128);
        let mask = 1u128 << i;
        self.0 |= mask;
    }
}

impl Default for SmallBitVector {
    fn default() -> Self {
        Self(0)
    }
}

/// A block has a bit vector indicating the minimum element in the prefix (suffix) of the
/// block up to each bit's index. This way a simple select(rank(k)) query can be used to find the
/// minimum element in the block prefix (suffix) of length k.
/// The space requirement for this structure is (sub-)linear in the block size.
struct Block {
    prefix_minima: SmallBitVector,
    suffix_minima: SmallBitVector,
}

/// A data structure for fast range minimum queries with linear space overhead. Practically, the
/// space overhead is O(n log n), because the block size is constant, however this increases speed
/// and will only be a problem for very large data sets.
pub struct FastRmq {
    data: Vec<u64>,
    block_minima: SmallNaiveRmq,
    block_min_indices: Vec<u8>,
    blocks: Vec<Block>,
}

impl FastRmq {
    pub fn new(data: Vec<u64>) -> Self {
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
            block_minima: SmallNaiveRmq::new(block_minima),
            block_min_indices,
            blocks,
        }
    }

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

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>()
            + self.block_minima.heap_size()
            + self.block_min_indices.len()
            + self.blocks.len() * size_of::<Block>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngCore, SeedableRng};
    use rand::rngs::StdRng;

    #[test]
    fn test_small_bit_vector_rank0() {
        let mut sbv = SmallBitVector::default();
        sbv.set_bit(1);
        sbv.set_bit(3);
        sbv.set_bit(64);
        sbv.set_bit(65);

        assert_eq!(sbv.rank0(0), 0);
        assert_eq!(sbv.rank0(1), 1);
        assert_eq!(sbv.rank0(2), 1);
        assert_eq!(sbv.rank0(3), 2);
        assert_eq!(sbv.rank0(4), 2);

        assert_eq!(sbv.rank0(64), 62);
        assert_eq!(sbv.rank0(65), 62);
        assert_eq!(sbv.rank0(66), 62);
        assert_eq!(sbv.rank0(67), 63);
    }

    #[test]
    fn test_small_bit_vector_select0() {
        let mut sbv = SmallBitVector::default();
        sbv.set_bit(1);
        sbv.set_bit(3);
        sbv.set_bit(64);
        sbv.set_bit(65);

        assert_eq!(sbv.select0(0), 0);
        assert_eq!(sbv.select0(1), 2);
        assert_eq!(sbv.select0(2), 4);
        assert_eq!(sbv.select0(3), 5);
        assert_eq!(sbv.select0(64), 68);
    }

    #[test]
    fn test_fast_rmq() {
        const L: usize = 2 * BLOCK_SIZE;

        let mut numbers_vec = Vec::with_capacity(L);
        for i in 0..L {
            numbers_vec.push(i as u64);
        }

        let rmq = FastRmq::new(numbers_vec.clone());

        for i in 0..L {
            for j in i..L {
                let min = i + numbers_vec[i..=j]
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &x)| x)
                    .unwrap()
                    .0;
                assert_eq!(rmq.range_min(i, j), min, "i = {}, j = {}", i, j);
            }
        }
    }

    #[test]
    fn test_fast_rmq_unsorted() {
        let mut rng = rand::thread_rng();
        const L: usize = 2 * BLOCK_SIZE;

        let mut numbers_vec = Vec::with_capacity(L);
        for _ in 0..L {
            numbers_vec.push(rng.next_u64());
        }

        let rmq = FastRmq::new(numbers_vec.clone());

        for i in 0..L {
            for j in i..L {
                let min = numbers_vec[i..=j].iter().min().unwrap();
                assert_eq!(
                    numbers_vec[rmq.range_min(i, j)],
                    *min,
                    "i = {}, j = {}",
                    i,
                    j
                );
            }
        }
    }
}
