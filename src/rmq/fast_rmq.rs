use std::cmp::min_by;

use crate::rmq::small_naive::SmallNaiveRmq;
use crate::{FastBitVector, RsVector, RsVectorBuilder};

/// Size of the blocks the data is split into. One block is indexable with a u8, hence its size.
const BLOCK_SIZE: usize = 256;

/// A block has a bit vector indicating the minimum element in the prefix (suffix) of the
/// block up to each bit's index. This way a simple select(rank(k)) query can be used to find the
/// minimum element in the block prefix (suffix) of length k.
/// The space requirement for this structure is (sub-)linear in the block size.
pub struct Block {
    prefix_minima: FastBitVector,
    suffix_minima: FastBitVector,
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
            let mut prefix_minima = RsVectorBuilder::<FastBitVector>::with_capacity(block.len());
            let mut suffix_minima = RsVectorBuilder::<FastBitVector>::with_capacity(block.len());

            let mut prefix_minimum = block[0];
            let mut block_minimum = block[0];
            let mut block_minimum_index = 0u8;

            prefix_minima.append_bit(1u8);

            for i in 1..block.len() {
                if block[i] < prefix_minimum {
                    prefix_minimum = block[i];
                    prefix_minima.append_bit(1u8);
                } else {
                    prefix_minima.append_bit(0u8);
                }

                if block[i] < block_minimum {
                    block_minimum = block[i];
                    block_minimum_index = i as u8;
                }
            }

            let mut suffix_minimum = block[block.len() - 1];
            suffix_minima.append_bit(1u8);

            for i in 2..=block.len() {
                if block[block.len() - i] < suffix_minimum {
                    suffix_minimum = block[block.len() - i];
                    suffix_minima.append_bit(1u8);
                } else {
                    suffix_minima.append_bit(0u8);
                }
            }

            block_minima.push(block_minimum);
            block_min_indices.push(block_minimum_index);
            blocks.push(Block {
                prefix_minima: prefix_minima.build(),
                suffix_minima: suffix_minima.build(),
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
            return i + self.data[i..=j]
                .iter()
                .enumerate()
                .min_by_key(|(_, &x)| x)
                .unwrap()
                .0;
        }

        let partial_block_i_min = (block_i + 1) * BLOCK_SIZE
            - self.blocks[block_i].suffix_minima.select1(
                self.blocks[block_i]
                    .suffix_minima
                    .rank1(BLOCK_SIZE - (i % BLOCK_SIZE))
                    - 1,
            )
            - 1;

        let partial_block_j_min = block_j * BLOCK_SIZE
            + self.blocks[block_j]
                .prefix_minima
                .select1(self.blocks[block_j].prefix_minima.rank1(j % BLOCK_SIZE + 1) - 1);

        // if there are full blocks between the two partial blocks, we can use the block minima
        // to find the minimum in the range [block_i + 1, block_j - 1]
        if block_i + 1 < block_j {
            let intermediate_min_block = self.block_minima.range_min(block_i + 1, block_j - 1);
            let min_block_index = intermediate_min_block * BLOCK_SIZE
                + self.block_min_indices[intermediate_min_block] as usize;

            min_by(
                min_by(partial_block_i_min, partial_block_j_min, |&a, &b| {
                    self.data[a as usize].cmp(&self.data[b as usize])
                }),
                min_block_index,
                |&a, &b| self.data[a as usize].cmp(&self.data[b as usize]),
            )
        } else {
            min_by(partial_block_i_min, partial_block_j_min, |&a, &b| {
                self.data[a as usize].cmp(&self.data[b as usize])
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_rmq() {
        const L: usize = 500;

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
}