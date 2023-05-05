use std::ops::Rem;

const WORD_SIZE: usize = 64;
const BLOCK_SIZE: usize = 512;
const SUPER_BLOCK_SIZE: usize = 4096;
const PAGE_SIZE: usize = u32::MAX as usize;

struct BlockDescriptor {
    zeros: usize,
}

pub struct BitVector {
    data: Vec<u64>,
    len: usize,
    blocks: Vec<BlockDescriptor>,
    super_blocks: Vec<BlockDescriptor>,
    // TODO: pages
}

impl BitVector {
    /// Appends a new 0 to the end of `data`. If the last block is full, a new block is
    /// created. If the last super block is full, a new super block is created.
    fn begin_new_word(&mut self) {
        self.data.push(0);

        if self.data.len() > (self.blocks.len() * BLOCK_SIZE) / WORD_SIZE {
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
    }

    /// Appends a bit to the end of the vector. Accepts any numerical type that implements
    /// the `Rem` trait, and will only use the least significant bit (i.e. will calculate
    /// `bit % 2`). Any other bit of the input will be ignored.
    pub fn append_bit<T: Rem + From<u8>>(&mut self, bit: T)
    where
        T::Output: Into<u64>,
    {
        let bit = (bit % T::from(2u8)).into();

        if self.len % WORD_SIZE == 0 {
            self.begin_new_word();
        }

        let pos = self.len % WORD_SIZE;
        self.data[self.len / WORD_SIZE] |= bit << pos;
        self.len += 1;

        if bit == 0 {
            self.blocks.last_mut().unwrap().zeros += 1;
            self.super_blocks.last_mut().unwrap().zeros += 1;
        }
    }

    pub fn rank(&self, zero: bool, pos: usize) -> usize {
        let index = pos / WORD_SIZE;
        let block_index = pos / BLOCK_SIZE;
        let super_block_index = pos / SUPER_BLOCK_SIZE;
        let mut rank = 0;

        if super_block_index > 0 {
            if zero {
                rank += self.super_blocks[super_block_index - 1].zeros;
            } else {
                rank += (super_block_index * SUPER_BLOCK_SIZE)
                    - self.super_blocks[super_block_index - 1].zeros;
            }
        }

        if block_index > 0 {
            if zero {
                rank += self.blocks[block_index - 1].zeros;
            } else {
                rank += (block_index * BLOCK_SIZE) - self.blocks[block_index - 1].zeros;
            }
        }

        for i in
            ((super_block_index * SUPER_BLOCK_SIZE) + (block_index * BLOCK_SIZE)) / WORD_SIZE..index
        {
            if zero {
                rank += self.data[i].count_zeros() as usize;
            } else {
                rank += self.data[i].count_ones() as usize;
            }
        }

        if zero {
            rank += (!self.data[index] & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as usize;
        } else {
            rank += (self.data[index] & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as usize;
        }

        rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_bit() {
        let mut bv = BitVector {
            data: Vec::new(),
            len: 0,
            blocks: vec![BlockDescriptor { zeros: 0 }],
            super_blocks: vec![BlockDescriptor { zeros: 0 }],
        };

        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        assert_eq!(bv.data, vec![0b110]);
    }

    #[test]
    fn test_rank() {
        let mut bv = BitVector {
            data: Vec::new(),
            len: 0,
            blocks: vec![BlockDescriptor { zeros: 0 }],
            super_blocks: vec![BlockDescriptor { zeros: 0 }],
        };

        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        // first bit must always have rank 0
        assert_eq!(bv.rank(true, 0), 0);
        assert_eq!(bv.rank(true, 0), 0);

        assert_eq!(bv.rank(false, 2), 1);
        assert_eq!(bv.rank(false, 3), 2);
        assert_eq!(bv.rank(false, 4), 2);
        assert_eq!(bv.rank(true, 3), 1);
    }

    #[test]
    fn test_multi_words() {
        let bv = BitVector {
            data: vec![0, 0b110],
            len: 67,
            blocks: vec![BlockDescriptor { zeros: 0 }],
            super_blocks: vec![BlockDescriptor { zeros: 0 }],
        };

        assert_eq!(bv.rank(true, 64), 64);
        assert_eq!(bv.rank(true, 65), 65);
        assert_eq!(bv.rank(true, 66), 65);
    }
}
