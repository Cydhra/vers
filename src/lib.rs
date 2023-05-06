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

    /// Appends a word to the end of the vector. The word is interpreted as a sequence of
    /// bits, with the least significant bit being the first one (i.e. the word is
    /// interpreted as little-endian). The vector will be extended by the number of bits
    /// in the word.
    ///
    /// It is a logic error if the word has a partial word at its end before this operation
    /// (i.e. the length of the vector must be a multiple of the word size).
    pub fn append_word(&mut self, word: u64) {
        debug_assert!(self.len % WORD_SIZE == 0);

        self.begin_new_word();
        *self.data.last_mut().unwrap() = word;
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

        if block_index > 0 {
            rank += if zero {
                self.blocks[block_index - 1].zeros
            } else {
                (block_index * BLOCK_SIZE) - self.blocks[block_index - 1].zeros
            };
        }

        for i in
            ((super_block_index * SUPER_BLOCK_SIZE) + (block_index * BLOCK_SIZE)) / WORD_SIZE..index
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

        rank
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
            data: vec![0, 0b110],
            len: 67,
            blocks: vec![BlockDescriptor { zeros: 0 }],
            super_blocks: vec![BlockDescriptor { zeros: 0 }],
        };

        assert_eq!(bv.rank0(64), 64);
        assert_eq!(bv.rank0(65), 65);
        assert_eq!(bv.rank0(66), 65);
    }
}
