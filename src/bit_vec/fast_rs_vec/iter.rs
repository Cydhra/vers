use crate::bit_vec::fast_rs_vec::{BLOCK_SIZE, SELECT_BLOCK_SIZE, SUPER_BLOCK_SIZE};
use crate::bit_vec::WORD_SIZE;
use crate::util::pdep::Pdep;
use crate::util::unroll;
use crate::RsVec;
use std::iter::FusedIterator;
use std::num::NonZeroUsize;

impl RsVec {
    /// Get an iterator over the bits in the vector. The iterator will return the indices of the
    /// 0-bits or the 1-bits in the vector, depending on the constant `ZERO`
    /// (if true, 0-bits are returned).
    ///
    /// It uses the select data structure to speed up iteration.
    /// It is also faster than calling `select` on each rank, because the iterator exploits
    /// the linear access pattern.
    ///
    /// This method has convenience methods `iter0` and `iter1`.
    pub fn select_iter<const ZERO: bool>(&self) -> SelectIter<'_, ZERO> {
        SelectIter::new(self)
    }

    /// Get an iterator over the 0-bits in the vector that uses the select data structure to speed
    /// up iteration.
    /// It is faster than calling `select0` on each rank, because the iterator
    /// exploits the linear access pattern.
    ///
    /// See [`SelectIter`] for more information.
    pub fn iter0(&self) -> SelectIter<'_, true> {
        self.select_iter()
    }

    /// Get an iterator over the 1-bits in the vector that uses the select data structure to speed
    /// up iteration.
    /// It is faster than calling `select1` on each rank, because the iterator
    /// exploits the linear access pattern.
    ///
    /// See [`SelectIter`] for more information.
    pub fn iter1(&self) -> SelectIter<'_, false> {
        self.select_iter()
    }
}

/// An iterator that iterates over 1-bits or 0-bits and returns their indices.
/// It uses the select data structures to speed up iteration.
/// It is faster than iterating over all bits if the iterated bits are sparse.
/// This is also faster than manually calling `select` on each rank,
/// because the iterator exploits the linear access pattern for faster select queries.
///
/// The iterator can be constructed by calling [`iter0`] or [`iter1`].
///
/// # Example
/// ```rust
/// use vers_vecs::{BitVec, RsVec};
///
/// let mut bit_vec = BitVec::new();
/// bit_vec.append_word(u64::MAX);
/// bit_vec.append_word(u64::MAX);
/// bit_vec.flip_bit(4);
///
/// let rs_vec = RsVec::from_bit_vec(bit_vec);
///
/// let mut iter = rs_vec.iter0();
///
/// assert_eq!(iter.next(), Some(4));
/// assert_eq!(iter.next(), None);
/// ```
///
/// [`iter0`]: crate::RsVec::iter0
/// [`iter1`]: crate::RsVec::iter1
#[derive(Clone, Debug)]
#[must_use]
pub struct SelectIter<'a, const ZERO: bool> {
    vec: &'a RsVec,
    next_rank: usize,

    /// the last index in the super block structure where we found a bit
    last_super_block: usize,

    /// the last index in the block structure where we found a bit
    last_block: usize,

    /// the last offset from the last block boundary (0-7) where we found a bit
    last_word: usize,
}

impl<'a, const ZERO: bool> SelectIter<'a, ZERO> {
    /// Create a new iterator over the given bit-vector. Initialize the caches for select queries
    fn new(vec: &'a RsVec) -> Self {
        Self {
            vec,
            next_rank: 0,
            last_super_block: 0,
            last_block: 0,
            last_word: 0,
        }
    }

    /// Same implementation like select0, but uses cached indices of last query to speed up search
    fn select_next_0(&mut self) -> Option<usize> {
        let mut rank = self.next_rank;

        if rank >= self.vec.rank0 {
            return None;
        }

        let mut super_block = self.vec.select_blocks[rank / SELECT_BLOCK_SIZE].index_0;
        let mut block_index = 0;

        if self.vec.super_blocks.len() > (self.last_super_block + 1)
            && self.vec.super_blocks[self.last_super_block + 1].zeros > rank
        {
            // instantly jump to the last searched position
            super_block = self.last_super_block;
            rank -= self.vec.super_blocks[super_block].zeros;

            // check if current block contains the one and if yes, we don't need to search
            // this is true IF the last_block is either the last block in a super block,
            // in which case it must be this block, because we know the rank is within the super block,
            // OR if the next block has a rank higher than the current rank
            if self.last_block % (SUPER_BLOCK_SIZE / BLOCK_SIZE) == 15
                || self.vec.blocks.len() > self.last_block + 1
                    && self.vec.blocks[self.last_block + 1].zeros as usize > rank
            {
                // instantly jump to the last searched position
                block_index = self.last_block;
                rank -= self.vec.blocks[block_index].zeros as usize;

                // check if the current word contains the one and if yes, we don't need to search
                let word = self.vec.data[block_index * BLOCK_SIZE / WORD_SIZE + self.last_word];
                if (word.count_zeros() as usize) > rank {
                    self.next_rank += 1;
                    return Some(
                        block_index * BLOCK_SIZE
                            + self.last_word * WORD_SIZE
                            + (1 << rank).pdep(!word).trailing_zeros() as usize,
                    );
                }

                // otherwise we continue select as normal
            }
        } else {
            let mut upper_bound = self.vec.select_blocks[rank / SELECT_BLOCK_SIZE + 1].index_0;

            while upper_bound - super_block > 8 {
                let middle = super_block + ((upper_bound - super_block) >> 1);
                if self.vec.super_blocks[middle].zeros <= rank {
                    super_block = middle;
                } else {
                    upper_bound = middle;
                }
            }

            // linear search for super block that contains the rank
            while self.vec.super_blocks.len() > (super_block + 1)
                && self.vec.super_blocks[super_block + 1].zeros <= rank
            {
                super_block += 1;
            }

            self.last_super_block = super_block;
            rank -= self.vec.super_blocks[super_block].zeros;
        }

        // if the block index is not zero, we already found the block, and need only update the word
        if block_index == 0 {
            block_index = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
            self.vec.search_block0(rank, &mut block_index);

            self.last_block = block_index;
            rank -= self.vec.blocks[block_index].zeros as usize;
        }

        // linear search for word that contains the rank. Binary search is not possible here,
        // because we don't have accumulated popcounts for the words. We use pdep to find the
        // position of the rank-th zero bit in the word, if the word contains enough zeros, otherwise
        // we subtract the number of ones in the word from the rank and continue with the next word.
        let mut index_counter = 0;
        debug_assert!(BLOCK_SIZE / WORD_SIZE == 8, "change unroll constant");
        unroll!(7, |n = {0}| {
            let word = self.vec.data[block_index * BLOCK_SIZE / WORD_SIZE + n];
            if (word.count_zeros() as usize) <= rank {
                rank -= word.count_zeros() as usize;
                index_counter += WORD_SIZE;
            } else {
                self.last_word = n;
                self.next_rank += 1;
                return Some(block_index * BLOCK_SIZE
                    + index_counter
                    + (1 << rank).pdep(!word).trailing_zeros() as usize);
            }
        }, n += 1);

        // the last word must contain the rank-th zero bit, otherwise the rank is outside the
        // block, and thus outside the bitvector
        self.last_word = 7;
        self.next_rank += 1;
        Some(
            block_index * BLOCK_SIZE
                + index_counter
                + (1 << rank)
                    .pdep(!self.vec.data[block_index * BLOCK_SIZE / WORD_SIZE + 7])
                    .trailing_zeros() as usize,
        )
    }

    #[must_use]
    #[allow(clippy::assertions_on_constants)]
    fn select_next_1(&mut self) -> Option<usize> {
        let mut rank = self.next_rank;

        if rank >= self.vec.rank1 {
            return None;
        }

        let mut super_block = self.vec.select_blocks[rank / SELECT_BLOCK_SIZE].index_1;
        let mut block_index = 0;

        // check if the last super block still contains the rank, and if yes, we don't need to search
        if self.vec.super_blocks.len() > (self.last_super_block + 1)
            && (self.last_super_block + 1) * SUPER_BLOCK_SIZE
                - self.vec.super_blocks[self.last_super_block + 1].zeros
                > rank
        {
            // instantly jump to the last searched position
            super_block = self.last_super_block;
            let block_at_super_block = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
            rank -= super_block * SUPER_BLOCK_SIZE - self.vec.super_blocks[super_block].zeros;

            // check if current block contains the one and if yes, we don't need to search
            // this is true IF the last_block is either the last block in a super block,
            // in which case it must be this block, because we know the rank is within the super block,
            // OR if the next block has a rank higher than the current rank
            if self.last_block % (SUPER_BLOCK_SIZE / BLOCK_SIZE) == 15
                || self.vec.blocks.len() > self.last_block + 1
                    && (self.last_block + 1 - block_at_super_block) * BLOCK_SIZE
                        - self.vec.blocks[self.last_block + 1].zeros as usize
                        > rank
            {
                // instantly jump to the last searched position
                block_index = self.last_block;
                let block_at_super_block = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
                rank -= (block_index - block_at_super_block) * BLOCK_SIZE
                    - self.vec.blocks[block_index].zeros as usize;

                // check if the current word contains the one and if yes, we don't need to search
                let word = self.vec.data[block_index * BLOCK_SIZE / WORD_SIZE + self.last_word];
                if (word.count_ones() as usize) > rank {
                    self.next_rank += 1;
                    return Some(
                        block_index * BLOCK_SIZE
                            + self.last_word * WORD_SIZE
                            + (1 << rank).pdep(word).trailing_zeros() as usize,
                    );
                }

                // otherwise we continue select as normal
            }
        } else {
            // search for the super block that contains the rank beginning from the super block
            // returned by the select_blocks vector
            if self.vec.super_blocks.len() > (super_block + 1)
                && ((super_block + 1) * SUPER_BLOCK_SIZE
                    - self.vec.super_blocks[super_block + 1].zeros)
                    <= rank
            {
                let mut upper_bound = self.vec.select_blocks[rank / SELECT_BLOCK_SIZE + 1].index_1;

                // binary search for super block that contains the rank until only 8 blocks are left
                while upper_bound - super_block > 8 {
                    let middle = super_block + ((upper_bound - super_block) >> 1);
                    if ((middle + 1) * SUPER_BLOCK_SIZE - self.vec.super_blocks[middle].zeros)
                        <= rank
                    {
                        super_block = middle;
                    } else {
                        upper_bound = middle;
                    }
                }
            }

            // linear search for super block that contains the rank
            while self.vec.super_blocks.len() > (super_block + 1)
                && ((super_block + 1) * SUPER_BLOCK_SIZE
                    - self.vec.super_blocks[super_block + 1].zeros)
                    <= rank
            {
                super_block += 1;
            }

            self.last_super_block = super_block;
            rank -= super_block * SUPER_BLOCK_SIZE - self.vec.super_blocks[super_block].zeros;
        }

        // if the block index is not zero, we already found the block, and need only update the word
        if block_index == 0 {
            // full binary search for block that contains the rank, manually loop-unrolled, because
            // LLVM doesn't do it for us, but it gains just under 20% performance
            let block_at_super_block = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
            block_index = block_at_super_block;
            self.vec
                .search_block1(rank, block_at_super_block, &mut block_index);

            self.last_block = block_index;
            rank -= (block_index - block_at_super_block) * BLOCK_SIZE
                - self.vec.blocks[block_index].zeros as usize;
        }

        // linear search for word that contains the rank. Binary search is not possible here,
        // because we don't have accumulated popcounts for the words. We use pdep to find the
        // position of the rank-th zero bit in the word, if the word contains enough zeros, otherwise
        // we subtract the number of ones in the word from the rank and continue with the next word.
        let mut index_counter = 0;
        debug_assert!(BLOCK_SIZE / WORD_SIZE == 8, "change unroll constant");
        unroll!(7, |n = {0}| {
            let word = self.vec.data[block_index * BLOCK_SIZE / WORD_SIZE + n];
            if (word.count_ones() as usize) <= rank {
                rank -= word.count_ones() as usize;
                index_counter += WORD_SIZE;
            } else {
                self.last_word = n;
                self.next_rank += 1;
                return Some(block_index * BLOCK_SIZE
                    + index_counter
                    + (1 << rank).pdep(word).trailing_zeros() as usize);
            }
        }, n += 1);

        // the last word must contain the rank-th zero bit, otherwise the rank is outside the
        // block, and thus outside the bitvector
        self.last_word = 7;
        self.next_rank += 1;
        Some(
            block_index * BLOCK_SIZE
                + index_counter
                + (1 << rank)
                    .pdep(self.vec.data[block_index * BLOCK_SIZE / WORD_SIZE + 7])
                    .trailing_zeros() as usize,
        )
    }

    /// Advances the iterator by `n` elements. Returns an error if the iterator does not have
    /// enough elements left. Does not call `next` internally.
    /// This method is currently being added to the iterator trait, see
    /// [this issue](https://github.com/rust-lang/rust/issues/77404).
    /// As soon as it is stabilized, this method will be removed and replaced with a custom
    /// implementation in the iterator impl.
    pub(super) fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        if self.len() >= n {
            self.next_rank += n;
            Ok(())
        } else {
            let len = self.len();
            self.next_rank += len;
            Err(NonZeroUsize::new(n - len).unwrap())
        }
    }
}

impl<'a, const ZERO: bool> Iterator for SelectIter<'a, ZERO> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if ZERO {
            self.select_next_0()
        } else {
            self.select_next_1()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }

    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        if self.len() == 0 {
            None
        } else {
            self.advance_by(self.len() - 1)
                .ok()
                .and_then(|()| self.next())
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if ZERO {
            self.advance_by(n).ok().and_then(|()| self.select_next_0())
        } else {
            self.advance_by(n).ok().and_then(|()| self.select_next_1())
        }
    }
}

impl<'a, const ZERO: bool> FusedIterator for SelectIter<'a, ZERO> {}

impl<'a, const ZERO: bool> ExactSizeIterator for SelectIter<'a, ZERO> {
    fn len(&self) -> usize {
        if ZERO {
            self.vec.rank0 - self.next_rank
        } else {
            self.vec.rank1 - self.next_rank
        }
    }
}
