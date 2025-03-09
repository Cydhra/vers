use crate::bit_vec::fast_rs_vec::{BLOCK_SIZE, SELECT_BLOCK_SIZE, SUPER_BLOCK_SIZE};
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

    /// Convert vector into an iterator over the bits in the vector. The iterator will return the indices of the
    /// 0-bits or the 1-bits in the vector, depending on the constant `ZERO`
    /// (if true, 0-bits are returned).
    ///
    /// It uses the select data structure to speed up iteration.
    /// It is also faster than calling `select` on each rank, because the iterator exploits
    /// the linear access pattern.
    ///
    /// This method has convenience methods `into_iter0` and `into_iter1`.
    pub fn into_select_iter<const ZERO: bool>(self) -> SelectIntoIter<ZERO> {
        SelectIntoIter::new(self)
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

    /// Convert vector into an iterator over the 0-bits in the vector
    /// that uses the select data structure to speed up iteration.
    /// It is faster than calling `select0` on each rank, because the iterator
    /// exploits the linear access pattern.
    ///
    /// See [`SelectIntoIter`] for more information.
    pub fn into_iter0(self) -> SelectIntoIter<true> {
        self.into_select_iter()
    }

    /// Convert vector into an iterator over the 1-bits in the vector
    /// that uses the select data structure to speed up iteration.
    /// It is faster than calling `select1` on each rank, because the iterator
    /// exploits the linear access pattern.
    ///
    /// See [`SelectIntoIter`] for more information.
    pub fn into_iter1(self) -> SelectIntoIter<false> {
        self.into_select_iter()
    }
}

macro_rules! gen_iter_impl {
    ($($life:lifetime, )? $name:ident) => {
        impl<$($life,)? const ZERO: bool> $name<$($life,)? ZERO> {

            /// Create a new iterator over the given bit-vector. Initialize the caches for select queries
            fn new(vec: $(&$life)? RsVec) -> Self {
                if vec.is_empty() {
                    return Self {
                        vec,
                        next_rank: 0,
                        next_rank_back: None,
                        last_super_block: 0,
                        last_super_block_back: 0,
                        last_block: 0,
                        last_block_back: 0,
                    };
                }

                let blocks_len = vec.blocks.len();
                let super_blocks_len = vec.super_blocks.len();
                let rank0 = vec.rank0;
                let rank1 = vec.total_rank1();

                Self {
                    vec,
                    next_rank: 0,
                    next_rank_back: Some(if ZERO { rank0 } else { rank1 }).and_then(|x| if x > 0 { Some(x - 1) } else { None }),
                    last_super_block: 0,
                    last_super_block_back: super_blocks_len - 1,
                    last_block: 0,
                    last_block_back: blocks_len - 1,
                }
            }

            /// Same implementation like select0, but uses cached indices of last query to speed up search
            fn select_next_0(&mut self) -> Option<usize> {
                let mut rank = self.next_rank;

                if rank >= self.vec.rank0 || self.next_rank_back.is_none() || rank > self.next_rank_back.unwrap() {
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
                    }
                } else {
                    super_block = self.vec.search_super_block0(super_block, rank);
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

                self.next_rank += 1;
                Some(self.vec.search_word_in_block0(rank, block_index))
            }

            /// Same implementation like ``select_next_0``, but backwards
            fn select_next_0_back(&mut self) -> Option<usize> {
                let mut rank = self.next_rank_back?;

                if self.next_rank_back.is_none() || rank < self.next_rank {
                    return None;
                }

                let mut super_block = self.vec.select_blocks[rank / SELECT_BLOCK_SIZE].index_0;
                let mut block_index = 0;

                if self.vec.super_blocks[self.last_super_block_back].zeros < rank
                {
                    // instantly jump to the last searched position
                    super_block = self.last_super_block_back;
                    rank -= self.vec.super_blocks[super_block].zeros;

                    // check if current block contains the one and if yes, we don't need to search
                    // this is true IF the zeros before the last block are less than the rank,
                    // since the block before then can't contain it
                    if self.vec.blocks[self.last_block_back].zeros as usize <= rank
                    {
                        // instantly jump to the last searched position
                        block_index = self.last_block_back;
                        rank -= self.vec.blocks[block_index].zeros as usize;
                    }
                } else {
                    super_block = self.vec.search_super_block0(super_block, rank);
                    self.last_super_block_back = super_block;
                    rank -= self.vec.super_blocks[super_block].zeros;
                }

                // if the block index is not zero, we already found the block, and need only update the word
                if block_index == 0 {
                    block_index = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
                    self.vec.search_block0(rank, &mut block_index);

                    self.last_block_back = block_index;
                    rank -= self.vec.blocks[block_index].zeros as usize;
                }

                self.next_rank_back = self.next_rank_back.and_then(|x| if x > 0 { Some(x - 1) } else { None });
                Some(self.vec.search_word_in_block0(rank, block_index))
            }

            #[must_use]
            #[allow(clippy::assertions_on_constants)]
            fn select_next_1(&mut self) -> Option<usize> {
                let mut rank = self.next_rank;

                if rank >= self.vec.total_rank1() || self.next_rank_back.is_none() || rank > self.next_rank_back.unwrap() {
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
                    }
                } else {
                    super_block = self.vec.search_super_block1(super_block, rank);

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

                self.next_rank += 1;
                Some(self.vec.search_word_in_block1(rank, block_index))
            }

            #[must_use]
            #[allow(clippy::assertions_on_constants)]
            fn select_next_1_back(&mut self) -> Option<usize> {
                let mut rank = self.next_rank_back?;

                if self.next_rank_back.is_none() || rank < self.next_rank {
                    return None;
                }

                let mut super_block = self.vec.select_blocks[rank / SELECT_BLOCK_SIZE].index_1;
                let mut block_index = 0;

                // check if the last super block still contains the rank, and if yes, we don't need to search
                if (self.last_super_block_back) * SUPER_BLOCK_SIZE
                        - self.vec.super_blocks[self.last_super_block_back].zeros
                        < rank
                {
                    // instantly jump to the last searched position
                    super_block = self.last_super_block_back;
                    let block_at_super_block = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
                    rank -= super_block * SUPER_BLOCK_SIZE - self.vec.super_blocks[super_block].zeros;

                    // check if current block contains the one and if yes, we don't need to search
                    // this is true IF the ones before the last block are less than the rank,
                    // since the block before then can't contain it
                    if (self.last_block_back - block_at_super_block) * BLOCK_SIZE
                        - self.vec.blocks[self.last_block_back].zeros as usize
                            <= rank
                    {
                        // instantly jump to the last searched position
                        block_index = self.last_block_back;
                        let block_at_super_block = super_block * (SUPER_BLOCK_SIZE / BLOCK_SIZE);
                        rank -= (block_index - block_at_super_block) * BLOCK_SIZE
                            - self.vec.blocks[block_index].zeros as usize;
                    }
                } else {
                    super_block = self.vec.search_super_block1(super_block, rank);

                    self.last_super_block_back = super_block;
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

                    self.last_block_back = block_index;
                    rank -= (block_index - block_at_super_block) * BLOCK_SIZE
                        - self.vec.blocks[block_index].zeros as usize;
                }

                self.next_rank_back = self.next_rank_back.and_then(|x| if x > 0 { Some(x - 1) } else { None });
                Some(self.vec.search_word_in_block1(rank, block_index))
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

            /// Advances the iterator back by `n` elements. Returns an error if the iterator does not have
            /// enough elements left. Does not call `next_back` internally.
            /// This method is currently being added to the iterator trait, see
            /// [this issue](https://github.com/rust-lang/rust/issues/77404).
            /// As soon as it is stabilized, this method will be removed and replaced with a custom
            /// implementation in the double ended iterator impl.
            pub(super) fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
                if self.len() >= n {
                    self.next_rank_back = self.next_rank_back.map(|x| x - n);
                    Ok(())
                } else {
                    let len = self.len();
                    self.next_rank_back = self.next_rank_back.map(|x| x - len);
                    Err(NonZeroUsize::new(n - len).unwrap())
                }
            }
        }

        impl<$($life,)? const ZERO: bool> Iterator for $name<$($life,)? ZERO> {
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

        impl<$($life,)? const ZERO: bool> DoubleEndedIterator for $name<$($life,)? ZERO> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if ZERO {
                    self.select_next_0_back()
                } else {
                    self.select_next_1_back()
                }
            }

            fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                if ZERO {
                    self.advance_back_by(n).ok().and_then(|()| self.select_next_0_back())
                } else {
                    self.advance_back_by(n).ok().and_then(|()| self.select_next_1_back())
                }
            }
        }

        impl<$($life,)? const ZERO: bool> FusedIterator for $name<$($life,)? ZERO> {}

        impl<$($life,)? const ZERO: bool> ExactSizeIterator for $name<$($life,)? ZERO> {
            fn len(&self) -> usize {
                self.next_rank_back.map(|x| x + 1).unwrap_or_default().saturating_sub(self.next_rank)
            }
        }
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
    pub(crate) vec: &'a RsVec,
    next_rank: usize,

    // rank back is none, iff it points to element -1 (i.e. element 0 has been consumed by
    // a call to next_back()). It can be Some(..) even if the iterator is empty
    next_rank_back: Option<usize>,

    /// the last index in the super block structure where we found a bit
    last_super_block: usize,

    last_super_block_back: usize,

    /// the last index in the block structure where we found a bit
    last_block: usize,

    last_block_back: usize,
}

gen_iter_impl!('a, SelectIter);

/// An iterator that iterates over 1-bits or 0-bits and returns their indices.
/// It owns the iterated bit-vector.
/// It uses the select data structures to speed up iteration.
/// It is faster than iterating over all bits if the iterated bits are sparse.
/// This is also faster than manually calling `select` on each rank,
/// because the iterator exploits the linear access pattern for faster select queries.
///
/// The iterator can be constructed by calling [`into_iter0`] or [`into_iter1`].
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
/// [`into_iter0`]: crate::RsVec::into_iter0
/// [`into_iter1`]: crate::RsVec::into_iter1
#[derive(Clone, Debug)]
#[must_use]
// the naming convention of other iterators is broken here, because SelectIter existed before
// this owning iterator became necessary
pub struct SelectIntoIter<const ZERO: bool> {
    pub(crate) vec: RsVec,
    next_rank: usize,

    // rank back is none, iff it points to element -1 (i.e. element 0 has been consumed by
    // a call to next_back()). It can be Some(..) even if the iterator is empty
    next_rank_back: Option<usize>,

    /// the last index in the super block structure where we found a bit
    last_super_block: usize,

    last_super_block_back: usize,

    /// the last index in the block structure where we found a bit
    last_block: usize,

    last_block_back: usize,
}

gen_iter_impl!(SelectIntoIter);
