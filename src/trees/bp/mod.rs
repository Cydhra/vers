//! A succinct tree data structure backed by the balanced parentheses representation.

use crate::trees::mmt::MinMaxTree;
use crate::trees::{IsAncestor, LevelTree, SubtreeSize, Tree};
use crate::{BitVec, RsVec};
use std::cmp::{max, min};

const OPEN_PAREN: u64 = 1;
const CLOSE_PAREN: u64 = 0;

mod builder;
// re-export the builders toplevel
pub use builder::BpDfsBuilder;

#[cfg(feature = "u16_lookup")]
mod lookup;
#[cfg(feature = "u16_lookup")]
use lookup::*;

#[cfg(not(feature = "u16_lookup"))]
mod lookup_query;
#[cfg(not(feature = "u16_lookup"))]
use lookup_query::*;

/// A succinct binary tree data structure.
pub struct BpTree<const BLOCK_SIZE: usize = 512> {
    vec: RsVec,
    min_max_tree: MinMaxTree,
}

impl<const BLOCK_SIZE: usize> BpTree<BLOCK_SIZE> {
    /// Construct a new `BpTree` from a given bit vector.
    pub fn from_bit_vector(bv: BitVec) -> Self {
        let min_max_tree = MinMaxTree::excess_tree(&bv, BLOCK_SIZE);
        let vec = bv.into();
        Self { vec, min_max_tree }
    }

    /// Search for a position where the excess relative to the starting `index` is `relative_excess`.
    /// Returns `None` if no such position exists.
    /// The initial position is never considered in the search.
    /// Searches forward in the bit vector.
    ///
    /// # Arguments
    /// - `index`: The starting index.
    /// - `relative_excess`: The desired relative excess value.
    pub fn fwd_search(&self, index: usize, mut relative_excess: i64) -> Option<usize> {
        // check for greater than or equal length minus one, because the last element
        // won't ever have a result from fwd_search
        if index >= (self.vec.len() - 1) {
            return None;
        }

        let block_index = (index + 1) / BLOCK_SIZE;
        self.fwd_search_block(index, block_index, &mut relative_excess)
            .map_or_else(
                |_| {
                    // find the block that contains the desired relative excess
                    let block = self.min_max_tree.fwd_search(block_index, relative_excess);

                    // check the result block for the exact position
                    block.and_then(|(block, mut relative_excess)| {
                        self.fwd_search_block(block * BLOCK_SIZE - 1, block, &mut relative_excess)
                            .ok()
                    })
                },
                Some,
            )
    }

    /// Perform the forward search within one block. If this doesn't yield a result, the search
    /// continues in the min-max-tree.
    ///
    /// Returns Ok(index) if an index with the desired relative excess is found, or None(excess)
    /// with the excess at the end of the current block if no index with the desired relative excess
    /// is found.
    #[inline(always)]
    fn fwd_search_block(
        &self,
        start_index: usize,
        block_index: usize,
        relative_excess: &mut i64,
    ) -> Result<usize, ()> {
        let block_boundary = min((block_index + 1) * BLOCK_SIZE, self.vec.len());

        // the boundary at which we can start with table lookups
        let lookup_boundary = min(
            (start_index + 1).div_ceil(LOOKUP_BLOCK_SIZE as usize) * LOOKUP_BLOCK_SIZE as usize,
            block_boundary,
        );
        for i in start_index + 1..lookup_boundary {
            let bit = self.vec.get_unchecked(i);
            *relative_excess -= if bit == 1 { 1 } else { -1 };

            if *relative_excess == 0 {
                return Ok(i);
            }
        }

        // the boundary up to which we can use table lookups
        let upper_lookup_boundary = max(
            lookup_boundary,
            (block_boundary / LOOKUP_BLOCK_SIZE as usize) * LOOKUP_BLOCK_SIZE as usize,
        );

        for i in (lookup_boundary..upper_lookup_boundary).step_by(LOOKUP_BLOCK_SIZE as usize) {
            if let Ok(idx) = process_block_fwd(
                self.vec
                    .get_bits_unchecked(i, LOOKUP_BLOCK_SIZE as usize)
                    .try_into()
                    .unwrap(),
                relative_excess,
            ) {
                return Ok(i + idx as usize);
            }
        }

        // if the upper_lookup_boundary isn't the block_boundary (which happens in non-full blocks, i.e. the last
        // block in the vector)
        for i in upper_lookup_boundary..block_boundary {
            let bit = self.vec.get_unchecked(i);
            *relative_excess -= if bit == 1 { 1 } else { -1 };

            if *relative_excess == 0 {
                return Ok(i);
            }
        }

        Err(())
    }

    /// Search for a position where the excess relative to the starting `index` is `relative_excess`.
    /// Returns `None` if no such position exists.
    /// The initial position is never considered in the search.
    /// Searches backward in the bit vector.
    ///
    /// # Arguments
    /// - `index`: The starting index.
    /// - `relative_excess`: The desired relative excess value.
    pub fn bwd_search(&self, index: usize, mut relative_excess: i64) -> Option<usize> {
        if index >= self.vec.len() {
            return None;
        }

        // if the index is 0, we cant have a valid result anyway, and this would overflow the
        // subtraction below, so we report None
        if index == 0 {
            return None;
        }

        // calculate the block we start searching in. It starts at index - 1, so we don't accidentally
        // search the mM tree and immediately find `index` as the position
        let block_index = (index - 1) / BLOCK_SIZE;

        // check the current block
        self.bwd_search_block(index, block_index, &mut relative_excess)
            .map_or_else(
                |_| {
                    // find the block that contains the desired relative excess
                    let block = self.min_max_tree.bwd_search(block_index, relative_excess);

                    // check the result block for the exact position
                    block.and_then(|(block, mut relative_excess)| {
                        self.bwd_search_block((block + 1) * BLOCK_SIZE, block, &mut relative_excess)
                            .ok()
                    })
                },
                Some,
            )
    }

    /// Perform the backward search within one block. If this doesn't yield a result, the search
    /// continues in the min-max-tree.
    ///
    /// Returns Ok(index) if an index with the desired relative excess is found, or None(excess)
    /// with the excess at the end of the current block if no index with the desired relative excess
    /// is found.
    #[inline(always)]
    fn bwd_search_block(
        &self,
        start_index: usize,
        block_index: usize,
        relative_excess: &mut i64,
    ) -> Result<usize, ()> {
        let block_boundary = min(block_index * BLOCK_SIZE, self.vec.len());

        // the boundary at which we can start with table lookups
        let lookup_boundary = max(
            ((start_index - 1) / LOOKUP_BLOCK_SIZE as usize) * LOOKUP_BLOCK_SIZE as usize,
            block_boundary,
        );
        for i in (lookup_boundary..start_index).rev() {
            let bit = self.vec.get_unchecked(i);
            *relative_excess += if bit == 1 { 1 } else { -1 };

            if *relative_excess == 0 {
                return Ok(i);
            }
        }

        for i in (block_boundary..lookup_boundary)
            .step_by(LOOKUP_BLOCK_SIZE as usize)
            .rev()
        {
            if let Ok(idx) = process_block_bwd(
                self.vec
                    .get_bits_unchecked(i, LOOKUP_BLOCK_SIZE as usize)
                    .try_into()
                    .unwrap(),
                relative_excess,
            ) {
                return Ok(i + idx as usize);
            }
        }

        Err(())
    }

    /// Find the position of the matching closing parenthesis for the opening parenthesis at `index`.
    /// If the bit at `index` is not an opening parenthesis, the result is meaningless.
    /// If there is no matching closing parenthesis, `None` is returned.
    pub fn close(&self, index: usize) -> Option<usize> {
        if index >= self.vec.len() {
            return None;
        }

        self.fwd_search(index, -1)
    }

    /// Find the position of the matching opening parenthesis for the closing parenthesis at `index`.
    /// If the bit at `index` is not a closing parenthesis, the result is meaningless.
    /// If there is no matching opening parenthesis, `None` is returned.
    pub fn open(&self, index: usize) -> Option<usize> {
        if index >= self.vec.len() {
            return None;
        }

        self.bwd_search(index, -1)
    }

    /// Find the position of the opening parenthesis that encloses the position `index`.
    /// This works regardless of whether the bit at `index` is an opening or closing parenthesis.
    /// If there is no enclosing parenthesis, `None` is returned.
    pub fn enclose(&self, index: usize) -> Option<usize> {
        if index >= self.vec.len() {
            return None;
        }

        self.bwd_search(
            index,
            if self.vec.get_unchecked(index) == 1 {
                -1
            } else {
                -2
            },
        )
    }

    /// Get the excess of open parentheses up to and including the position `index`.
    /// The excess is the number of open parentheses minus the number of closing parentheses.
    /// If `index` is out of bounds, the total excess of the parentheses expression is returned.
    pub fn excess(&self, index: usize) -> i64 {
        debug_assert!(index < self.vec.len(), "Index out of bounds");
        self.vec.rank1(index + 1) as i64 - self.vec.rank0(index + 1) as i64
    }
}

impl<const BLOCK_SIZE: usize> Tree for BpTree<BLOCK_SIZE> {
    type NodeHandle = usize;

    fn root(&self) -> Option<Self::NodeHandle> {
        if !self.vec.is_empty() {
            Some(0)
        } else {
            None
        }
    }

    fn parent(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle> {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );

        self.enclose(node)
    }

    fn first_child(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle> {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );

        if let Some(bit) = self.vec.get(node + 1) {
            if bit == OPEN_PAREN {
                return Some(node + 1);
            }
        }

        None
    }

    fn next_sibling(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle> {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );
        self.close(node).and_then(|i| {
            self.vec
                .get(i + 1)
                .and_then(|bit| if bit == OPEN_PAREN { Some(i + 1) } else { None })
        })
    }

    fn previous_sibling(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle> {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );
        if node == 0 {
            None
        } else {
            self.vec.get(node - 1).and_then(|bit| {
                if bit == CLOSE_PAREN {
                    self.open(node - 1)
                } else {
                    None
                }
            })
        }
    }

    fn last_child(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle> {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );
        self.vec.get(node + 1).and_then(|bit| {
            if bit == OPEN_PAREN {
                if let Some(i) = self.close(node) {
                    self.open(i - 1)
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    fn node_index(&self, node: Self::NodeHandle) -> usize {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );
        self.vec.rank1(node)
    }

    fn node_handle(&self, index: usize) -> Self::NodeHandle {
        self.vec.select1(index)
    }

    fn is_leaf(&self, node: Self::NodeHandle) -> bool {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );
        self.vec.get(node + 1) == Some(CLOSE_PAREN)
    }

    fn depth(&self, node: Self::NodeHandle) -> u64 {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );
        (self.excess(node) as u64).saturating_sub(1)
    }

    fn size(&self) -> usize {
        self.vec.rank1(self.vec.len())
    }

    fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }
}

impl<const BLOCK_SIZE: usize> IsAncestor for BpTree<BLOCK_SIZE> {
    fn is_ancestor(
        &self,
        ancestor: Self::NodeHandle,
        descendant: Self::NodeHandle,
    ) -> Option<bool> {
        debug_assert!(
            self.vec.get(ancestor) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );
        debug_assert!(
            self.vec.get(descendant) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );

        self.close(ancestor)
            .map(|closing| ancestor <= descendant && descendant < closing)
    }
}

impl<const BLOCK_SIZE: usize> LevelTree for BpTree<BLOCK_SIZE> {
    fn level_ancestor(&self, node: Self::NodeHandle, level: u64) -> Option<Self::NodeHandle> {
        if level == 0 {
            return Some(node);
        }

        self.bwd_search(node, -(level as i64))
    }

    fn level_next(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle> {
        self.fwd_search(self.close(node)?, 1)
    }

    fn level_prev(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle> {
        self.open(self.bwd_search(node, 1)?)
    }

    fn level_leftmost(&self, level: u64) -> Option<Self::NodeHandle> {
        // fwd_search doesn't support returning the input position
        if level == 0 {
            return Some(0);
        }

        self.fwd_search(0, level as i64)
    }

    fn level_rightmost(&self, level: u64) -> Option<Self::NodeHandle> {
        self.open(self.bwd_search(self.size() * 2 - 1, level as i64)?)
    }
}

impl<const BLOCK_SIZE: usize> SubtreeSize for BpTree<BLOCK_SIZE> {
    fn subtree_size(&self, node: Self::NodeHandle) -> Option<usize> {
        self.close(node)
            .map(|c| self.vec.rank1(c) - self.vec.rank1(node))
    }
}

#[cfg(test)]
mod tests;
