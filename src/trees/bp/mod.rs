//! A succinct tree data structure backed by the balanced parentheses representation.

use crate::trees::mmt::MinMaxTree;
use crate::trees::Tree;
use crate::{BitVec, RsVec};
use std::cmp::min;

const OPEN_PAREN: u64 = 1;
const CLOSE_PAREN: u64 = 0;

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
    pub fn fwd_search(&self, index: usize, relative_excess: i64) -> Option<usize> {
        if index >= self.vec.len() {
            return None;
        }

        let block_index = index / BLOCK_SIZE;
        let block_boundary = min((block_index + 1) * BLOCK_SIZE, self.vec.len());

        let mut current_relative_excess = 0;

        // check the current block
        for i in index + 1..block_boundary {
            let bit = self.vec.get_unchecked(i);
            current_relative_excess += if bit == 1 { 1 } else { -1 };

            if current_relative_excess == relative_excess {
                return Some(i);
            }
        }

        // find the block that contains the desired relative excess
        let block = self
            .min_max_tree
            .fwd_search(block_index, relative_excess - current_relative_excess);

        // check the result block for the exact position
        block.and_then(|(block, relative_excess)| {
            current_relative_excess = 0;
            for i in block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE {
                let bit = self.vec.get_unchecked(i);
                current_relative_excess += if bit == 1 { 1 } else { -1 };

                if current_relative_excess == relative_excess {
                    return Some(i);
                }
            }

            unreachable!("If the block isn't None, the loop should always return Some(i)")
        })
    }

    /// Search for a position where the excess relative to the starting `index` is `relative_excess`.
    /// Returns `None` if no such position exists.
    /// The initial position is never considered in the search.
    /// Searches backward in the bit vector.
    ///
    /// # Arguments
    /// - `index`: The starting index.
    /// - `relative_excess`: The desired relative excess value.
    pub fn bwd_search(&self, index: usize, relative_excess: i64) -> Option<usize> {
        if index >= self.vec.len() {
            return None;
        }

        let block_index = index / BLOCK_SIZE;
        let block_boundary = min(block_index * BLOCK_SIZE, self.vec.len());

        let mut current_relative_excess = 0;

        // check the current block
        for i in (block_boundary..index).rev() {
            let bit = self.vec.get_unchecked(i);
            current_relative_excess += if bit == 1 { -1 } else { 1 };

            if current_relative_excess == relative_excess {
                return Some(i);
            }
        }

        // find the block that contains the desired relative excess
        let block = self
            .min_max_tree
            .bwd_search(block_index, relative_excess - current_relative_excess);

        // check the result block for the exact position
        block.and_then(|(block, relative_excess)| {
            current_relative_excess = 0;
            for i in (block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE).rev() {
                let bit = self.vec.get_unchecked(i);
                current_relative_excess += if bit == 1 { -1 } else { 1 };

                if current_relative_excess == relative_excess {
                    return Some(i);
                }
            }

            unreachable!("If the block isn't None, the loop should always return Some(i)")
        })
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
        self.vec.get(node - 1).and_then(|bit| {
            if bit == CLOSE_PAREN {
                self.open(node - 1)
            } else {
                None
            }
        })
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

#[cfg(test)]
mod tests;
