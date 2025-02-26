//! A succinct tree data structure backed by the balanced parentheses representation.

use crate::bit_vec::fast_rs_vec::SelectIntoIter;
use crate::trees::mmt::MinMaxTree;
use crate::trees::{IsAncestor, LevelTree, SubtreeSize, Tree};
use crate::{BitVec, RsVec};
use std::cmp::{max, min};

/// The default block size for the tree, used in several const generics
const DEFAULT_BLOCK_SIZE: usize = 512;

const OPEN_PAREN: u64 = 1;
const CLOSE_PAREN: u64 = 0;

mod builder;
// re-export the builders toplevel
pub use builder::BpBuilder;

#[cfg(feature = "bp_u16_lookup")]
mod lookup;
#[cfg(feature = "bp_u16_lookup")]
use lookup::{process_block_bwd, process_block_fwd, LOOKUP_BLOCK_SIZE};

#[cfg(not(feature = "bp_u16_lookup"))]
mod lookup_query;
#[cfg(not(feature = "bp_u16_lookup"))]
use lookup_query::{process_block_bwd, process_block_fwd, LOOKUP_BLOCK_SIZE};

/// A succinct tree data structure based on balanced parenthesis expressions.
/// A tree with `n` nodes is encoded in a bit vector using `2n` bits plus the rank/select overhead
/// of the [`RsVec`] implementation.
/// Additionally, a small pointerless heap data structure stores
/// additional meta information required to perform most tree operations.
///
/// The tree is thus pointer-less and succinct.
/// It supports tree navigation operations between parent, child, and sibling nodes, both in
/// depth-first search order and in level order.
/// All operations run in `O(log n)` time with small overheads.
///
/// ## Lookup Table
/// The tree internally uses a lookup table for subqueries on blocks of bits.
/// The lookup table requires 4 KiB of memory and is compiled into the binary.
/// If the `bp_u16_lookup` feature is enabled, a larger lookup table is used, which requires 128 KiB of
/// memory, but answers queries faster.
///
/// ## Block Size
/// The tree has a block size of 512 bits by default, which can be changed by setting the
/// `BLOCK_SIZE` generic parameter.
/// This block size is expected to be a good choice for most applications,
/// as it will fit a cache line.
///
/// If you want to tune the parameter,
/// the block size should be chosen based on the expected size of the tree and the available memory.
/// Smaller block sizes increase the size of the supporting data structure but reduce the time
/// complexity of some operations by a constant amount.
/// Larger block sizes are best combined with the `bp_u16_lookup` feature to keep the query time
/// low.
/// In any case, benchmarking for the specific use case is recommended for tuning.
///
/// ## Unbalanced Parentheses
/// The tree is implemented in a way to theoretically support unbalanced parenthesis expressions
/// (which encode invalid trees) without panicking.
/// However, some operations may behave erratically if the parenthesis expression isn't balanced.
/// Generally, operations specify if they require a balanced tree.
///
/// The results of the operations are unspecified,
/// meaning no guarantees are made about the stability of the results across versions
/// (except the operations not panicking).
/// However, for research purposes, this behavior can be useful and should yield expected results
/// in most cases.
///
/// Only the basic operations like [`fwd_search`] and [`bwd_search`],
/// as well as the tree navigation operations
/// (defined by the traits [`Tree`], [`IsAncestor`], [`LevelTree`], and [`SubtreeSize`]),
/// are included in this guarantee.
/// Additional operations like iterators may panic if the tree is unbalanced (this is documented per
/// operation).
///
/// # Examples
///
/// The high-level approach to building a tree is to use the [`BpBuilder`] to construct the tree
/// using depth-first traversal of all its nodes.
/// ```rust
/// # #![allow(long_running_const_eval)] // for some reason this is needed for test cases
/// use vers_vecs::{BitVec, BpBuilder, BpTree, TreeBuilder, Tree};
///
/// let mut builder = BpBuilder::<512>::new();
///
/// // build the tree by depth-first traversal
/// builder.enter_node();
/// builder.enter_node();
/// builder.enter_node();
/// builder.leave_node();
/// builder.enter_node();
/// builder.leave_node();
/// builder.leave_node();
/// builder.enter_node();
/// builder.leave_node();
/// builder.leave_node();
///
/// let tree = builder.build().unwrap();
/// let root = tree.root().unwrap();
/// assert_eq!(root, 0);
/// assert_eq!(tree.first_child(root), Some(1));
/// assert_eq!(tree.next_sibling(1), Some(7));
/// assert_eq!(tree.next_sibling(7), None);
///
/// assert_eq!(root, 0);
/// assert_eq!(tree.depth(2), 2);
/// assert_eq!(tree.depth(7), 1);
/// ```
///
/// Alternatively, the tree can be constructed from a [`BitVec`] containing the parenthesis
/// expression directly.
/// This is also how trees with unbalanced parenthesis expressions can be constructed.
///
/// ```rust
/// # #![allow(long_running_const_eval)]
/// use vers_vecs::{BitVec, BpTree, Tree};
/// let bv = BitVec::pack_sequence_u8(&[0b1101_0111, 0b0010_0100], 8);
/// let tree = BpTree::<4>::from_bit_vector(bv);
///
/// let nodes = tree.dfs_iter().collect::<Vec<_>>();
/// assert_eq!(nodes, vec![0, 1, 2, 4, 6, 7, 10, 13]);
/// ```
///
/// [`RsVec`]: RsVec
/// [`fwd_search`]: BpTree::fwd_search
/// [`bwd_search`]: BpTree::bwd_search
/// [`Tree`]: Tree
/// [`IsAncestor`]: IsAncestor
/// [`LevelTree`]: LevelTree
/// [`SubtreeSize`]: SubtreeSize
/// [`BpBuilder`]: BpBuilder
/// [`BitVec`]: BitVec
#[derive(Clone, Debug)]
pub struct BpTree<const BLOCK_SIZE: usize = DEFAULT_BLOCK_SIZE> {
    vec: RsVec,
    min_max_tree: MinMaxTree,
}

impl<const BLOCK_SIZE: usize> BpTree<BLOCK_SIZE> {
    /// Construct a new `BpTree` from a given bit vector.
    #[must_use]
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
                |()| {
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
                |()| {
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
    #[must_use]
    pub fn close(&self, index: usize) -> Option<usize> {
        if index >= self.vec.len() {
            return None;
        }

        self.fwd_search(index, -1)
    }

    /// Find the position of the matching opening parenthesis for the closing parenthesis at `index`.
    /// If the bit at `index` is not a closing parenthesis, the result is meaningless.
    /// If there is no matching opening parenthesis, `None` is returned.
    #[must_use]
    pub fn open(&self, index: usize) -> Option<usize> {
        if index >= self.vec.len() {
            return None;
        }

        self.bwd_search(index, -1)
    }

    /// Find the position of the opening parenthesis that encloses the position `index`.
    /// This works regardless of whether the bit at `index` is an opening or closing parenthesis.
    /// If there is no enclosing parenthesis, `None` is returned.
    #[must_use]
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
    #[must_use]
    pub fn excess(&self, index: usize) -> i64 {
        debug_assert!(index < self.vec.len(), "Index out of bounds");
        self.vec.rank1(index + 1) as i64 - self.vec.rank0(index + 1) as i64
    }

    /// Iterate over the nodes of the tree.
    /// The iterator yields the nodes in depth-first (pre-)order.
    /// This method is an alias for [`dfs_iter`].
    ///
    /// If the tree is unbalanced, the iterator returns the node handles in the order they appear in
    /// the parenthesis expression, and it will return handles that don't have a matching closing
    /// parenthesis.
    ///
    /// [`dfs_iter`]: BpTree::dfs_iter
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = <BpTree<BLOCK_SIZE> as Tree>::NodeHandle> + use<'_, BLOCK_SIZE> {
        self.dfs_iter()
    }

    /// Iterate over the nodes of the tree in depth-first (pre-)order.
    /// This is the most efficient way to iterate over all nodes of the tree.
    ///
    /// If the tree is unbalanced, the iterator returns the node handles in the order they appear in
    /// the parenthesis expression, and it will return handles that don't have a matching closing
    /// parenthesis.
    pub fn dfs_iter(
        &self,
    ) -> impl Iterator<Item = <BpTree<BLOCK_SIZE> as Tree>::NodeHandle> + use<'_, BLOCK_SIZE> {
        self.vec.iter1()
    }

    /// Iterate over the nodes of a valid tree in depth-first (post-)order.
    /// This is slower than the pre-order iteration.
    ///
    /// # Panics
    /// The iterator may panic at any point if the parenthesis expression is unbalanced.
    pub fn dfs_post_iter(
        &self,
    ) -> impl Iterator<Item = <BpTree<BLOCK_SIZE> as Tree>::NodeHandle> + use<'_, BLOCK_SIZE> {
        self.vec.iter0().map(|n| self.open(n).unwrap())
    }

    /// Iterate over a subtree rooted at `node` in depth-first (pre-)order.
    /// The iteration starts with the node itself.
    ///
    /// Calling this method on an invalid node handle, or an unbalanced parenthesis expression,
    /// will produce an iterator over an unspecified subset of nodes.
    pub fn subtree_iter(
        &self,
        node: <BpTree<BLOCK_SIZE> as Tree>::NodeHandle,
    ) -> impl Iterator<Item = <BpTree<BLOCK_SIZE> as Tree>::NodeHandle> + use<'_, BLOCK_SIZE> {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );

        let index = self.vec.rank1(node);
        let close = self.close(node).unwrap_or(node);
        let subtree_size = self.vec.rank1(close) - index;

        self.vec.iter1().skip(index).take(subtree_size)
    }

    /// Iterate over a subtree rooted at `node` in depth-first (post-)order.
    /// This is slower than the pre-order iteration.
    /// The iteration ends with the node itself.
    ///
    /// Calling this method on an invalid node handle, or an unbalanced parenthesis expression,
    /// will produce an iterator over an unspecified subset of nodes.
    pub fn subtree_post_iter(
        &self,
        node: <BpTree<BLOCK_SIZE> as Tree>::NodeHandle,
    ) -> impl Iterator<Item = <BpTree<BLOCK_SIZE> as Tree>::NodeHandle> + use<'_, BLOCK_SIZE> {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );

        let index = self.vec.rank0(node);
        let close = self.close(node).unwrap_or(node);
        let subtree_size = self.vec.rank0(close) + 1 - index;

        self.vec
            .iter0()
            .skip(index)
            .take(subtree_size)
            .map(|n| self.open(n).unwrap())
    }

    /// Returns the number of bytes used on the heap for this tree. This does not include
    /// allocated space that is not used (e.g. by the allocation behavior of `Vec`).
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.vec.heap_size() + self.min_max_tree.heap_size()
    }
}

impl<const BLOCK_SIZE: usize> Tree for BpTree<BLOCK_SIZE> {
    type NodeHandle = usize;

    fn root(&self) -> Option<Self::NodeHandle> {
        if self.vec.is_empty() {
            None
        } else {
            Some(0)
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
        let excess: u64 = self.excess(node).try_into().unwrap_or(0);
        excess.saturating_sub(1)
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

        #[allow(clippy::cast_possible_wrap)]
        // if the level exceeds 2^63, we accept that the result is wrong
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

        #[allow(clippy::cast_possible_wrap)]
        // if the level exceeds 2^63, we accept that the result is wrong
        self.fwd_search(0, level as i64)
    }

    fn level_rightmost(&self, level: u64) -> Option<Self::NodeHandle> {
        #[allow(clippy::cast_possible_wrap)]
        // if the level exceeds 2^63, we accept that the result is wrong
        self.open(self.bwd_search(self.size() * 2 - 1, level as i64)?)
    }
}

impl<const BLOCK_SIZE: usize> SubtreeSize for BpTree<BLOCK_SIZE> {
    fn subtree_size(&self, node: Self::NodeHandle) -> Option<usize> {
        debug_assert!(
            self.vec.get(node) == Some(OPEN_PAREN),
            "Node handle is invalid"
        );

        self.close(node)
            .map(|c| self.vec.rank1(c) - self.vec.rank1(node))
    }
}

impl<const BLOCK_SIZE: usize> IntoIterator for BpTree<BLOCK_SIZE> {
    type Item = <BpTree<BLOCK_SIZE> as Tree>::NodeHandle;
    type IntoIter = SelectIntoIter<false>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter1()
    }
}

#[cfg(test)]
mod tests;
