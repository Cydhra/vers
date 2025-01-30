//! Tree data structures. Currently only the [BP][bp] tree is exposed.
//! The trees are succinct, approaching the information-theoretic lower bound for the space complexity:
//! They need O(n) bits to store a tree with n nodes, and o(n) extra bits to support queries.
//!
//! For details, see the submodules.

pub mod bp;

pub(crate) mod mmt;

/// A trait for succinct tree data structures defining the most basic tree navigation operations.
pub trait Tree {
    /// A type that represents a node during tree navigation.
    type NodeHandle;

    /// Returns the root node of the tree, if the tree isn't empty.
    fn root(&self) -> Option<Self::NodeHandle>;

    /// Returns the parent of a node, if it exists.
    /// If `node` is not a valid node handle, the result is meaningless.
    fn parent(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Returns the left child of a node, if it exists.
    /// If `node` is not a valid node handle, the result is meaningless.
    fn first_child(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Returns the left sibling of a node, if it exists.
    /// If `node` is not a valid node handle, the result is meaningless.
    fn next_sibling(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Returns the right sibling of a node, if it exists.
    /// If `node` is not a valid node handle, the result is meaningless.
    fn previous_sibling(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Returns the rightmost child of a node, if it exists.
    /// If `node` is not a valid node handle, the result is meaningless.
    fn last_child(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Convert a node handle into a contiguous index, allowing associated data to be stored in a vector.
    /// If `node` is not a valid node handle, the result is meaningless.
    fn node_index(&self, node: Self::NodeHandle) -> usize;

    /// Convert a contiguous index that enumerates all nodes into a node handle.
    /// This operation is the inverse of `node_index`.
    /// The index must be in the range `0..self.size()`.
    ///
    /// If the index is out of bounds, the behavior is unspecified.
    fn node_handle(&self, index: usize) -> Self::NodeHandle;

    /// Returns true if the node is a leaf.
    /// If `node` is not a valid node handle, the result is meaningless.
    fn is_leaf(&self, node: Self::NodeHandle) -> bool;

    /// Returns the depth of the node in the tree.
    /// The root node has depth 0.
    /// If `node` is not a valid node handle, the result is meaningless.
    fn depth(&self, node: Self::NodeHandle) -> u64;

    /// Returns the number of nodes in the tree.
    fn size(&self) -> usize;

    /// Returns true, if the tree has no nodes.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

/// A trait for succinct tree data structures that support [`subtree_size`] queries.
///
/// [`subtree_size`]: SubtreeSize::subtree_size
pub trait SubtreeSize: Tree {
    /// Returns the number of nodes in the subtree rooted at the given node.
    /// todo: define whether that includes the node itself or not.
    fn subtree_size(&self, node: Self::NodeHandle) -> usize;
}

/// A trait for succinct tree data structures that support [`is_ancestor`] queries.
///
/// [`is_ancestor`]: IsAncestor::is_ancestor
pub trait IsAncestor: Tree {
    /// Returns true if `ancestor` is an ancestor of the `descendant` node.
    /// Note that a node is considered an ancestor of itself.
    ///
    /// Returns `None` if the parenthesis expression is unbalanced and `ancestor` does not have a
    /// closing parenthesis.
    fn is_ancestor(&self, ancestor: Self::NodeHandle, descendant: Self::NodeHandle)
        -> Option<bool>;
}

/// A trait for succinct tree data structures that support level-order traversal.
pub trait LevelTree: Tree {
    /// Returns the `level`'th ancestor of the given node, if it exists. If the level is 0, `node`
    /// is returned. If `node` is not a valid node handle, the result is meaningless.
    fn level_ancestor(&self, node: Self::NodeHandle, level: u64) -> Option<Self::NodeHandle>;

    /// Returns the next node in the level order traversal of the tree, if it exists.
    fn level_next(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Returns the previous node in the level order traversal of the tree, if it exists.
    fn level_prev(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Returns the leftmost node at the given level, if it exists.
    fn level_leftmost(&self, level: u64) -> Option<Self::NodeHandle>;

    /// Returns the rightmost node at the given level, if it exists.
    fn level_rightmost(&self, level: u64) -> Option<Self::NodeHandle>;
}

/// This trait provides the functionality to build a tree by visiting its nodes in depth first
/// search order. The caller should call [`enter_node`] for each node visited in pre-order depth-first
/// traversal, and [`leave_node`] once the node's subtree was visited (i.e. post-order).
///
/// Once the full tree has been visited, the caller must call [`build`] to create an instance of the
/// implementing tree type.
pub trait DfsTreeBuilder {
    /// The tree type constructed with this interface
    type Tree;

    /// Called to create a new node in the tree builder
    fn enter_node(&mut self);

    /// Called after the subtree of a node in the tree has already been visited.
    fn leave_node(&mut self);

    /// Finalize the tree instance. Returns `Err(excess)` if the constructed tree is invalid
    /// (i.e. there are nodes for which [`leave_node`] has not been called,
    /// or there are more calls to `leave_node` than to [`enter_node`];
    /// the number of extraneous calls to `enter_node` is returned in the error).
    fn build(self) -> Result<Self::Tree, i64>;
}
