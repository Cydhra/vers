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

    /// Returns the root node of the tree.
    fn root(&self) -> Self::NodeHandle;

    /// Returns the parent of a node, if it exists.
    fn parent(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Returns the left child of a node, if it exists.
    fn first_child(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Returns the left sibling of a node, if it exists.
    fn left_sibling(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Returns the right sibling of a node, if it exists.
    fn right_sibling(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Returns the rightmost child of a node, if it exists.
    fn last_child(&self, node: Self::NodeHandle) -> Option<Self::NodeHandle>;

    /// Convert a node handle into a contiguous index, allowing associated data to be stored in a vector.
    fn node_index(&self, node: Self::NodeHandle) -> usize;

    /// Convert a contiguous index that enumerates all nodes into a node handle.
    /// This operation is the inverse of `node_index`.
    fn node_handle(&self, index: usize) -> Self::NodeHandle;

    /// Returns true, if the node is a leaf.
    fn is_leaf(&self, node: Self::NodeHandle) -> bool;

    /// Returns the depth of the node in the tree.
    fn depth(&self, node: Self::NodeHandle) -> usize;

    /// Returns the number of nodes in the tree.
    fn size(&self) -> usize;

    /// Returns true, if the tree has no nodes.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

/// A trait for succinct tree data structures that support subtree size queries.
pub trait SubtreeSize: Tree {
    /// Returns the number of nodes in the subtree rooted at the given node.
    /// todo: define whether that includes the node itself or not.
    fn subtree_size(&self, node: Self::NodeHandle) -> usize;
}

/// A trait for succinct tree data structures that support level-order traversal.
pub trait LevelTree: Tree {
    /// Returns the `level`'th ancestor of the given node, if it exists.
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
