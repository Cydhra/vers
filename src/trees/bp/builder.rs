use crate::trees::bp::{BpTree, DEFAULT_BLOCK_SIZE};
use crate::trees::DfsTreeBuilder;
use crate::BitVec;

/// A builder for [`BpTrees`] using depth-first traversal of the tree. See the documentation of
/// [`DfsTreeBuilder`].
///
/// [`BpTree`]: BpTree
pub struct BpDfsBuilder<const BLOCK_SIZE: usize = DEFAULT_BLOCK_SIZE> {
    excess: i64,
    bit_vec: BitVec,
}

impl<const BLOCK_SIZE: usize> BpDfsBuilder<BLOCK_SIZE> {
    /// Create new empty `DfsTreeBuilder`
    #[must_use]
    pub fn new() -> Self {
        Self {
            excess: 0,
            bit_vec: BitVec::new(),
        }
    }

    /// Create a new empty `DfsTreeBuilder` with the given capacity for nodes.
    #[must_use]
    pub fn with_capacity(capacity: u64) -> Self {
        Self {
            excess: 0,
            bit_vec: BitVec::with_capacity((capacity * 2) as usize),
        }
    }
}

impl<const BLOCK_SIZE: usize> Default for BpDfsBuilder<BLOCK_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const BLOCK_SIZE: usize> DfsTreeBuilder for BpDfsBuilder<BLOCK_SIZE> {
    type Tree = BpTree<BLOCK_SIZE>;

    fn enter_node(&mut self) {
        self.excess += 1;
        self.bit_vec.append_bit(1);
    }

    fn leave_node(&mut self) {
        self.excess -= 1;
        self.bit_vec.append_bit(0);
    }

    fn build(self) -> Result<Self::Tree, i64> {
        if self.excess != 0 {
            Err(self.excess)
        } else {
            Ok(BpTree::from_bit_vector(self.bit_vec))
        }
    }
}
