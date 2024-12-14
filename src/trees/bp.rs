use crate::trees::mmt::MinMaxTree;
use crate::RsVec;

/// A succinct binary tree data structure.
pub struct BpTree {
    vec: RsVec,
    min_max_tree: MinMaxTree,
}
