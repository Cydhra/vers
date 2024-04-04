use crate::RsVec;
use crate::trees::MinMaxTree;

/// A succinct binary tree data structure.
pub struct BpTree {
    vec: RsVec,
    min_max_tree: MinMaxTree,
}
