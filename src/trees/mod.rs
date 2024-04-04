use std::num::NonZeroUsize;
use std::ops::Index;

mod bp;

/// A singular node in a binary min-max tree that is part of the BpTree data structure.
struct MinMaxNode {
    start_excess: usize,
    min_excess: usize,
    max_excess: usize,
}

/// A binary min-max tree that is part of the BpTree data structure.
struct MinMaxTree {
    nodes: Vec<MinMaxNode>,
}

impl MinMaxTree {

    fn start_excess(&self, index: usize) -> usize {
        self.nodes[index].start_excess
    }

    fn min_excess(&self, index: usize) -> usize {
        self.nodes[index].min_excess
    }

    fn max_excess(&self, index: usize) -> usize {
        self.nodes[index].max_excess
    }

    fn parent(&self, index: NonZeroUsize) -> usize {
        debug_assert!(index.get() < self.nodes.len(), "request parent for non-existent node");
        (index.get() - 1) / 2
    }

    /// Get the index of the left child of the node at `index` if it exists
    fn left_child(&self, index: usize) -> Option<NonZeroUsize> {
        if index * 2 + 1 < self.nodes.len() {
            NonZeroUsize::new(index * 2 + 1)
        } else {
            None
        }
    }

    /// Get the index of the right child of the node at `index` if it exists
    fn right_child(&self, index: usize) -> Option<NonZeroUsize> {
        if index * 2 + 2 < self.nodes.len() {
            NonZeroUsize::new(index * 2 + 2)
        } else {
            None
        }
    }

    /// Get the index of the right sibling of the node at `index` if it exists
    fn right_sibling(&self, index: NonZeroUsize) -> Option<NonZeroUsize> {
        if index.get() % 2 == 1 {
            if index.get() + 1 >= self.nodes.len() {
                None
            } else {
                index.checked_add(1)
            }
        } else {
            None
        }
    }

    /// Get the index of the left sibling of the node at `index` if it exists
    fn left_sibling(&self, index: NonZeroUsize) -> Option<NonZeroUsize> {
        if index.get() % 2 == 0 {
            // index is at least 2
            NonZeroUsize::new(index.get() - 1)
        } else {
            None
        }
    }
}



