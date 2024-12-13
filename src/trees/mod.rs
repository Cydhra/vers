use crate::BitVec;
use std::num::NonZeroUsize;

mod bp;

/// A singular node in a binary min-max tree that is part of the BpTree data structure.
#[derive(Debug, Clone, Default)]
struct MinMaxNode {
    /// excess from l..=r in the node [l, r]
    total_excess: isize,

    /// minimum (relative) excess in the node [l, r]
    min_excess: isize,

    /// maximum (relative) excess in the node [l, r]
    max_excess: isize,
}

/// A binary min-max tree that is part of the BpTree data structure.
struct MinMaxTree {
    nodes: Vec<MinMaxNode>,
}

impl MinMaxTree {
    fn excess_tree(bit_vec: &BitVec, block_size: usize) -> Self {
        let num_leaves = (bit_vec.len() + block_size - 1) / block_size;
        let num_internal_nodes = (1 << (num_leaves as f64).log2().ceil() as usize) - 1;

        let mut nodes = vec![MinMaxNode::default(); num_leaves + num_internal_nodes];
        let mut total_excess = 0;
        let mut min_excess = 0;
        let mut max_excess = 0;

        // bottom up construction
        for i in 0..bit_vec.len() {
            if i > 0 && i % block_size == 0 {
                nodes[num_internal_nodes + i / block_size - 1] = MinMaxNode {
                    total_excess,
                    min_excess,
                    max_excess,
                };
                total_excess = 0;
                min_excess = 0;
                max_excess = 0;
            }
            total_excess += if bit_vec.is_bit_set_unchecked(i) {
                1
            } else {
                -1
            };
            min_excess = min_excess.min(total_excess);
            max_excess = max_excess.max(total_excess);
        }
        nodes[num_internal_nodes + num_leaves - 1] = MinMaxNode {
            total_excess,
            min_excess,
            max_excess,
        };

        let mut current_level_size = num_leaves.next_power_of_two() / 2;
        let mut current_level_start = num_internal_nodes - current_level_size;
        loop {
            for i in 0..current_level_size {
                let left_child_index = (current_level_start + i) * 2 + 1;
                let right_child_index = (current_level_start + i) * 2 + 2;

                if left_child_index < nodes.len() {
                    if right_child_index < nodes.len() {
                        let left_child = &nodes[left_child_index];
                        let right_child = &nodes[right_child_index];
                        nodes[current_level_start + i] = MinMaxNode {
                            total_excess: left_child.total_excess + right_child.total_excess,
                            min_excess: left_child
                                .min_excess
                                .min(left_child.total_excess + right_child.min_excess),
                            max_excess: left_child
                                .max_excess
                                .max(left_child.total_excess + right_child.max_excess),
                        };
                    } else {
                        nodes[current_level_start + i] = nodes[left_child_index].clone();
                    }
                }
            }

            // if this was the root level, break the loop
            if current_level_size == 1 {
                break;
            }

            current_level_size /= 2;
            current_level_start -= current_level_size;
        }

        Self { nodes }
    }

    fn total_excess(&self, index: usize) -> isize {
        self.nodes[index].total_excess
    }

    fn min_excess(&self, index: usize) -> isize {
        self.nodes[index].min_excess
    }

    fn max_excess(&self, index: usize) -> isize {
        self.nodes[index].max_excess
    }

    fn parent(&self, index: NonZeroUsize) -> usize {
        debug_assert!(
            index.get() < self.nodes.len(),
            "request parent for non-existent node"
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitVec;

    #[test]
    fn test_simple_excess_tree() {
        #[rustfmt::skip]
        let bv = BitVec::from_bits(&[
            1, 1, 1, 0, 0, 1, 1, 1,
            0, 1, 0, 1, 1, 1, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0,
        ]);

        let tree = MinMaxTree::excess_tree(&bv, 8);

        // three internal nodes, three leaves
        assert_eq!(tree.nodes.len(), 6);

        // leaf nodes
        assert_eq!(tree.nodes[3].total_excess, 4);
        assert_eq!(tree.nodes[3].min_excess, 0);
        assert_eq!(tree.nodes[3].max_excess, 4);

        assert_eq!(tree.nodes[4].total_excess, 0);
        assert_eq!(tree.nodes[4].min_excess, -1);
        assert_eq!(tree.nodes[4].max_excess, 2);

        assert_eq!(tree.nodes[5].total_excess, -4);
        assert_eq!(tree.nodes[5].min_excess, -4);
        assert_eq!(tree.nodes[5].max_excess, 1);

        // root node
        assert_eq!(tree.nodes[0].total_excess, 0); // the tree should be balanced
        assert_eq!(tree.nodes[0].min_excess, 0);
        assert_eq!(tree.nodes[0].max_excess, 6);

        // left child of the root
        assert_eq!(tree.nodes[1].total_excess, 4);
        assert_eq!(tree.nodes[1].min_excess, 0);
        assert_eq!(tree.nodes[1].max_excess, 6);

        // right child of the root
        assert_eq!(tree.nodes[2].total_excess, -4);
        assert_eq!(tree.nodes[2].min_excess, -4);
        assert_eq!(tree.nodes[2].max_excess, 1);
    }
}
