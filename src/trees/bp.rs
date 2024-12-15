use crate::trees::mmt::MinMaxTree;
use crate::{BitVec, RsVec};
use std::cmp::min;

/// A succinct binary tree data structure.
pub struct BpTree<const BLOCK_SIZE: usize = 512> {
    vec: RsVec,
    min_max_tree: MinMaxTree,
}

impl<const BLOCK_SIZE: usize> BpTree<BLOCK_SIZE> {
    fn from_bit_vector(bv: BitVec) -> Self {
        let min_max_tree = MinMaxTree::excess_tree(&bv, BLOCK_SIZE);
        let vec = bv.into();
        Self { vec, min_max_tree }
    }

    fn fwd_search(&self, index: usize, relative_excess: i64) -> Option<usize> {
        if index >= self.vec.len() {
            return None;
        }

        let block_index = index / BLOCK_SIZE;
        let block_boundary = min((block_index + 1) * BLOCK_SIZE, self.vec.len());

        let mut current_relative_excess = 0;

        for i in index + 1..block_boundary {
            let bit = self.vec.get_unchecked(i);
            current_relative_excess += if bit == 1 { 1 } else { -1 };

            if current_relative_excess == relative_excess {
                return Some(i);
            }
        }

        let block = self
            .min_max_tree
            .fwd_search(block_index, relative_excess - current_relative_excess);

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitVec;

    #[test]
    fn test_forward_search() {
        #[rustfmt::skip]
        let bv = BitVec::from_bits(&[
            1, 1, 1, 1, 0, 0, 1, 1,
            0, 1, 0, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 0, 0,
        ]);

        let bp_tree = BpTree::<8>::from_bit_vector(bv);

        // search within block
        assert_eq!(bp_tree.fwd_search(3, -1), Some(4));
        assert_eq!(bp_tree.fwd_search(2, -1), Some(5));
        assert_eq!(bp_tree.fwd_search(12, -1), Some(13));
        assert_eq!(bp_tree.fwd_search(20, -1), Some(21));

        // search across blocks
        assert_eq!(bp_tree.fwd_search(0, -1), Some(23));
        assert_eq!(bp_tree.fwd_search(1, -1), Some(22));

        // search with weird relative excess
        assert_eq!(bp_tree.fwd_search(3, 0), Some(7));
        assert_eq!(bp_tree.fwd_search(3, -2), Some(5));
        assert_eq!(bp_tree.fwd_search(1, -2), Some(23));
    }
}
