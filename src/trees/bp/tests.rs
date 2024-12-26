use super::*;
use crate::BitVec;
use rand::{RngCore, SeedableRng};
use rand::rngs::StdRng;

#[test]
fn test_fwd_search() {
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

#[test]
fn test_fwd_single_block() {
    #[rustfmt::skip]
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 0, 0,
    ]);

    let bp_tree = BpTree::<512>::from_bit_vector(bv);

    assert_eq!(bp_tree.fwd_search(3, -1), Some(4));
    assert_eq!(bp_tree.fwd_search(2, -1), Some(5));
    assert_eq!(bp_tree.fwd_search(12, -1), Some(13));
    assert_eq!(bp_tree.fwd_search(20, -1), Some(21));
    assert_eq!(bp_tree.fwd_search(0, -1), Some(23));
    assert_eq!(bp_tree.fwd_search(1, -1), Some(22));
    assert_eq!(bp_tree.fwd_search(3, 0), Some(7));
    assert_eq!(bp_tree.fwd_search(3, -2), Some(5));
    assert_eq!(bp_tree.fwd_search(1, -2), Some(23));
}

#[test]
fn test_fwd_illegal_queries() {
    #[rustfmt::skip]
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 0, 0,
    ]);

    let tree = BpTree::<8>::from_bit_vector(bv.clone());

    assert_eq!(tree.fwd_search(24, 0), None);
    assert_eq!(tree.fwd_search(25, 0), None);

    assert_eq!(tree.fwd_search(0, -2), None);
    assert_eq!(tree.fwd_search(22, 1), None);

    let tree = BpTree::<64>::from_bit_vector(bv);

    assert_eq!(tree.fwd_search(24, 0), None);
    assert_eq!(tree.fwd_search(25, 0), None);

    assert_eq!(tree.fwd_search(0, -2), None);
    assert_eq!(tree.fwd_search(22, 1), None);
}

#[test]
fn test_fwd_unbalanced_expression() {
    // test whether forward search works with unbalanced parenthesis expressions
    #[rustfmt::skip]
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1,
    ]);

    let tree = BpTree::<8>::from_bit_vector(bv);

    assert_eq!(tree.fwd_search(0, -1), Some(13));
    assert_eq!(tree.fwd_search(1, -1), Some(12));
    assert_eq!(tree.fwd_search(7, -1), Some(8));
    assert_eq!(tree.fwd_search(16, 1), None);
    assert_eq!(tree.fwd_search(5, 2), Some(7));
}

#[test]
fn test_fwd_block_boundary() {
    let bv = BitVec::from_bits(&[1, 1, 0, 1, 0, 0,]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    // test if a query returns the correct result if the result is the first bit in a block
    // and not in the initial block
    assert_eq!(tree.fwd_search(3, -1), Some(4));

    // test if the query returns the correct result if the result is the last bit in a block
    // and not in the initial block
    assert_eq!(tree.fwd_search(3, -2), Some(5));
}

#[test]
fn test_fwd_negative_block() {
    let bv = BitVec::from_bits(&[1, 1, 1, 1, 0, 0, 0, 0,]);
    let tree = BpTree::<2>::from_bit_vector(bv);

    // regression: test if a query correctly returns none (instead of crashing) if the following
    // block has a negative maximum excess (as a previous bug clamped it to 0).
    assert_eq!(tree.fwd_search(3, 0), None);
}

#[test]
fn test_fwd_last_element() {
    // a query beginning from the last element cannot return a valid result, but
    // the binary mM tree right of it may be uninitialized, and so not ending the query early
    // may yield invalid results or break assertions
    #[rustfmt::skip]
    let bv = BitVec::from_bits(&[
        1, 1, 0, 1,  0, 1, 0, 1,
        0, 1, 0, 1,  0, 1, 0, 1,
        0, 1, 0, 1,  0, 1, 0, 1,
    ]);

    let tree = BpTree::<4>::from_bit_vector(bv);
    assert!(tree.fwd_search(23, 0).is_none());
}

#[test]
fn test_lookup_extreme_pop() {
    // test whether a table lookup works if the bit pattern is only ones or only zeros
    let bv = BitVec::from_bits(&[1; 64]);
    let tree = BpTree::<512>::from_bit_vector(bv);

    assert_eq!(tree.fwd_search(0, 40), Some(40));

    let bv = BitVec::from_bits(&[0; 64]);
    let tree = BpTree::<512>::from_bit_vector(bv);

    assert_eq!(tree.fwd_search(0, -40), Some(40));
}

#[test]
fn test_fwd_fuzzy() {
    // we're fuzzing forward search a bit
    const L: usize = 1000;
    const L_BITS: usize = L * size_of::<u64>() * 8;

    // we generate a vector using a seeded random generator and check that every query works as expected
    let mut rng = StdRng::from_seed([0; 32]);
    let mut bit_vec = BitVec::with_capacity(L_BITS);

    for _ in 0..L {
        bit_vec.append_word(rng.next_u64());
    }

    // pre-calculate all absolute excess values
    let mut excess_values = vec![0i16; L_BITS];
    let mut excess = 0;
    for (idx, bit) in bit_vec.iter().enumerate() {
        if bit == 1 {
            excess += 1;
            excess_values[idx] = excess;
        } else {
            excess -= 1;
            excess_values[idx] = excess;
        }
    }

    let bp = BpTree::<128>::from_bit_vector(bit_vec);

    // test any query from valid nodes with the given relative excess values
    for relative_excess in [-3, -2, -1, 0, 1, 2, 3] {
        for node_handle in bp.vec.iter1() {
            let absolute_excess = bp.excess(node_handle) + relative_excess;
            let expected = excess_values[node_handle + 1..]
                .iter()
                .position(|&excess| excess as i64 == absolute_excess)
                .map(|i| i + node_handle + 1);
            let actual = bp.fwd_search(node_handle, relative_excess);
            assert_eq!(
                expected,
                actual,
                "fwd search returned wrong position for relative excess {} searching from index ({}): expected ({:?}) but got ({:?}).",
                relative_excess,
                node_handle,
                expected,
                actual,
            )
        }
    }
}

#[test]
fn test_bwd_search() {
    #[rustfmt::skip]
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 0, 0,
    ]);

    let bp_tree = BpTree::<8>::from_bit_vector(bv);

    // search within block
    assert_eq!(bp_tree.bwd_search(4, -1), Some(3));
    assert_eq!(bp_tree.bwd_search(5, -1), Some(2));
    assert_eq!(bp_tree.bwd_search(13, -1), Some(12));
    assert_eq!(bp_tree.bwd_search(21, -1), Some(20));

    // search across blocks
    assert_eq!(bp_tree.bwd_search(23, -1), Some(0));
    assert_eq!(bp_tree.bwd_search(22, -1), Some(1));

    // search with weird relative excess
    assert_eq!(bp_tree.bwd_search(7, 0), Some(5));
    assert_eq!(bp_tree.bwd_search(5, -2), Some(1));
    assert_eq!(bp_tree.bwd_search(23, 0), Some(1));
}

#[test]
fn test_bwd_single_block() {
    #[rustfmt::skip]
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 0, 0,
    ]);

    let bp_tree = BpTree::<512>::from_bit_vector(bv);

    assert_eq!(bp_tree.bwd_search(4, -1), Some(3));
    assert_eq!(bp_tree.bwd_search(5, -1), Some(2));
    assert_eq!(bp_tree.bwd_search(13, -1), Some(12));
    assert_eq!(bp_tree.bwd_search(21, -1), Some(20));
    assert_eq!(bp_tree.bwd_search(23, -1), Some(0));
    assert_eq!(bp_tree.bwd_search(22, -1), Some(1));
    assert_eq!(bp_tree.bwd_search(7, 0), Some(5));
    assert_eq!(bp_tree.bwd_search(5, -2), Some(1));
    assert_eq!(bp_tree.bwd_search(23, 0), Some(1));
}

#[test]
fn test_bwd_illegal_queries() {
    #[rustfmt::skip]
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 0, 0,
    ]);

    let tree = BpTree::<8>::from_bit_vector(bv.clone());

    assert_eq!(tree.bwd_search(0, 0), None);
    assert_eq!(tree.bwd_search(1, 0), None);

    assert_eq!(tree.bwd_search(23, -2), None);
    assert_eq!(tree.bwd_search(22, -3), None);

    let tree = BpTree::<64>::from_bit_vector(bv);

    assert_eq!(tree.bwd_search(0, 0), None);
    assert_eq!(tree.bwd_search(1, 0), None);

    assert_eq!(tree.bwd_search(23, -2), None);
    assert_eq!(tree.bwd_search(22, -3), None);
}

#[test]
fn test_bwd_left_block_boundary() {
    // test if a query returns the correct result if the result is the first bit after
    // a block boundary (the left-most one even for backward search)
    let bv = BitVec::from_bits(&[1, 1, 0, 1, 0, 0,]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    assert_eq!(tree.bwd_search(5, 0), Some(3));
}

#[test]
fn test_bwd_right_block_boundary() {
    #[rustfmt::skip]
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0,
    ]);

    let bp_tree = BpTree::<4>::from_bit_vector(bv);

    // test the correct result is returned if result is exactly at a right block boundary
    assert_eq!(bp_tree.bwd_search(11, -1), Some(4));
}

#[test]
fn test_bwd_block_traversal() {
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 0,
    ]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    // if we request excess 0 backwards at a block boundary
    // we test if that actually traverses the vector instead of reporting
    // back the input index (which happens if the mM tree is queried without modification
    // of relative excess, which happens at block boundaries)
    assert_eq!(tree.bwd_search(4, 0), None);
}

#[test]
fn test_bwd_fuzzy() {
    // we're fuzzing forward search a bit
    const L: usize = 1000;
    const L_BITS: usize = L * size_of::<u64>() * 8;

    // we generate a vector using a seeded random generator and check that every query works as expected
    let mut rng = StdRng::from_seed([0; 32]);
    let mut bit_vec = BitVec::with_capacity(L_BITS);

    for _ in 0..L {
        bit_vec.append_word(rng.next_u64());
    }

    // pre-calculate all absolute excess values
    let mut excess_values = vec![0i16; L_BITS + 1];
    let mut excess = 0;
    for (idx, bit) in bit_vec.iter().enumerate() {
        if bit == 1 {
            excess += 1;
            excess_values[idx + 1] = excess;
        } else {
            excess -= 1;
            excess_values[idx + 1] = excess;
        }
    }


    let bp = BpTree::<128>::from_bit_vector(bit_vec);

    // test any query from valid nodes with the given relative excess values
    for relative_excess in [-3, -2, -1, 0, 1, 2, 3] {
        for node_handle in bp.vec.iter0() {
            let absolute_excess = if node_handle == 0 { 0 } else { bp.excess(node_handle - 1) + relative_excess };
            let expected = excess_values[..node_handle]
                .iter()
                .rposition(|&excess| excess as i64 == absolute_excess);

            let actual = bp.bwd_search(node_handle, relative_excess);
            assert_eq!(
                expected,
                actual,
                "bwd search returned wrong position for relative excess {} searching from index ({}): expected ({:?}) but got ({:?}).",
                relative_excess,
                node_handle,
                expected,
                actual,
            )
        }
    }
}

#[test]
fn test_close() {
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);

    let tree = BpTree::<8>::from_bit_vector(bv);

    for i in 0..24 {
        assert_eq!(tree.close(i), Some(47 - i));
    }

    assert_eq!(tree.close(100), None);
}

#[test]
fn test_open() {
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);

    let tree = BpTree::<8>::from_bit_vector(bv);

    for i in 24..48 {
        assert_eq!(tree.open(i), Some(47 - i));
    }

    assert_eq!(tree.open(100), None);
}

#[test]
fn test_enclose() {
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);

    let tree = BpTree::<8>::from_bit_vector(bv);

    for i in 1..24 {
        assert_eq!(tree.enclose(i), Some(i - 1));
    }

    assert_eq!(tree.enclose(0), None);

    for i in 24..46 {
        assert_eq!(tree.enclose(i), Some(46 - i));
    }

    assert_eq!(tree.enclose(47), None);

    assert_eq!(tree.enclose(100), None);
}

#[test]
fn test_parent() {
    let bv = BitVec::from_bits(&[
        1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    ]);

    let tree = BpTree::<8>::from_bit_vector(bv.clone());

    assert_eq!(tree.excess(27), 0, "tree is not balanced");

    let mut stack = Vec::new();
    let mut head = None;
    for (idx, bit) in bv.iter().enumerate() {
        if bit == 1 {
            assert_eq!(
                tree.parent(idx),
                head,
                "parent of node {} is incorrect",
                idx
            );
            stack.push(head);
            head = Some(idx);
        } else {
            head = stack.pop().expect("stack underflow despite balanced tree");
        }
    }
}

#[test]
fn test_children() {
    let bv = BitVec::from_bits(&[1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]);

    let tree = BpTree::<8>::from_bit_vector(bv);

    assert_eq!(tree.excess(17), 0, "tree is not balanced");
    assert_eq!(tree.first_child(0), Some(1));
    assert_eq!(tree.previous_sibling(1), None);
    assert_eq!(tree.next_sibling(1), Some(7));
    assert_eq!(tree.previous_sibling(7), Some(1));
    assert_eq!(tree.next_sibling(7), Some(9));
    assert_eq!(tree.previous_sibling(9), Some(7));
    assert_eq!(tree.next_sibling(9), Some(11));
    assert_eq!(tree.previous_sibling(11), Some(9));
    assert_eq!(tree.next_sibling(11), None);
    assert_eq!(tree.last_child(0), Some(11));

    assert_eq!(tree.first_child(11), Some(12));
    assert_eq!(tree.next_sibling(12), Some(14));
    assert_eq!(tree.previous_sibling(14), Some(12));
    assert_eq!(tree.next_sibling(14), None);
    assert_eq!(tree.last_child(11), Some(14));

    assert_eq!(tree.first_child(12), None);
    assert_eq!(tree.last_child(12), None);

    assert_eq!(tree.first_child(14), None);
    assert_eq!(tree.last_child(14), None);
}

#[test]
fn test_contiguous_index() {
    let bv = BitVec::from_bits(&[1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]);
    let tree = BpTree::<4>::from_bit_vector(bv.clone());
    let rs: RsVec = bv.into();

    for (rank, index_in_bv) in rs.iter1().enumerate() {
        assert_eq!(tree.node_index(index_in_bv), rank);
        assert_eq!(tree.node_handle(rank), index_in_bv);
    }
}

#[test]
fn test_depth() {
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);

    let mut depth = 0;

    let tree = BpTree::<8>::from_bit_vector(bv.clone());
    for i in 0..24 {
        if bv.get(i) == Some(1) {
            assert_eq!(tree.depth(i), depth);
            depth += 1;
        } else {
            depth -= 1;
        }
    }
}

#[test]
fn test_is_leaf() {
    let bits = vec![
        1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
    ];
    let bv = BitVec::from_bits(&bits);
    let leaves = bits[..]
        .windows(2)
        .map(|window| {
            if window[0] == 1 && window[1] == 0 {
                true
            } else {
                false
            }
        })
        .collect::<Vec<_>>();
    let tree = BpTree::<8>::from_bit_vector(bv.clone());

    for (idx, is_leaf) in leaves.iter().enumerate() {
        // if the bit is 1, check if that node is a leaf. If it's 0, it's not a valid node handle.
        if bits[idx] == 1 {
            assert_eq!(tree.is_leaf(idx), *is_leaf);
        }
    }
}

#[test]
fn test_root() {
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    let tree = BpTree::<8>::from_bit_vector(bv);
    assert_eq!(tree.root(), Some(0));

    let tree = BpTree::<16>::from_bit_vector(BitVec::new());
    assert_eq!(tree.root(), None);
}
