use super::*;
use crate::BitVec;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

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
    let bv = BitVec::from_bits(&[1, 1, 0, 1, 0, 0]);
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
    let bv = BitVec::from_bits(&[1, 1, 1, 1, 0, 0, 0, 0]);
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

    for excess in 1..64 {
        assert_eq!(tree.fwd_search(0, excess), Some(excess as usize));
    }

    let bv = BitVec::from_bits(&[0; 64]);
    let tree = BpTree::<512>::from_bit_vector(bv);

    for excess in 1..64 {
        assert_eq!(tree.fwd_search(0, -excess), Some(excess as usize));
    }
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
    let bv = BitVec::from_bits(&[1, 1, 0, 1, 0, 0]);
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
    let bv = BitVec::from_bits(&[1, 1, 1, 1, 0]);
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
            let absolute_excess = if node_handle == 0 {
                0
            } else {
                bp.excess(node_handle - 1) + relative_excess
            };
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
    // test whether `node_index` and `node_handle` return correct indices / node handles.

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
        .map(|window| window[0] == 1 && window[1] == 0)
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
fn test_is_ancestor() {
    // (()((())()))
    // ab cde  f
    let bits = vec![1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0];
    let bv = BitVec::from_bits(&bits);
    let tree = BpTree::<8>::from_bit_vector(bv);
    let a = tree.root().unwrap();
    let b = tree.first_child(a).unwrap();
    let c = tree.next_sibling(b).unwrap();
    let d = tree.first_child(c).unwrap();
    let e = tree.first_child(d).unwrap();
    let f = tree.next_sibling(d).unwrap();

    assert!(tree.is_ancestor(a, b).unwrap());
    assert!(tree.is_ancestor(a, c).unwrap());
    assert!(tree.is_ancestor(a, d).unwrap());
    assert!(tree.is_ancestor(a, e).unwrap());
    assert!(tree.is_ancestor(a, f).unwrap());

    assert!(!tree.is_ancestor(b, a).unwrap());
    assert!(!tree.is_ancestor(b, c).unwrap());
    assert!(tree.is_ancestor(c, d).unwrap());
    assert!(tree.is_ancestor(c, e).unwrap());
    assert!(tree.is_ancestor(c, f).unwrap());
    assert!(!tree.is_ancestor(f, e).unwrap());
    assert!(!tree.is_ancestor(e, d).unwrap());

    assert!(tree.is_ancestor(a, a).unwrap());
    assert!(tree.is_ancestor(b, b).unwrap());
}

#[test]
fn test_root() {
    let bv = BitVec::from_bits(&[
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    let tree = BpTree::<8>::from_bit_vector(bv);
    assert_eq!(tree.root(), Some(0));
    assert_eq!(tree.previous_sibling(0), None);
    assert_eq!(tree.next_sibling(0), None);

    let tree = BpTree::<16>::from_bit_vector(BitVec::new());
    assert_eq!(tree.root(), None);
}

#[test]
fn test_level_ancestor() {
    let bv = BitVec::from_bits(&[1, 1, 1, 0, 0, 1, 0, 0]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    assert_eq!(tree.level_ancestor(2, 0), Some(2));
    assert_eq!(tree.level_ancestor(2, 1), Some(1));
    assert_eq!(tree.level_ancestor(2, 2), Some(0));
    assert_eq!(tree.level_ancestor(2, 3), None);

    assert_eq!(tree.level_ancestor(0, 1), None);
    assert_eq!(tree.level_ancestor(5, 1), Some(0));
    assert_eq!(tree.level_ancestor(5, 2), None);
}

#[test]
fn test_level_next() {
    let bv = BitVec::from_bits(&[
        1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, // intentionally unbalanced
    ]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    assert_eq!(tree.level_next(0), None); // unbalanced query
    assert_eq!(tree.level_next(1), Some(5));
    assert_eq!(tree.level_next(2), Some(8));
    assert_eq!(tree.level_next(5), Some(7));
    assert_eq!(tree.level_next(7), None);
    assert_eq!(tree.level_next(8), None);
}

#[test]
fn test_level_prev() {
    let bv = BitVec::from_bits(&[1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    assert_eq!(tree.level_prev(0), None);
    assert_eq!(tree.level_prev(1), None);
    assert_eq!(tree.level_prev(2), None);
    assert_eq!(tree.level_prev(5), Some(1));
    assert_eq!(tree.level_prev(7), Some(5));
    assert_eq!(tree.level_prev(8), Some(2));
    assert_eq!(tree.level_prev(11), Some(7));
    assert_eq!(tree.level_prev(12), Some(8));
    assert_eq!(tree.level_prev(13), None);
}

#[test]
fn test_level_leftmost() {
    let bv = BitVec::from_bits(&[1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    assert_eq!(tree.level_leftmost(0), Some(0));
    assert_eq!(tree.level_leftmost(1), Some(1));
    assert_eq!(tree.level_leftmost(2), Some(2));
    assert_eq!(tree.level_leftmost(3), Some(13));
    assert_eq!(tree.level_leftmost(4), None);
    assert_eq!(tree.level_leftmost(10), None);
}

#[test]
fn test_level_rightmost() {
    let bv = BitVec::from_bits(&[1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    assert_eq!(tree.level_rightmost(0), Some(0));
    assert_eq!(tree.level_rightmost(1), Some(11));
    assert_eq!(tree.level_rightmost(2), Some(12));
    assert_eq!(tree.level_rightmost(3), Some(13));
    assert_eq!(tree.level_rightmost(4), None);
    assert_eq!(tree.level_rightmost(10), None);
}

#[test]
fn test_subtree_size() {
    let bv = BitVec::from_bits(&[1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    assert_eq!(tree.subtree_size(0), Some(9));
    assert_eq!(tree.subtree_size(1), Some(2));
    assert_eq!(tree.subtree_size(2), Some(1));
    assert_eq!(tree.subtree_size(5), Some(1));
    assert_eq!(tree.subtree_size(7), Some(2));
    assert_eq!(tree.subtree_size(8), Some(1));
    assert_eq!(tree.subtree_size(11), Some(3));
    assert_eq!(tree.subtree_size(12), Some(2));
    assert_eq!(tree.subtree_size(13), Some(1));
}

#[test]
fn test_malformed_tree_positive() {
    // test that an unbalanced expression doesn't panic.
    // most results are meaningless, but we don't want to panic and leave the data structure
    // for further queries in a consistent state.

    // the tree has not enough closing brackets
    let bv = BitVec::from_bits(&[1, 1, 1, 0, 1, 1, 0, 1, 1, 0]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    test_all_functions(&tree);
}

#[test]
fn test_malformed_tree_negative() {
    // test that an unbalanced expression doesn't panic.
    // most results are meaningless, but we don't want to panic and leave the data structure
    // for further queries in a consistent state.

    // the tree has too many closing brackets
    let bv = BitVec::from_bits(&[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]);
    let tree = BpTree::<4>::from_bit_vector(bv);

    test_all_functions(&tree);
}

// helper function to run all functions on a tree once, without any asserts.
fn test_all_functions(tree: &BpTree<4>) {
    tree.root();

    for i in 0..tree.vec.len() {
        tree.fwd_search(i, 0);
        tree.bwd_search(i, 0);
        tree.fwd_search(i, 1);
        tree.bwd_search(i, 1);
        tree.close(i);
        tree.open(i);
        tree.enclose(i);

        if tree.vec.get(i).unwrap() == OPEN_PAREN {
            tree.parent(i);
            tree.first_child(i);
            tree.last_child(i);
            tree.previous_sibling(i);
            tree.next_sibling(i);
            tree.is_leaf(i);
            tree.depth(i);
            tree.is_ancestor(i, i);

            for j in 0..i {
                if tree.vec.get(j).unwrap() == OPEN_PAREN {
                    tree.is_ancestor(i, j);
                }
            }

            tree.level_ancestor(i, 0);
            tree.level_next(i);
            tree.level_prev(i);
            tree.level_leftmost(i as u64);
            tree.level_rightmost(i as u64);
            tree.subtree_size(i);
        }
    }
}

#[test]
fn fuzz_tree_navigation() {
    // fuzzing the tree navigation operations on an unbalanced tree
    // because those are easier to generate uniformly.

    const L: usize = 1 << 14;
    const L_BITS: usize = L * size_of::<u64>() * 8;

    // we generate a vector using a seeded random generator and check that every query works as expected
    let mut rng = StdRng::from_seed([0; 32]);
    let mut bit_vec = BitVec::with_capacity(L_BITS);

    for _ in 0..L / 64 {
        bit_vec.append_word(rng.next_u64());
    }

    let tree = BpTree::<32>::from_bit_vector(bit_vec.clone());
    let mut parent_stack = Vec::new();

    // keep track of last sibling for each node
    let mut last_sibling_stack = Vec::new();
    last_sibling_stack.push(None);

    // keep track of how many siblings we encountered on the current node level yet
    let mut sibling_count_stack = Vec::new();

    tree.vec.iter().enumerate().for_each(|(idx, bit)| {
        if bit == OPEN_PAREN {
            assert_eq!(tree.parent(idx), parent_stack.last().copied());
            assert_eq!(
                tree.previous_sibling(idx),
                last_sibling_stack.last().copied().unwrap()
            );

            if let Some(num_child) = sibling_count_stack.last() {
                let mut child = tree.first_child(tree.parent(idx).unwrap()).unwrap();
                for _ in 0..*num_child {
                    child = tree.next_sibling(child).unwrap();
                }
                assert_eq!(child, idx);
            }

            parent_stack.push(idx);
            last_sibling_stack.push(None);
            sibling_count_stack.push(0);
        } else {
            let last_parent = parent_stack.pop();

            // check the last child, and previous_sibling
            if let Some(parent) = last_parent {
                let mut child = tree.first_child(parent);
                let mut reverse_child = tree.last_child(parent);
                let num_children = *sibling_count_stack.last().unwrap();

                for _ in 1..num_children {
                    child = tree.next_sibling(child.unwrap());
                    reverse_child = tree.previous_sibling(reverse_child.unwrap());
                }

                assert_eq!(child, tree.last_child(parent));
                assert_eq!(reverse_child, tree.first_child(parent));
            }

            // pop the last sibling element for the current level, and pop the last sibling element for the parent level,
            // replacing it with the parent node
            last_sibling_stack.pop();
            last_sibling_stack.pop();
            last_sibling_stack.push(last_parent);

            // update sibling count
            sibling_count_stack.pop();
            if let Some(counter) = sibling_count_stack.pop() {
                sibling_count_stack.push(counter + 1);
            }
        }
    });
}
