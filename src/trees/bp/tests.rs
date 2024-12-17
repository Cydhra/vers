use super::*;
use crate::BitVec;

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
