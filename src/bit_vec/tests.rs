use super::BitVec;

#[test]
fn simple_bit_vec_test() {
    let mut bv = BitVec::default();

    bv.append_word(2);
    assert_eq!(bv.len(), 64);

    bv.append_bit(1);
    assert_eq!(bv.len(), 65);

    assert_eq!(bv.get(0), Some(0));
    assert_eq!(bv.get(1), Some(1));
    (2..64).for_each(|i| assert_eq!(bv.get(i), Some(0), "mismatch at {}", i));
    assert_eq!(bv.get(64), Some(1));

    bv.flip_bit(1);
    (0..64).for_each(|i| assert_eq!(bv.get(i), Some(0), "mismatch after flip at {}", i));

    bv.drop_last(64);
    assert_eq!(bv.len(), 1);
    assert_eq!(bv.get(0), Some(0));
    assert_eq!(bv.get(1), None);
}

#[test]
fn test_alloc_ones() {
    let bv = BitVec::from_ones(42);
    assert_eq!(bv.len(), 42);
    assert_eq!(bv.data.len(), 1);

    (0..42).for_each(|i| assert_eq!(bv.get(i), Some(1), "mismatch at {}", i));

    // test that unused bits are still zero, because this is what the data structure assumes
    assert_eq!(bv.data[0] >> 42, 0);
}

#[test]
fn test_illegal_queries() {
    let bv = BitVec::from_zeros(128);
    assert_eq!(bv.len(), 128);

    assert!(bv.get(128).is_none());
    assert!(bv.get(129).is_none());

    let result = std::panic::catch_unwind(|| {
        let mut bv = BitVec::from_zeros(128);
        bv.flip_bit(128)
    });
    assert!(result.is_err());

    let result = std::panic::catch_unwind(|| {
        let mut bv = BitVec::from_zeros(128);
        bv.flip_bit(129)
    });
    assert!(result.is_err());
}

#[test]
fn drop_last_test() {
    let mut bv = BitVec::from_ones(128);
    bv.drop_last(65);
    assert_eq!(bv.len(), 63);
    assert_eq!(bv.data.len(), 1);
    (0..63).for_each(|i| assert_eq!(bv.get(i), Some(1), "mismatch at {}", i));

    bv.append_bits(0, 8);
    assert_eq!(bv.len(), 71);
    assert_eq!(bv.data.len(), 2);
    (0..63).for_each(|i| assert_eq!(bv.get(i), Some(1), "mismatch after append at {}", i));
    (63..71).for_each(|i| assert_eq!(bv.get(i), Some(0), "mismatch of appended bits at {}", i));

    bv.drop_last(128);
    assert_eq!(bv.len(), 0);
    assert_eq!(bv.data.len(), 0);
}