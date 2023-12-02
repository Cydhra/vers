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
fn test_empty_vec() {
    let bv = BitVec::new();

    assert_eq!(bv.get(0), None);
    assert_eq!(bv.get(1), None);
    assert_eq!(bv.get_bits(0, 0), None);
    assert_eq!(bv.get_bits(0, 10), None);
}

#[test]
fn test_alloc_ones() {
    let bv = BitVec::from_ones(42);
    assert_eq!(bv.len(), 42);
    assert_eq!(bv.data.len(), 1);

    (0..42).for_each(|i| assert_eq!(bv.get(i), Some(1), "mismatch at {}", i));
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

    let result = std::panic::catch_unwind(|| {
        let mut bv = BitVec::from_zeros(128);
        bv.append_bits(0, 65);
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

#[test]
fn drop_and_append_word_test() {
    let mut bv = BitVec::from_ones(64);
    bv.drop_last(32);
    bv.append_word(0);

    assert_eq!(bv.len(), 96);
    (0..32).for_each(|i| assert_eq!(bv.get(i), Some(1), "mismatch at {}", i));
    (32..96).for_each(|i| assert_eq!(bv.get(i), Some(0), "mismatch at {}", i));
}

#[test]
fn drop_and_append_stride_test() {
    let mut bv = BitVec::from_ones(64);
    bv.drop_last(32);
    bv.append_bits(0, 64);

    assert_eq!(bv.len(), 96);
    (0..32).for_each(|i| assert_eq!(bv.get(i), Some(1), "mismatch at {}", i));
    (32..96).for_each(|i| assert_eq!(bv.get(i), Some(0), "mismatch at {}", i));
}

#[test]
fn drop_and_append_bit_test() {
    let mut bv = BitVec::from_ones(64);
    bv.drop_last(32);
    bv.append(false);
    bv.append_bit(0);
    bv.append_bit_u8(0);
    bv.append_bit_u32(0);

    assert_eq!(bv.len(), 36);
    (0..32).for_each(|i| assert_eq!(bv.get(i), Some(1), "mismatch at {}", i));
    (32..36).for_each(|i| assert_eq!(bv.get(i), Some(0), "mismatch at {}", i));
}

#[test]
fn append_and_drop_test() {
    let mut bv = BitVec::from_zeros(1000);
    bv.append_word(420);
    bv.drop_last(420);
    assert_eq!(bv.len(), 1000 + 64 - 420);
    (0..1000 + 64 - 420).for_each(|i| assert_eq!(bv.get(i), Some(0), "mismatch at {}", i));
}

#[test]
fn test_drop_all_append() {
    let mut bv = BitVec::from_zeros(100);
    bv.drop_last(1000);
    bv.append_word(u64::MAX);
    assert_eq!(bv.len(), 64);
    (0..64).for_each(|i| assert_eq!(bv.get(i), Some(1), "mismatch at {}", i));
}

#[test]
fn test_iter() {
    let mut bv = BitVec::from_zeros(10);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);

    let mut iter = bv.iter();
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    let mut iter = bv.iter();
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next_back(), Some(1));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next_back(), Some(1));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next_back(), Some(1));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next_back(), None);

    let mut iter = bv.iter();
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next_back(), Some(1));
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_custom_iter_behavior() {
    let mut bv = BitVec::from_zeros(10);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);

    assert!(bv.iter().advance_by(10).is_ok());
    assert!(bv.iter().advance_back_by(10).is_ok());
    assert!(bv.iter().advance_by(11).is_err());
    assert!(bv.iter().advance_back_by(11).is_err());

    let mut iter = bv.iter();
    assert!(iter.advance_by(5).is_ok());
    assert!(iter.advance_back_by(6).is_err());
    assert!(iter.advance_by(6).is_err());
    assert!(iter.advance_back_by(5).is_ok());

    assert_eq!(bv.iter().skip(2).next(), Some(0));
    assert_eq!(bv.iter().count(), 10);
    assert_eq!(bv.iter().skip(2).count(), 8);
    assert_eq!(bv.iter().last(), Some(0));
    assert_eq!(bv.iter().nth(3), Some(1));
    assert_eq!(bv.iter().nth(12), None);

    assert_eq!(bv.clone().into_iter().skip(2).next(), Some(0));
    assert_eq!(bv.clone().into_iter().count(), 10);
    assert_eq!(bv.clone().into_iter().skip(2).count(), 8);
    assert_eq!(bv.clone().into_iter().last(), Some(0));
    assert_eq!(bv.clone().into_iter().nth(3), Some(1));
    assert_eq!(bv.clone().into_iter().nth(12), None);

    let mut iter = bv.iter();
    assert!(iter.advance_by(2).is_ok());
    assert!(iter.advance_back_by(4).is_ok());
    assert_eq!(iter.clone().collect::<Vec<_>>(), vec![0, 1, 0, 1]);
    assert_eq!(iter.clone().rev().collect::<Vec<_>>(), vec![1, 0, 1, 0]);
}

#[test]
fn test_empty_iter() {
    let bv = BitVec::from_zeros(0);
    let mut iter = bv.iter();
    assert_eq!(iter.clone().count(), 0);

    assert!(iter.next().is_none());
    assert!(iter.next_back().is_none());
    assert!(iter.nth(20).is_none());
    assert!(iter.nth_back(20).is_none());
    assert!(iter.advance_by(0).is_ok());
    assert!(iter.advance_back_by(0).is_ok());
    assert!(iter.advance_by(100).is_err());
    assert!(iter.advance_back_by(100).is_err());

    let bv = BitVec::from_zeros(1);
    let mut iter = bv.iter();
    assert_eq!(iter.clone().count(), 1);
    assert!(iter.advance_by(1).is_ok());
    assert_eq!(iter.clone().count(), 0);
    assert!(iter.advance_by(0).is_ok());
    assert!(iter.advance_back_by(0).is_ok());
    assert!(iter.advance_by(1).is_err());
    assert!(iter.advance_back_by(1).is_err());

    let bv = BitVec::from_ones(1);
    let mut iter = bv.iter();
    assert_eq!(iter.clone().count(), 1);
    assert!(iter.advance_back_by(1).is_ok());
    assert_eq!(iter.clone().count(), 0);
    assert!(iter.advance_back_by(0).is_ok());
    assert!(iter.advance_by(0).is_ok());
    assert!(iter.advance_back_by(1).is_err());
    assert!(iter.advance_by(1).is_err());
}

#[test]
fn test_get_bits() {
    let mut bv = BitVec::from_zeros(200);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);
    bv.flip_bit(197);
    bv.flip_bit(199);

    assert_eq!(bv.get_bits_unchecked(1, 3), 0b101);
    assert_eq!(bv.get_bits_unchecked(1, 4), 0b101);
    assert_eq!(bv.get_bits_unchecked(2, 2), 0b10);
    assert_eq!(bv.get_bits_unchecked(197, 3), 0b101);
    assert_eq!(bv.get_bits_unchecked(198, 2), 0b10);

    assert_eq!(bv.get_bits(0, 65), None);
    assert_eq!(bv.get_bits(300, 2), None);
    assert_eq!(bv.get_bits(190, 12), None);
    assert_eq!(bv.get_bits(0, 64), Some(0b101010));
}

#[test]
fn test_set_bit() {
    let mut bv = BitVec::from_zeros(1025);
    for i in (0..1025).step_by(25) {
        assert!(bv.set(i, 0x1).is_ok());
    }

    for i in 0..1025 {
        assert_eq!(bv.get(i), Some((i % 25 == 0) as u64));
    }
}

#[test]
fn test_count_bits() {
    let mut bv = BitVec::from_ones(2000);
    bv.flip_bit(24);
    bv.flip_bit(156);
    bv.flip_bit(1999);

    assert_eq!(bv.count_ones(), 1997);
    assert_eq!(bv.count_zeros(), 3);
}

#[test]
fn test_count_empty_vec() {
    let bv = BitVec::new();
    assert_eq!(bv.count_ones(), 0);
    assert_eq!(bv.count_zeros(), 0);
}

#[test]
fn test_masked() {
    let mut original = BitVec::from_zeros(100);
    assert!(original.set(30, 1).is_ok());
    assert!(original.set(31, 1).is_ok());
    assert!(original.set(32, 1).is_ok());

    let mut mask_or = BitVec::from_zeros(100);
    assert!(mask_or.set(29, 1).is_ok());

    let masked = original.mask_or(&mask_or).expect("failed to mask 'or'");
    assert_eq!(masked.get_bits(0, 28), Some(0));
    assert_eq!(masked.get_bits(29, 4), Some(15));
    assert_eq!(masked.get_bits(33, 64 - 33), Some(0));
    assert_eq!(masked.get_bits(64, 36), Some(0));

    let mut mask_and = BitVec::from_zeros(100);
    assert!(mask_and.set(31, 1).is_ok());

    let masked = original.mask_and(&mask_and).expect("failed to mask 'and'");
    assert_eq!(masked.get_bits(0, 31), Some(0));
    assert_eq!(masked.get_bits(31, 1), Some(1));
    assert_eq!(masked.get_bits(32, 64 - 32), Some(0));
    assert_eq!(masked.get_bits(64, 36), Some(0));

    let mut mask_xor = BitVec::from_zeros(100);
    assert!(mask_xor.set(30, 1).is_ok());
    assert!(mask_xor.set(70, 1).is_ok());

    let masked = original.mask_xor(&mask_xor).expect("failed to mask 'xor'");
    assert_eq!(masked.get_bits(0, 31), Some(0));
    assert_eq!(masked.get_bits(31, 2), Some(3));
    assert_eq!(masked.get_bits(33, 64 - 33), Some(0));
    assert_eq!(masked.get_bits(64, 36), Some(1 << 6));
}

#[test]
fn test_masked_empty_vec() {
    let bv = BitVec::new();
    let mask = BitVec::new();
    let bv = bv.mask_or(&mask).expect("failed to mask vector");

    assert_eq!(bv.count_ones(), 0);
    assert_eq!(bv.get(0), None);
    assert_eq!(bv.get(1), None);
    assert_eq!(bv.get_bits(0, 0), None);
    assert_eq!(bv.get_bits(0, 10), None);
}

#[test]
fn test_masked_get_bits() {
    let mut bv = BitVec::from_zeros(200);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);
    bv.flip_bit(197);
    bv.flip_bit(199);

    let mut mask = BitVec::from_zeros(200);
    mask.flip_bit(1);
    mask.flip_bit(2);
    mask.flip_bit(3);

    let bv = bv.mask_and(&mask).expect("failed to mask vector");

    assert_eq!(bv.get_bits_unchecked(1, 3), 0b101);
    assert_eq!(bv.get_bits_unchecked(1, 4), 0b101);
    assert_eq!(bv.get_bits_unchecked(2, 2), 0b10);
    assert_eq!(bv.get_bits_unchecked(197, 3), 0);
    assert_eq!(bv.get_bits_unchecked(198, 2), 0);

    assert_eq!(bv.get_bits(0, 65), None);
    assert_eq!(bv.get_bits(300, 2), None);
    assert_eq!(bv.get_bits(190, 12), None);
    assert_eq!(bv.get_bits(0, 64), Some(0b1010));
}

#[test]
fn test_masked_count_bits() {
    let mut bv = BitVec::from_ones(2000);
    bv.flip_bit(24);
    bv.flip_bit(156);
    bv.flip_bit(1999);

    let mut mask = BitVec::from_zeros(2000);
    mask.flip_bit(24);
    mask.flip_bit(1999);
    mask.flip_bit(0);

    let bv = bv.mask_and(&mask).expect("failed to mask vector");

    assert_eq!(bv.count_ones(), 1);
    assert_eq!(bv.count_zeros(), 1999);
}

#[test]
fn test_masked_to_bit_vec() {
    let mut bv = BitVec::from_zeros(5);
    bv.flip_bit(0);

    let mut bv2 = BitVec::from_zeros(5);
    bv2.flip_bit(1);

    let combined = bv
        .mask_or(&bv2)
        .expect("failed to mask vector")
        .to_bit_vec();
    assert_eq!(combined.get_bits(0, 2), Some(3));
    assert_eq!(combined.get_bits(2, 3), Some(0));
    assert_eq!(combined.len(), 5);
}

#[test]
fn test_from_bits() {
    let bv = BitVec::from_bits(&[1, 0, 1]);
    assert_eq!(bv.len, 3);
    assert_eq!(bv.get_bits(0, 3), Some(0b101));

    let bv = BitVec::from_bits_u64(&[1, 0, 1]);
    assert_eq!(bv.len, 3);
    assert_eq!(bv.get_bits(0, 3), Some(0b101));
}
