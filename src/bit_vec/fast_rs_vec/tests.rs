use super::*;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::num::NonZeroUsize;

#[test]
fn test_append_bit() {
    let mut bv = BitVec::new();
    bv.append_bit_u8(0u8);
    bv.append_bit_u8(1u8);
    bv.append_bit_u8(1u8);
    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.data[..1], vec![0b110]);
}

#[test]
fn test_random_data_rank() {
    let mut bv = BitVec::with_capacity(LENGTH);
    let mut rng = StdRng::from_seed([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);
    let sample = Uniform::new(0, 2);
    static LENGTH: usize = 4 * SUPER_BLOCK_SIZE;

    for _ in 0..LENGTH {
        bv.append_bit(sample.sample(&mut rng));
    }

    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.len(), LENGTH);

    for _ in 0..100 {
        let rnd_index = rng.gen_range(0..LENGTH);
        let actual_rank1 = bv.rank1(rnd_index);
        let actual_rank0 = bv.rank0(rnd_index);

        let data = &bv.data;
        let mut expected_rank1 = 0;
        let mut expected_rank0 = 0;

        let data_index = rnd_index / WORD_SIZE;
        let bit_index = rnd_index % WORD_SIZE;

        for i in 0..data_index {
            expected_rank1 += data[i].count_ones() as usize;
            expected_rank0 += data[i].count_zeros() as usize;
        }

        if bit_index > 0 {
            expected_rank1 += (data[data_index] & (1 << bit_index) - 1).count_ones() as usize;
            expected_rank0 += (!data[data_index] & (1 << bit_index) - 1).count_ones() as usize;
        }

        assert_eq!(actual_rank1, expected_rank1);
        assert_eq!(actual_rank0, expected_rank0);
    }
}

#[test]
fn test_append_bit_long() {
    let mut bv = BitVec::new();
    let len = SUPER_BLOCK_SIZE + 1;
    for _ in 0..len {
        bv.append_bit_u8(0u8);
        bv.append_bit_u8(1u8);
    }

    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.len(), len * 2);
    assert_eq!(bv.rank0(2 * len - 1), len);
    assert_eq!(bv.rank1(2 * len - 1), len - 1);
}

#[test]
fn test_rank() {
    let mut bv = BitVec::default();
    bv.append_bit_u8(0u8);
    bv.append_bit_u8(1u8);
    bv.append_bit_u8(1u8);
    bv.append_bit_u8(0u8);
    bv.append_bit_u8(1u8);
    bv.append_bit_u8(1u8);
    let bv = RsVec::from_bit_vec(bv);

    // first bit must always have rank 0
    assert_eq!(bv.rank0(0), 0);
    assert_eq!(bv.rank1(0), 0);

    assert_eq!(bv.rank1(2), 1);
    assert_eq!(bv.rank1(3), 2);
    assert_eq!(bv.rank1(4), 2);
    assert_eq!(bv.rank0(3), 1);
}

#[test]
fn test_multi_words_rank() {
    let mut bv = BitVec::default();
    bv.append_word(0);
    bv.append_bit_u8(0u8);
    bv.append_bit_u8(1u8);
    bv.append_bit_u8(1u8);
    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.rank0(63), 63);
    assert_eq!(bv.rank0(64), 64);
    assert_eq!(bv.rank0(65), 65);
    assert_eq!(bv.rank0(66), 65);
}

#[test]
fn test_only_zeros_rank() {
    let mut bv = BitVec::default();
    for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
        bv.append_word(0);
    }
    bv.append_bit_u8(0u8);
    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

    for i in 0..bv.len() {
        assert_eq!(bv.rank0(i), i);
        assert_eq!(bv.rank1(i), 0);
    }
}

#[test]
fn test_only_ones_rank() {
    let mut bv = BitVec::default();
    for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
        bv.append_word(u64::MAX);
    }
    bv.append_bit_u8(1u8);
    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

    for i in 0..bv.len() {
        assert_eq!(bv.rank0(i), 0);
        assert_eq!(bv.rank1(i), i);
    }
}

#[test]
fn test_simple_select() {
    let mut bv = BitVec::default();
    bv.append_word(0b10110);
    let bv = RsVec::from_bit_vec(bv);
    assert_eq!(bv.select0(0), 0);
    assert_eq!(bv.select1(1), 2);
    assert_eq!(bv.select0(1), 3);
}

#[test]
fn test_multi_words_select() {
    let mut bv = BitVec::default();
    bv.append_word(0);
    bv.append_word(0);
    bv.append_word(0b10110);
    let bv = RsVec::from_bit_vec(bv);
    assert_eq!(bv.select1(0), 129);
    assert_eq!(bv.select1(1), 130);
    assert_eq!(bv.select0(32), 32);
    assert_eq!(bv.select0(128), 128);
    assert_eq!(bv.select0(129), 131);
}

#[test]
fn test_only_zeros_select() {
    let mut bv = BitVec::default();
    for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
        bv.append_word(0);
    }
    bv.append_bit_u8(0u8);
    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

    for i in 0..bv.len() {
        assert_eq!(bv.select0(i), i);
    }
}

#[test]
fn test_only_ones_select() {
    let mut bv = BitVec::default();

    for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
        bv.append_word(u64::MAX);
    }
    bv.append_bit_u8(1u8);
    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

    for i in 0..bv.len() {
        assert_eq!(bv.select1(i), i);
    }
}

#[test]
fn random_data_select0() {
    let mut bv = BitVec::with_capacity(LENGTH);
    let mut rng = StdRng::from_seed([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);
    let sample = Uniform::new(0, 2);
    static LENGTH: usize = 4 * SUPER_BLOCK_SIZE;

    for _ in 0..LENGTH {
        bv.append_bit_u8(sample.sample(&mut rng) as u8);
    }

    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.len(), LENGTH);

    for _ in 0..500 {
        let rnd_rank0 = rng.gen_range(0..bv.rank0);
        let actual_index0 = bv.select0(rnd_rank0);

        let data = &bv.data;
        let mut rank_counter = 0;
        let mut expected_index0 = 0;

        let mut index = 0;
        loop {
            let zeros = data[index].count_zeros() as usize;
            if rank_counter + zeros > rnd_rank0 {
                break;
            } else {
                rank_counter += zeros;
                expected_index0 += WORD_SIZE;
                index += 1;
            }
        }

        let mut bit_index = 0;
        loop {
            if data[index] & (1 << bit_index) == 0 {
                if rank_counter == rnd_rank0 {
                    break;
                } else {
                    rank_counter += 1;
                }
            }
            expected_index0 += 1;
            bit_index += 1;
        }

        assert_eq!(actual_index0, expected_index0);
    }
}

#[test]
fn random_data_select1() {
    let mut bv = BitVec::with_capacity(LENGTH);
    let mut rng = StdRng::from_seed([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);
    let sample = Uniform::new(0, 2);
    static LENGTH: usize = 4 * SUPER_BLOCK_SIZE;

    for _ in 0..LENGTH {
        bv.append_bit_u8(sample.sample(&mut rng) as u8);
    }

    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.len(), LENGTH);

    for _ in 0..500 {
        let rnd_rank1 = rng.gen_range(0..bv.rank1);
        let actual_index1 = bv.select1(rnd_rank1);

        let data = &bv.data;
        let mut rank_counter = 0;
        let mut expected_index1 = 0;

        let mut index = 0;
        loop {
            let ones = data[index].count_ones() as usize;
            if rank_counter + ones > rnd_rank1 {
                break;
            } else {
                rank_counter += ones;
                expected_index1 += WORD_SIZE;
                index += 1;
            }
        }

        let mut bit_index = 0;
        loop {
            if data[index] & (1 << bit_index) > 0 {
                if rank_counter == rnd_rank1 {
                    break;
                } else {
                    rank_counter += 1;
                }
            }
            expected_index1 += 1;
            bit_index += 1;
        }

        assert_eq!(actual_index1, expected_index1);
    }
}

#[test]
fn test_total_ranks() {
    let mut bv = BitVec::default();
    bv.append_word(0b10110);
    bv.append_word(0b10110);
    bv.append_word(0b10110);
    bv.append_word(0b10110);
    let bv = RsVec::from_bit_vec(bv);
    assert_eq!(bv.rank0, 4 * 61);
    assert_eq!(bv.rank1, 4 * 3);

    let mut bv = BitVec::default();
    for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
        bv.append_word(0b10110);
    }
    bv.append_bit_u8(0u8);
    bv.append_bit_u8(1u8);
    bv.append_bit_u8(0u8);

    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.rank0, 2 * (SUPER_BLOCK_SIZE / WORD_SIZE) * 61 + 2);
    assert_eq!(bv.rank1, 2 * (SUPER_BLOCK_SIZE / WORD_SIZE) * 3 + 1);
}

#[test]
fn test_large_query() {
    let mut bv = BitVec::default();
    bv.append_bit_u8(1u8);
    bv.append_bit_u8(0u8);
    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.rank0(10), 1);
    assert_eq!(bv.rank1(10), 1);
}

// test ranking and selecting positions and data on and outside of the vector boundaries
#[test]
fn test_large_selects() {
    let mut bit_vec = BitVec::new();
    bit_vec.append_bit(0);
    bit_vec.append_word(u64::MAX);

    let rs_vec = RsVec::from_bit_vec(bit_vec);
    assert_eq!(rs_vec.rank1(65), 64);
    assert_eq!(rs_vec.rank1(66), 64);

    assert_eq!(rs_vec.select0(1), 65); // select0(1) is not in the vector
    assert_eq!(rs_vec.select1(63), 64); // last 1 in the vector
    assert_eq!(rs_vec.select1(64), 65); // select1(64) is not in the vector
    assert_eq!(rs_vec.select1(65), 65); // select1(65) is not in the vector
}

#[test]
fn test_empty_vec() {
    let bv = BitVec::new();
    let rs_vec = RsVec::from_bit_vec(bv);
    assert_eq!(rs_vec.len(), 0);
    assert_eq!(rs_vec.rank0(0), 0);
    assert_eq!(rs_vec.rank1(0), 0);
    assert_eq!(rs_vec.select0(0), 0);
    assert_eq!(rs_vec.select1(0), 0);
}

#[test]
fn test_block_boundaries() {
    fn test_ranks(mut bv: BitVec) {
        bv.flip_bit(10);
        bv.flip_bit(11);
        bv.flip_bit(bv.len() - 11);
        bv.flip_bit(bv.len() - 10);
        let rs_vec = RsVec::from_bit_vec(bv);

        assert_eq!(rs_vec.rank0(10), 10);
        assert_eq!(rs_vec.rank0(11), 10);

        assert_eq!(rs_vec.rank1(10), 0);
        assert_eq!(rs_vec.rank1(11), 1);

        assert_eq!(rs_vec.rank1(rs_vec.len() - 2), 4);
        assert_eq!(rs_vec.rank1(rs_vec.len() - 1), 4);
        assert_eq!(rs_vec.rank1(rs_vec.len()), 4);

        assert_eq!(rs_vec.rank0(rs_vec.len() - 2), rs_vec.len() - 6);
        assert_eq!(rs_vec.rank0(rs_vec.len() - 1), rs_vec.len() - 5);
        assert_eq!(rs_vec.rank0(rs_vec.len()), rs_vec.len() - 4);

        assert_eq!(rs_vec.select1(0), 10);
        assert_eq!(rs_vec.select1(1), 11);
        assert_eq!(rs_vec.select1(2), rs_vec.len() - 11);
        assert_eq!(rs_vec.select1(3), rs_vec.len() - 10);
    }

    let bv = BitVec::from_zeros(BLOCK_SIZE - 1);
    test_ranks(bv);

    let bv = BitVec::from_zeros(BLOCK_SIZE);
    test_ranks(bv);

    let bv = BitVec::from_zeros(BLOCK_SIZE + 1);
    test_ranks(bv);

    let bv = BitVec::from_zeros(SUPER_BLOCK_SIZE - 1);
    test_ranks(bv);

    let bv = BitVec::from_zeros(SUPER_BLOCK_SIZE);
    test_ranks(bv);

    let bv = BitVec::from_zeros(SUPER_BLOCK_SIZE + 1);
    test_ranks(bv);

    let bv = BitVec::from_zeros(SUPER_BLOCK_SIZE + 10);
    test_ranks(bv);

    let bv = BitVec::from_zeros(SUPER_BLOCK_SIZE + 11);
    test_ranks(bv);
}

#[test]
fn test_iter() {
    let mut bv = BitVec::from_zeros(10);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);
    let rs = RsVec::from_bit_vec(bv);

    let mut iter = rs.iter();
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

    let mut iter = rs.iter();
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

    let mut iter = rs.iter();
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
    let rs = RsVec::from_bit_vec(bv);

    assert!(rs.iter().advance_by(10).is_ok());
    assert!(rs.iter().advance_back_by(10).is_ok());
    assert!(rs.iter().advance_by(11).is_err());
    assert!(rs.iter().advance_back_by(11).is_err());

    let mut iter = rs.iter();
    assert!(iter.advance_by(5).is_ok());
    assert!(iter.advance_back_by(6).is_err());
    assert!(iter.advance_by(6).is_err());
    assert!(iter.advance_back_by(5).is_ok());

    assert_eq!(rs.iter().skip(2).next(), Some(0));
    assert_eq!(rs.iter().count(), 10);
    assert_eq!(rs.iter().skip(2).count(), 8);
    assert_eq!(rs.iter().last(), Some(0));
    assert_eq!(rs.iter().nth(3), Some(1));
    assert_eq!(rs.iter().nth(12), None);

    assert_eq!(rs.clone().into_iter().skip(2).next(), Some(0));
    assert_eq!(rs.clone().into_iter().count(), 10);
    assert_eq!(rs.clone().into_iter().skip(2).count(), 8);
    assert_eq!(rs.clone().into_iter().last(), Some(0));
    assert_eq!(rs.clone().into_iter().nth(3), Some(1));
    assert_eq!(rs.clone().into_iter().nth(12), None);

    let mut iter = rs.iter();
    assert_eq!(iter.nth(2), Some(0));
    assert_eq!(iter.nth_back(4), Some(1));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.clone().count(), 1);
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.count(), 0);

    let mut iter = rs.iter();
    assert!(iter.advance_by(2).is_ok());
    assert!(iter.advance_back_by(4).is_ok());
    assert_eq!(iter.clone().collect::<Vec<_>>(), vec![0, 1, 0, 1]);
    assert_eq!(iter.clone().rev().collect::<Vec<_>>(), vec![1, 0, 1, 0]);
}

#[test]
fn test_empty_iter() {
    let bv = BitVec::from_zeros(0);
    let rs = RsVec::from_bit_vec(bv);
    let mut iter = rs.iter();
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
    let rs = RsVec::from_bit_vec(bv);
    let mut iter = rs.iter();
    assert_eq!(iter.clone().count(), 1);
    assert!(iter.advance_by(1).is_ok());
    assert_eq!(iter.clone().count(), 0);
    assert!(iter.advance_by(0).is_ok());
    assert!(iter.advance_back_by(0).is_ok());
    assert!(iter.advance_by(1).is_err());
    assert!(iter.advance_back_by(1).is_err());

    let bv = BitVec::from_ones(1);
    let rs = RsVec::from_bit_vec(bv);
    let mut iter = rs.iter();
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
    let mut bv = BitVec::from_zeros(2 * SUPER_BLOCK_SIZE);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);
    bv.flip_bit(SUPER_BLOCK_SIZE - 1);
    bv.flip_bit(SUPER_BLOCK_SIZE + 1);
    let rs = RsVec::from_bit_vec(bv);

    assert_eq!(rs.get_bits_unchecked(1, 3), 0b101);
    assert_eq!(rs.get_bits_unchecked(1, 4), 0b101);
    assert_eq!(rs.get_bits_unchecked(2, 2), 0b10);
    assert_eq!(rs.get_bits_unchecked(SUPER_BLOCK_SIZE - 1, 3), 0b101);
    assert_eq!(rs.get_bits_unchecked(SUPER_BLOCK_SIZE, 3), 0b10);

    assert_eq!(rs.get_bits(0, 65), None);
    assert_eq!(rs.get_bits(3 * SUPER_BLOCK_SIZE, 2), None);
    assert_eq!(rs.get_bits(2 * SUPER_BLOCK_SIZE - 10, 12), None);
    assert_eq!(rs.get_bits(0, 64), Some(0b101010));
}

#[test]
fn test_select1_iterator() {
    let mut bv = BitVec::from_zeros(2 * SUPER_BLOCK_SIZE);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);
    bv.flip_bit(BLOCK_SIZE);
    bv.flip_bit(BLOCK_SIZE + 1);
    bv.flip_bit(SUPER_BLOCK_SIZE - 1);
    bv.flip_bit(SUPER_BLOCK_SIZE);
    bv.flip_bit(SUPER_BLOCK_SIZE + 1);
    let rs = RsVec::from_bit_vec(bv);

    let mut iter = rs.iter1();
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(5));
    assert_eq!(iter.next(), Some(BLOCK_SIZE));
    assert_eq!(iter.next(), Some(BLOCK_SIZE + 1));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE - 1));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE + 1));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_select0_iterator() {
    let mut bv = BitVec::from_ones(2 * SUPER_BLOCK_SIZE);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);
    bv.flip_bit(BLOCK_SIZE);
    bv.flip_bit(BLOCK_SIZE + 1);
    bv.flip_bit(SUPER_BLOCK_SIZE - 1);
    bv.flip_bit(SUPER_BLOCK_SIZE);
    bv.flip_bit(SUPER_BLOCK_SIZE + 1);
    let rs = RsVec::from_bit_vec(bv);

    let mut iter = rs.iter0();
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(5));
    assert_eq!(iter.next(), Some(BLOCK_SIZE));
    assert_eq!(iter.next(), Some(BLOCK_SIZE + 1));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE - 1));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE + 1));
    assert_eq!(iter.next(), None);
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512bw",
))]
#[test]
fn test_select1_bs_iterator() {
    let mut bv = BitVec::from_zeros(2 * SUPER_BLOCK_SIZE);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);
    bv.flip_bit(BLOCK_SIZE);
    bv.flip_bit(BLOCK_SIZE + 1);
    bv.flip_bit(SUPER_BLOCK_SIZE - 1);
    bv.flip_bit(SUPER_BLOCK_SIZE);
    bv.flip_bit(SUPER_BLOCK_SIZE + 1);
    let rs = RsVec::from_bit_vec(bv);

    let mut iter = rs.bit_set_iter1();
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(5));
    assert_eq!(iter.next(), Some(BLOCK_SIZE));
    assert_eq!(iter.next(), Some(BLOCK_SIZE + 1));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE - 1));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE + 1));
    assert_eq!(iter.next(), None);
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512bw",
))]
#[test]
fn test_select0_bs_iterator() {
    let mut bv = BitVec::from_ones(2 * SUPER_BLOCK_SIZE);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);
    bv.flip_bit(BLOCK_SIZE);
    bv.flip_bit(BLOCK_SIZE + 1);
    bv.flip_bit(SUPER_BLOCK_SIZE - 1);
    bv.flip_bit(SUPER_BLOCK_SIZE);
    bv.flip_bit(SUPER_BLOCK_SIZE + 1);
    let rs = RsVec::from_bit_vec(bv);

    let mut iter = rs.bit_set_iter0();
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(5));
    assert_eq!(iter.next(), Some(BLOCK_SIZE));
    assert_eq!(iter.next(), Some(BLOCK_SIZE + 1));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE - 1));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE));
    assert_eq!(iter.next(), Some(SUPER_BLOCK_SIZE + 1));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_empty_select_iterator() {
    let bv = BitVec::new();
    let rs = RsVec::from_bit_vec(bv);
    let mut iter = rs.iter1();
    assert_eq!(iter.clone().count(), 0);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_select_iter_custom_impls() {
    let mut bv = BitVec::from_ones(2 * SUPER_BLOCK_SIZE);
    bv.flip_bit(1);
    bv.flip_bit(3);
    bv.flip_bit(5);
    bv.flip_bit(BLOCK_SIZE);
    bv.flip_bit(BLOCK_SIZE + 1);
    bv.flip_bit(SUPER_BLOCK_SIZE - 1);
    bv.flip_bit(SUPER_BLOCK_SIZE);
    bv.flip_bit(SUPER_BLOCK_SIZE + 1);
    let rs = RsVec::from_bit_vec(bv);

    let iter = rs.iter0();
    assert_eq!(iter.clone().last(), Some(SUPER_BLOCK_SIZE + 1));
    assert_eq!(iter.clone().nth(2), Some(5));
    assert_eq!(iter.clone().nth(10), None);
    assert_eq!(iter.clone().advance_by(2), Ok(()));
    assert_eq!(
        iter.clone().advance_by(10),
        Err(NonZeroUsize::new(2).unwrap())
    );
    assert_eq!(iter.count(), 8);
}

#[test]
fn test_sparse_equals() {
    for a in 0..u8::MAX as u64 {
        for b in 0..u8::MAX as u64 {
            let mut bv1 = BitVec::with_capacity(8);
            let mut bv2 = BitVec::with_capacity(8);
            bv1.append_bits(a, 9);
            bv2.append_bits(b, 9);
            let rs1 = RsVec::from_bit_vec(bv1);
            let rs2 = RsVec::from_bit_vec(bv2);

            assert_eq!(
                rs1.sparse_equals::<true>(&rs2),
                a == b,
                "sparse_equals::0 gives wrong result for a = {}, b = {}",
                a,
                b
            );
            assert_eq!(
                rs1.sparse_equals::<false>(&rs2),
                a == b,
                "sparse_equals::1 gives wrong result for a = {}, b = {}",
                a,
                b
            );
        }
    }

    let mut bv = BitVec::from_zeros(2 * SUPER_BLOCK_SIZE);
    bv.flip_bit(1);

    let rs1 = RsVec::from_bit_vec(bv.clone());
    let rs2 = RsVec::from_bit_vec(bv.clone());

    assert_eq!(rs1.sparse_equals::<false>(&rs2), true);
    assert_eq!(rs1.sparse_equals::<true>(&rs2), true);

    bv.flip_bit(3);
    let rs2 = RsVec::from_bit_vec(bv.clone());

    assert_eq!(rs1.sparse_equals::<false>(&rs2), false);
    assert_eq!(rs1.sparse_equals::<true>(&rs2), false);

    bv.flip_bit(3);
    bv.flip_bit(2 * SUPER_BLOCK_SIZE - 1);
    let rs1 = RsVec::from_bit_vec(bv.clone());

    assert_eq!(rs1.sparse_equals::<false>(&rs2), false);
    assert_eq!(rs1.sparse_equals::<true>(&rs2), false);
}

#[test]
fn test_full_equals() {
    for a in 0..u8::MAX as u64 {
        for b in 0..u8::MAX as u64 {
            let mut bv1 = BitVec::with_capacity(8);
            let mut bv2 = BitVec::with_capacity(8);
            bv1.append_bits(a, 9);
            bv2.append_bits(b, 9);
            let rs1 = RsVec::from_bit_vec(bv1);
            let rs2 = RsVec::from_bit_vec(bv2);

            assert_eq!(
                rs1.full_equals(&rs2),
                a == b,
                "full_equals gives wrong result for a = {}, b = {}",
                a,
                b
            );
        }
    }

    let mut bv = BitVec::from_zeros(2 * SUPER_BLOCK_SIZE);
    bv.flip_bit(1);

    let rs1 = RsVec::from_bit_vec(bv.clone());
    let rs2 = RsVec::from_bit_vec(bv.clone());

    assert_eq!(rs1.full_equals(&rs2), true);

    bv.flip_bit(3);
    let rs2 = RsVec::from_bit_vec(bv.clone());

    assert_eq!(rs1.full_equals(&rs2), false);

    bv.flip_bit(3);
    bv.flip_bit(2 * SUPER_BLOCK_SIZE - 1);
    let rs1 = RsVec::from_bit_vec(bv.clone());

    assert_eq!(rs1.full_equals(&rs2), false);
}

// fuzzing test for iter1 and iter0 as last ditch fail-safe
#[test]
fn test_random_data_iter() {
    let mut rng = StdRng::from_seed([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);

    for fill_ratio in [10, 50, 90] {
        for length in [
            BLOCK_SIZE / 2,
            BLOCK_SIZE,
            SUPER_BLOCK_SIZE,
            4 * SUPER_BLOCK_SIZE,
        ] {
            for _ in 0..20 {
                let mut bv = BitVec::with_capacity(length);
                let sample = Uniform::new(0, 100);
                for _ in 0..length {
                    bv.append_bit((sample.sample(&mut rng) < fill_ratio) as u64);
                }

                let bv = RsVec::from_bit_vec(bv);
                let output_on_bits: Vec<_> = bv.iter1().collect();
                let output_off_bits: Vec<_> = bv.iter0().collect();

                for idx in output_on_bits {
                    assert_eq!(bv.get(idx), Some(1), "bit {} is not 1", idx);
                }

                for idx in output_off_bits {
                    assert_eq!(bv.get(idx), Some(0), "bit {} is not 0", idx);
                }
            }
        }
    }
}

// test a randomly generated bit vector for correct values in blocks
#[test]
fn test_block_layout() {
    static LENGTH: usize = 4 * SUPER_BLOCK_SIZE;
    let mut bv = BitVec::with_capacity(LENGTH);
    let mut rng = StdRng::from_seed([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);
    let sample = Uniform::new(0, 2);

    for _ in 0..LENGTH {
        bv.append_bit(sample.sample(&mut rng));
    }

    let bv = RsVec::from_bit_vec(bv);
    assert_eq!(bv.len(), LENGTH);

    let mut zero_counter = 0u32;
    for (block_index, block) in bv.blocks.iter().enumerate() {
        if block_index % (SUPER_BLOCK_SIZE / BLOCK_SIZE) == 0 {
            zero_counter = 0;
        }
        assert_eq!(
            zero_counter,
            block.zeros as u32,
            "zero count mismatch in block {} of {}",
            block_index,
            bv.blocks.len()
        );
        for word in bv.data[block_index * BLOCK_SIZE / WORD_SIZE..]
            .iter()
            .take(BLOCK_SIZE / WORD_SIZE)
        {
            zero_counter += word.count_zeros();
        }
    }
}

// Github issue https://github.com/Cydhra/vers/issues/6 regression test
#[test]
fn test_iter1_regression_i6() {
    static LENGTH: usize = 4 * SUPER_BLOCK_SIZE;
    let mut bv = BitVec::with_capacity(LENGTH);
    let mut rng = StdRng::from_seed([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);
    let sample = Uniform::new(0, 2);

    for _ in 0..LENGTH {
        bv.append_bit(sample.sample(&mut rng));
    }

    let bv = RsVec::from_bit_vec(bv);

    assert_eq!(bv.len(), LENGTH);

    for bit1 in bv.iter1() {
        assert_eq!(bv.get(bit1), Some(1));
    }

    for bit0 in bv.iter0() {
        assert_eq!(bv.get(bit0), Some(0));
    }

    let mut all_bits: Vec<_> = bv.iter0().chain(bv.iter1()).collect();
    all_bits.sort();
    assert_eq!(all_bits.len(), LENGTH);
}
