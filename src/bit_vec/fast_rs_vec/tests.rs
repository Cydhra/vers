use super::*;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
fn random_data_select() {
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

    for _ in 0..100 {
        // since we need a random rank, do not generate a number within the full length of
        // the vector, as only approximately half of the bits are set.
        let rnd_rank = rng.gen_range(0..LENGTH / 2 - BLOCK_SIZE);
        let actual_index0 = bv.select0(rnd_rank);

        let data = &bv.data;
        let mut rank_counter = 0;
        let mut expected_index0 = 0;

        let mut index = 0;
        loop {
            let zeros = data[index].count_zeros() as usize;
            if rank_counter + zeros > rnd_rank {
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
                if rank_counter == rnd_rank {
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
}
