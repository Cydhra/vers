use super::*;
use crate::RsVectorBuilder;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn test_append_bit() {
    let mut bv = RsVectorBuilder::new();
    bv.append_bit(0u8);
    bv.append_bit(1u8);
    bv.append_bit(1u8);
    let bv = bv.build();

    assert_eq!(bv.data[..1], vec![0b110]);
}

#[test]
fn test_random_data_rank() {
    let mut bv = RsVectorBuilder::with_capacity(LENGTH);
    let mut rng = StdRng::from_seed([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);
    let sample = Uniform::new(0, 2);
    static LENGTH: usize = 4 * SUPER_BLOCK_SIZE;

    for _ in 0..LENGTH {
        bv.append_bit(sample.sample(&mut rng) as u8);
    }

    let bv = bv.build();

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
    let mut bv = RsVectorBuilder::new();
    let len = SUPER_BLOCK_SIZE + 1;
    for _ in 0..len {
        bv.append_bit(0u8);
        bv.append_bit(1u8);
    }

    let bv = bv.build();

    assert_eq!(bv.len(), len * 2);
    assert_eq!(bv.rank0(2 * len - 1), len);
    assert_eq!(bv.rank1(2 * len - 1), len - 1);
}

#[test]
fn test_rank() {
    let mut bv = RsVectorBuilder::default();
    bv.append_bit(0u8);
    bv.append_bit(1u8);
    bv.append_bit(1u8);
    bv.append_bit(0u8);
    bv.append_bit(1u8);
    bv.append_bit(1u8);
    let bv = bv.build();

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
    let mut bv = RsVectorBuilder::default();
    bv.append_word(0);
    bv.append_bit(0u8);
    bv.append_bit(1u8);
    bv.append_bit(1u8);
    let bv = bv.build();

    assert_eq!(bv.rank0(63), 63);
    assert_eq!(bv.rank0(64), 64);
    assert_eq!(bv.rank0(65), 65);
    assert_eq!(bv.rank0(66), 65);
}

#[test]
fn test_only_zeros_rank() {
    let mut bv = RsVectorBuilder::default();
    for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
        bv.append_word(0);
    }
    bv.append_bit(0u8);
    let bv = bv.build();

    assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

    for i in 0..bv.len() {
        assert_eq!(bv.rank0(i), i);
        assert_eq!(bv.rank1(i), 0);
    }
}

#[test]
fn test_only_ones_rank() {
    let mut bv = RsVectorBuilder::default();
    for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
        bv.append_word(u64::MAX);
    }
    bv.append_bit(1u8);
    let bv = bv.build();

    assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

    for i in 0..bv.len() {
        assert_eq!(bv.rank0(i), 0);
        assert_eq!(bv.rank1(i), i);
    }
}

#[test]
fn test_simple_select() {
    let mut bv = RsVectorBuilder::default();
    bv.append_word(0b10110);
    let bv = bv.build();
    assert_eq!(bv.select0(0), 0);
    assert_eq!(bv.select1(1), 2);
    assert_eq!(bv.select0(1), 3);
}

#[test]
fn test_multi_words_select() {
    let mut bv = RsVectorBuilder::default();
    bv.append_word(0);
    bv.append_word(0);
    bv.append_word(0b10110);
    let bv = bv.build();
    assert_eq!(bv.select1(0), 129);
    assert_eq!(bv.select1(1), 130);
    assert_eq!(bv.select0(32), 32);
    assert_eq!(bv.select0(128), 128);
    assert_eq!(bv.select0(129), 131);
}

#[test]
fn test_only_zeros_select() {
    let mut bv = RsVectorBuilder::default();
    for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
        bv.append_word(0);
    }
    bv.append_bit(0u8);
    let bv = bv.build();

    assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

    for i in 0..bv.len() {
        assert_eq!(bv.select0(i), i);
    }
}

#[test]
fn test_only_ones_select() {
    let mut bv = RsVectorBuilder::default();

    for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
        bv.append_word(u64::MAX);
    }
    bv.append_bit(1u8);
    let bv = bv.build();

    assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

    for i in 0..bv.len() {
        assert_eq!(bv.select1(i), i);
    }
}

#[test]
fn random_data_select() {
    let mut bv = RsVectorBuilder::with_capacity(LENGTH);
    let mut rng = StdRng::from_seed([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);
    let sample = Uniform::new(0, 2);
    static LENGTH: usize = 4 * SUPER_BLOCK_SIZE;

    for _ in 0..LENGTH {
        bv.append_bit(sample.sample(&mut rng) as u8);
    }

    let bv = bv.build();

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
    let mut bv = RsVectorBuilder::default();
    bv.append_word(0b10110);
    bv.append_word(0b10110);
    bv.append_word(0b10110);
    bv.append_word(0b10110);
    let bv = bv.build();
    assert_eq!(bv.rank0, 4 * 61);
    assert_eq!(bv.rank1, 4 * 3);

    let mut bv = RsVectorBuilder::default();
    for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
        bv.append_word(0b10110);
    }
    bv.append_bit(0u8);
    bv.append_bit(1u8);
    bv.append_bit(0u8);

    let bv = bv.build();

    assert_eq!(bv.rank0, 2 * (SUPER_BLOCK_SIZE / WORD_SIZE) * 61 + 2);
    assert_eq!(bv.rank1, 2 * (SUPER_BLOCK_SIZE / WORD_SIZE) * 3 + 1);
}

#[test]
fn test_large_query() {
    let mut bv = RsVectorBuilder::default();
    bv.append_bit(1u8);
    bv.append_bit(0u8);
    let bv = bv.build();

    assert_eq!(bv.rank0(10), 1);
    assert_eq!(bv.rank1(10), 1);
}
