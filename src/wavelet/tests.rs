use super::*;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::{max, min};

#[test]
fn test_wavelet_encoding_pc() {
    let data = vec![1, 5, 3];
    let wavelet = WaveletMatrix::from_bit_vec_pc(&BitVec::pack_sequence_u8(&data, 3), 3);

    assert_eq!(wavelet.len(), 3);
    assert_eq!(
        wavelet.get_value_unchecked(0),
        BitVec::pack_sequence_u8(&[1], 3)
    );
    assert_eq!(
        wavelet.get_value_unchecked(1),
        BitVec::pack_sequence_u8(&[5], 3)
    );
    assert_eq!(
        wavelet.get_value_unchecked(2),
        BitVec::pack_sequence_u8(&[3], 3)
    );
}

#[test]
fn test_wavelet_encoding_randomized() {
    let mut rng = StdRng::from_seed([1; 32]);

    for _ in 0..100 {
        let data: Vec<u8> = (0..rng.gen_range(500..1000))
            .map(|_| rng.gen_range(0..=u8::MAX))
            .collect();
        let data_u64: Vec<u64> = data.iter().map(|&x| x as u64).collect();
        let wavelet = WaveletMatrix::from_bit_vec(&BitVec::pack_sequence_u8(&data, 8), 8);
        let wavelet_from_slice = WaveletMatrix::from_slice(&data_u64, 8);
        let wavelet_prefix_counting =
            WaveletMatrix::from_bit_vec_pc(&BitVec::pack_sequence_u8(&data, 8), 8);

        assert_eq!(wavelet.len(), data.len());

        for (i, v) in data.iter().enumerate() {
            assert_eq!(wavelet.get_u64_unchecked(i), *v as u64);
            assert_eq!(wavelet_from_slice.get_u64_unchecked(i), *v as u64);
            assert_eq!(wavelet_prefix_counting.get_u64_unchecked(i), *v as u64);
        }
    }
}

#[test]
fn test_wavelet_encoding_large_alphabet() {
    let mut data = BitVec::pack_sequence_u64(&[0, 0, u64::MAX, 0, 1, 2], 127);

    // write 127 bits into the fourth element
    for i in 0..127 {
        data.set(127 * 3 + i, 1).unwrap();
    }

    let wavelet = WaveletMatrix::from_bit_vec(&data, 127);
    assert_eq!(wavelet.len(), 6);

    assert_eq!(wavelet.get_value_unchecked(0), BitVec::from_zeros(127));
    assert_eq!(wavelet.get_value_unchecked(1), BitVec::from_zeros(127));
    assert_eq!(
        wavelet.get_value_unchecked(2),
        BitVec::pack_sequence_u64(&[u64::MAX], 127)
    );
    assert_eq!(wavelet.get_value_unchecked(3), BitVec::from_ones(127));
    assert_eq!(
        wavelet.get_value_unchecked(4),
        BitVec::pack_sequence_u8(&[1], 127)
    );
    assert_eq!(
        wavelet.get_value_unchecked(5),
        BitVec::pack_sequence_u8(&[2], 127)
    );
}

#[test]
fn test_rank_range() {
    let data = BitVec::pack_sequence_u64(&[1, 4, 4, 1, 3, 1, 4, 3, 2, 0], 4);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    let symbol_0 = BitVec::from_zeros(4);
    assert_eq!(wavelet.rank_range_unchecked(0..10, &symbol_0), 1);
    assert_eq!(wavelet.rank_range_unchecked(0..8, &symbol_0), 0);
    assert_eq!(wavelet.rank_range_unchecked(0..9, &symbol_0), 0);
    assert_eq!(wavelet.rank_range_unchecked(9..10, &symbol_0), 1);

    let symbol_1 = BitVec::pack_sequence_u8(&[1], 4);
    assert_eq!(wavelet.rank_range_unchecked(0..10, &symbol_1), 3);
    assert_eq!(wavelet.rank_range_unchecked(0..6, &symbol_1), 3);
    assert_eq!(wavelet.rank_range_unchecked(0..5, &symbol_1), 2);
    assert_eq!(wavelet.rank_range_unchecked(5..6, &symbol_1), 1);

    let symbol_5 = BitVec::pack_sequence_u8(&[5], 4);
    assert_eq!(wavelet.rank_range_unchecked(0..10, &symbol_5), 0);
    assert_eq!(wavelet.rank_range_unchecked(0..5, &symbol_5), 0);
    assert_eq!(wavelet.rank_range_unchecked(5..10, &symbol_5), 0);
    assert_eq!(wavelet.rank_range_unchecked(1..2, &symbol_5), 0);

    // test if out-of-bounds range returns None
    assert_eq!(wavelet.rank_range(0..11, &symbol_0), None);
    assert_eq!(wavelet.rank_range(10..10, &symbol_1), None);

    // test if empty range returns 0
    assert_eq!(wavelet.rank_range_unchecked(9..9, &symbol_0), 0);
}

#[test]
fn test_empty_vec_rank() {
    let data = BitVec::new();
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    assert_eq!(wavelet.len(), 0);
    assert_eq!(wavelet.rank_range(0..0, &BitVec::from_zeros(4)), None);
    assert_eq!(wavelet.rank_range(0..10, &BitVec::from_zeros(4)), None);
    assert_eq!(wavelet.rank_offset(10, 0, &BitVec::from_zeros(4)), None);
    assert_eq!(wavelet.rank_offset(0, 10, &BitVec::from_zeros(4)), None);
    assert_eq!(wavelet.rank_offset(0, 0, &BitVec::from_zeros(4)), None);
    assert_eq!(wavelet.rank(0, &BitVec::from_zeros(4)), Some(0));
    assert_eq!(wavelet.rank(10, &BitVec::from_zeros(4)), None);
}

#[test]
fn test_rank_randomized() {
    let mut rng = StdRng::from_seed([100; 32]);

    let data: Vec<u8> = (0..1000).map(|_| rng.gen_range(0..=u8::MAX)).collect();

    let wavelet = WaveletMatrix::from_bit_vec(&BitVec::pack_sequence_u8(&data, 8), 8);

    let mut symbols = data.clone();
    symbols.sort();
    symbols.dedup();

    for symbol in symbols {
        let symbol_bit_vec = BitVec::pack_sequence_u8(&[symbol], 8);
        let mut rank = 0;
        for (i, v) in data.iter().enumerate() {
            assert_eq!(wavelet.rank_unchecked(i, &symbol_bit_vec), rank);
            if *v == symbol {
                rank += 1;
            }
        }
    }
}

#[test]
fn test_rank_bounds() {
    let data = BitVec::pack_sequence_u64(&[1, 4, 4, 1, 3, 1, 4, 3, 2, 0], 4);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    let long_data = BitVec::from_zeros(90);
    let long_wavelet = WaveletMatrix::from_bit_vec(&long_data, 90);

    let symbol_0 = BitVec::from_zeros(4);

    assert_eq!(wavelet.rank_offset(0, 11, &symbol_0), None);
    assert_eq!(wavelet.rank_offset_u64(0, 11, 0), None);
    assert_eq!(wavelet.rank_offset(11, 0, &symbol_0), None);
    assert_eq!(wavelet.rank_offset_u64(11, 0, 0), None);
    assert_eq!(wavelet.rank_offset(11, 11, &symbol_0), None);
    assert_eq!(wavelet.rank_offset_u64(11, 11, 0), None);
    assert_eq!(wavelet.rank_offset(0, 0, &symbol_0), Some(0));
    assert_eq!(wavelet.rank_offset_u64(0, 0, 0), Some(0));

    assert_eq!(wavelet.rank_range(0..11, &symbol_0), None);
    assert_eq!(wavelet.rank_range_u64(0..11, 0), None);
    assert_eq!(wavelet.rank_range(0..0, &symbol_0), Some(0));
    assert_eq!(wavelet.rank_range_u64(0..0, 0), Some(0));
    assert_eq!(wavelet.rank_range(11..11, &symbol_0), None);
    assert_eq!(wavelet.rank_range_u64(11..11, 0), None);
    assert_eq!(long_wavelet.rank_offset(0, 1, &long_data), Some(1));
    assert_eq!(long_wavelet.rank_offset_u64(0, 1, 0), None);
}

#[test]
fn test_select_range() {
    let data = BitVec::pack_sequence_u64(&[1, 4, 4, 1, 3, 1, 4, 3, 2, 0], 4);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    let symbol_0 = BitVec::from_zeros(4);
    let symbol_7 = BitVec::from_ones(4);
    let symbol_4 = BitVec::pack_sequence_u8(&[4], 4);

    assert_eq!(wavelet.select_offset_unchecked(0, 0, &symbol_0), 9);
    assert_eq!(wavelet.select_offset_unchecked(0, 1, &symbol_0), 10);
    assert_eq!(wavelet.select_offset_unchecked(0, 0, &symbol_7), 10);
    assert_eq!(wavelet.select_offset_unchecked(0, 1, &symbol_4), 2);
    assert_eq!(wavelet.select_offset_unchecked(2, 0, &symbol_4), 2);
    assert_eq!(wavelet.select_offset_unchecked(2, 1, &symbol_4), 6);
    assert_eq!(wavelet.select_offset_unchecked(2, 9, &symbol_4), 10);
}

#[test]
fn test_select_bounds() {
    let data = BitVec::pack_sequence_u64(&[1, 4, 4, 1, 3, 1, 4, 3, 2, 0], 4);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    let long_data = BitVec::from_zeros(90);
    let long_wavelet = WaveletMatrix::from_bit_vec(&long_data, 90);

    let symbol_0 = BitVec::from_zeros(4);
    let symbol_error = BitVec::from_ones(4);

    assert_eq!(wavelet.select_offset(10, 0, &symbol_0), None);
    assert_eq!(wavelet.select_offset_u64(10, 0, 0), None);
    assert_eq!(wavelet.select_offset(0, 11, &symbol_0), None);
    assert_eq!(wavelet.select_offset_u64(0, 11, 0), None);
    assert_eq!(wavelet.select_offset(0, 0, &symbol_error), None);
    assert_eq!(wavelet.select_offset_u64(0, 0, 15), None);

    assert_eq!(wavelet.select(10, &symbol_0), None);
    assert_eq!(wavelet.select_u64(10, 0), None);
    assert_eq!(wavelet.select(0, &symbol_error), None);
    assert_eq!(wavelet.select_u64(0, 15), None);

    assert_eq!(long_wavelet.select_offset(0, 0, &long_data), Some(0));
    assert_eq!(long_wavelet.select_offset_u64(0, 1, 0), None);
}

#[test]
fn test_quantile() {
    let mut sequence = [1, 4, 4, 1, 3, 1, 4, 3, 2, 0];
    let data = BitVec::pack_sequence_u64(&sequence, 4);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    sequence.sort();

    for (i, v) in sequence.iter().enumerate() {
        assert_eq!(
            wavelet.quantile(0..10, i),
            Some(BitVec::pack_sequence_u8(&[*v as u8], 4))
        );
        assert_eq!(wavelet.quantile_u64(0..10, i), Some(*v));
    }

    assert_eq!(wavelet.quantile(0..10, 10), None);

    assert_eq!(
        wavelet.quantile(0..5, 0),
        Some(BitVec::pack_sequence_u8(&[1], 4))
    );
    assert_eq!(wavelet.quantile_u64(0..5, 0), Some(1));
    assert_eq!(
        wavelet.quantile(1..5, 0),
        Some(BitVec::pack_sequence_u8(&[1], 4))
    );
    assert_eq!(wavelet.quantile_u64(1..5, 0), Some(1));
    assert_eq!(
        wavelet.quantile(1..4, 1),
        Some(BitVec::pack_sequence_u8(&[4], 4))
    );
    assert_eq!(wavelet.quantile_u64(1..4, 1), Some(4));
    assert_eq!(
        wavelet.quantile(1..5, 1),
        Some(BitVec::pack_sequence_u8(&[3], 4))
    );
    assert_eq!(wavelet.quantile_u64(1..5, 1), Some(3));
}

#[test]
fn test_quantile_randomized() {
    let mut rng = StdRng::from_seed([100; 32]);

    let data: Vec<u8> = (0..1000).map(|_| rng.gen_range(0..=u8::MAX)).collect();

    let wavelet = WaveletMatrix::from_bit_vec(&BitVec::pack_sequence_u8(&data, 8), 8);

    for _ in 0..1000 {
        let range_i = rng.gen_range(0..data.len());
        let range_j = rng.gen_range(0..data.len());
        let range = min(range_i, range_j)..max(range_i, range_j);

        let k = if range.is_empty() {
            0
        } else {
            rng.gen_range(range.clone()) - range.start
        };

        let mut range_data = data[range.clone()].to_vec();
        range_data.sort_unstable();

        assert_eq!(
            wavelet.quantile_u64(range.clone(), k),
            if range.is_empty() {
                None
            } else {
                Some(range_data[k] as u64)
            }
        );
        assert_eq!(
            wavelet.range_max_u64(range.clone()),
            range_data.last().map(|&x| x as u64)
        );
        assert_eq!(
            wavelet.range_min_u64(range.clone()),
            range_data.first().map(|&x| x as u64)
        );
        assert_eq!(
            wavelet.range_median_u64(range.clone()),
            if range.is_empty() {
                None
            } else {
                Some(range_data[(range_data.len() - 1) / 2] as u64)
            }
        );
    }
}

// test bounds for the quantile-convenience functions
#[test]
fn test_convenience_bounds() {
    let data = BitVec::pack_sequence_u64(&[1, 4, 4, 1, 3, 1, 4, 3, 2, 0], 4);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    let long_data = BitVec::from_zeros(90);
    let long_wavelet = WaveletMatrix::from_bit_vec(&long_data, 90);

    assert_eq!(
        wavelet.range_max(0..10),
        Some(BitVec::pack_sequence_u8(&[4], 4))
    );
    assert_eq!(
        wavelet.range_max_unchecked(0..10),
        BitVec::pack_sequence_u8(&[4], 4)
    );
    assert_eq!(wavelet.range_max_u64(0..10), Some(4));
    assert_eq!(wavelet.range_max_u64_unchecked(0..10), 4);
    assert_eq!(wavelet.range_max(0..11), None);
    assert_eq!(wavelet.range_max_u64(0..11), None);
    assert_eq!(wavelet.range_max(1..1), None);
    assert_eq!(wavelet.range_max_u64(1..1), None);
    assert_eq!(long_wavelet.range_max(0..1), Some(long_data.clone()));
    assert_eq!(long_wavelet.range_max_u64(0..1), None);

    assert_eq!(
        wavelet.range_min(0..10),
        Some(BitVec::pack_sequence_u8(&[0], 4))
    );
    assert_eq!(
        wavelet.range_min_unchecked(0..10),
        BitVec::pack_sequence_u8(&[0], 4)
    );
    assert_eq!(wavelet.range_min_u64(0..10), Some(0));
    assert_eq!(wavelet.range_min_u64_unchecked(0..10), 0);
    assert_eq!(wavelet.range_min(0..11), None);
    assert_eq!(wavelet.range_min_u64(0..11), None);
    assert_eq!(wavelet.range_min(1..1), None);
    assert_eq!(wavelet.range_min_u64(1..1), None);
    assert_eq!(long_wavelet.range_min(0..1), Some(long_data.clone()));
    assert_eq!(long_wavelet.range_min_u64(0..1), None);

    assert_eq!(
        wavelet.range_median(0..10),
        Some(BitVec::pack_sequence_u8(&[2], 4))
    );
    assert_eq!(
        wavelet.range_median_unchecked(0..10),
        BitVec::pack_sequence_u8(&[2], 4)
    );
    assert_eq!(wavelet.range_median_u64(0..10), Some(2));
    assert_eq!(wavelet.range_median_u64_unchecked(0..10), 2);
    assert_eq!(wavelet.range_median(0..11), None);
    assert_eq!(wavelet.range_median_u64(0..11), None);
    assert_eq!(wavelet.range_median(1..1), None);
    assert_eq!(wavelet.range_median_u64(1..1), None);
    assert_eq!(long_wavelet.range_median(0..1), Some(long_data.clone()));
    assert_eq!(long_wavelet.range_median_u64(0..1), None);
}

#[test]
fn test_empty_matrix() {
    let wavelet = WaveletMatrix::from_bit_vec(&BitVec::new(), 4);

    assert_eq!(wavelet.len(), 0);
    assert_eq!(wavelet.rank(0, &BitVec::from_zeros(4)), Some(0));
    assert_eq!(wavelet.rank(100, &BitVec::from_ones(4)), None);

    assert_eq!(wavelet.select(0, &BitVec::from_zeros(4)), None);
    assert_eq!(wavelet.select(100, &BitVec::from_ones(4)), None);

    assert_eq!(wavelet.quantile(0..0, 0), None);
    assert_eq!(wavelet.quantile(0..100, 10), None);

    assert_eq!(wavelet.range_max(0..0), None);
    assert_eq!(wavelet.range_max(0..100), None);
    assert_eq!(wavelet.range_min(0..0), None);
    assert_eq!(wavelet.range_min(0..100), None);
}

#[test]
fn test_predecessor() {
    let data = BitVec::pack_sequence_u64(&[1, 10, 1, 3, 9, 5], 4);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    assert_eq!(
        wavelet.predecessor(0..6, &BitVec::pack_sequence_u8(&[0], 4)),
        None
    );
    assert_eq!(wavelet.predecessor_u64(0..6, 0), None);
    assert_eq!(
        wavelet.predecessor(0..6, &BitVec::pack_sequence_u8(&[1], 4)),
        Some(BitVec::pack_sequence_u8(&[1], 4))
    );
    assert_eq!(wavelet.predecessor_u64(0..6, 1), Some(1));
    assert_eq!(
        wavelet.predecessor(0..6, &BitVec::pack_sequence_u8(&[2], 4)),
        Some(BitVec::pack_sequence_u8(&[1], 4))
    );
    assert_eq!(wavelet.predecessor_u64(0..6, 2), Some(1));
    assert_eq!(
        wavelet.predecessor(0..6, &BitVec::pack_sequence_u8(&[3], 4)),
        Some(BitVec::pack_sequence_u8(&[3], 4))
    );
    assert_eq!(wavelet.predecessor_u64(0..6, 3), Some(3));
    assert_eq!(
        wavelet.predecessor(0..6, &BitVec::pack_sequence_u8(&[4], 4)),
        Some(BitVec::pack_sequence_u8(&[3], 4))
    );
    assert_eq!(wavelet.predecessor_u64(0..6, 4), Some(3));

    assert_eq!(
        wavelet.predecessor(0..6, &BitVec::pack_sequence_u8(&[9], 4)),
        Some(BitVec::pack_sequence_u8(&[9], 4))
    );
    assert_eq!(wavelet.predecessor_u64(0..6, 9), Some(9));
    assert_eq!(
        wavelet.predecessor(0..6, &BitVec::pack_sequence_u8(&[10], 4)),
        Some(BitVec::pack_sequence_u8(&[10], 4))
    );
    assert_eq!(wavelet.predecessor_u64(0..6, 10), Some(10));
    assert_eq!(
        wavelet.predecessor(0..6, &BitVec::pack_sequence_u8(&[11], 4)),
        Some(BitVec::pack_sequence_u8(&[10], 4))
    );
    assert_eq!(wavelet.predecessor_u64(0..6, 11), Some(10));
    assert_eq!(
        wavelet.predecessor(0..6, &BitVec::pack_sequence_u8(&[15], 4)),
        Some(BitVec::pack_sequence_u8(&[10], 4))
    );
    assert_eq!(wavelet.predecessor_u64(0..6, 15), Some(10));

    assert_eq!(
        wavelet.predecessor(2..4, &BitVec::pack_sequence_u8(&[5], 4)),
        Some(BitVec::pack_sequence_u8(&[3], 4))
    );
    assert_eq!(wavelet.predecessor_u64(2..4, 5), Some(3));
    assert_eq!(
        wavelet.predecessor(0..3, &BitVec::pack_sequence_u8(&[3], 4)),
        Some(BitVec::pack_sequence_u8(&[1], 4))
    );
    assert_eq!(wavelet.predecessor_u64(0..3, 3), Some(1));
    assert_eq!(
        wavelet.predecessor(3..6, &BitVec::pack_sequence_u8(&[10], 4)),
        Some(BitVec::pack_sequence_u8(&[9], 4))
    );
    assert_eq!(wavelet.predecessor_u64(3..6, 10), Some(9));

    assert_eq!(
        wavelet.predecessor(0..6, &BitVec::pack_sequence_u8(&[0], 4)),
        None
    );
    assert_eq!(wavelet.predecessor_u64(0..6, 0), None);
    assert_eq!(
        wavelet.predecessor(3..5, &BitVec::pack_sequence_u8(&[1], 4)),
        None
    );
    assert_eq!(wavelet.predecessor_u64(3..5, 1), None);
    assert_eq!(
        wavelet.predecessor(5..6, &BitVec::pack_sequence_u8(&[4], 4)),
        None
    );
    assert_eq!(wavelet.predecessor_u64(5..6, 4), None);
}

#[test]
fn test_predecessor_large_gap() {
    let data = BitVec::pack_sequence_u64(&[3, 9000], 16);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 16);

    assert_eq!(
        wavelet.predecessor(0..2, &BitVec::pack_sequence_u16(&[0], 16)),
        None
    );
    assert_eq!(
        wavelet.predecessor(0..2, &BitVec::pack_sequence_u16(&[3], 16)),
        Some(BitVec::pack_sequence_u16(&[3], 16))
    );
    assert_eq!(
        wavelet.predecessor(0..2, &BitVec::pack_sequence_u16(&[4], 16)),
        Some(BitVec::pack_sequence_u16(&[3], 16))
    );
    assert_eq!(
        wavelet.predecessor(0..2, &BitVec::pack_sequence_u16(&[8000], 16)),
        Some(BitVec::pack_sequence_u16(&[3], 16))
    );
    assert_eq!(
        wavelet.predecessor(0..2, &BitVec::pack_sequence_u16(&[8999], 16)),
        Some(BitVec::pack_sequence_u16(&[3], 16))
    );
    assert_eq!(
        wavelet.predecessor(0..2, &BitVec::pack_sequence_u16(&[9000], 16)),
        Some(BitVec::pack_sequence_u16(&[9000], 16))
    );
    assert_eq!(
        wavelet.predecessor(0..2, &BitVec::pack_sequence_u16(&[10000], 16)),
        Some(BitVec::pack_sequence_u16(&[9000], 16))
    );
}

#[test]
fn test_successor() {
    let data = BitVec::pack_sequence_u64(&[1, 10, 1, 3, 9, 5], 4);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    assert_eq!(
        wavelet.successor(0..6, &BitVec::pack_sequence_u8(&[0], 4)),
        Some(BitVec::pack_sequence_u8(&[1], 4))
    );
    assert_eq!(wavelet.successor_u64(0..6, 0), Some(1));
    assert_eq!(
        wavelet.successor(0..6, &BitVec::pack_sequence_u8(&[1], 4)),
        Some(BitVec::pack_sequence_u8(&[1], 4))
    );
    assert_eq!(wavelet.successor_u64(0..6, 1), Some(1));
    assert_eq!(
        wavelet.successor(0..6, &BitVec::pack_sequence_u8(&[2], 4)),
        Some(BitVec::pack_sequence_u8(&[3], 4))
    );
    assert_eq!(wavelet.successor_u64(0..6, 2), Some(3));
    assert_eq!(
        wavelet.successor(0..6, &BitVec::pack_sequence_u8(&[3], 4)),
        Some(BitVec::pack_sequence_u8(&[3], 4))
    );
    assert_eq!(wavelet.successor_u64(0..6, 3), Some(3));
    assert_eq!(
        wavelet.successor(0..6, &BitVec::pack_sequence_u8(&[4], 4)),
        Some(BitVec::pack_sequence_u8(&[5], 4))
    );
    assert_eq!(wavelet.successor_u64(0..6, 4), Some(5));

    assert_eq!(
        wavelet.successor(0..6, &BitVec::pack_sequence_u8(&[9], 4)),
        Some(BitVec::pack_sequence_u8(&[9], 4))
    );
    assert_eq!(wavelet.successor_u64(0..6, 9), Some(9));
    assert_eq!(
        wavelet.successor(0..6, &BitVec::pack_sequence_u8(&[10], 4)),
        Some(BitVec::pack_sequence_u8(&[10], 4))
    );

    assert_eq!(
        wavelet.successor(0..6, &BitVec::pack_sequence_u8(&[11], 4)),
        None
    );
    assert_eq!(wavelet.successor_u64(0..6, 11), None);
    assert_eq!(
        wavelet.successor(0..6, &BitVec::pack_sequence_u8(&[15], 4)),
        None
    );
    assert_eq!(wavelet.successor_u64(0..6, 15), None);
}

#[test]
fn test_successor_large_gap() {
    let data = BitVec::pack_sequence_u64(&[3, 9000], 16);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 16);

    assert_eq!(
        wavelet.successor(0..2, &BitVec::pack_sequence_u16(&[0], 16)),
        Some(BitVec::pack_sequence_u16(&[3], 16))
    );
    assert_eq!(
        wavelet.successor(0..2, &BitVec::pack_sequence_u16(&[3], 16)),
        Some(BitVec::pack_sequence_u16(&[3], 16))
    );
    assert_eq!(
        wavelet.successor(0..2, &BitVec::pack_sequence_u16(&[4], 16)),
        Some(BitVec::pack_sequence_u16(&[9000], 16))
    );

    assert_eq!(
        wavelet.successor(0..2, &BitVec::pack_sequence_u16(&[9000], 16)),
        Some(BitVec::pack_sequence_u16(&[9000], 16))
    );

    assert_eq!(
        wavelet.successor(0..2, &BitVec::pack_sequence_u16(&[10000], 16)),
        None
    );
}

#[test]
fn test_pred_succ_randomized() {
    let mut rng = StdRng::from_seed([100; 32]);

    let data: Vec<u64> = (0..1000).map(|_| rng.gen_range(0..=u64::MAX)).collect();
    let wavelet = WaveletMatrix::from_bit_vec(&BitVec::pack_sequence_u64(&data, 64), 64);

    let mut sorted_data = data.clone();
    sorted_data.sort();

    for _ in 0..1000 {
        let query = rng.gen_range(0..u64::MAX);

        let pred = sorted_data
            .iter()
            .rev()
            .position(|&x| x <= query)
            .map(|x| sorted_data.len() - x - 1);
        let succ = sorted_data.iter().position(|&x| x >= query);

        assert_eq!(
            wavelet.predecessor_u64(0..1000, query),
            pred.map(|x| sorted_data[x]),
        );
        assert_eq!(
            wavelet.successor_u64(0..1000, query),
            succ.map(|x| sorted_data[x])
        );
    }
}

// test iterators exist and work correctly
#[test]
fn test_wavelet_iter() {
    let data = BitVec::pack_sequence_u64(&[1, 4, 4, 1, 3, 1, 4, 3, 2, 0], 4);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    let mut iter = wavelet.iter();
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[1], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[4], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[4], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[1], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[3], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[1], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[4], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[3], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[2], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[0], 4)));
    assert_eq!(iter.next(), None);

    let mut iter = wavelet.iter();
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[1], 4)));
    assert_eq!(iter.next_back(), Some(BitVec::pack_sequence_u64(&[0], 4)));
    assert_eq!(iter.next_back(), Some(BitVec::pack_sequence_u64(&[2], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[4], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[4], 4)));
    assert_eq!(iter.next_back(), Some(BitVec::pack_sequence_u64(&[3], 4)));
    assert_eq!(iter.next_back(), Some(BitVec::pack_sequence_u64(&[4], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[1], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[3], 4)));
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[1], 4)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);

    let mut iter = wavelet.iter_u64().unwrap();
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(4));
    assert_eq!(iter.next(), Some(4));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(4));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), None);
}

// test into_iter exists and works
#[test]
fn test_wavelet_into_iter() {
    let data = BitVec::pack_sequence_u64(&[1], 1);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 1);

    let mut iter = wavelet.clone().into_iter();
    assert_eq!(iter.next(), Some(BitVec::pack_sequence_u64(&[1], 1)));
    assert_eq!(iter.next(), None);

    let mut iter = wavelet.into_iter_u64().unwrap();
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), None);
}

// fuzz the iterators. don't bother with into_iter, as they should have the same implementation
// anyway, and we already tested that they exist in the previous test
#[test]
fn test_wavelet_iter_randomized() {
    let mut rng = StdRng::from_seed([100; 32]);

    for _ in 0..50 {
        let data: Vec<u8> = (0..1000).map(|_| rng.gen_range(0..=u8::MAX)).collect();
        let wavelet = WaveletMatrix::from_bit_vec(&BitVec::pack_sequence_u8(&data, 8), 8);

        let mut iter = wavelet.iter();
        for v in &data {
            assert_eq!(iter.next(), Some(BitVec::pack_sequence_u8(&[*v], 8)));
        }
        assert_eq!(iter.next(), None);

        let mut iter = wavelet.iter_u64().unwrap();
        for v in &data {
            assert_eq!(iter.next(), Some(*v as u64));
        }
        assert_eq!(iter.next(), None);
    }
}

#[test]
fn test_wavelet_empty_iter() {
    let data = BitVec::new();
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    let mut iter = wavelet.iter();
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next(), None);

    let mut iter = wavelet.iter_u64().unwrap();
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sorted_iter() {
    let mut data = [
        1, 4, 4, 1, 3, 1, 4, 3, 13, 11, 12, 13, 2, 0, 4, 6, 7, 5, 8, 9, 10,
    ];
    let wavelet = WaveletMatrix::from_bit_vec(&BitVec::pack_sequence_u64(&data, 4), 4);

    data.sort();

    let mut iter = wavelet.iter_sorted();
    let mut iter64 = wavelet.iter_sorted_u64().unwrap();
    for i in 0..data.len() {
        assert_eq!(
            iter.next(),
            Some(BitVec::pack_sequence_u64(&data[i..i + 1], 4))
        );
        assert_eq!(iter64.next(), Some(data[i]));
    }
    assert_eq!(iter.next(), None);
    assert_eq!(iter64.next(), None);
}
