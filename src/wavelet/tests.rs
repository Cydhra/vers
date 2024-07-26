use super::*;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::{max, min};

#[test]
fn test_wavelet_encoding_randomized() {
    let mut rng = StdRng::from_seed([1; 32]);

    for _ in 0..100 {
        let data: Vec<u8> = (0..rng.gen_range(500..1000))
            .map(|_| rng.gen_range(0..=u8::MAX))
            .collect();
        let wavelet = WaveletMatrix::from_bit_vec(&BitVec::pack_sequence_u8(&data, 8), 8);
        assert_eq!(wavelet.len(), data.len());

        for i in 0..data.len() {
            assert_eq!(wavelet.get_u64_unchecked(i), data[i] as u64);
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
        for i in 0..data.len() {
            assert_eq!(wavelet.rank_unchecked(i, &symbol_bit_vec), rank);
            if data[i] == symbol {
                rank += 1;
            }
        }
    }
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
fn test_quantile() {
    let mut sequence = [1, 4, 4, 1, 3, 1, 4, 3, 2, 0];
    let data = BitVec::pack_sequence_u64(&sequence, 4);
    let wavelet = WaveletMatrix::from_bit_vec(&data, 4);

    sequence.sort();

    for i in 0..sequence.len() {
        assert_eq!(
            wavelet.quantile(0..10, i),
            Some(BitVec::pack_sequence_u8(&[sequence[i] as u8], 4))
        );
        assert_eq!(wavelet.quantile_u64(0..10, i), Some(sequence[i]));
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

        let mut range_data = data[range.clone()].iter().copied().collect::<Vec<u8>>();
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
                Some(range_data[range_data.len() / 2] as u64)
            }
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

    let mut iter = wavelet.iter_u64();
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

    let mut iter = wavelet.into_iter_u64();
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
        for i in 0..data.len() {
            assert_eq!(iter.next(), Some(BitVec::pack_sequence_u8(&[data[i]], 8)));
        }
        assert_eq!(iter.next(), None);

        let mut iter = wavelet.iter_u64();
        for i in 0..data.len() {
            assert_eq!(iter.next(), Some(data[i] as u64));
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

    let mut iter = wavelet.iter_u64();
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next(), None);
}
