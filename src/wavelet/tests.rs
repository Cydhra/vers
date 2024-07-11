use super::*;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn test_wavelet_encoding_randomized() {
    let mut rng = StdRng::from_seed([1; 32]);

    for _ in 0..100 {
        let data: Vec<u8> = (0..rng.gen_range(500..1000))
            .map(|_| rng.gen_range(0..=u8::MAX))
            .collect();
        let wavelet = WaveletMatrix::from_bit_vec(BitVec::pack_sequence_u8(&data, 8), 8);
        assert_eq!(wavelet.len(), data.len());

        for i in 0..data.len() {
            assert_eq!(wavelet.get_u64(i), data[i] as u64);
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

    let wavelet = WaveletMatrix::from_bit_vec(data, 127);
    assert_eq!(wavelet.len(), 6);

    assert_eq!(wavelet.get_value(0), BitVec::from_zeros(127));
    assert_eq!(wavelet.get_value(1), BitVec::from_zeros(127));
    assert_eq!(
        wavelet.get_value(2),
        BitVec::pack_sequence_u64(&[u64::MAX], 127)
    );
    assert_eq!(wavelet.get_value(3), BitVec::from_ones(127));
    assert_eq!(wavelet.get_value(4), BitVec::pack_sequence_u8(&[1], 127));
    assert_eq!(wavelet.get_value(5), BitVec::pack_sequence_u8(&[2], 127));
}
