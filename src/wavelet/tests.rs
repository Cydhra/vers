use super::*;

#[test]
fn test_wavelet_encoding() {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let wavelet = WaveletMatrix::from_bit_vec(BitVec::pack_sequence_u8(&data, 4), 4);

    for i in 0..data.len() {
        assert_eq!(wavelet.get_u64(i), data[i].into());
    }
}
