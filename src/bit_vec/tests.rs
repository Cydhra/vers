use super::BitVec;

#[test]
fn test_alloc_ones() {
    let bv = BitVec::from_ones(42);
    assert_eq!(bv.len(), 42);
    assert_eq!(bv.data.len(), 1);

    // test that unused bits are still zero, because this is what the data structure assumes
    assert_eq!(bv.data[0] >> 42, 0);
}