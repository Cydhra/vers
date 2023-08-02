use super::*;
use rand::RngCore;

#[test]
fn test_small_bit_vector_rank0() {
    let mut sbv = SmallBitVector::default();
    sbv.set_bit(1);
    sbv.set_bit(3);
    sbv.set_bit(64);
    sbv.set_bit(65);

    assert_eq!(sbv.rank0(0), 0);
    assert_eq!(sbv.rank0(1), 1);
    assert_eq!(sbv.rank0(2), 1);
    assert_eq!(sbv.rank0(3), 2);
    assert_eq!(sbv.rank0(4), 2);

    assert_eq!(sbv.rank0(64), 62);
    assert_eq!(sbv.rank0(65), 62);
    assert_eq!(sbv.rank0(66), 62);
    assert_eq!(sbv.rank0(67), 63);
}

#[test]
fn test_small_bit_vector_select0() {
    let mut sbv = SmallBitVector::default();
    sbv.set_bit(1);
    sbv.set_bit(3);
    sbv.set_bit(64);
    sbv.set_bit(65);

    assert_eq!(sbv.select0(0), 0);
    assert_eq!(sbv.select0(1), 2);
    assert_eq!(sbv.select0(2), 4);
    assert_eq!(sbv.select0(3), 5);
    assert_eq!(sbv.select0(64), 68);
}

#[test]
fn test_fast_rmq() {
    const L: usize = 2 * BLOCK_SIZE;

    let mut numbers_vec = Vec::with_capacity(L);
    for i in 0..L {
        numbers_vec.push(i as u64);
    }

    let rmq = FastRmq::from_vec(numbers_vec.clone());

    for i in 0..L {
        for j in i..L {
            let min = i + numbers_vec[i..=j]
                .iter()
                .enumerate()
                .min_by_key(|(_, &x)| x)
                .unwrap()
                .0;
            assert_eq!(rmq.range_min(i, j), min, "i = {}, j = {}", i, j);
        }
    }
}

#[test]
fn test_fast_rmq_unsorted() {
    let mut rng = rand::thread_rng();
    const L: usize = 2 * BLOCK_SIZE;

    let mut numbers_vec = Vec::with_capacity(L);
    for _ in 0..L {
        numbers_vec.push(rng.next_u64());
    }

    let rmq = FastRmq::from_vec(numbers_vec.clone());

    for i in 0..L {
        for j in i..L {
            let min = numbers_vec[i..=j].iter().min().unwrap();
            assert_eq!(
                numbers_vec[rmq.range_min(i, j)],
                *min,
                "i = {}, j = {}",
                i,
                j
            );
        }
    }
}

#[test]
fn test_iter() {
    let rmq = FastRmq::from_vec(vec![1, 2, 3, 4, 5]);
    let mut iter = rmq.iter();
    assert_eq!(iter.next(), Some(&1));
    assert_eq!(iter.next(), Some(&2));
    assert_eq!(iter.next(), Some(&3));
    assert_eq!(iter.next(), Some(&4));
    assert_eq!(iter.next(), Some(&5));
    assert_eq!(iter.next(), None);
}
