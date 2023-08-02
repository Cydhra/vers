use crate::rmq::binary_rmq::BinaryRmq;
use rand::RngCore;

#[test]
fn small_test() {
    let rmq = BinaryRmq::from_vec(vec![9, 6, 10, 4, 0, 8, 3, 7, 1, 2, 5]);

    assert_eq!(rmq.range_min(0, 0), 0);
    assert_eq!(rmq.range_min(0, 1), 1);
    assert_eq!(rmq.range_min(0, 2), 1);
    assert_eq!(rmq.range_min(0, 3), 3);
    assert_eq!(rmq.range_min(5, 8), 8);
    assert_eq!(rmq.range_min(5, 9), 8);
    assert_eq!(rmq.range_min(9, 10), 9);
    assert_eq!(rmq.range_min(0, 10), 4);
}

#[test]
fn randomized_test() {
    let mut rng = rand::thread_rng();
    const L: usize = 100;

    let mut numbers_vec = Vec::with_capacity(L);
    for _ in 0..L {
        numbers_vec.push(rng.next_u64());
    }

    let rmq = BinaryRmq::from_vec(numbers_vec.clone());

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
    let rmq = BinaryRmq::from_vec(vec![1, 2, 3, 4, 5]);
    let mut iter = rmq.iter();
    assert_eq!(iter.next(), Some(&1));
    assert_eq!(iter.next(), Some(&2));
    assert_eq!(iter.next(), Some(&3));
    assert_eq!(iter.next(), Some(&4));
    assert_eq!(iter.next(), Some(&5));
    assert_eq!(iter.next(), None);
}
