use crate::EliasFanoVec;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, RngCore, SeedableRng};

#[test]
fn test_elias_fano() {
    let ef = EliasFanoVec::from_slice(&[0, 1, 4, 7]);

    assert_eq!(ef.len(), 4);
    assert_eq!(ef.get_unchecked(0), 0);
    assert_eq!(ef.get_unchecked(1), 1);
    assert_eq!(ef.get_unchecked(2), 4);
    assert_eq!(ef.get_unchecked(3), 7);

    assert_eq!(ef.predecessor_unchecked(0), 0);
    assert_eq!(ef.predecessor_unchecked(1), 1);
    assert_eq!(ef.predecessor_unchecked(2), 1);
    assert_eq!(ef.predecessor_unchecked(5), 4);
    assert_eq!(ef.predecessor_unchecked(8), 7);
}

// test the edge case in which the predecessor query doesn't find bounds around the result,
// but the result is the last element before the bounds.
#[test]
fn test_edge_case() {
    let ef = EliasFanoVec::from_slice(&[0, 1, u64::MAX - 10, u64::MAX - 1]);
    assert_eq!(ef.predecessor_unchecked(u64::MAX - 11), 1);
}

// test a query that is way larger than any element in the vector
#[test]
fn test_large_query() {
    let ef = EliasFanoVec::from_slice(&[0, 1, 2, 3]);
    assert_eq!(ef.predecessor_unchecked(u64::MAX), 3);
}

// test whether duplicates are handled correctly by predecessor queries and reconstruction
#[test]
fn test_duplicates() {
    let ef = EliasFanoVec::from_slice(&[0, 0, 0, 1, 1, 1, 2, 2, 2]);
    assert_eq!(ef.predecessor_unchecked(0), 0);
    assert_eq!(ef.predecessor_unchecked(1), 1);
    assert_eq!(ef.predecessor_unchecked(2), 2);

    assert_eq!(ef.get_unchecked(2), 0);
    assert_eq!(ef.get_unchecked(3), 1);
    assert_eq!(ef.get_unchecked(5), 1);
    assert_eq!(ef.get_unchecked(8), 2);
}

// a randomized test to catch edge cases. If the test fails, efforts should be made to
// reproduce the failing case and add it to the test suite.
#[test]
fn test_randomized_elias_fano() {
    let mut rng = thread_rng();
    let mut seq = vec![0u64; 1000];
    for v in seq.iter_mut() {
        *v = rng.gen();
    }
    seq.sort_unstable();

    let ef = EliasFanoVec::from_slice(&seq);

    assert_eq!(ef.len(), seq.len());

    for (i, &v) in seq.iter().enumerate() {
        assert_eq!(ef.get_unchecked(i), v);
    }

    for _ in 0..1000 {
        let mut random_splitter: u64 = rng.gen();

        // make sure we don't generate erroneous queries
        while random_splitter < seq[0] {
            random_splitter = rng.gen();
        }

        let pred = ef.predecessor_unchecked(random_splitter);
        assert!(seq.iter().filter(|&&x| x == pred).count() >= 1);

        assert_eq!(
            pred,
            seq[seq.partition_point(|&x| x <= random_splitter) - 1]
        );
    }
}

// a test case that checks for correctness of the predecessor query in a
// clustered vector (i.e. a vector with large gaps between elements)
#[test]
fn test_clustered_ef() {
    let mut seq = Vec::with_capacity(4000);

    for i in 0..1000 {
        seq.push(i);
    }

    for i in 250000..251000 {
        seq.push(i);
    }

    for i in 500000000..500001000 {
        seq.push(i);
    }

    for i in 750000000000..750000001000 {
        seq.push(i);
    }

    let ef = EliasFanoVec::from_slice(&seq);
    for (i, &x) in seq.iter().enumerate() {
        assert_eq!(ef.get_unchecked(i), x, "expected {:b}", x);
        assert_eq!(ef.predecessor_unchecked(x), x);
        assert_eq!(ef.successor_unchecked(x), x);
    }

    for (x, p) in [
        (1001, 999),
        (5000, 999),
        (50000, 999),
        (249999, 999),
        (500001001, 500000999),
        (500002000, 500000999),
    ] {
        assert_eq!(ef.predecessor_unchecked(x), p);
    }

    for (x, s) in [
        (1001, 250000),
        (249999, 250000),
        (1000000, 500000000),
        (499999999, 500000000),
    ] {
        assert_eq!(ef.successor_unchecked(x), s);
    }
}

// a randomized test case that checks for correctness of the predecessor query in a
// clustered vector (i.e. a vector with large gaps between elements)
#[test]
fn large_clustered_rng() {
    cluster_test(1 << 16)
}

fn cluster_test(l: usize) {
    let mut rng = thread_rng();
    let dist_high = Uniform::new(u64::MAX / 2 - 200, u64::MAX / 2 - 1);
    let dist_low = Uniform::new(0, l as u64);
    let query_distribution = Uniform::new(0, l);

    // prepare a sequence of low values with a few high values at the end
    let mut sequence = (&mut rng)
        .sample_iter(dist_low)
        .take(l - 100)
        .collect::<Vec<u64>>();
    sequence.sort_unstable();
    let mut sequence_top = (&mut rng)
        .sample_iter(dist_high)
        .take(100)
        .collect::<Vec<u64>>();
    sequence_top.sort_unstable();
    sequence.append(&mut sequence_top);
    let bad_ef_vec = EliasFanoVec::from_slice(&sequence);

    // query random values from the actual sequences, to force long searches in the lower vec
    for _ in 0..1000 {
        let elem = sequence[rng.sample(query_distribution)];
        let supposed = sequence.partition_point(|&n| n <= elem) - 1;
        let supposed_succ = sequence.partition_point(|&n| n < elem);
        assert_eq!(bad_ef_vec.predecessor_unchecked(elem), sequence[supposed]);
        assert_eq!(
            bad_ef_vec.successor_unchecked(elem),
            sequence[supposed_succ]
        );
    }
}

#[test]
fn test_iter() {
    let ef = EliasFanoVec::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);

    // borrowing iter test
    let mut iter = ef.iter();
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(4));
    assert_eq!(iter.next(), Some(5));
    assert_eq!(iter.next(), Some(6));
    assert_eq!(iter.next(), Some(7));
    assert_eq!(iter.next(), Some(8));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    let mut iter = ef.iter();
    assert_eq!(iter.next_back(), Some(8));
    assert_eq!(iter.next_back(), Some(7));
    assert_eq!(iter.next_back(), Some(6));
    assert_eq!(iter.next_back(), Some(5));
    assert_eq!(iter.next_back(), Some(4));
    assert_eq!(iter.next_back(), Some(3));
    assert_eq!(iter.next_back(), Some(2));
    assert_eq!(iter.next_back(), Some(1));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next_back(), None);

    let mut iter = ef.iter();
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next_back(), Some(8));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next_back(), Some(7));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next_back(), Some(6));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next_back(), Some(5));
    assert_eq!(iter.next(), Some(4));
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next(), None);

    // owning iter and into_iter test
    for (i, elem) in ef.into_iter().enumerate() {
        assert_eq!(elem, i as u64);
    }
}

#[test]
fn test_custom_iter_behavior() {
    let ef = EliasFanoVec::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
    #[allow(clippy::iter_skip_next)]
    let next = ef.iter().skip(2).next(); // explicit test for skip()
    assert_eq!(next, Some(2));
    assert_eq!(ef.iter().count(), 9);
    assert_eq!(ef.iter().skip(2).count(), 7);
    assert_eq!(ef.iter().last(), Some(8));
    assert_eq!(ef.iter().nth(2), Some(2));
    assert_eq!(ef.iter().nth(10), None);
    assert_eq!(ef.iter().skip(3).min(), Some(3));

    assert!(ef.iter().advance_by(9).is_ok());
    assert!(ef.iter().advance_back_by(9).is_ok());
    assert!(ef.iter().advance_by(10).is_err());
    assert!(ef.iter().advance_back_by(10).is_err());

    let mut iter = ef.iter();
    assert!(iter.advance_by(5).is_ok());
    assert!(iter.advance_back_by(6).is_err());
    assert!(iter.advance_by(6).is_err());
    assert!(iter.advance_back_by(4).is_ok());

    #[allow(clippy::iter_skip_next)]
    let next = ef.clone().into_iter().skip(2).next(); // explicit test for skip()
    assert_eq!(next, Some(2));
    assert_eq!(ef.clone().into_iter().count(), 9);
    assert_eq!(ef.clone().into_iter().skip(2).count(), 7);
    assert_eq!(ef.clone().into_iter().last(), Some(8));
    assert_eq!(ef.clone().into_iter().nth(2), Some(2));
    assert_eq!(ef.clone().into_iter().nth(10), None);
    assert_eq!(ef.clone().into_iter().skip(3).min(), Some(3));

    let mut iter = ef.iter();
    assert_eq!(iter.nth(2), Some(2));
    assert_eq!(iter.nth_back(4), Some(4));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.clone().count(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);

    let mut iter = ef.iter();
    assert!(iter.advance_by(5).is_ok());
    assert!(iter.advance_back_by(2).is_ok());
    assert_eq!(iter.clone().collect::<Vec<_>>(), vec![5, 6]);
    assert_eq!(iter.clone().max(), Some(6));
    assert_eq!(iter.clone().min(), Some(5));
    assert_eq!(iter.clone().rev().collect::<Vec<_>>(), vec![6, 5]);
}

#[test]
fn test_empty_iter() {
    let ef = EliasFanoVec::from_slice(&[]);
    let mut iter = ef.iter();
    assert_eq!(iter.clone().count(), 0);

    assert!(iter.next().is_none());
    assert!(iter.next_back().is_none());
    assert!(iter.nth(20).is_none());
    assert!(iter.nth_back(20).is_none());
    assert!(iter.advance_by(0).is_ok());
    assert!(iter.advance_back_by(0).is_ok());
    assert!(iter.advance_by(100).is_err());
    assert!(iter.advance_back_by(100).is_err());

    let ef = EliasFanoVec::from_slice(&[0]);
    let mut iter = ef.iter();
    assert_eq!(iter.clone().count(), 1);
    assert!(iter.advance_by(1).is_ok());
    assert_eq!(iter.clone().count(), 0);
    assert!(iter.advance_by(0).is_ok());
    assert!(iter.advance_back_by(0).is_ok());
    assert!(iter.advance_by(1).is_err());
    assert!(iter.advance_back_by(1).is_err());

    let ef = EliasFanoVec::from_slice(&[1]);
    let mut iter = ef.iter();
    assert_eq!(iter.clone().count(), 1);
    assert!(iter.advance_back_by(1).is_ok());
    assert_eq!(iter.clone().count(), 0);
    assert!(iter.advance_back_by(0).is_ok());
    assert!(iter.advance_by(0).is_ok());
    assert!(iter.advance_back_by(1).is_err());
    assert!(iter.advance_by(1).is_err());
}

#[test]
fn test_iter_randomized() {
    let mut rng = StdRng::from_seed([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);

    for _ in 0..100 {
        let len = (rng.next_u64() % 4000) as usize;
        let mut seq = vec![0u64; len];
        for v in seq.iter_mut() {
            *v = rng.next_u64();
        }

        seq.sort_unstable();

        let ef = EliasFanoVec::from_slice(&seq);

        let mut iter = ef.iter();
        let mut compare = seq.iter();

        for _ in 0..len {
            if rng.gen_bool(0.5) {
                assert_eq!(iter.next().unwrap(), *compare.next().unwrap());
            } else {
                assert_eq!(iter.next_back().unwrap(), *compare.next_back().unwrap());
            }
        }
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }
}

#[test]
fn test_successor() {
    let ef = EliasFanoVec::from_slice(&[0, 1, 4, 7]);
    assert_eq!(ef.len(), 4);

    assert_eq!(ef.successor_unchecked(0), 0);
    assert_eq!(ef.successor_unchecked(1), 1);
    assert_eq!(ef.successor_unchecked(2), 4);
    assert_eq!(ef.successor_unchecked(5), 7);
    assert_eq!(ef.successor(8), None);
}

#[test]
fn test_edge_case_successor() {
    let ef = EliasFanoVec::from_slice(&[0, 1, u64::MAX - 10, u64::MAX - 1]);
    assert_eq!(ef.successor_unchecked(2), u64::MAX - 10);
    assert_eq!(ef.successor_unchecked(u64::MAX - 11), u64::MAX - 10);
    assert_eq!(ef.successor_unchecked(u64::MAX - 10), u64::MAX - 10);
    assert_eq!(ef.successor_unchecked(u64::MAX - 9), u64::MAX - 1);
}

#[test]
fn test_large_query_successor() {
    let ef = EliasFanoVec::from_slice(&[0, 1, 2, 3]);
    assert_eq!(ef.successor(u64::MAX), None);
}

// test whether duplicates are handled correctly by predecessor queries and reconstruction
#[test]
fn test_duplicates_successor() {
    let ef = EliasFanoVec::from_slice(&[0, 0, 0, 1, 1, 1, 2, 2, 2]);
    assert_eq!(ef.successor_unchecked(0), 0);
    assert_eq!(ef.successor_unchecked(1), 1);
    assert_eq!(ef.successor_unchecked(2), 2);
    assert_eq!(ef.successor(3), None);
}

// a randomized test to catch edge cases. If the test fails, efforts should be made to
// reproduce the failing case and add it to the test suite.
#[test]
fn test_randomized_elias_fano_successor() {
    let mut rng = thread_rng();
    let mut seq = vec![0u64; 1000];
    for v in seq.iter_mut() {
        *v = rng.gen();
    }
    seq.sort_unstable();

    let ef = EliasFanoVec::from_slice(&seq);

    assert_eq!(ef.len(), seq.len());

    for (i, &v) in seq.iter().enumerate() {
        assert_eq!(ef.get_unchecked(i), v);
    }

    for _ in 0..1000 {
        let mut random_splitter: u64 = rng.gen();

        // make sure we don't generate erroneous queries
        while random_splitter > seq[seq.len() - 1] {
            random_splitter = rng.gen();
        }

        let succ = ef.successor_unchecked(random_splitter);
        assert!(seq.iter().filter(|&&x| x == succ).count() >= 1);

        assert_eq!(succ, seq[seq.partition_point(|&x| x <= random_splitter)]);
    }
}

#[test]
fn test_rank() {
    let ef = EliasFanoVec::from_slice(&vec![0, 1, 1, 4, 4, 4, 4, 4, 7, 99, 101, 102, 150]);

    assert_eq!(ef.rank(0), 0);
    assert_eq!(ef.rank(1), 1);
    assert_eq!(ef.rank(4), 3);
    assert_eq!(ef.rank(5), 8);
    assert_eq!(ef.rank(7), 8);
    assert_eq!(ef.rank(8), 9);
    assert_eq!(ef.rank(99), 9);
    assert_eq!(ef.rank(3000), 13);
}

#[test]
fn test_oob_rank() {
    let ef = EliasFanoVec::from_slice(&vec![1000]);
    assert_eq!(ef.rank(1000), 0);
    assert_eq!(ef.rank(1001), 1);
    assert_eq!(ef.rank(1002), 1);
    assert_eq!(ef.rank(10000), 1);
    assert_eq!(ef.rank(u64::MAX), 1);
    assert_eq!(ef.rank(0), 0);
    assert_eq!(ef.rank(1), 0);
}

#[test]
fn test_rank_binary_search() {
    const MAX_LEN: usize = 60;
    // test various configurations of the elias fano vec that require binary search
    let mut slice = Vec::with_capacity(MAX_LEN + 20);
    for length in 10..MAX_LEN {
        slice.clear();
        slice.push(0);
        slice.push(1);
        for _ in 0..length {
            slice.push(10);
        }
        slice.push(20);
        slice.push(30);

        let ef = EliasFanoVec::from_slice(&slice);

        assert_eq!(ef.rank(0), 0);
        assert_eq!(ef.rank(1), 1);
        assert_eq!(ef.rank(10), 2);
        assert_eq!(ef.rank(11), length as u64 + 2);
    }

    // test various configurations where the binary search returns elements in the middle
    for length in 10..MAX_LEN {
        slice.clear();
        slice.push(0);
        slice.push(1);
        for _ in 0..16 {
            slice.push(10);
        }
        for _ in 0..length {
            slice.push(11);
        }
        slice.push(20);
        slice.push(30);

        let ef = EliasFanoVec::from_slice(&slice);

        assert_eq!(ef.rank(0), 0);
        assert_eq!(ef.rank(1), 1);
        assert_eq!(ef.rank(10), 2);
        assert_eq!(ef.rank(11), 18);
        assert_eq!(ef.rank(12), 18 + length as u64);
    }
}

#[test]
fn test_delta() {
    let ef = EliasFanoVec::from_slice(&vec![0, 1, 4, 7]);

    assert_eq!(ef.delta(0), Some(0));
    assert_eq!(ef.delta(1), Some(1));
    assert_eq!(ef.delta(2), Some(3));
    assert_eq!(ef.delta(3), Some(3));
    assert_eq!(ef.delta(4), None);
    assert_eq!(ef.delta(5), None);
}

#[test]
fn test_delta_non_zero() {
    // test whether an EF vector that doesnt start at 0 is handled correctly
    let ef = EliasFanoVec::from_slice(&vec![100, 101, 102, 103]);
    assert_eq!(ef.delta(0), Some(100));
    assert_eq!(ef.delta(1), Some(1));
}

#[test]
fn test_empty_ef_vec() {
    let ef = EliasFanoVec::from_slice(&[]);
    assert_eq!(ef.len(), 0);
    assert_eq!(ef.successor(0), None);
    assert_eq!(ef.successor(u64::MAX), None);
    assert_eq!(ef.predecessor(0), None);
    assert_eq!(ef.predecessor(u64::MAX), None);
    assert_eq!(ef.get(0), None);
    assert_eq!(ef.rank(3), 0);
    assert_eq!(ef.delta(0), None);
}
