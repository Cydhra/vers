use crate::EliasFanoVec;
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

#[test]
fn test_elias_fano() {
    let ef = EliasFanoVec::new(&vec![0, 1, 4, 7]);

    assert_eq!(ef.len(), 4);
    assert_eq!(ef.get_unchecked(0), 0);
    assert_eq!(ef.get_unchecked(1), 1);
    assert_eq!(ef.get_unchecked(2), 4);
    assert_eq!(ef.get_unchecked(3), 7);

    assert_eq!(ef.pred(0), 0);
    assert_eq!(ef.pred(1), 1);
    assert_eq!(ef.pred(2), 1);
    assert_eq!(ef.pred(5), 4);
    assert_eq!(ef.pred(8), 7);
}

// test the edge case in which the predecessor query doesn't find bounds around the result,
// but the result is the last element before the bounds.
#[test]
fn test_edge_case() {
    let ef = EliasFanoVec::new(&vec![0, 1, u64::MAX - 10, u64::MAX - 1]);
    assert_eq!(ef.pred(u64::MAX - 11), 1);
}

// test a query that is way larger than any element in the vector
#[test]
fn test_large_query() {
    let ef = EliasFanoVec::new(&vec![0, 1, 2, 3]);
    assert_eq!(ef.pred(u64::MAX), 3);
}

// test whether duplicates are handled correctly by predecessor queries and reconstruction
#[test]
fn test_duplicates() {
    let ef = EliasFanoVec::new(&vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);
    assert_eq!(ef.pred(0), 0);
    assert_eq!(ef.pred(1), 1);
    assert_eq!(ef.pred(2), 2);

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
    for i in 0..1000 {
        seq[i] = rng.gen();
    }
    seq.sort_unstable();

    let ef = EliasFanoVec::new(&seq);

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

        let pred = ef.pred(random_splitter);
        assert!(seq.iter().filter(|&&x| x == pred).count() >= 1);

        assert_eq!(
            ef.pred(random_splitter),
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

    let ef = EliasFanoVec::new(&seq);
    for (i, &x) in seq.iter().enumerate() {
        assert_eq!(ef.get_unchecked(i), x, "expected {:b}", x);
        assert_eq!(ef.pred(x), x);
    }

    for (x, p) in [
        (1001, 999),
        (5000, 999),
        (50000, 999),
        (249999, 999),
        (500001001, 500000999),
        (500002000, 500000999),
    ] {
        assert_eq!(ef.pred(x), p);
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
    let bad_ef_vec = EliasFanoVec::new(&sequence);

    // query random values from the actual sequences, to force long searches in the lower vec
    for _ in 0..1000 {
        let elem = sequence[(&mut rng).sample(query_distribution)];
        let supposed = sequence.partition_point(|&n| n <= elem) - 1;
        assert_eq!(bad_ef_vec.pred(elem), sequence[supposed]);
    }
}

#[test]
fn test_iter() {
    let ef = EliasFanoVec::new(&vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);

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
    drop(iter); // end borrow for next test

    // owning iter and into_iter test
    let mut i = 0;
    for elem in ef {
        assert_eq!(elem, i);
        i += 1;
    }
}