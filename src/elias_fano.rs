//! Elias-Fano encoding for sorted vectors of u64 values. It reduces the space required to represent
//! all numbers (compression ratio dependent on data) and allows for constant time predecessor
//! queries.

use crate::BitVec;
use crate::{RsVec, RsVectorBuilder};
use std::cmp::max;

/// We use linear search for small 1-blocks in the upper vector because it is generally more memory-
/// friendly. But for large clusters this takes too long, so we switch to binary search.
/// We use 4 because benchmarks suggested that this was the best trade-off between speed for average
/// case and for worst case.
const BIN_SEARCH_THRESHOLD: usize = 4;

/// An elias-fano encoded vector of u64 values. The vector is immutable, which will be exploited by
/// limiting the word length of elements to the minimum required to represent all elements.
/// The space requirement for this structure is thus linear in the number of elements with a small
/// constant factor (smaller than one, unless the required word length is close to 64 bit).
///
/// # Predecessor Queries
/// This data structure supports constant time predecessor queries on average.
/// See [`EliasFanoVec::pred`] for more information.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EliasFanoVec {
    upper_vec: RsVec,
    lower_vec: BitVec,
    universe_zero: u64,
    universe_max: u64,
    lower_len: usize,
    len: usize,
}

impl EliasFanoVec {
    /// Create a new Elias-Fano vector by compressing the given data. The data must be sorted in
    /// ascending order. The resulting vector is immutable, which will be exploited by limiting the
    /// word length of elements to the minimum required to represent the universe bound.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(data: &Vec<u64>) -> Self {
        // calculate the largest element the vector needs to represent.
        // By limiting the universe size, we can limit the number of bits
        // required to represent each element, and also spread the elements out more evenly through
        // the upper vector. We also subtract the first element from all elements to make the
        // universe start at zero and possibly save some bits for dense distributions.
        let universe_zero = data[0];
        let universe_bound = data[data.len() - 1] - universe_zero;

        let log_n = ((data.len() + 2) as f64).log2().ceil() as usize;
        let bits_per_number = (max(universe_bound, 2) as f64).log2().ceil() as usize;
        let bits_for_upper_values = (max(data.len(), 2) as f64).log2().ceil() as usize;
        let lower_width = max(bits_per_number, log_n) - bits_for_upper_values;

        let mut upper_vec =
            BitVec::from_zeros(2 + data.len() + (universe_bound >> lower_width) as usize);
        let mut lower_vec = BitVec::with_capacity(data.len() * lower_width);

        for (i, &word) in data.iter().enumerate() {
            let word = word - universe_zero;

            let upper = (word >> lower_width) as usize;
            let lower = word & ((1 << lower_width) - 1);

            upper_vec.flip_bit_unchecked(upper + i + 1);
            lower_vec.append_bits(lower, lower_width);
        }

        Self {
            upper_vec: RsVectorBuilder::from_bit_vec(upper_vec),
            lower_vec,
            universe_zero,
            universe_max: data[data.len() - 1],
            lower_len: lower_width,
            len: data.len(),
        }
    }

    /// Returns the number of elements in the vector.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the element at the given index.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn get(&self, index: usize) -> u64 {
        let upper = self.upper_vec.select1(index) - index - 1;
        let lower = self
            .lower_vec
            .get_bits_unchecked(index * self.lower_len, self.lower_len);
        ((upper << self.lower_len) as u64 | lower) + self.universe_zero
    }

    /// Returns the largest element that is smaller than or equal to the given element.
    /// If the given element is smaller than the smallest element in the vector,
    /// `u64::MAX` is returned.
    ///
    /// # Runtime
    /// This query runs in constant time on average. The worst case runtime is logarithmic in the
    /// number of elements in the vector. The worst case occurs when values in the vector are very
    /// dense with only very few elements that are much larger than most.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn pred(&self, n: u64) -> u64 {
        // bound the query to the universe size
        if n > self.universe_max {
            return self.get(self.len() - 1);
        }

        if n < self.universe_zero {
            return u64::MAX;
        }

        let n = n - self.universe_zero;

        // split the query into the upper and lower part
        let upper_query = (n >> self.lower_len) as usize;
        let lower_query = n & ((1 << self.lower_len) - 1);

        // calculate the lower bound within the lower vector where our predecessor can be found. Since
        // each bit-prefix in the universe has exactly one corresponding zero in the upper vector,
        // we can use select0 to find the start of the block of values with the same upper value.
        let lower_bound_upper_index = self.upper_vec.select0(upper_query);
        let lower_bound_lower_index = lower_bound_upper_index - upper_query;

        // get the first value from the lower vector that corresponds to the query prefix
        let mut lower_candidate = self
            .lower_vec
            .get_bits_unchecked(lower_bound_lower_index * self.lower_len, self.lower_len);

        // calculate the upper part of the result. This only works if the next value in the upper
        // vector is set, otherwise the there is no value in the entire vector with this bit-prefix,
        // and we need to search the largest prefix smaller than the query.
        let result_upper = (upper_query << self.lower_len) as u64;

        // check if the next bit is set. If it is not, or if the result would be larger than the
        // query, we need to search for the block of values before the current prefix and return its
        // last element.
        if self.upper_vec.get_unchecked(lower_bound_upper_index + 1) > 0
            && (result_upper | lower_candidate) <= n
        {
            // search for the largest element in the lower vector that is smaller than the query.
            // Abort the search once the upper vector contains another zero, as this marks the end
            // of the block of values with the same upper prefix.
            let mut cursor = 1;
            while self.upper_vec.get_unchecked(lower_bound_upper_index + cursor + 1) > 0 {
                let next_candidate = self.lower_vec.get_bits_unchecked(
                    (lower_bound_lower_index + cursor) * self.lower_len,
                    self.lower_len,
                );

                // if we found a value that is larger than the query, return the previous value
                if next_candidate > lower_query {
                    return (result_upper | lower_candidate) + self.universe_zero;
                } else if next_candidate == lower_query {
                    return (result_upper | next_candidate) + self.universe_zero;
                } else {
                    lower_candidate = next_candidate;
                }
                cursor += 1;

                // if linear search takes too long, we can use select0 to find the next zero in the
                // upper vector, and then use binary search
                if cursor == BIN_SEARCH_THRESHOLD {
                    let block_end = self.upper_vec.select0(upper_query + 1) - upper_query - 2;
                    let mut upper_bound = block_end;
                    let mut lower_bound = lower_bound_lower_index + cursor - 1;

                    // binary search the largest element smaller than the query
                    while lower_bound < upper_bound - 1 {
                        let middle = lower_bound + ((upper_bound - lower_bound) >> 1);

                        let middle_candidate = self
                            .lower_vec
                            .get_bits_unchecked(middle * self.lower_len, self.lower_len);

                        if middle_candidate > lower_query {
                            upper_bound = middle;
                        } else if middle_candidate == lower_query {
                            return (result_upper | middle_candidate) + self.universe_zero;
                        } else {
                            lower_candidate = middle_candidate;
                            lower_bound = middle;
                        }
                    }

                    // check the last element in the binary search interval is smaller than the query,
                    // if it is in the 1-block
                    if lower_bound < block_end {
                        let check_candidate = self
                            .lower_vec
                            .get_bits_unchecked((lower_bound + 1) * self.lower_len, self.lower_len);

                        if check_candidate <= lower_query {
                            return (result_upper | check_candidate) + self.universe_zero;
                        }
                    }

                    break;
                }
            }
            (result_upper | lower_candidate) + self.universe_zero
        } else {
            // return the largest element directly in front of the calculated bounds. This is
            // done when the vector does not contain an element with the query's most significant
            // bit prefix, or when the element at the lower bound is larger than the query.
            self.get(lower_bound_lower_index - 1)
        }
    }

    /// Returns the number of bytes on the heap for this vector. Does not include allocated memory
    /// that isn't used.
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.upper_vec.heap_size() + self.lower_vec.heap_size()
    }
}

#[cfg(test)]
mod tests {
    use crate::EliasFanoVec;
    use rand::distributions::Uniform;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_elias_fano() {
        let ef = EliasFanoVec::new(&vec![0, 1, 4, 7]);

        assert_eq!(ef.len(), 4);
        assert_eq!(ef.get(0), 0);
        assert_eq!(ef.get(1), 1);
        assert_eq!(ef.get(2), 4);
        assert_eq!(ef.get(3), 7);

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

        assert_eq!(ef.get(2), 0);
        assert_eq!(ef.get(3), 1);
        assert_eq!(ef.get(5), 1);
        assert_eq!(ef.get(8), 2);
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
            assert_eq!(ef.get(i), v);
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
            assert_eq!(ef.get(i), x, "expected {:b}", x);
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
}
