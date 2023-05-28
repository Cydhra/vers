use crate::bit_vec::{BitVec, BuildingStrategy};
use crate::RsVector;
use std::cmp::max;

pub struct EliasFanoVec<B: RsVector> {
    upper_vec: B,
    lower_vec: BitVec,
    universe_zero: u64,
    universe_max: u64,
    lower_len: usize,
}

impl<B: RsVector + BuildingStrategy<Vector = B>> EliasFanoVec<B> {
    /// Create a new Elias-Fano vector by compressing the given data. The data must be sorted in
    /// ascending order. The resulting vector is immutable, which will be exploited by limiting the
    /// word length of elements to the minimum required to represent the universe bound.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(data: &Vec<u64>) -> Self {
        // calculate the largest element the vector needs to represent. If there are more elements
        // in the vector than the largest element is able to represent, the length of the vector
        // will be used instead. By limiting the universe size, we can limit the number of bits
        // required to represent each element, and also spread the elements out more evenly through
        // the upper vector. We also subtract the first element from all elements to make the
        // universe start at zero and possibly save some bits for dense distributions.
        let mut universe_zero = data[0];
        let mut universe_bound = data[data.len() - 1] - universe_zero;
        if data.len() > universe_bound as usize {
            universe_zero = 0;
            universe_bound = max(data.len() as u64, data[data.len() - 1]);
        }

        // Calculate the number of bits that will be stored in the lower vector per element. This
        // is the log2 of the universe size rounded up (Rounding up is forced by adding one, so if
        // the log is even, it will be rounded up regardless).
        let lower_width =
            (data.len().leading_zeros() + 1 - universe_bound.leading_zeros()) as usize;

        let mut upper_vec = BitVec::from_zeros(data.len() * 2 + 1);
        let mut lower_vec = BitVec::with_capacity(data.len() * lower_width);

        for (i, &word) in data.iter().enumerate() {
            let word = word - universe_zero;
            let upper = (word >> lower_width) as usize;
            let lower = word & ((1 << lower_width) - 1);

            upper_vec.flip_bit(upper + i + 1);
            lower_vec.append_bits(lower, lower_width);
        }

        Self {
            upper_vec: B::from_bit_vec(upper_vec),
            lower_vec,
            universe_zero,
            universe_max: data[data.len() - 1],
            lower_len: lower_width,
        }
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.lower_vec.len() / self.lower_len
    }

    /// Returns true if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the element at the given index.
    #[allow(clippy::cast_possible_truncation)]
    pub fn get(&self, index: usize) -> u64 {
        let upper = self.upper_vec.select1(index) - index - 1;
        let lower = self
            .lower_vec
            .get_bits(index * self.lower_len, self.lower_len);
        ((upper << self.lower_len) as u64 | lower) + self.universe_zero
    }

    /// Returns the largest element that is smaller than the given element. If the given element is
    /// smaller than the smallest element in the vector, the code will panic or produce a logical
    /// error.
    #[allow(clippy::cast_possible_truncation)]
    pub fn pred(&self, n: u64) -> u64 {
        // bound the query to the universe size
        if n > self.universe_max {
            return self.get(self.len() - 1);
        }

        let n = n - self.universe_zero;

        // split the query into the upper and lower part
        let upper = (n >> self.lower_len) as usize;
        let lower = n & ((1 << self.lower_len) - 1);

        // calculate the bounds within the lower vector where our predecessor can be found. Since
        // each bit-prefix in the universe has exactly one corresponding zero in the upper vector,
        // we can use select0 to find the start of the block of values with the same upper value.
        let lower_bound_upper_index = self.upper_vec.select0(upper);
        let lower_bound_lower_index = lower_bound_upper_index - upper;

        // get the first value from the lower vector that corresponds to the query prefix
        let mut lower_candidate = self
            .lower_vec
            .get_bits(lower_bound_lower_index * self.lower_len, self.lower_len);

        // calculate the upper part of the result. This only works if the next value in the upper
        // vector is set, otherwise the there is no value in the entire vector with this bit-prefix,
        // and we need to search the largest prefix smaller than the query.
        let mut result_upper = (upper << self.lower_len) as u64;

        // check if the next bit is set. If it is not, or if the result would be larger than the
        // query, we need to search for the block of values before the current prefix and return its
        // last element.
        if self.upper_vec.get(lower_bound_upper_index + 1) > 0 && (result_upper | lower_candidate) <= n {
            // search for the largest element in the lower vector that is smaller than the query.
            // Abort the search once the upper vector contains another zero, as this marks the end
            // of the block of values with the same upper prefix.
            let mut cursor = 1;
            while self.upper_vec.get(lower_bound_upper_index + cursor + 1) > 0 {
                let next_candidate = self.lower_vec.get_bits((lower_bound_lower_index + cursor) * self.lower_len, self.lower_len);

                // if we found a value that is larger than the query, return the previous value
                if next_candidate > lower {
                    return (result_upper | lower_candidate) + self.universe_zero;
                } else if next_candidate == lower {
                    return (result_upper | next_candidate) + self.universe_zero;
                } else {
                    lower_candidate = next_candidate;
                }
                cursor += 1;
            }
        } else {
            // return the largest element directly in front of the calculated bounds. This is
            // done when the bounds are equal (i.e. the vector does not contain an element with the
            // query's most significant bit prefix), or when the bounds aren't equal but the element
            // at the lower bound is larger than the query.
            result_upper =
                ((self.upper_vec.select1(lower_bound_lower_index - 1) - lower_bound_lower_index) << self.lower_len) as u64;
            lower_candidate = self
                .lower_vec
                .get_bits((lower_bound_lower_index - 1) * self.lower_len, self.lower_len);
        }

        (result_upper | lower_candidate) + self.universe_zero
    }

    /// Returns the number of bytes on the heap for this vector. Does not include allocated memory
    /// that isn't used.
    pub fn heap_size(&self) -> usize {
        self.upper_vec.heap_size() + self.lower_vec.heap_size()
    }
}

#[cfg(test)]
mod tests {
    use crate::{EliasFanoVec, FastBitVector};
    use rand::{thread_rng, Rng};

    #[test]
    fn test_elias_fano() {
        let ef = EliasFanoVec::<FastBitVector>::new(&vec![0, 1, 4, 7]);

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
        let ef = EliasFanoVec::<FastBitVector>::new(&vec![0, 1, u64::MAX - 10, u64::MAX - 1]);
        assert_eq!(ef.pred(u64::MAX - 11), 1);
    }

    // test a query that is way larger than any element in the vector
    #[test]
    fn test_large_query() {
        let ef = EliasFanoVec::<FastBitVector>::new(&vec![0, 1, 2, 3]);
        assert_eq!(ef.pred(u64::MAX), 3);
    }

    // test whether duplicates are handled correctly by predecessor queries and reconstruction
    #[test]
    fn test_duplicates() {
        let ef = EliasFanoVec::<FastBitVector>::new(&vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);
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

        let ef = EliasFanoVec::<FastBitVector>::new(&seq);

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

            assert_eq!(
                ef.pred(random_splitter),
                seq[seq.partition_point(|&x| x <= random_splitter) - 1]
            );
        }
    }
}
