use std::cmp::min_by;
use std::mem::size_of;

/// As with the naive implementation, this data-structure pre-calculates some queries.
/// The minimum element in intervals 2^k for all k is precalculated and each query is turned into
/// two overlapping sub-queries. This leads to constant-time queries and O(n log n) space overhead.
pub struct SmallNaiveRmq {
    data: Vec<u64>,

    // store indices relative to start of range. There is no way to have ranges exceeding 2^32 bits
    // for reasonable data sizes, because that would exceed 2^43 bits of memory for the input data
    // alone.
    results: Vec<u32>,
}

impl SmallNaiveRmq {
    pub fn new(data: Vec<u64>) -> Self {
        // the results are stored in a one-dimensional array, where the k'th element of each row i is
        // the index of the minimum element in the interval [i, i + 2^k). The length of the row is
        // ceil(log2(data.len())) + 1, which wastes 1/2 + 1/4 + 1/8... = 1 * log n words of memory,
        // but saves us a large amount of page faults for big vectors, when compared to having a
        // two-dimensional array with dynamic length in the second dimension.
        let len = data.len();
        let row_length = len.next_power_of_two().trailing_zeros() as usize + 1;
        let mut results = vec![0u32; len * row_length];

        // initialize the first column of the results array with the indices of the elements in the
        // data array. This is setup for the dynamic programming approach to calculating the rest of
        // the results.
        for i in 0..len {
            results[i * row_length] = 0;
        }

        // calculate the rest of the results using dynamic programming (it uses the minima of smaller
        // intervals to calculate the minima of larger intervals).
        for i in 0..data.len().next_power_of_two().trailing_zeros() {
            let i = i as usize;
            for j in 0..data.len() {
                let offset = 1 << i;
                let arg_min: usize = if j + offset < data.len() {
                    if data[results[j * row_length + i] as usize + j]
                        < data[results[(j + offset) * row_length + i] as usize + (j + offset)]
                    {
                        results[j * row_length + i] as usize + j
                    } else {
                        results[(j + offset) * row_length + i] as usize + (j + offset)
                    }
                } else {
                    // TODO check if all -1 offsets are correct.
                    if data.len() - offset - 1 > j {
                        if data[results[j * row_length + i] as usize + j]
                            < data[results[(data.len() - offset - 1) * row_length + i - 1] as usize
                                + (data.len() - offset - 1)]
                        {
                            results[j * row_length + i] as usize + j
                        } else {
                            results[(data.len() - offset - 1) * row_length + i - 1] as usize
                                + (data.len() - offset - 1)
                        }
                    } else {
                        j
                    }
                };

                results[j * row_length + i + 1] = (arg_min - j) as u32;
            }
        }

        Self { data, results }
    }

    /// Calculates the index of the minimum element in the range [i, j].
    pub fn range_min(&self, i: usize, j: usize) -> usize {
        let row_len = self.data.len().next_power_of_two().trailing_zeros() as usize + 1;
        let log_dist = (usize::BITS - (j - i).leading_zeros()).saturating_sub(1) as usize;
        let dist = (1 << log_dist) - 1;

        // the minimum of the two sub-queries with powers of two is the minimum of the whole query.
        min_by(
            self.results[i * row_len + log_dist] as usize + i,
            self.results[(j - dist) * row_len + log_dist] as usize + (j - dist),
            |a, b| self.data[*a].cmp(&self.data[*b]),
        )
    }

    pub fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>() + self.results.len() * size_of::<u32>()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn small_naive_rmq_test() {
        let rmq = super::SmallNaiveRmq::new(vec![9, 6, 10, 4, 0, 8, 3, 7, 1, 2, 5]);

        assert_eq!(rmq.range_min(0, 0), 0);
        assert_eq!(rmq.range_min(0, 1), 1);
        assert_eq!(rmq.range_min(0, 2), 1);
        assert_eq!(rmq.range_min(0, 3), 3);
        assert_eq!(rmq.range_min(5, 8), 8);
        assert_eq!(rmq.range_min(5, 9), 8);
        assert_eq!(rmq.range_min(9, 10), 9);
        assert_eq!(rmq.range_min(0, 10), 4);
    }
}
