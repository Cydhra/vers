//! This module contains a range minimum query data structure. It pre-computes the
//! minimum element in intervals 2^k for all k and uses this information to answer queries in
//! constant time. This uses O(n log n) space overhead.

use std::cmp::min_by;
use std::collections::Bound;
use std::mem::size_of;
use std::ops::{Deref, RangeBounds};

/// A Range Minimum Query data structure that pre-calculates some queries.
/// The minimum element in intervals 2^k for all k is precalculated and each query is turned into
/// two overlapping sub-queries. This leads to constant-time queries and O(n log n) space overhead.
/// The pre-calculation is done in O(n log n) time.
/// This RMQ data structure is slightly faster than the [fast RMQ][crate::rmq::fast_rmq::FastRmq]
/// for small inputs, but has a much higher space overhead, which makes it slower for large inputs.
/// It does not support input sizes exceeding 2^32 elements.
///
/// # Example
/// ```rust
/// use vers_vecs::BinaryRmq;
///
/// let data = vec![4, 10, 3, 11, 2, 12];
/// let rmq = BinaryRmq::from_vec(data);
///
/// assert_eq!(rmq.range_min(0, 1), 0);
/// assert_eq!(rmq.range_min(0, 2), 2);
/// assert_eq!(rmq.range_min(0, 3), 2);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BinaryRmq {
    data: Vec<u64>,

    // store indices relative to start of range. There is no way to have ranges exceeding 2^32 bits
    // but since we have fast_rmq for larger inputs, which does not have any downsides at that point,
    // we can just use u32 here (which gains cache efficiency for both implementations).
    results: Vec<u32>,
}

impl BinaryRmq {
    /// Create a new RMQ data structure for the given data. This uses O(n log n) space and
    /// precalculates the minimum element in intervals 2^k for all k for all elements.
    #[must_use]
    pub fn from_vec(data: Vec<u64>) -> Self {
        // the results are stored in a one-dimensional array, where the k'th element of each row i is
        // the index of the minimum element in the interval [i, i + 2^k). The length of the row is
        // ceil(log2(data.len())) + 1, which wastes 1/2 + 1/4 + 1/8... = 1 * log n words of memory,
        // but saves us a large amount of page faults for big vectors, when compared to having a
        // two-dimensional array with dynamic length in the second dimension.
        let len = data.len();
        assert!(u32::try_from(len).is_ok(), "input too large for binary rmq");

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
                #[allow(clippy::collapsible_else_if)] // readability
                let arg_min: usize = if j + offset < data.len() {
                    if data[results[j * row_length + i] as usize + j]
                        < data[results[(j + offset) * row_length + i] as usize + (j + offset)]
                    {
                        results[j * row_length + i] as usize + j
                    } else {
                        results[(j + offset) * row_length + i] as usize + (j + offset)
                    }
                } else {
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

                #[allow(clippy::cast_possible_truncation)]
                // we know that the result is in bounds, since the input is bounded to 2^32 elements
                {
                    results[j * row_length + i + 1] = (arg_min - j) as u32;
                }
            }
        }

        Self { data, results }
    }

    /// Convenience function for [`BinaryRmq::range_min`] for using range operators.
    /// The range is clamped to the length of the data structure, so this function will not panic,
    /// unless called on an empty data structure, because that does not have a valid index.
    ///
    /// # Example
    /// ```rust
    /// use vers_vecs::BinaryRmq;
    /// let rmq = BinaryRmq::from_vec(vec![5, 4, 3, 2, 1]);
    /// assert_eq!(rmq.range_min_with_range(0..3), 2);
    /// assert_eq!(rmq.range_min_with_range(0..=3), 3);
    /// ```
    ///
    /// # Panics
    /// This function will panic if the data structure is empty.
    #[must_use]
    pub fn range_min_with_range<T: RangeBounds<usize>>(&self, range: T) -> usize {
        let start = match range.start_bound() {
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i + 1,
            Bound::Unbounded => 0,
        }
        .clamp(0, self.len() - 1);

        let end = match range.end_bound() {
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i - 1,
            Bound::Unbounded => self.len() - 1,
        }
        .clamp(0, self.len() - 1);
        self.range_min(start, end)
    }

    /// Returns the index of the minimum element in the range [i, j] in O(1) time.
    /// This has a constant query time. The range is inclusive.
    ///
    /// # Panics
    /// Calling this function with i > j will produce either a panic or an incorrect result.
    /// Calling this function where one of the indices is out of bounds will produce a panic or an
    /// incorrect result.
    #[must_use]
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

    /// Returns the amount of memory used by this data structure in bytes. This does not include
    /// space allocated but not in use (e.g. unused capacity of vectors).
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>() + self.results.len() * size_of::<u32>()
    }
}

/// Implements Deref to delegate to the underlying data structure. This allows the user to use
/// indexing syntax on the RMQ data structure to access the underlying data, as well as iterators,
/// etc.
impl Deref for BinaryRmq {
    type Target = Vec<u64>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[cfg(test)]
mod tests;
