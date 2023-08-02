//! Elias-Fano encoding for sorted vectors of u64 values. It reduces the space required to represent
//! all numbers (compression ratio dependent on data) and allows for constant time predecessor
//! queries.
//!
//! It implements [`iter`][EliasFanoVec::iter] and [`IntoIterator`][IntoIterator] to allow for
//! iteration over the values in the vector.
//!
//! # Example
//! ```rust
//! use vers_vecs::EliasFanoVec;
//!
//! let mut elias_fano_vec = EliasFanoVec::from_slice(&[0, 9, 29, 109]);
//!
//! assert_eq!(elias_fano_vec.len(), 4);
//! assert_eq!(elias_fano_vec.get_unchecked(0), 0);
//! assert_eq!(elias_fano_vec.get_unchecked(3), 109);
//!
//! assert_eq!(elias_fano_vec.pred(9), 9);
//! assert_eq!(elias_fano_vec.pred(10), 9);
//! assert_eq!(elias_fano_vec.pred(420000), 109);
//! ```

use crate::BitVec;
use crate::RsVec;
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
    /// word length of elements to the minimum required to represent all elements.
    ///
    /// # Panics
    /// The function might panic if the input data is not in ascending order.
    /// Alternatively, it might produce a data structure which contains garbage data.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_slice(data: &[u64]) -> Self {
        debug_assert!(data.windows(2).all(|w| w[0] <= w[1]), "Data must be sorted in ascending order");

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
            upper_vec: RsVec::from_bit_vec(upper_vec),
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

    /// Returns the element at the given index, or `None` if the index exceeds the length of the
    /// vector.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len() {
            return None;
        }

        Some(self.get_unchecked(index))
    }

    /// Returns the element at the given index.
    ///
    /// # Panics
    /// If the index exceeds the length of the vector, the function will panic.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_unchecked(&self, index: usize) -> u64 {
        let upper = self.upper_vec.select1(index) - index - 1;
        let lower = self
            .lower_vec
            .get_bits_unchecked(index * self.lower_len, self.lower_len);
        ((upper << self.lower_len) as u64 | lower) + self.universe_zero
    }

    /// Returns the largest element that is smaller than or equal to the query.
    /// If the query is smaller than the smallest element in the vector,
    /// `u64::MAX` is returned.
    ///
    /// This query runs in constant time on average. The worst case runtime is logarithmic in the
    /// number of elements in the vector. The worst case occurs when values in the vector are very
    /// dense with only very few elements that are much larger than most.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn pred(&self, n: u64) -> u64 {
        // bound the query to the universe size
        if n > self.universe_max {
            return self.get_unchecked(self.len() - 1);
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
            while self
                .upper_vec
                .get_unchecked(lower_bound_upper_index + cursor + 1)
                > 0
            {
                let next_candidate = self.lower_vec.get_bits_unchecked(
                    (lower_bound_lower_index + cursor) * self.lower_len,
                    self.lower_len,
                );

                // if we found a value that is larger than the query, return the previous value
                if next_candidate > lower_query {
                    return (result_upper | lower_candidate) + self.universe_zero;
                } else if next_candidate == lower_query {
                    return (result_upper | next_candidate) + self.universe_zero;
                }

                lower_candidate = next_candidate;
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
            self.get_unchecked(lower_bound_lower_index - 1)
        }
    }

    /// Returns an iterator over the values in the vector.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        EliasFanoVecRefIter { ef: self, index: 0 }
    }

    /// Returns the number of bytes on the heap for this vector. Does not include allocated memory
    /// that isn't used.
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.upper_vec.heap_size() + self.lower_vec.heap_size()
    }
}

/// A borrowing iterator over the values in an Elias-Fano encoded vector.
/// This iterator is created by [`EliasFanoVec::iter`] and [`EliasFanoVec::into_iter`] on references.
pub struct EliasFanoVecRefIter<'a> {
    ef: &'a EliasFanoVec,
    index: usize,
}

impl<'a> Iterator for EliasFanoVecRefIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.ef.get(self.index).map(|v| {
            self.index += 1;
            v
        })
    }
}

/// An owning iterator over the values in an Elias-Fano encoded vector.
/// This iterator is created by [`EliasFanoVec::into_iter`].
pub struct EliasFanoVecIter {
    ef: EliasFanoVec,
    index: usize,
}

impl Iterator for EliasFanoVecIter {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.ef.get(self.index).map(|v| {
            self.index += 1;
            v
        })
    }
}

impl IntoIterator for EliasFanoVec {
    type Item = u64;
    type IntoIter = EliasFanoVecIter;

    fn into_iter(self) -> Self::IntoIter {
        EliasFanoVecIter { ef: self, index: 0 }
    }
}

impl<'a> IntoIterator for &'a EliasFanoVec {
    type Item = u64;
    type IntoIter = EliasFanoVecRefIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        EliasFanoVecRefIter { ef: self, index: 0 }
    }
}

impl<'a> IntoIterator for &'a mut EliasFanoVec {
    type Item = u64;
    type IntoIter = EliasFanoVecRefIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        EliasFanoVecRefIter { ef: self, index: 0 }
    }
}

#[cfg(test)]
mod tests;
