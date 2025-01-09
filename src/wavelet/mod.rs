//! This module contains the implementation of a [wavelet matrix].
//! The wavelet matrix is a data structure that encodes a sequence of `n` `k`-bit words
//! into a matrix of bit vectors, allowing for fast rank and select queries on the encoded sequence.
//!
//! This implementation further supports quantile queries, and range-predecessor and -successor queries.
//! All operations are `O(k)` time complexity.
//!
//! See the struct documentation for more information.
//!
//! [wavelet matrix]: WaveletMatrix

use crate::util::impl_vector_iterator;
use crate::{BitVec, RsVec};
use std::mem;
use std::ops::Range;

/// A wavelet matrix implementation implemented as described in
/// [Navarro and Claude, 2021](http://dx.doi.org/10.1007/978-3-642-34109-0_18).
/// The implementation is designed to allow for extremely large alphabet sizes, without
/// sacrificing performance for small alphabets.
///
/// There are two constructor algorithms available:
/// - [`from_bit_vec`] and [`from_slice`] construct the wavelet matrix by repeatedly sorting the elements.
///   These constructors have linear space overhead and run in `O(kn * log n)` time complexity.
/// - [`from_bit_vec_pc`] and [`from_slice_pc`] construct the wavelet matrix by counting the
///   prefixes of the elements. These constructors have a space complexity of `O(2^k)` and run
///   in `O(kn)`, which makes this constructor preferable for large sequences over small alphabets.
///
/// They encode a sequence of `n` `k`-bit words into a wavelet matrix which supports constant-time
/// rank and select queries on elements of its `k`-bit alphabet.
/// All query functions are mirrored for both `BitVec` and `u64` query elements, so
/// if `k <= 64`, no heap allocation is needed for the query element.
///
/// Other than rank and select queries, the matrix also supports quantile queries (range select i), and
/// range-predecessor and -successor queries, all of which are loosely based on
/// [Külekci and Thankachan](https://doi.org/10.1016/j.jda.2017.01.002) with better time complexity.
///
/// All operations implemented on the matrix are `O(k)` time complexity.
/// The space complexity of the wavelet matrix is `O(n * k)` with a small linear overhead
/// (see [`RsVec`]).
///
/// # Examples
/// ```
/// use vers_vecs::{BitVec, WaveletMatrix};
///
/// // pack elements from a 3-bit alphabet into a bit vector and construct a wavelet matrix from them
/// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
/// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
///
/// // query the wavelet matrix
/// assert_eq!(wavelet_matrix.get_u64(0), Some(1));
/// assert_eq!(wavelet_matrix.get_u64(1), Some(4));
///
// // rank and select queries
/// assert_eq!(wavelet_matrix.rank_u64(3, 4), Some(2));
/// assert_eq!(wavelet_matrix.rank_u64(3, 1), Some(1));
/// assert_eq!(wavelet_matrix.select_u64(0, 7), Some(5));
///
/// // statistics
/// assert_eq!(wavelet_matrix.range_median_u64(0..3), Some(4));
/// assert_eq!(wavelet_matrix.predecessor_u64(0..6, 3), Some(2));
/// ```
///
/// [`RsVec`]: RsVec
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WaveletMatrix {
    data: Box<[RsVec]>,
}

impl WaveletMatrix {
    /// Generic constructor that constructs the wavelet matrix by repeatedly sorting the elements.
    /// The runtime complexity is `O(kn * log n)`.
    ///
    /// # Parameters
    /// - `bits_per_element`: The number of bits in each word. Cannot exceed 1 << 16.
    /// - `num_elements`: The number of elements in the sequence.
    /// - `bit_lookup`: A closure that returns the `bit`-th bit of the `element`-th word.
    #[inline(always)] // should get rid of closures in favor of static calls
    fn permutation_sorting<LOOKUP: Fn(usize, usize) -> u64>(
        bits_per_element: u16,
        num_elements: usize,
        bit_lookup: LOOKUP,
    ) -> Self {
        let element_len = bits_per_element as usize;

        let mut data = vec![BitVec::from_zeros(num_elements); element_len];

        // insert the first bit of each word into the first bit vector
        // for each following level, insert the next bit of each word into the next bit vector
        // sorted stably by the previous bit vector
        let mut permutation = (0..num_elements).collect::<Vec<_>>();
        let mut next_permutation = vec![0; num_elements];

        for (level, data) in data.iter_mut().enumerate() {
            let mut total_zeros = 0;
            for (i, p) in permutation.iter().enumerate() {
                if bit_lookup(*p, element_len - level - 1) == 0 {
                    total_zeros += 1;
                } else {
                    data.set(i, 1).unwrap();
                }
            }

            // scan through the generated bit array and move the elements to the correct position
            // for the next permutation
            if level < element_len - 1 {
                let mut zero_boundary = 0;
                let mut one_boundary = total_zeros;
                for (i, p) in permutation.iter().enumerate() {
                    if data.get_unchecked(i) == 0 {
                        next_permutation[zero_boundary] = *p;
                        zero_boundary += 1;
                    } else {
                        next_permutation[one_boundary] = *p;
                        one_boundary += 1;
                    }
                }

                mem::swap(&mut permutation, &mut next_permutation);
            }
        }

        Self {
            data: data.into_iter().map(BitVec::into).collect(),
        }
    }

    /// Create a new wavelet matrix from a sequence of `n` `k`-bit words.
    /// The constructor runs in `O(kn * log n)` time complexity.
    ///
    /// # Parameters
    /// - `bit_vec`: A packed sequence of `n` `k`-bit words. The `i`-th word begins in the
    ///   `bits_per_element * i`-th bit of the bit vector. Words are stored from least to most
    ///    significant bit.
    /// - `bits_per_element`: The number `k` of bits in each word. Cannot exceed 1 << 16.
    ///
    /// # Panics
    /// Panics if the number of bits in the bit vector is not a multiple of the number of bits per element.
    #[must_use]
    pub fn from_bit_vec(bit_vec: &BitVec, bits_per_element: u16) -> Self {
        assert_eq!(bit_vec.len() % bits_per_element as usize, 0, "The number of bits in the bit vector must be a multiple of the number of bits per element.");
        let num_elements = bit_vec.len() / bits_per_element as usize;
        Self::permutation_sorting(bits_per_element, num_elements, |element, bit| {
            bit_vec.get_unchecked(element * bits_per_element as usize + bit)
        })
    }

    /// Create a new wavelet matrix from a sequence of `n` `k`-bit words.
    /// The constructor runs in `O(kn * log n)` time complexity.
    ///
    /// # Parameters
    /// - `sequence`: A slice of `n` u64 values, each encoding a `k`-bit word.
    /// - `bits_per_element`: The number `k` of bits in each word. Cannot exceed 64.
    ///
    /// # Panics
    /// Panics if the number of bits per element exceeds 64.
    #[must_use]
    pub fn from_slice(sequence: &[u64], bits_per_element: u16) -> Self {
        assert!(
            bits_per_element <= 64,
            "The number of bits per element cannot exceed 64."
        );
        Self::permutation_sorting(bits_per_element, sequence.len(), |element, bit| {
            (sequence[element] >> bit) & 1
        })
    }

    /// Generic constructor that constructs the wavelet matrix by counting the prefixes of the elements.
    /// The runtime complexity is `O(kn)`.
    /// This constructor is only recommended for small alphabets.
    ///
    /// # Parameters
    /// - `bits_per_element`: The number of bits in each word. Cannot exceed 64.
    /// - `num_elements`: The number of elements in the sequence.
    /// - `bit_lookup`: A closure that returns the `bit`-th bit of the `element`-th word.
    /// - `element_lookup`: A closure that returns the `element`-th word.
    #[inline(always)] // should get rid of closures in favor of static calls
    fn prefix_counting<LOOKUP: Fn(usize, usize) -> u64, ELEMENT: Fn(usize) -> u64>(
        bits_per_element: u16,
        num_elements: usize,
        bit_lookup: LOOKUP,
        element_lookup: ELEMENT,
    ) -> Self {
        let element_len = bits_per_element as usize;
        let mut histogram = vec![0usize; 1 << bits_per_element];
        let mut borders = vec![0usize; 1 << bits_per_element];
        let mut data = vec![BitVec::from_zeros(num_elements); element_len];

        for i in 0..num_elements {
            histogram[element_lookup(i) as usize] += 1;
            data[0].set_unchecked(i, bit_lookup(i, element_len - 1));
        }

        for level in (1..element_len).rev() {
            // combine histograms of prefixes
            for h in 0..1 << level {
                histogram[h] = histogram[2 * h] + histogram[2 * h + 1];
            }

            // compute borders of current level, using bit reverse patterns because of the weird
            // node ordering in wavelet matrices
            borders[0] = 0;
            for h in 1usize..1 << level {
                let h_minus_1 = (h - 1).reverse_bits() >> (64 - level);
                borders[h.reverse_bits() >> (64 - level)] =
                    borders[h_minus_1] + histogram[h_minus_1];
            }

            for i in 0..num_elements {
                let bit = bit_lookup(i, element_len - level - 1);
                data[level].set_unchecked(
                    borders[element_lookup(i) as usize >> (element_len - level)],
                    bit,
                );
                borders[element_lookup(i) as usize >> (element_len - level)] += 1;
            }
        }

        Self {
            data: data.into_iter().map(BitVec::into).collect(),
        }
    }

    /// Create a new wavelet matrix from a sequence of `n` `k`-bit words using the prefix counting
    /// algorithm [Dinklage et al.](https://doi.org/10.1145/3457197)
    /// The constructor runs in `O(kn)` time complexity but requires `O(2^k)` space during construction,
    /// so it is only recommended for small alphabets.
    /// Use the [`from_bit_vec`] or [`from_slice`] constructors for larger alphabets.
    ///
    /// # Parameters
    /// - `bit_vec`: A packed sequence of `n` `k`-bit words. The `i`-th word begins in the
    ///   `bits_per_element * i`-th bit of the bit vector. Words are stored from least to most
    ///   significant bit.
    /// - `bits_per_element`: The number `k` of bits in each word. Cannot exceed 1 << 16.
    ///
    /// # Panics
    /// Panics if the number of bits in the bit vector is not a multiple of the number of bits per element,
    /// or if the number of bits per element exceeds 64.
    ///
    /// [`from_bit_vec`]: WaveletMatrix::from_bit_vec
    /// [`from_slice`]: WaveletMatrix::from_slice
    #[must_use]
    pub fn from_bit_vec_pc(bit_vec: &BitVec, bits_per_element: u16) -> Self {
        assert_eq!(bit_vec.len() % bits_per_element as usize, 0, "The number of bits in the bit vector must be a multiple of the number of bits per element.");
        assert!(
            bits_per_element <= 64,
            "The number of bits per element cannot exceed 64."
        );
        let num_elements = bit_vec.len() / bits_per_element as usize;
        Self::prefix_counting(
            bits_per_element,
            num_elements,
            |element, bit| bit_vec.get_unchecked(element * bits_per_element as usize + bit),
            |element| {
                bit_vec.get_bits_unchecked(
                    element * bits_per_element as usize,
                    bits_per_element as usize,
                )
            },
        )
    }

    /// Create a new wavelet matrix from a sequence of `n` `k`-bit words using the prefix counting
    /// algorithm [Dinklage et al.](https://doi.org/10.1145/3457197)
    /// The constructor runs in `O(kn)` time complexity but requires `O(2^k)` space during construction,
    /// so it is only recommended for small alphabets.
    /// Use the [`from_bit_vec`] or [`from_slice`] constructors for larger alphabets.
    ///
    /// # Parameters
    /// - `sequence`: A slice of `n` u64 values, each encoding a `k`-bit word.
    /// - `bits_per_element`: The number `k` of bits in each word. Cannot exceed 64.
    ///
    /// # Panics
    /// Panics if the number of bits per element exceeds 64.
    ///
    /// [`from_bit_vec`]: WaveletMatrix::from_bit_vec
    /// [`from_slice`]: WaveletMatrix::from_slice
    #[must_use]
    pub fn from_slice_pc(sequence: &[u64], bits_per_element: u16) -> Self {
        assert!(
            bits_per_element <= 64,
            "The number of bits per element cannot exceed 64."
        );
        Self::prefix_counting(
            bits_per_element,
            sequence.len(),
            |element, bit| (sequence[element] >> bit) & 1,
            |element| sequence[element],
        )
    }

    /// Generic function to read a value from the wavelet matrix and consume it with a closure.
    /// The function is used by the `get_value` and `get_u64` functions, deduplicating code.
    #[inline(always)]
    fn reconstruct_value_unchecked<F: FnMut(u64)>(&self, mut i: usize, mut target_func: F) {
        for level in 0..self.bits_per_element() {
            let bit = self.data[level].get_unchecked(i);
            target_func(bit);
            if bit == 0 {
                i = self.data[level].rank0(i);
            } else {
                i = self.data[level].rank0 + self.data[level].rank1(i);
            }
        }
    }

    /// Get the `i`-th element of the encoded sequence in a `k`-bit word.
    /// The `k`-bit word is returned as a `BitVec`.
    /// The first element of the bit vector is the least significant bit.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.get_value(0), Some(BitVec::pack_sequence_u8(&[1], 3)));
    /// assert_eq!(wavelet_matrix.get_value(1), Some(BitVec::pack_sequence_u8(&[4], 3)));
    /// assert_eq!(wavelet_matrix.get_value(100), None);
    /// ```
    #[must_use]
    pub fn get_value(&self, i: usize) -> Option<BitVec> {
        if self.data.is_empty() || i >= self.data[0].len() {
            None
        } else {
            Some(self.get_value_unchecked(i))
        }
    }

    /// Get the `i`-th element of the encoded sequence in a `k`-bit word.
    /// The `k`-bit word is returned as a `BitVec`.
    /// The first element of the bit vector is the least significant bit.
    /// This function does not perform bounds checking.
    /// Use [`get_value`] for a checked version.
    ///
    /// # Panics
    /// May panic if the index is out of bounds. May instead return an empty bit vector.
    ///
    /// [`get_value`]: WaveletMatrix::get_value
    #[must_use]
    pub fn get_value_unchecked(&self, i: usize) -> BitVec {
        let mut value = BitVec::from_zeros(self.bits_per_element());
        let mut level = self.bits_per_element() - 1;
        self.reconstruct_value_unchecked(i, |bit| {
            value.set_unchecked(level, bit);
            level = level.saturating_sub(1);
        });
        value
    }

    /// Get the `i`-th element of the encoded sequence as a `u64`.
    /// The `u64` is constructed from the `k`-bit word stored in the wavelet matrix.
    ///
    /// Returns `None` if the index is out of bounds, or if the number of bits per element exceeds 64.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.get_u64(0), Some(1));
    /// assert_eq!(wavelet_matrix.get_u64(1), Some(4));
    /// assert_eq!(wavelet_matrix.get_u64(100), None);
    /// ```
    #[must_use]
    pub fn get_u64(&self, i: usize) -> Option<u64> {
        if self.bits_per_element() > 64 || self.data.is_empty() || i >= self.data[0].len() {
            None
        } else {
            Some(self.get_u64_unchecked(i))
        }
    }

    /// Get the `i`-th element of the encoded sequence as a `u64` numeral.
    /// The element is encoded in the lowest `k` bits of the numeral.
    /// If the number of bits per element exceeds 64, the value is truncated.
    /// This function does not perform bounds checking.
    /// Use [`get_u64`] for a checked version.
    ///
    /// # Panic
    /// May panic if the value of `i` is out of bounds. May instead return 0.
    ///
    /// [`get_u64`]: WaveletMatrix::get_u64
    #[must_use]
    pub fn get_u64_unchecked(&self, i: usize) -> u64 {
        let mut value = 0;
        self.reconstruct_value_unchecked(i, |bit| {
            value <<= 1;
            value |= bit;
        });
        value
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence in the `range`.
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    ///
    /// This method does not perform bounds checking, nor does it check if the `symbol` is a valid
    /// `k`-bit word.
    /// Use [`rank_range`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds,
    /// or if the number of bits in `symbol` is lower than `k`.
    /// May instead return 0.
    ///
    /// [`BitVec`]: BitVec
    /// [`rank_range`]: WaveletMatrix::rank_range
    #[must_use]
    pub fn rank_range_unchecked(&self, mut range: Range<usize>, symbol: &BitVec) -> usize {
        for (level, data) in self.data.iter().enumerate() {
            if symbol.get_unchecked((self.bits_per_element() - 1) - level) == 0 {
                range.start = data.rank0(range.start);
                range.end = data.rank0(range.end);
            } else {
                range.start = data.rank0 + data.rank1(range.start);
                range.end = data.rank0 + data.rank1(range.end);
            }
        }

        range.end - range.start
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence in the `range`.
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    ///
    /// Returns `None` if the `range` is out of bounds (greater than the length of the encoded sequence,
    /// but since it is exclusive, it may be equal to the length),
    /// or if the number of bits in `symbol` is not equal to `k`.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.rank_range(0..3, &BitVec::pack_sequence_u8(&[4], 3)), Some(2));
    /// assert_eq!(wavelet_matrix.rank_range(2..4, &BitVec::pack_sequence_u8(&[4], 3)), Some(1));
    /// ```
    ///
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn rank_range(&self, range: Range<usize>, symbol: &BitVec) -> Option<usize> {
        if range.start >= self.len()
            || range.end > self.len()
            || symbol.len() != self.bits_per_element()
        {
            None
        } else {
            Some(self.rank_range_unchecked(range, symbol))
        }
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence in the `range`.
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    /// The interval is half-open, meaning `rank_range_u64(0..0, symbol)` returns 0.
    ///
    /// This method does not perform bounds checking, nor does it check if the elements of the
    /// wavelet matrix can be represented in a u64 numeral.
    /// Use [`rank_range_u64`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds.
    /// May instead return 0.
    /// If the number of bits in wavelet matrix elements exceed `64`, the behavior is
    /// platform-dependent.
    ///
    /// [`rank_range_u64`]: WaveletMatrix::rank_range_u64
    #[must_use]
    pub fn rank_range_u64_unchecked(&self, mut range: Range<usize>, symbol: u64) -> usize {
        for (level, data) in self.data.iter().enumerate() {
            if (symbol >> ((self.bits_per_element() - 1) - level)) & 1 == 0 {
                range.start = data.rank0(range.start);
                range.end = data.rank0(range.end);
            } else {
                range.start = data.rank0 + data.rank1(range.start);
                range.end = data.rank0 + data.rank1(range.end);
            }
        }

        range.end - range.start
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence in the `range`.
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    /// The interval is half-open, meaning `rank_range_u64(0..0, symbol)` returns 0.
    ///
    /// Returns `None` if the `range` is out of bounds (greater than the length of the encoded sequence,
    /// but since it is exclusive, it may be equal to the length),
    /// or if the number of bits in the wavelet matrix elements exceed `64`.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.rank_range_u64(0..3, 4), Some(2));
    /// assert_eq!(wavelet_matrix.rank_range_u64(2..4, 4), Some(1));
    /// ```
    #[must_use]
    pub fn rank_range_u64(&self, range: Range<usize>, symbol: u64) -> Option<usize> {
        if range.start >= self.len() || range.end > self.len() || self.bits_per_element() > 64 {
            None
        } else {
            Some(self.rank_range_u64_unchecked(range, symbol))
        }
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence between the
    /// `offset`-th and `i`-th element (exclusive).
    /// This is equivalent to ```rank_range_unchecked(offset..i, symbol)```.
    /// The interval is half-open, meaning ```rank_offset_unchecked(0, 0, symbol)``` returns 0.
    ///
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    ///
    /// This method does not perform bounds checking, nor does it check if the `symbol` is a valid
    /// `k`-bit word.
    /// Use [`rank_offset`] for a checked version.
    ///
    /// # Panics
    /// May panic if `offset` is out of bounds,
    /// or if `offset + i` is larger than the length of the encoded sequence,
    /// or if `offset` is greater than `i`,
    /// or if the number of bits in `symbol` is lower than `k`.
    /// May instead return 0.
    ///
    /// [`BitVec`]: BitVec
    /// [`rank_offset`]: WaveletMatrix::rank_offset
    #[must_use]
    pub fn rank_offset_unchecked(&self, offset: usize, i: usize, symbol: &BitVec) -> usize {
        self.rank_range_unchecked(offset..i, symbol)
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence between the
    /// `offset`-th and `i`-th element (exclusive).
    /// This is equivalent to ``rank_range(offset..i, symbol)``.
    /// The interval is half-open, meaning ``rank_offset(0, 0, symbol)`` returns 0,
    /// because the interval is empty.
    ///
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    ///
    /// Returns `None` if `offset` is out of bounds,
    /// or if `i` is larger than the length of the encoded sequence,
    /// or if `offset` is greater than `i`,
    /// or if the number of bits in `symbol` is not equal to `k`.
    /// `i` may equal the length of the encoded sequence,
    /// which will return the number of occurrences of the symbol up to the end of the sequence.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.rank_offset(0, 3, &BitVec::pack_sequence_u8(&[4], 3)), Some(2));
    /// assert_eq!(wavelet_matrix.rank_offset(2, 4, &BitVec::pack_sequence_u8(&[4], 3)), Some(1));
    /// ```
    ///
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn rank_offset(&self, offset: usize, i: usize, symbol: &BitVec) -> Option<usize> {
        if offset > i
            || offset >= self.len()
            || i > self.len()
            || symbol.len() != self.bits_per_element()
        {
            None
        } else {
            Some(self.rank_offset_unchecked(offset, i, symbol))
        }
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence between the
    /// `offset`-th and `i`-th element (exclusive).
    /// This is equivalent to ``rank_range(offset..i, symbol)``.
    /// The interval is half-open, meaning ``rank_offset(0, 0, symbol)`` returns 0.
    ///
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    ///
    /// This method does not perform bounds checking, nor does it check if the elements of the
    /// wavelet matrix can be represented in a u64 numeral.
    /// Use [`rank_offset_u64`] for a checked version.
    ///
    /// # Panics
    /// May panic if `offset` is out of bounds,
    /// or if `i` is larger than the length of the encoded sequence,
    /// or if `offset` is greater than `i`,
    /// or if the number of bits in wavelet matrix elements exceed `64`.
    /// May instead return 0.
    ///
    /// [`rank_offset_u64`]: WaveletMatrix::rank_offset_u64
    #[must_use]
    pub fn rank_offset_u64_unchecked(&self, offset: usize, i: usize, symbol: u64) -> usize {
        self.rank_range_u64_unchecked(offset..i, symbol)
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence between the
    /// `offset`-th and `i`-th element (exclusive).
    /// This is equivalent to ``rank_range(offset..i, symbol)``.
    /// The interval is half-open, meaning ``rank_offset(0, 0, symbol)`` returns 0.
    ///
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    ///
    /// Returns `None` if `offset` is out of bounds,
    /// or if `i` is larger than the length of the encoded sequence,
    /// or if `offset` is greater than `i`,
    /// or if the number of bits in the wavelet matrix elements exceed `64`.
    /// `i` may equal the length of the encoded sequence,
    /// which will return the number of occurrences of the symbol up to the end of the sequence.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.rank_offset_u64(0, 3, 4), Some(2));
    /// assert_eq!(wavelet_matrix.rank_offset_u64(2, 4, 4), Some(1));
    /// ```
    #[must_use]
    pub fn rank_offset_u64(&self, offset: usize, i: usize, symbol: u64) -> Option<usize> {
        if offset > i || offset >= self.len() || i > self.len() || self.bits_per_element() > 64 {
            None
        } else {
            Some(self.rank_offset_u64_unchecked(offset, i, symbol))
        }
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence up to the `i`-th
    /// element (exclusive).
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    ///
    /// This method does not perform bounds checking, nor does it check if the `symbol` is a valid
    /// `k`-bit word.
    /// Use [`rank`] for a checked version.
    ///
    /// # Panics
    /// May panic if `i` is out of bounds, or if the number of bits in `symbol` is lower than `k`.
    /// May instead return 0.
    /// If the number of bits in `symbol` exceeds `k`, the remaining bits are ignored.
    ///
    /// [`BitVec`]: BitVec
    /// [`rank`]: WaveletMatrix::rank
    #[must_use]
    pub fn rank_unchecked(&self, i: usize, symbol: &BitVec) -> usize {
        self.rank_range_unchecked(0..i, symbol)
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence up to the `i`-th
    /// element (exclusive).
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    ///
    /// Returns `None` if `i` is out of bounds (greater than the length of the encoded sequence, but
    /// since it is exclusive, it may be equal to the length),
    /// or if the number of bits in `symbol` is not equal to `k`.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.rank(3, &BitVec::pack_sequence_u8(&[4], 3)), Some(2));
    /// assert_eq!(wavelet_matrix.rank(3, &BitVec::pack_sequence_u8(&[1], 3)), Some(1));
    /// ```
    ///
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn rank(&self, i: usize, symbol: &BitVec) -> Option<usize> {
        if i > self.len() || symbol.len() != self.bits_per_element() {
            None
        } else {
            Some(self.rank_range_unchecked(0..i, symbol))
        }
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence up to the `i`-th
    /// element (exclusive).
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    ///
    /// This method does not perform bounds checking, nor does it check if the elements of the
    /// wavelet matrix can be represented in a u64 numeral.
    /// Use [`rank_u64`] for a checked version.
    ///
    /// # Panics
    /// May panic if `i` is out of bounds,
    /// or if the number of bits in wavelet matrix elements exceed `64`.
    /// May instead return 0.
    ///
    /// [`rank_u64`]: WaveletMatrix::rank_u64
    #[must_use]
    pub fn rank_u64_unchecked(&self, i: usize, symbol: u64) -> usize {
        self.rank_range_u64_unchecked(0..i, symbol)
    }

    /// Get the number of occurrences of the given `symbol` in the encoded sequence up to the `i`-th
    /// element (exclusive).
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    ///
    /// Returns `None` if `i` is out of bounds (greater than the length of the encoded sequence, but
    /// since it is exclusive, it may be equal to the length),
    /// or if the number of bits in the wavelet matrix elements exceed `64`.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.rank_u64(3, 4), Some(2));
    /// assert_eq!(wavelet_matrix.rank_u64(3, 1), Some(1));
    /// ```
    #[must_use]
    pub fn rank_u64(&self, i: usize, symbol: u64) -> Option<usize> {
        if i > self.len() || self.bits_per_element() > 64 {
            None
        } else {
            Some(self.rank_range_u64_unchecked(0..i, symbol))
        }
    }

    /// Get the index of the `rank`-th occurrence of the given `symbol` in the encoded sequence,
    /// starting from the `offset`-th element.
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    ///
    /// This method does not perform bounds checking, nor does it check if the `symbol` is a valid
    /// `k`-bit word.
    /// Use [`select_offset`] for a checked version.
    ///
    /// Returns the index of the `rank`-th occurrence of the `symbol` in the encoded sequence,
    /// or the length of the encoded sequence if the `rank`-th occurrence does not exist.
    ///
    /// # Panics
    /// May panic if the `offset` is out of bounds,
    /// or if the number of bits in `symbol` is lower than `k`.
    /// May instead return the length of the encoded sequence.
    ///
    /// [`BitVec`]: BitVec
    /// [`select_offset`]: WaveletMatrix::select_offset
    #[must_use]
    pub fn select_offset_unchecked(&self, offset: usize, rank: usize, symbol: &BitVec) -> usize {
        let mut range_start = offset;

        for (level, data) in self.data.iter().enumerate() {
            if symbol.get_unchecked((self.bits_per_element() - 1) - level) == 0 {
                range_start = data.rank0(range_start);
            } else {
                range_start = data.rank0 + data.rank1(range_start);
            }
        }

        let mut range_end = range_start + rank;

        for (level, data) in self.data.iter().enumerate().rev() {
            if symbol.get_unchecked((self.bits_per_element() - 1) - level) == 0 {
                range_end = data.select0(range_end);
            } else {
                range_end = data.select1(range_end - data.rank0);
            }
        }

        range_end
    }

    /// Get the index of the `rank`-th occurrence of the given `symbol` in the encoded sequence,
    /// starting from the `offset`-th element.
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    ///
    /// Returns `None` if `offset` is out of bounds, or if the number of bits in `symbol` is not equal to `k`,
    /// or if the `rank`-th occurrence of the `symbol` does not exist.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.select_offset(0, 0, &BitVec::pack_sequence_u8(&[4], 3)), Some(1));
    /// assert_eq!(wavelet_matrix.select_offset(0, 1, &BitVec::pack_sequence_u8(&[4], 3)), Some(2));
    /// assert_eq!(wavelet_matrix.select_offset(2, 0, &BitVec::pack_sequence_u8(&[4], 3)), Some(2));
    /// assert_eq!(wavelet_matrix.select_offset(2, 1, &BitVec::pack_sequence_u8(&[4], 3)), None);
    /// ```
    ///
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn select_offset(&self, offset: usize, rank: usize, symbol: &BitVec) -> Option<usize> {
        if offset >= self.len() || symbol.len() != self.bits_per_element() {
            None
        } else {
            let idx = self.select_offset_unchecked(offset, rank, symbol);
            if idx < self.len() {
                Some(idx)
            } else {
                None
            }
        }
    }

    /// Get the index of the `rank`-th occurrence of the given `symbol` in the encoded sequence,
    /// starting from the `offset`-th element.
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    ///
    /// This method does not perform bounds checking, nor does it check if the elements of the
    /// wavelet matrix can be represented in a u64 numeral.
    /// Use [`select_offset_u64`] for a checked version.
    ///
    /// Returns the index of the `rank`-th occurrence of the `symbol` in the encoded sequence,
    /// or the length of the encoded sequence if the `rank`-th occurrence does not exist.
    ///
    /// # Panics
    /// May panic if the `offset` is out of bounds,
    /// or if the number of bits in wavelet matrix elements exceed `64`.
    /// May instead return the length of the encoded sequence.
    ///
    /// [`select_offset_u64`]: WaveletMatrix::select_offset_u64
    #[must_use]
    pub fn select_offset_u64_unchecked(&self, offset: usize, rank: usize, symbol: u64) -> usize {
        let mut range_start = offset;

        for (level, data) in self.data.iter().enumerate() {
            if (symbol >> ((self.bits_per_element() - 1) - level)) & 1 == 0 {
                range_start = data.rank0(range_start);
            } else {
                range_start = data.rank0 + data.rank1(range_start);
            }
        }

        let mut range_end = range_start + rank;

        for (level, data) in self.data.iter().enumerate().rev() {
            if (symbol >> ((self.bits_per_element() - 1) - level)) & 1 == 0 {
                range_end = data.select0(range_end);
            } else {
                range_end = data.select1(range_end - data.rank0);
            }
        }

        range_end
    }

    /// Get the index of the `rank`-th occurrence of the given `symbol` in the encoded sequence,
    /// starting from the `offset`-th element.
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    ///
    /// Returns `None` if `offset` is out of bounds, or if the number of bits in the wavelet matrix
    /// elements exceed `64`, or if the `rank`-th occurrence of the `symbol` does not exist.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.select_offset_u64(0, 0, 4), Some(1));
    /// assert_eq!(wavelet_matrix.select_offset_u64(0, 1, 4), Some(2));
    /// assert_eq!(wavelet_matrix.select_offset_u64(2, 0, 4), Some(2));
    /// assert_eq!(wavelet_matrix.select_offset_u64(2, 1, 4), None);
    /// ```
    #[must_use]
    pub fn select_offset_u64(&self, offset: usize, rank: usize, symbol: u64) -> Option<usize> {
        if offset >= self.len() || self.bits_per_element() > 64 {
            None
        } else {
            let idx = self.select_offset_u64_unchecked(offset, rank, symbol);
            if idx < self.len() {
                Some(idx)
            } else {
                None
            }
        }
    }

    /// Get the index of the `rank`-th occurrence of the given `symbol` in the encoded sequence.
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    ///
    /// This method does not perform bounds checking, nor does it check if the `symbol` is a valid
    /// `k`-bit word.
    /// Use [`select`] for a checked version.
    ///
    /// Returns the index of the `rank`-th occurrence of the `symbol` in the encoded sequence,
    /// or the length of the encoded sequence if the `rank`-th occurrence does not exist.
    ///
    /// # Panics
    /// May panic if the number of bits in `symbol` is not equal to `k`.
    /// May instead return the length of the encoded sequence.
    ///
    /// [`BitVec`]: BitVec
    /// [`select`]: WaveletMatrix::select
    #[must_use]
    pub fn select_unchecked(&self, rank: usize, symbol: &BitVec) -> usize {
        self.select_offset_unchecked(0, rank, symbol)
    }

    /// Get the index of the `rank`-th occurrence of the given `symbol` in the encoded sequence.
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    ///
    /// Returns `None` if the number of bits in `symbol` is not equal to `k`,
    /// or if the `rank`-th occurrence of the `symbol` does not exist.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.select(0, &BitVec::pack_sequence_u8(&[4], 3)), Some(1));
    /// assert_eq!(wavelet_matrix.select(1, &BitVec::pack_sequence_u8(&[4], 3)), Some(2));
    /// ```
    ///
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn select(&self, rank: usize, symbol: &BitVec) -> Option<usize> {
        if symbol.len() == self.bits_per_element() {
            let idx = self.select_unchecked(rank, symbol);
            if idx < self.len() {
                Some(idx)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get the index of the `rank`-th occurrence of the given `symbol` in the encoded sequence.
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    ///
    /// This method does not perform bounds checking, nor does it check if the elements of the
    /// wavelet matrix can be represented in a u64 numeral.
    /// Use [`select_u64`] for a checked version.
    ///
    /// Returns the index of the `rank`-th occurrence of the `symbol` in the encoded sequence,
    /// or the length of the encoded sequence if the `rank`-th occurrence does not exist.
    ///
    /// # Panics
    /// May panic if the number of bits in wavelet matrix elements exceed `64`.
    /// May instead return the length of the encoded sequence.
    ///
    /// [`select_u64`]: WaveletMatrix::select_u64
    #[must_use]
    pub fn select_u64_unchecked(&self, rank: usize, symbol: u64) -> usize {
        self.select_offset_u64_unchecked(0, rank, symbol)
    }

    /// Get the index of the `rank`-th occurrence of the given `symbol` in the encoded sequence.
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    ///
    /// Returns `None` if the number of bits in the wavelet matrix elements exceed `64`,
    /// or if the `rank`-th occurrence of the `symbol` does not exist.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.select_u64(0, 4), Some(1));
    /// assert_eq!(wavelet_matrix.select_u64(1, 4), Some(2));
    /// ```
    #[must_use]
    pub fn select_u64(&self, rank: usize, symbol: u64) -> Option<usize> {
        if self.bits_per_element() > 64 {
            None
        } else {
            let idx = self.select_u64_unchecked(rank, symbol);
            if idx < self.len() {
                Some(idx)
            } else {
                None
            }
        }
    }

    /// Get the `k`-th smallest element in the encoded sequence in the specified `range`,
    /// where `k = 0` returns the smallest element.
    /// The `range` is a half-open interval, meaning that the `end` index is exclusive.
    /// The `k`-th smallest element is returned as a `BitVec`,
    /// where the least significant bit is the first element.
    ///
    /// This method does not perform bounds checking.
    /// It returns a nonsensical result if the `k` is greater than the size of the range.
    /// Use [`quantile`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds. May instead return an empty bit vector.
    ///
    /// [`quantile`]: WaveletMatrix::quantile
    #[must_use]
    pub fn quantile_unchecked(&self, range: Range<usize>, k: usize) -> BitVec {
        let result = BitVec::from_zeros(self.bits_per_element());

        self.partial_quantile_search_unchecked(range, k, 0, result)
    }

    /// Internal function to reuse the quantile code for the predecessor and successor search.
    /// This function performs the quantile search starting at the given level with the given prefix.
    ///
    /// The function does not perform any checks, so the caller must ensure that the range is valid,
    /// and that the prefix is a valid prefix for the given level.
    #[inline(always)]
    fn partial_quantile_search_unchecked(
        &self,
        mut range: Range<usize>,
        mut k: usize,
        start_level: usize,
        mut prefix: BitVec,
    ) -> BitVec {
        debug_assert!(prefix.len() == self.bits_per_element());
        debug_assert!(!range.is_empty());
        debug_assert!(range.end <= self.len());

        for (level, data) in self.data.iter().enumerate().skip(start_level) {
            let zeros_start = data.rank0(range.start);
            let zeros_end = data.rank0(range.end);
            let zeros = zeros_end - zeros_start;

            // if k < zeros, the element is among the zeros
            if k < zeros {
                range.start = zeros_start;
                range.end = zeros_end;
            } else {
                // the element is among the ones, so we set the bit to 1, and move the range
                // into the 1-partition of the next level
                prefix.set_unchecked((self.bits_per_element() - 1) - level, 1);
                k -= zeros;
                range.start = data.rank0 + (range.start - zeros_start); // range.start - zeros_start is the rank1 of range.start
                range.end = data.rank0 + (range.end - zeros_end); // same here
            }
        }

        prefix
    }

    /// Get the `k`-th smallest element in the encoded sequence in the specified `range`,
    /// where `k = 0` returns the smallest element.
    /// The `range` is a half-open interval, meaning that the `end` index is exclusive.
    /// The `k`-th smallest element is returned as a `BitVec`,
    /// where the least significant bit is the first element.
    ///
    /// Returns `None` if the `range` is out of bounds, or if `k` is greater than the size of the range.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.quantile(0..3, 0), Some(BitVec::pack_sequence_u8(&[1], 3)));
    /// assert_eq!(wavelet_matrix.quantile(0..3, 1), Some(BitVec::pack_sequence_u8(&[4], 3)));
    /// assert_eq!(wavelet_matrix.quantile(1..4, 0), Some(BitVec::pack_sequence_u8(&[1], 3)));
    /// ```
    #[must_use]
    pub fn quantile(&self, range: Range<usize>, k: usize) -> Option<BitVec> {
        if range.start >= self.len() || range.end > self.len() || k >= range.end - range.start {
            None
        } else {
            Some(self.quantile_unchecked(range, k))
        }
    }

    /// Get the `i`-th smallest element in the entire wavelet matrix.
    /// The `i`-th smallest element is returned as a `BitVec`,
    /// where the least significant bit is the first element.
    ///
    /// This method does not perform bounds checking.
    /// Use [`get_sorted`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `i` is out of bounds, or returns an empty bit vector.
    #[must_use]
    pub fn get_sorted_unchecked(&self, i: usize) -> BitVec {
        self.quantile_unchecked(0..self.len(), i)
    }

    /// Get the `i`-th smallest element in the wavelet matrix.
    /// The `i`-th smallest element is returned as a `BitVec`,
    /// where the least significant bit is the first element.
    /// This method call is equivalent to `self.quantile(0..self.len(), i)`.
    ///
    /// Returns `None` if the `i` is out of bounds.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.get_sorted(0), Some(BitVec::pack_sequence_u8(&[1], 3)));
    /// assert_eq!(wavelet_matrix.get_sorted(1), Some(BitVec::pack_sequence_u8(&[1], 3)));
    /// assert_eq!(wavelet_matrix.get_sorted(2), Some(BitVec::pack_sequence_u8(&[2], 3)));
    /// ```
    #[must_use]
    pub fn get_sorted(&self, i: usize) -> Option<BitVec> {
        if i >= self.len() {
            None
        } else {
            Some(self.get_sorted_unchecked(i))
        }
    }

    /// Get the `k`-th smallest element in the encoded sequence in the specified `range`,
    /// where `k = 0` returns the smallest element.
    /// The `range` is a half-open interval, meaning that the `end` index is exclusive.
    /// The `k`-th smallest element is returned as a `u64` numeral.
    /// If the number of bits per element exceeds 64, the value is truncated.
    ///
    /// This method does not perform bounds checking.
    /// It returns a nonsensical result if the `k` is greater than the size of the range.
    /// Use [`quantile_u64`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds.
    /// May instead return 0.
    ///
    /// [`quantile_u64`]: WaveletMatrix::quantile_u64
    #[must_use]
    pub fn quantile_u64_unchecked(&self, range: Range<usize>, k: usize) -> u64 {
        self.partial_quantile_search_u64_unchecked(range, k, 0, 0)
    }

    /// Internal function to reuse the quantile code for the predecessor and successor search.
    /// This function performs the quantile search starting at the given level with the given prefix.
    ///
    /// The function does not perform any checks, so the caller must ensure that the range is valid,
    /// and that the prefix is a valid prefix for the given level (i.e. the prefix is not shifted
    /// to another level).
    #[inline(always)]
    fn partial_quantile_search_u64_unchecked(
        &self,
        mut range: Range<usize>,
        mut k: usize,
        start_level: usize,
        mut prefix: u64,
    ) -> u64 {
        debug_assert!(self.bits_per_element() <= 64);
        debug_assert!(!range.is_empty());
        debug_assert!(range.end <= self.len());

        for data in self.data.iter().skip(start_level) {
            prefix <<= 1;
            let zeros_start = data.rank0(range.start);
            let zeros_end = data.rank0(range.end);
            let zeros = zeros_end - zeros_start;

            if k < zeros {
                range.start = zeros_start;
                range.end = zeros_end;
            } else {
                prefix |= 1;
                k -= zeros;
                range.start = data.rank0 + (range.start - zeros_start);
                range.end = data.rank0 + (range.end - zeros_end);
            }
        }

        prefix
    }

    /// Get the `k`-th smallest element in the encoded sequence in the specified `range`,
    /// where `k = 0` returns the smallest element.
    /// The `range` is a half-open interval, meaning that the `end` index is exclusive.
    /// The `k`-th smallest element is returned as a `u64` numeral.
    ///
    /// Returns `None` if the `range` is out of bounds, or if the number of bits per element exceeds 64,
    /// or if `k` is greater than the size of the range.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.quantile_u64(0..3, 0), Some(1));
    /// assert_eq!(wavelet_matrix.quantile_u64(0..3, 1), Some(4));
    /// assert_eq!(wavelet_matrix.quantile_u64(1..4, 0), Some(1));
    /// ```
    #[must_use]
    pub fn quantile_u64(&self, range: Range<usize>, k: usize) -> Option<u64> {
        if range.start >= self.len()
            || range.end > self.len()
            || self.bits_per_element() > 64
            || k >= range.end - range.start
        {
            None
        } else {
            Some(self.quantile_u64_unchecked(range, k))
        }
    }

    /// Get the `i`-th smallest element in the wavelet matrix.
    /// The `i`-th smallest element is returned as a u64 numeral.
    ///
    /// If the number of bits per element exceeds 64, the value is truncated.
    ///
    /// This method does not perform bounds checking.
    /// Use [`get_sorted_u64`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `i` is out of bounds, or returns an empty bit vector.
    ///
    /// [`get_sorted_u64`]: WaveletMatrix::get_sorted_u64
    #[must_use]
    pub fn get_sorted_u64_unchecked(&self, i: usize) -> u64 {
        self.quantile_u64_unchecked(0..self.len(), i)
    }

    /// Get the `i`-th smallest element in the entire wavelet matrix.
    /// The `i`-th smallest element is returned as a u64 numeral.
    ///
    /// Returns `None` if the `i` is out of bounds, or if the number of bits per element exceeds 64.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.get_sorted_u64(0), Some(1));
    /// assert_eq!(wavelet_matrix.get_sorted_u64(1), Some(1));
    /// assert_eq!(wavelet_matrix.get_sorted_u64(2), Some(2));
    /// ```
    #[must_use]
    pub fn get_sorted_u64(&self, i: usize) -> Option<u64> {
        if i >= self.len() || self.bits_per_element() > 64 {
            None
        } else {
            Some(self.get_sorted_u64_unchecked(i))
        }
    }

    /// Get the smallest element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The smallest element is returned as a `BitVec`,
    ///
    /// This method does not perform bounds checking.
    /// Use [`range_min`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return an empty bit vector.
    ///
    /// [`range_min`]: WaveletMatrix::range_min
    #[must_use]
    pub fn range_min_unchecked(&self, range: Range<usize>) -> BitVec {
        self.quantile_unchecked(range, 0)
    }

    /// Get the smallest element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The smallest element is returned as a `BitVec`,
    ///
    /// Returns `None` if the `range` is out of bounds or if the range is empty.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.range_min(0..3), Some(BitVec::pack_sequence_u8(&[1], 3)));
    /// assert_eq!(wavelet_matrix.range_min(1..4), Some(BitVec::pack_sequence_u8(&[1], 3)));
    /// assert_eq!(wavelet_matrix.range_min(1..3), Some(BitVec::pack_sequence_u8(&[4], 3)));
    /// ```
    #[must_use]
    pub fn range_min(&self, range: Range<usize>) -> Option<BitVec> {
        self.quantile(range, 0)
    }

    /// Get the smallest element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The smallest element is returned as a `u64` numeral.
    /// If the number of bits per element exceeds 64, the value is truncated.
    ///
    /// This method does not perform bounds checking.
    /// Use [`range_min_u64`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return 0.
    ///
    /// [`range_min_u64`]: WaveletMatrix::range_min_u64
    #[must_use]
    pub fn range_min_u64_unchecked(&self, range: Range<usize>) -> u64 {
        self.quantile_u64_unchecked(range, 0)
    }

    /// Get the smallest element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The smallest element is returned as a `u64` numeral.
    ///
    /// Returns `None` if the `range` is out of bounds, if the range is empty, or if the number of bits
    /// per element exceeds 64.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.range_min_u64(0..3), Some(1));
    /// assert_eq!(wavelet_matrix.range_min_u64(1..4), Some(1));
    /// assert_eq!(wavelet_matrix.range_min_u64(1..3), Some(4));
    /// ```
    #[must_use]
    pub fn range_min_u64(&self, range: Range<usize>) -> Option<u64> {
        self.quantile_u64(range, 0)
    }

    /// Get the largest element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The largest element is returned as a `BitVec`,
    /// where the least significant bit is the first element.
    ///
    /// This method does not perform bounds checking.
    /// Use [`range_max`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return an empty bit vector.
    ///
    /// [`range_max`]: WaveletMatrix::range_max
    #[must_use]
    pub fn range_max_unchecked(&self, range: Range<usize>) -> BitVec {
        let k = range.end - range.start - 1;
        self.quantile_unchecked(range, k)
    }

    /// Get the largest element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The largest element is returned as a `BitVec`,
    /// where the least significant bit is the first element.
    ///
    /// Returns `None` if the `range` is out of bounds or if the range is empty.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.range_max(0..3), Some(BitVec::pack_sequence_u8(&[4], 3)));
    /// assert_eq!(wavelet_matrix.range_max(3..6), Some(BitVec::pack_sequence_u8(&[7], 3)));
    /// ```
    #[must_use]
    pub fn range_max(&self, range: Range<usize>) -> Option<BitVec> {
        if range.is_empty() {
            None
        } else {
            let k = range.end - range.start - 1;
            self.quantile(range, k)
        }
    }

    /// Get the largest element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The largest element is returned as a `u64` numeral.
    /// If the number of bits per element exceeds 64, the value is truncated.
    ///
    /// This method does not perform bounds checking.
    /// Use [`range_max_u64`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return 0.
    ///
    /// [`range_max_u64`]: WaveletMatrix::range_max_u64
    #[must_use]
    pub fn range_max_u64_unchecked(&self, range: Range<usize>) -> u64 {
        let k = range.end - range.start - 1;
        self.quantile_u64_unchecked(range, k)
    }

    /// Get the largest element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The largest element is returned as a `u64` numeral.
    ///
    /// Returns `None` if the `range` is out of bounds, if the range is empty, or if the number of bits
    /// per element exceeds 64.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.range_max_u64(0..3), Some(4));
    /// assert_eq!(wavelet_matrix.range_max_u64(3..6), Some(7));
    /// ```
    #[must_use]
    pub fn range_max_u64(&self, range: Range<usize>) -> Option<u64> {
        if range.is_empty() {
            None
        } else {
            let k = range.end - range.start - 1;
            self.quantile_u64(range, k)
        }
    }

    /// Get the median element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The median element is returned as a `BitVec`,
    /// where the least significant bit is the first element.
    ///
    /// If the range does not contain an odd number of elements, the position is rounded down.
    ///
    /// This method does not perform bounds checking.
    /// Use [`range_median`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return an empty bit vector.
    ///
    /// [`range_median`]: WaveletMatrix::range_median
    #[must_use]
    pub fn range_median_unchecked(&self, range: Range<usize>) -> BitVec {
        let k = (range.end - 1 - range.start) / 2;
        self.quantile_unchecked(range, k)
    }

    /// Get the median element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The median element is returned as a `BitVec`,
    /// where the least significant bit is the first element.
    ///
    /// If the range does not contain an odd number of elements, the position is rounded down.
    ///
    /// Returns `None` if the `range` is out of bounds or if the range is empty.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.range_median(0..3), Some(BitVec::pack_sequence_u8(&[4], 3)));
    /// assert_eq!(wavelet_matrix.range_median(1..4), Some(BitVec::pack_sequence_u8(&[4], 3)));
    /// assert_eq!(wavelet_matrix.range_median(0..6), Some(BitVec::pack_sequence_u8(&[2], 3)));
    /// ```
    #[must_use]
    pub fn range_median(&self, range: Range<usize>) -> Option<BitVec> {
        if range.is_empty() {
            None
        } else {
            let k = (range.end - 1 - range.start) / 2;
            self.quantile(range, k)
        }
    }

    /// Get the median element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The median element is returned as a `u64` numeral.
    /// If the number of bits per element exceeds 64, the value is truncated.
    ///
    /// If the range does not contain an odd number of elements, the position is rounded down.
    ///
    /// This method does not perform bounds checking.
    /// Use [`range_median_u64`] for a checked version.
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return 0.
    ///
    /// [`range_median_u64`]: WaveletMatrix::range_median_u64
    #[must_use]
    pub fn range_median_u64_unchecked(&self, range: Range<usize>) -> u64 {
        let k = (range.end - 1 - range.start) / 2;
        self.quantile_u64_unchecked(range, k)
    }

    /// Get the median element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The median element is returned as a `u64` numeral.
    ///
    /// If the range does not contain an odd number of elements, the position is rounded down.
    ///
    /// Returns `None` if the `range` is out of bounds, if the range is empty, or if the number of bits
    /// per element exceeds 64.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.range_median_u64(0..3), Some(4));
    /// assert_eq!(wavelet_matrix.range_median_u64(1..4), Some(4));
    /// assert_eq!(wavelet_matrix.range_median_u64(0..6), Some(2));
    /// ```
    #[must_use]
    pub fn range_median_u64(&self, range: Range<usize>) -> Option<u64> {
        if range.is_empty() || self.bits_per_element() > 64 || range.end > self.len() {
            None
        } else {
            let k = (range.end - 1 - range.start) / 2;
            self.quantile_u64(range, k)
        }
    }

    /// Get the predecessor of the given `symbol` in the given `range`.
    /// This is a private generic helper function to implement the public `predecessor` functions.
    ///
    /// The read and write access to the query and result values are abstracted by the `Reader` and `Writer` closures.
    #[inline(always)] // even though the function is pretty large, inlining probably gets rid of the closure calls in favor of static calls
    fn predecessor_generic_unchecked<
        T: Clone,
        Reader: Fn(usize, &T) -> u64,
        Writer: Fn(u64, usize, &mut T),
        Quantile: Fn(&Self, Range<usize>, usize, usize, T) -> T,
    >(
        &self,
        mut range: Range<usize>,
        symbol: &T,
        mut result_value: T,
        bit_reader: Reader,
        result_writer: Writer,
        quantile_search: Quantile,
    ) -> Option<T> {
        // the bit-prefix at the last node where we could go to an interval with smaller elements,
        // i.e. where we need to go if the current prefix has no elements smaller than the query
        let mut last_smaller_prefix = result_value.clone();
        // the level of the last node where we could go to an interval with smaller elements
        let mut last_one_level: Option<usize> = None;
        // the range of the last node where we could go to an interval with smaller elements
        let mut next_smaller_range: Option<Range<usize>> = None;

        for (level, data) in self.data.iter().enumerate() {
            let query_bit = bit_reader(level, symbol);

            // read the amount of elements with the current result-prefix plus one 0 bit.
            // if the query_bit is 1, we can calculate the amount of elements with the result-prefix
            // plus one 1 from there
            let zeros_start = data.rank0(range.start);
            let zeros_end = data.rank0(range.end);
            let elements_zero = zeros_end - zeros_start;

            if query_bit == 0 {
                if elements_zero == 0 {
                    // if our query bit is zero in this level and suddenly our new interval is empty,
                    // all elements that were in the previous interval are bigger than the query element,
                    // because they all have a 1 in the current level.
                    // this means the predecessor is the largest element in the last smaller interval,
                    // i.e. the interval that has a 0 bit at the last level where our prefix had a 1 bit.

                    return next_smaller_range.map(|r| {
                        let idx = r.end - r.start - 1;
                        result_writer(0, last_one_level.unwrap(), &mut last_smaller_prefix);
                        quantile_search(
                            self,
                            r,
                            idx,
                            last_one_level.unwrap() + 1,
                            last_smaller_prefix,
                        )
                    });
                }

                // update the prefix
                result_writer(0, level, &mut result_value);

                // update the range to the interval of the new prefix
                range.start = zeros_start;
                range.end = zeros_end;
            } else {
                if elements_zero == range.end - range.start {
                    // if our query element is 1 in this level and suddenly our new interval is empty,
                    // all elements that were in the previous interval are smaller than the query element,
                    // because they all have a 0 in the current level.
                    // this means the predecessor is the largest element in the last level's interval.

                    let idx = range.end - range.start - 1;
                    return Some(quantile_search(self, range, idx, level, result_value));
                }

                // if the other interval is not empty, we update the last smaller interval to the last interval where we can switch to a 0 prefix
                if !(zeros_start..zeros_end).is_empty() {
                    last_one_level = Some(level);
                    next_smaller_range = Some(zeros_start..zeros_end);
                    last_smaller_prefix = result_value.clone();
                }

                // update the prefix
                result_writer(1, level, &mut result_value);

                // update the range to the interval of the new prefix
                range.start = data.rank0 + (range.start - zeros_start);
                range.end = data.rank0 + (range.end - zeros_end);
            }
        }

        Some(result_value)
    }

    /// Get the predecessor of the given `symbol` in the given `range`.
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    /// The `symbol` does not have to be in the encoded sequence.
    /// The predecessor is the largest element in the `range` that is smaller than or equal
    /// to the `symbol`.
    ///
    /// Returns `None` if the number of bits in the `symbol` is not equal to `k`,
    /// if the range is empty, if the wavelet matrix is empty, if the range is out of bounds,
    /// or if the `symbol` is smaller than all elements in the range.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.predecessor(0..3, &BitVec::pack_sequence_u8(&[7], 3)), Some(BitVec::pack_sequence_u8(&[4], 3)));
    /// assert_eq!(wavelet_matrix.predecessor(0..3, &BitVec::pack_sequence_u8(&[4], 3)), Some(BitVec::pack_sequence_u8(&[4], 3)));
    /// assert_eq!(wavelet_matrix.predecessor(0..6, &BitVec::pack_sequence_u8(&[7], 3)), Some(BitVec::pack_sequence_u8(&[7], 3)));
    /// ```
    ///
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn predecessor(&self, range: Range<usize>, symbol: &BitVec) -> Option<BitVec> {
        if symbol.len() != self.bits_per_element()
            || range.is_empty()
            || self.is_empty()
            || range.end > self.len()
        {
            return None;
        }

        self.predecessor_generic_unchecked(
            range,
            symbol,
            BitVec::from_zeros(self.bits_per_element()),
            |level, symbol| symbol.get_unchecked((self.bits_per_element() - 1) - level),
            |bit, level, result| {
                result.set_unchecked((self.bits_per_element() - 1) - level, bit);
            },
            Self::partial_quantile_search_unchecked,
        )
    }

    /// Get the predecessor of the given `symbol` in the given `range`.
    /// The `symbol` is a `k`-bit word encoded in a `u64` numeral,
    /// where k is less than or equal to 64.
    /// The `symbol` does not have to be in the encoded sequence.
    /// The predecessor is the largest element in the `range` that is smaller than or equal
    /// to the `symbol`.
    ///
    /// Returns `None` if the number of bits in the matrix is greater than 64,
    /// if the range is empty, if the wavelet matrix is empty, if the range is out of bounds,
    /// or if the `symbol` is smaller than all elements in the range.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.predecessor_u64(0..3, 7), Some(4));
    /// assert_eq!(wavelet_matrix.predecessor_u64(0..3, 4), Some(4));
    /// assert_eq!(wavelet_matrix.predecessor_u64(0..6, 7), Some(7));
    /// ```
    #[must_use]
    pub fn predecessor_u64(&self, range: Range<usize>, symbol: u64) -> Option<u64> {
        if self.bits_per_element() > 64
            || range.is_empty()
            || self.is_empty()
            || range.end > self.len()
        {
            return None;
        }

        self.predecessor_generic_unchecked(
            range,
            &symbol,
            0,
            |level, symbol| symbol >> ((self.bits_per_element() - 1) - level) & 1,
            |bit, _level, result| {
                // we ignore the level here, and instead rely on the fact that the bits are set in order.
                // we have to do that, because the quantile_search_u64 does the same.
                *result <<= 1;
                *result |= bit;
            },
            Self::partial_quantile_search_u64_unchecked,
        )
    }

    #[inline(always)]
    fn successor_generic_unchecked<
        T: Clone,
        Reader: Fn(usize, &T) -> u64,
        Writer: Fn(u64, usize, &mut T),
        Quantile: Fn(&Self, Range<usize>, usize, usize, T) -> T,
    >(
        &self,
        mut range: Range<usize>,
        symbol: &T,
        mut result_value: T,
        bit_reader: Reader,
        result_writer: Writer,
        quantile_search: Quantile,
    ) -> Option<T> {
        // the bit-prefix at the last node where we could go to an interval with larger elements,
        // i.e. where we need to go if the current prefix has no elements larger than the query
        let mut last_larger_prefix = result_value.clone();
        // the level of the last node where we could go to an interval with larger elements
        let mut last_zero_level: Option<usize> = None;
        // the range of the last node where we could go to an interval with larger elements
        let mut next_larger_range: Option<Range<usize>> = None;

        for (level, data) in self.data.iter().enumerate() {
            let query_bit = bit_reader(level, symbol);

            // read the amount of elements with the current result-prefix plus one 0 bit.
            // if the query_bit is 1, we can calculate the amount of elements with the result-prefix
            // plus one 1 from there
            let zeros_start = data.rank0(range.start);
            let zeros_end = data.rank0(range.end);
            let elements_zero = zeros_end - zeros_start;

            if query_bit == 0 {
                if elements_zero == 0 {
                    // if our query element is 0 in this level and suddenly our new interval is empty,
                    // all elements that were in the previous interval are larger than the query element,
                    // because they all have a 1 in the current level.
                    // this means the successor is the smallest element in the last level's interval.

                    return Some(quantile_search(self, range, 0, level, result_value));
                }

                // if the other interval is not empty, we update the last interval where we can switch to a prefix with a 1
                if !(data.rank0 + (range.start - zeros_start)..data.rank0 + (range.end - zeros_end))
                    .is_empty()
                {
                    last_zero_level = Some(level);
                    next_larger_range = Some(
                        data.rank0 + (range.start - zeros_start)
                            ..data.rank0 + (range.end - zeros_end),
                    );
                    last_larger_prefix = result_value.clone();
                }

                // update the prefix
                result_writer(0, level, &mut result_value);

                // update the range to the interval of the new prefix
                range.start = zeros_start;
                range.end = zeros_end;
            } else {
                if elements_zero == range.end - range.start {
                    // if our query bit is 1 in this level and suddenly our new interval is empty,
                    // all elements that were in the previous interval are smaller than the query element,
                    // because they all have a 0 in the current level.
                    // this means the successor is the smallest element in the last interval with a larger prefix,
                    // i.e. the interval that has a 1 bit at the last level where our prefix had a 0 bit.

                    return next_larger_range.map(|r| {
                        result_writer(1, last_zero_level.unwrap(), &mut last_larger_prefix);
                        quantile_search(
                            self,
                            r,
                            0,
                            last_zero_level.unwrap() + 1,
                            last_larger_prefix,
                        )
                    });
                }

                // update the prefix
                result_writer(1, level, &mut result_value);

                // update the range to the interval of the new prefix
                range.start = data.rank0 + (range.start - zeros_start);
                range.end = data.rank0 + (range.end - zeros_end);
            }
        }

        Some(result_value)
    }

    /// Get the successor of the given `symbol` in the given range.
    /// The `symbol` is a `k`-bit word encoded in a [`BitVec`],
    /// where the least significant bit is the first element, and `k` is the number of bits per element
    /// in the wavelet matrix.
    /// The `symbol` does not have to be in the encoded sequence.
    /// The successor is the smallest element in the range that is greater than or equal
    /// to the `symbol`.
    ///
    /// Returns `None` if the number of bits in the `symbol` is not equal to `k`,
    /// if the range is empty, if the wavelet matrix is empty, if the range is out of bounds,
    /// or if the `symbol` is greater than all elements in the range.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.successor(0..3, &BitVec::pack_sequence_u8(&[2], 3)), Some(BitVec::pack_sequence_u8(&[4], 3)));
    /// assert_eq!(wavelet_matrix.successor(0..3, &BitVec::pack_sequence_u8(&[5], 3)), None);
    /// assert_eq!(wavelet_matrix.successor(0..6, &BitVec::pack_sequence_u8(&[2], 3)), Some(BitVec::pack_sequence_u8(&[2], 3)));
    /// ```
    ///
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn successor(&self, range: Range<usize>, symbol: &BitVec) -> Option<BitVec> {
        if symbol.len() != self.bits_per_element()
            || range.is_empty()
            || self.is_empty()
            || range.end > self.len()
        {
            return None;
        }

        self.successor_generic_unchecked(
            range,
            symbol,
            BitVec::from_zeros(self.bits_per_element()),
            |level, symbol| symbol.get_unchecked((self.bits_per_element() - 1) - level),
            |bit, level, result| {
                result.set_unchecked((self.bits_per_element() - 1) - level, bit);
            },
            Self::partial_quantile_search_unchecked,
        )
    }

    /// Get the successor of the given `symbol` in the given range.
    /// The `symbol` is a `k`-bit word encoded in a `u64` numeral,
    /// where k is less than or equal to 64.
    /// The `symbol` does not have to be in the encoded sequence.
    /// The successor is the smallest element in the range that is greater than or equal
    /// to the `symbol`.
    ///
    /// Returns `None` if the number of bits in the matrix is greater than 64,
    /// if the range is empty, if the wavelet matrix is empty, if the range is out of bounds,
    /// or if the `symbol` is greater than all elements in the range.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// assert_eq!(wavelet_matrix.successor_u64(0..3, 2), Some(4));
    /// assert_eq!(wavelet_matrix.successor_u64(0..3, 5), None);
    /// assert_eq!(wavelet_matrix.successor_u64(0..6, 2), Some(2));
    /// ```
    #[must_use]
    pub fn successor_u64(&self, range: Range<usize>, symbol: u64) -> Option<u64> {
        if self.bits_per_element() > 64
            || range.is_empty()
            || self.is_empty()
            || range.end > self.len()
        {
            return None;
        }

        self.successor_generic_unchecked(
            range,
            &symbol,
            0,
            |level, symbol| symbol >> ((self.bits_per_element() - 1) - level) & 1,
            |bit, _level, result| {
                // we ignore the level here, and instead rely on the fact that the bits are set in order.
                // we have to do that, because the quantile_search_u64 does the same.
                *result <<= 1;
                *result |= bit;
            },
            Self::partial_quantile_search_u64_unchecked,
        )
    }

    /// Get an iterator over the elements of the encoded sequence.
    /// The iterator yields `u64` elements.
    /// If the number of bits per element exceeds 64, `None` is returned.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// let mut iter = wavelet_matrix.iter_u64().unwrap();
    /// assert_eq!(iter.collect::<Vec<_>>(), vec![1, 4, 4, 1, 2, 7]);
    /// ```
    #[must_use]
    pub fn iter_u64(&self) -> Option<WaveletNumRefIter> {
        if self.bits_per_element() > 64 {
            None
        } else {
            Some(WaveletNumRefIter::new(self))
        }
    }

    /// Turn the encoded sequence into an iterator.
    /// The iterator yields `u64` elements.
    /// If the number of bits per element exceeds 64, `None` is returned.
    #[must_use]
    pub fn into_iter_u64(self) -> Option<WaveletNumIter> {
        if self.bits_per_element() > 64 {
            None
        } else {
            Some(WaveletNumIter::new(self))
        }
    }

    /// Get an iterator over the sorted elements of the encoded sequence.
    /// The iterator yields `BitVec` elements.
    ///
    /// See also [`iter_sorted_u64`] for an iterator that yields `u64` elements.
    #[must_use]
    pub fn iter_sorted(&self) -> WaveletSortedRefIter {
        WaveletSortedRefIter::new(self)
    }

    /// Turn the encoded sequence into an iterator over the sorted sequence.
    /// The iterator yields `BitVec` elements.
    #[must_use]
    pub fn into_iter_sorted(self) -> WaveletSortedIter {
        WaveletSortedIter::new(self)
    }

    /// Get an iterator over the sorted elements of the encoded sequence.
    /// The iterator yields `u64` elements.
    /// If the number of bits per element exceeds 64, `None` is returned.
    ///
    /// # Example
    /// ```
    /// use vers_vecs::{BitVec, WaveletMatrix};
    ///
    /// let bit_vec = BitVec::pack_sequence_u8(&[1, 4, 4, 1, 2, 7], 3);
    /// let wavelet_matrix = WaveletMatrix::from_bit_vec(&bit_vec, 3);
    ///
    /// let mut iter = wavelet_matrix.iter_sorted_u64().unwrap();
    /// assert_eq!(iter.collect::<Vec<_>>(), vec![1, 1, 2, 4, 4, 7]);
    /// ```
    #[must_use]
    pub fn iter_sorted_u64(&self) -> Option<WaveletSortedNumRefIter> {
        if self.bits_per_element() > 64 {
            None
        } else {
            Some(WaveletSortedNumRefIter::new(self))
        }
    }

    /// Turn the encoded sequence into an iterator over the sorted sequence.
    /// The iterator yields `u64` elements.
    /// If the number of bits per element exceeds 64, `None` is returned.
    #[must_use]
    pub fn into_iter_sorted_u64(self) -> Option<WaveletSortedNumIter> {
        if self.bits_per_element() > 64 {
            None
        } else {
            Some(WaveletSortedNumIter::new(self))
        }
    }

    /// Get the number of bits per element in the alphabet of the encoded sequence.
    #[must_use]
    #[inline(always)]
    pub fn bits_per_element(&self) -> usize {
        self.data.len()
    }

    /// Get the number of bits per element in the alphabet of the encoded sequence.
    #[deprecated(since = "1.5.1", note = "please use `bits_per_element` instead")]
    pub fn bit_len(&self) -> u16 {
        self.bits_per_element() as u16
    }

    /// Get the number of elements stored in the encoded sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        if self.data.is_empty() {
            0
        } else {
            self.data[0].len()
        }
    }

    /// Check if the wavelet matrix is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of bytes allocated on the heap for the wavelet matrix.
    /// This does not include memory that is allocated but unused due to allocation policies of
    /// internal data structures.
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.data.iter().map(RsVec::heap_size).sum::<usize>()
    }
}

impl_vector_iterator!(
    WaveletMatrix,
    WaveletIter,
    WaveletRefIter,
    get_value_unchecked,
    get_value,
    BitVec
);

impl_vector_iterator!(
    WaveletMatrix,
    WaveletNumIter,
    WaveletNumRefIter,
    get_u64_unchecked,
    get_u64,
    u64,
    special
);

impl_vector_iterator!(
    WaveletMatrix,
    WaveletSortedIter,
    WaveletSortedRefIter,
    get_sorted_unchecked,
    get_sorted,
    BitVec,
    special
);

impl_vector_iterator!(
    WaveletMatrix,
    WaveletSortedNumIter,
    WaveletSortedNumRefIter,
    get_sorted_u64_unchecked,
    get_sorted_u64,
    u64,
    special
);

#[cfg(test)]
mod tests;
