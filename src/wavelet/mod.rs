use crate::util::impl_vector_iterator;
use crate::{BitVec, RsVec};
use std::mem;
use std::ops::Range;

/// A wavelet matrix implementation implemented as described in
/// [Navarro and Claude, 2021](http://dx.doi.org/10.1007/978-3-642-34109-0_18).
///
/// Encodes a sequence of `n` `k`-bit words into a wavelet matrix which supports constant-time
/// rank and select queries on elements of its `k`-bit alphabet.
/// All query functions are mirrored for both `BitVec` and `u64` query elements, so
/// if `k <= 64`, no heap allocation is needed for the query element.
///
/// All operations implemented on the matrix are `O(k)` time complexity.
/// The space complexity of the wavelet matrix is `O(n * k)` with a small linear overhead
/// (see [`RsVec`]).
///
/// Other than rank and select queries, the matrix also supports quantile queries (range select i), and
/// range-predecessor and -successor queries, all of which are loosely based on
/// [Külekci and Thankachan](https://doi.org/10.1016/j.jda.2017.01.002) with better time complexity.
///
/// [`RsVec`]: RsVec
#[derive(Clone, Debug)]
pub struct WaveletMatrix {
    data: Box<[RsVec]>,
    bits_per_element: u16,
}

impl WaveletMatrix {
    /// Create a new wavelet matrix from a sequence of `n` `k`-bit words.
    ///
    /// # Parameters
    /// - `bit_vec`: The sequence of `n` `k`-bit words to encode. The `i`-th word begins in the
    ///   `bits_per_element * i`-th bit of the bit vector. Words are stored from least significant
    ///    bit to most significant bit.
    /// - `bits_per_element`: The number of bits in each word. Cannot exceed 1 << 16.
    ///
    /// # Panics
    /// Panics if the number of bits in the bit vector is not a multiple of the number of bits per element.
    #[must_use]
    pub fn from_bit_vec(bit_vec: &BitVec, bits_per_element: u16) -> Self {
        assert_eq!(bit_vec.len() % bits_per_element as usize, 0, "The number of bits in the bit vector must be a multiple of the number of bits per element.");
        let element_len = bits_per_element as usize;
        let num_elements = bit_vec.len() / element_len;

        let mut data = vec![BitVec::from_zeros(num_elements); element_len];

        // insert the first bit of each word into the first bit vector
        // for each following level, insert the next bit of each word into the next bit vector
        // sorted stably by the previous bit vector
        let mut permutation = (0..num_elements).collect::<Vec<_>>();
        let mut next_permutation = vec![0; num_elements];

        for level in 0..element_len {
            let mut total_zeros = 0;
            for i in 0..num_elements {
                if bit_vec.get_unchecked(permutation[i] * element_len + element_len - level - 1)
                    == 0
                {
                    total_zeros += 1;
                } else {
                    data[level].set(i, 1).unwrap();
                }
            }

            // scan through the generated bit array and move the elements to the correct position
            // for the next permutation
            if level < element_len - 1 {
                let mut zero_boundary = 0;
                let mut one_boundary = total_zeros;
                for i in 0..num_elements {
                    if data[level].get_unchecked(i) == 0 {
                        next_permutation[zero_boundary] = permutation[i];
                        zero_boundary += 1;
                    } else {
                        next_permutation[one_boundary] = permutation[i];
                        one_boundary += 1;
                    }
                }

                mem::swap(&mut permutation, &mut next_permutation);
            }
        }

        Self {
            data: data.into_iter().map(BitVec::into).collect(),
            bits_per_element,
        }
    }

    /// Generic function to read a value from the wavelet matrix and consume it with a closure.
    /// The function is used by the `get_value` and `get_u64` functions, deduplicating code.
    #[inline(always)]
    fn reconstruct_value_unchecked<F: FnMut(u64)>(&self, mut i: usize, mut target_func: F) {
        for level in 0..self.bits_per_element as usize {
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
        let mut value = BitVec::from_zeros(self.bits_per_element as usize);
        let mut level = self.bits_per_element - 1;
        self.reconstruct_value_unchecked(i, |bit| {
            value.set_unchecked(level as usize, bit);
            level = level.saturating_sub(1);
        });
        value
    }

    /// Get the `i`-th element of the encoded sequence as a `u64`.
    /// The `u64` is constructed from the `k`-bit word stored in the wavelet matrix.
    ///
    /// # Parameters
    /// - `i`: The index of the element to retrieve.
    ///
    /// # Panics
    /// Panics if the number of bits per element exceeds 64.
    #[must_use]
    pub fn get_u64(&self, i: usize) -> Option<u64> {
        if self.bits_per_element > 64 || self.data.is_empty() || i >= self.data[0].len() {
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
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds,
    /// or if the number of bits in `symbol` is lower than `k`.
    /// May instead return 0.
    ///
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn rank_range_unchecked(&self, mut range: Range<usize>, symbol: &BitVec) -> usize {
        for (level, data) in self.data.iter().enumerate() {
            if symbol.get_unchecked((self.bits_per_element - 1) as usize - level) == 0 {
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
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn rank_range(&self, range: Range<usize>, symbol: &BitVec) -> Option<usize> {
        if range.start >= self.len()
            || range.end > self.len()
            || symbol.len() != self.bits_per_element as usize
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
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds.
    /// May instead return 0.
    /// If the number of bits in wavelet matrix elements exceed `64`, the behavior is
    /// platform-dependent.
    #[must_use]
    pub fn rank_range_u64_unchecked(&self, mut range: Range<usize>, symbol: u64) -> usize {
        for (level, data) in self.data.iter().enumerate() {
            if (symbol >> ((self.bits_per_element - 1) as usize - level)) & 1 == 0 {
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
    #[must_use]
    pub fn rank_range_u64(&self, range: Range<usize>, symbol: u64) -> Option<usize> {
        if range.start >= self.len() || range.end > self.len() || self.bits_per_element > 64 {
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
    ///
    /// # Panics
    /// May panic if `offset` is out of bounds,
    /// or if `offset + i` is larger than the length of the encoded sequence,
    /// or if `offset` is greater than `i`,
    /// or if the number of bits in `symbol` is lower than `k`.
    /// May instead return 0.
    ///
    /// [`BitVec`]: BitVec
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
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn rank_offset(&self, offset: usize, i: usize, symbol: &BitVec) -> Option<usize> {
        if offset > i
            || offset >= self.len()
            || i > self.len()
            || symbol.len() != self.bits_per_element as usize
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
    ///
    /// # Panics
    /// May panic if `offset` is out of bounds,
    /// or if `i` is larger than the length of the encoded sequence,
    /// or if `offset` is greater than `i`,
    /// or if the number of bits in wavelet matrix elements exceed `64`.
    /// May instead return 0.
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
    #[must_use]
    pub fn rank_offset_u64(&self, offset: usize, i: usize, symbol: u64) -> Option<usize> {
        if offset > i || offset >= self.len() || i > self.len() || self.bits_per_element > 64 {
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
    ///
    /// # Panics
    /// May panic if `i` is out of bounds, or if the number of bits in `symbol` is lower than `k`.
    /// May instead return 0.
    /// If the number of bits in `symbol` exceeds `k`, the remaining bits are ignored.
    ///
    /// [`BitVec`]: BitVec
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
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn rank(&self, i: usize, symbol: &BitVec) -> Option<usize> {
        if i > self.len() || symbol.len() != self.bits_per_element as usize {
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
    ///
    /// # Panics
    /// May panic if `i` is out of bounds,
    /// or if the number of bits in wavelet matrix elements exceed `64`.
    /// May instead return 0.
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
    #[must_use]
    pub fn rank_u64(&self, i: usize, symbol: u64) -> Option<usize> {
        if i > self.len() || self.bits_per_element > 64 {
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
    #[must_use]
    pub fn select_offset_unchecked(&self, offset: usize, rank: usize, symbol: &BitVec) -> usize {
        let mut range_start = offset;

        for (level, data) in self.data.iter().enumerate() {
            if symbol.get_unchecked((self.bits_per_element - 1) as usize - level) == 0 {
                range_start = data.rank0(range_start);
            } else {
                range_start = data.rank0 + data.rank1(range_start);
            }
        }

        let mut range_end = range_start + rank;

        for (level, data) in self.data.iter().enumerate().rev() {
            if symbol.get_unchecked((self.bits_per_element - 1) as usize - level) == 0 {
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
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn select_offset(&self, offset: usize, rank: usize, symbol: &BitVec) -> Option<usize> {
        if offset >= self.len() || symbol.len() != self.bits_per_element as usize {
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
    ///
    /// Returns the index of the `rank`-th occurrence of the `symbol` in the encoded sequence,
    /// or the length of the encoded sequence if the `rank`-th occurrence does not exist.
    ///
    /// # Panics
    /// May panic if the `offset` is out of bounds,
    /// or if the number of bits in wavelet matrix elements exceed `64`.
    /// May instead return the length of the encoded sequence.
    #[must_use]
    pub fn select_offset_u64_unchecked(&self, offset: usize, rank: usize, symbol: u64) -> usize {
        let mut range_start = offset;

        for (level, data) in self.data.iter().enumerate() {
            if (symbol >> ((self.bits_per_element - 1) as usize - level)) & 1 == 0 {
                range_start = data.rank0(range_start);
            } else {
                range_start = data.rank0 + data.rank1(range_start);
            }
        }

        let mut range_end = range_start + rank;

        for (level, data) in self.data.iter().enumerate().rev() {
            if (symbol >> ((self.bits_per_element - 1) as usize - level)) & 1 == 0 {
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
    #[must_use]
    pub fn select_offset_u64(&self, offset: usize, rank: usize, symbol: u64) -> Option<usize> {
        if offset >= self.len() || self.bits_per_element > 64 {
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
    ///
    /// Returns the index of the `rank`-th occurrence of the `symbol` in the encoded sequence,
    /// or the length of the encoded sequence if the `rank`-th occurrence does not exist.
    ///
    /// # Panics
    /// May panic if the number of bits in `symbol` is not equal to `k`.
    /// May instead return the length of the encoded sequence.
    ///
    /// [`BitVec`]: BitVec
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
    /// [`BitVec`]: BitVec
    #[must_use]
    pub fn select(&self, rank: usize, symbol: &BitVec) -> Option<usize> {
        if symbol.len() != self.bits_per_element as usize {
            None
        } else {
            let idx = self.select_unchecked(rank, symbol);
            if idx < self.len() {
                Some(idx)
            } else {
                None
            }
        }
    }

    /// Get the index of the `rank`-th occurrence of the given `symbol` in the encoded sequence.
    /// The `symbol` is a `k`-bit word encoded in a u64 numeral,
    /// where k is less than or equal to 64.
    ///
    /// This method does not perform bounds checking, nor does it check if the elements of the
    /// wavelet matrix can be represented in a u64 numeral.
    ///
    /// Returns the index of the `rank`-th occurrence of the `symbol` in the encoded sequence,
    /// or the length of the encoded sequence if the `rank`-th occurrence does not exist.
    ///
    /// # Panics
    /// May panic if the number of bits in wavelet matrix elements exceed `64`.
    /// May instead return the length of the encoded sequence.
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
    #[must_use]
    pub fn select_u64(&self, rank: usize, symbol: u64) -> Option<usize> {
        if self.bits_per_element > 64 {
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
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds. May instead return an empty bit vector.
    #[must_use]
    pub fn quantile_unchecked(&self, range: Range<usize>, k: usize) -> BitVec {
        let result = BitVec::from_zeros(self.bits_per_element as usize);

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
        debug_assert!(prefix.len() == self.bits_per_element as usize);
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
                prefix.set_unchecked((self.bits_per_element - 1) as usize - level, 1);
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
    ///
    /// # Panics
    /// May panic if the `i` is out of bounds, or returns an empty bit vector.
    pub fn get_sorted_unchecked(&self, i: usize) -> BitVec {
        self.quantile_unchecked(0..self.len(), i)
    }

    /// Get the `i`-th smallest element in the entire wavelet matrix.
    /// The `i`-th smallest element is returned as a `BitVec`,
    /// where the least significant bit is the first element.
    ///
    /// Returns `None` if the `i` is out of bounds.
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
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds.
    /// May instead return 0.
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
        debug_assert!(self.bits_per_element <= 64);
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
    #[must_use]
    pub fn quantile_u64(&self, range: Range<usize>, k: usize) -> Option<u64> {
        if range.start >= self.len()
            || range.end > self.len()
            || self.bits_per_element > 64
            || k >= range.end - range.start
        {
            None
        } else {
            Some(self.quantile_u64_unchecked(range, k))
        }
    }

    /// Get the `i`-th smallest element in the entire wavelet matrix.
    /// The `i`-th smallest element is returned as a u64 numeral.
    ///
    /// If the number of bits per element exceeds 64, the value is truncated.
    ///
    /// This method does not perform bounds checking.
    ///
    /// # Panics
    /// May panic if the `i` is out of bounds, or returns an empty bit vector.
    pub fn get_sorted_u64_unchecked(&self, i: usize) -> u64 {
        self.quantile_u64_unchecked(0..self.len(), i)
    }

    /// Get the `i`-th smallest element in the entire wavelet matrix.
    /// The `i`-th smallest element is returned as a u64 numeral.
    ///
    /// Returns `None` if the `i` is out of bounds, or if the number of bits per element exceeds 64.
    pub fn get_sorted_u64(&self, i: usize) -> Option<u64> {
        if i >= self.len() || self.bits_per_element > 64 {
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
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return an empty bit vector.
    #[must_use]
    pub fn range_min_unchecked(&self, range: Range<usize>) -> BitVec {
        self.quantile_unchecked(range, 0)
    }

    /// Get the smallest element in the encoded sequence in the specified `range`.
    /// The range is a half-open interval, meaning that the `end` index is exclusive.
    /// The smallest element is returned as a `BitVec`,
    ///
    /// Returns `None` if the `range` is out of bounds or if the range is empty.
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
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return 0.
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
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return an empty bit vector.
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
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return 0.
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
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return an empty bit vector.
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
    ///
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return 0.
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
    /// # Panics
    /// May panic if the `range` is out of bounds or if the range is empty.
    /// May instead return 0.
    #[must_use]
    pub fn range_median_u64(&self, range: Range<usize>) -> Option<u64> {
        if range.is_empty() {
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
        Writer: Fn(u64, usize, &mut T) -> (),
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
                        quantile_search(self, r, idx, last_one_level.unwrap(), last_smaller_prefix)
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
    #[must_use]
    pub fn predecessor(&self, range: Range<usize>, symbol: &BitVec) -> Option<BitVec> {
        if symbol.len() != self.bits_per_element as usize
            || range.is_empty()
            || self.is_empty()
            || range.end > self.len()
        {
            return None;
        }

        self.predecessor_generic_unchecked(
            range,
            symbol,
            BitVec::from_zeros(self.bits_per_element as usize),
            |level, symbol| symbol.get_unchecked((self.bits_per_element - 1) as usize - level),
            |bit, level, result| {
                result.set_unchecked((self.bits_per_element - 1) as usize - level, bit)
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
    #[must_use]
    pub fn predecessor_u64(&self, range: Range<usize>, symbol: u64) -> Option<u64> {
        if self.bits_per_element > 64
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
            |level, symbol| symbol >> ((self.bits_per_element - 1) as usize - level) & 1,
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
        Writer: Fn(u64, usize, &mut T) -> (),
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
                        quantile_search(self, r, 0, last_zero_level.unwrap(), last_larger_prefix)
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
    #[must_use]
    pub fn successor(&self, range: Range<usize>, symbol: &BitVec) -> Option<BitVec> {
        if symbol.len() != self.bits_per_element as usize
            || range.is_empty()
            || self.is_empty()
            || range.end > self.len()
        {
            return None;
        }

        self.successor_generic_unchecked(
            range,
            symbol,
            BitVec::from_zeros(self.bits_per_element as usize),
            |level, symbol| symbol.get_unchecked((self.bits_per_element - 1) as usize - level),
            |bit, level, result| {
                result.set_unchecked((self.bits_per_element - 1) as usize - level, bit)
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
    #[must_use]
    pub fn successor_u64(&self, range: Range<usize>, symbol: u64) -> Option<u64> {
        if self.bits_per_element > 64
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
            |level, symbol| symbol >> ((self.bits_per_element - 1) as usize - level) & 1,
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
    #[must_use]
    pub fn iter_u64(&self) -> Option<WaveletNumRefIter> {
        if self.bits_per_element > 64 {
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
        if self.bits_per_element > 64 {
            None
        } else {
            Some(WaveletNumIter::new(self))
        }
    }

    /// Get an iterator over the sorted elements of the encoded sequence.
    /// The iterator yields `BitVec` elements.
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
    #[must_use]
    pub fn iter_sorted_u64(&self) -> Option<WaveletSortedNumRefIter> {
        if self.bits_per_element > 64 {
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
        if self.bits_per_element > 64 {
            None
        } else {
            Some(WaveletSortedNumIter::new(self))
        }
    }

    /// Get the number of bits per element in the alphabet of the encoded sequence.
    #[must_use]
    pub fn bit_len(&self) -> u16 {
        self.bits_per_element
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
