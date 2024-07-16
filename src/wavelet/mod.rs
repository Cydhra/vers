use crate::{BitVec, RsVec};
use std::mem;
use std::ops::Range;

/// A wavelet matrix implementation implemented as described in
/// [Navarro and Claude, 2021](http://dx.doi.org/10.1007/978-3-642-34109-0_18).
///
/// Encodes a sequence of `n` `k`-bit words into a wavelet matrix which supports constant-time
/// rank and select queries on elements of its `k`-bit alphabet.
/// The wavelet matrix supports queries where elements are encoded
/// either in a `BitVec` or as `u64` numerals if `bits_per_element <= 64`.
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
    ///
    /// Returns `None` if the `range` is out of bounds (greater than the length of the encoded sequence,
    /// but since it is exclusive, it may be equal to the length),
    /// or if the number of bits in the wavelet matrix elements exceed `64`. // todo panic instead?
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
    /// The interval is half-open, meaning ```rank_offset_unchecked(0, 0, symbol)``` tries to
    /// compute the rank of an empty interval, which returns an unspecified value.
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
    /// The interval is half-open, meaning ``rank_offset(0, 0, symbol)`` returns None,
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
        if offset >= i
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
    /// The interval is half-open, meaning ``rank_offset(0, 0, symbol)`` tries to compute the rank
    /// of an empty interval, which returns an unspecified value.
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
    /// The interval is half-open, meaning ``rank_offset(0, 0, symbol)`` returns None,
    /// because the interval is empty.
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
        if offset >= i || offset >= self.len() || i > self.len() || self.bits_per_element > 64 {
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
    /// or if the number of bits in the wavelet matrix elements exceed `64`. // todo panic instead?
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

#[cfg(test)]
mod tests;
