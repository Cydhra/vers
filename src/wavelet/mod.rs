use crate::{BitVec, RsVec};

/// Encode a sequence of `n` `k`-bit words in a wavelet matrix.
/// The wavelet matrix allows for rank and select queries for `k`-bit symbols on the encoded sequence.
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
    ///  bit to most significant bit.
    /// - `bits_per_element`: The number of bits in each word. Cannot exceed 1 << 16.
    ///
    /// # Panics
    /// Panics if the number of bits in the bit vector is not a multiple of the number of bits per element.
    #[must_use]
    pub fn from_bit_vec(bit_vec: BitVec, bits_per_element: u16) -> Self {
        assert_eq!(bit_vec.len() % bits_per_element as usize, 0, "The number of bits in the bit vector must be a multiple of the number of bits per element.");
        let element_len = bits_per_element as usize;
        let num_elements = bit_vec.len() / element_len;

        let mut data = vec![BitVec::from_zeros(num_elements); element_len];

        // insert the first bit of each word into the first bit vector
        // for each following level, insert the next bit of each word into the next bit vector
        // sorted stably by the previous bit vector
        let mut permutation = (0..num_elements).collect::<Vec<_>>();
        for level in 0..element_len {
            for i in 0..num_elements {
                data[level]
                    .set(
                        i,
                        bit_vec
                            .get_unchecked(permutation[i] * element_len + element_len - level - 1),
                    )
                    .unwrap();
            }
            permutation.sort_by_key(|&i| data[level].get_unchecked(i));
        }

        Self {
            data: data.into_iter().map(RsVec::from_bit_vec).collect(),
            bits_per_element,
        }
    }

    /// Read a bit from the wavelet matrix and store it somewhere using the provided function.
    #[inline(always)]
    fn read_bit<F: FnMut(u64)>(&self, level: usize, i: usize, mut target: F) {
        target(self.data[level].get_unchecked(i));
    }

    /// Get the `i`-th element of the encoded sequence in a `k`-bit word.
    /// The `k`-bit word is returned as a `BitVec`.
    /// The first element of the bit vector is the least significant bit.
    #[must_use]
    pub fn get_value(&self, i: usize) -> BitVec {
        let mut value = BitVec::from_zeros(self.bits_per_element as usize);
        for level in 0..self.bits_per_element {
            self.read_bit(level as usize, i, |bit| {
                value
                    .set((self.bits_per_element - level - 1) as usize, bit)
                    .unwrap()
            });
        }
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
    pub fn get_u64(&self, mut i: usize) -> u64 {
        assert!(
            self.bits_per_element <= 64,
            "The number of bits per element must be at most 64."
        );
        let mut value = 0;
        for level in 0..self.bits_per_element {
            self.read_bit(level as usize, i, |bit| {
                value <<= 1;
                value |= bit;
            });
            if value % 2 == 1 {
                i = self.data[level as usize].rank1(i);
            } else {
                i = self.data[level as usize].rank1 + self.data[level as usize].rank0(i);
            }
            i = self.data[level as usize].rank1(i);
        }
        value
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
