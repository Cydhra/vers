use crate::BitVec;

/// A bit vector that is masked with another bit vector via a masking function. Offers the same
/// functions like an unmasked vector. The mask is applied lazily.
pub struct MaskedBitVec<'a, 'b> {
    vec: &'a BitVec,
    mask: &'b BitVec,
    bin_op: fn(u64, u64) -> u64,
}

impl<'a, 'b> MaskedBitVec<'a, 'b> {
    #[inline]
    pub(crate) fn new(
        vec: &'a BitVec,
        mask: &'b BitVec,
        bin_op: fn(u64, u64) -> u64,
    ) -> Result<Self, String> {
        if vec.len != mask.len {
            return Err(String::from(
                "mask cannot have different length than vector",
            ));
        }

        Ok(MaskedBitVec { vec, mask, bin_op })
    }

    /// Iterate over the limbs of the masked vector
    #[inline]
    #[must_use]
    fn iter_limbs<'s>(&'s self) -> impl Iterator<Item = u64> + 's
    where
        'a: 's,
        'b: 's,
    {
        self.vec
            .iter()
            .zip(self.mask)
            .map(|(a, b)| (self.bin_op)(a, b))
    }

    /// Return the bit at the given position.
    /// The bit takes the least significant bit of the returned u64 word.
    /// If the position is larger than the length of the vector, None is returned.
    #[inline]
    #[must_use]
    pub fn get(&self, pos: usize) -> Option<u64> {
        if pos >= self.vec.len {
            None
        } else {
            Some(self.get_unchecked(pos))
        }
    }

    /// Return the bit at the given position.
    /// The bit takes the least significant bit of the returned u64 word.
    ///
    /// # Panics
    /// If the position is larger than the length of the vector,
    /// the function will either return unpredictable data, or panic.
    /// Use [`get`] to properly handle this case with an `Option`.
    ///
    /// [`get`]: MaskedBitVec::get
    #[inline]
    #[must_use]
    pub fn get_unchecked(&self, pos: usize) -> u64 {
        ((self.bin_op)(
            self.vec.data[pos / super::WORD_SIZE],
            self.mask.data[pos / super::WORD_SIZE],
        ) >> (pos % super::WORD_SIZE))
            & 1
    }

    /// Return whether the bit at the given position is set.
    /// If the position is larger than the length of the vector, None is returned.
    #[inline]
    #[must_use]
    pub fn is_bit_set(&self, pos: usize) -> Option<bool> {
        if pos >= self.vec.len {
            None
        } else {
            Some(self.is_bit_set_unchecked(pos))
        }
    }

    /// Return whether the bit at the given position is set.
    ///
    /// # Panics
    /// If the position is larger than the length of the vector,
    /// the function will either return unpredictable data, or panic.
    /// Use [`is_bit_set`] to properly handle this case with an `Option`.
    ///
    /// [`is_bit_set`]: MaskedBitVec::is_bit_set
    #[inline]
    #[must_use]
    pub fn is_bit_set_unchecked(&self, pos: usize) -> bool {
        self.get_unchecked(pos) != 0
    }

    /// Return multiple bits at the given position. The number of bits to return is given by `len`.
    /// At most 64 bits can be returned.
    /// If the position at the end of the query is larger than the length of the vector,
    /// None is returned (even if the query partially overlaps with the vector).
    /// If the length of the query is larger than 64, None is returned.
    #[inline]
    #[must_use]
    pub fn get_bits(&self, pos: usize, len: usize) -> Option<u64> {
        if len > super::WORD_SIZE || len == 0 {
            return None;
        }
        if pos + len > self.vec.len {
            None
        } else {
            Some(self.get_bits_unchecked(pos, len))
        }
    }

    /// Return multiple bits at the given position. The number of bits to return is given by `len`.
    /// At most 64 bits can be returned.
    ///
    /// This function is always inlined, because it gains a lot from loop optimization and
    /// can utilize the processor pre-fetcher better if it is.
    ///
    /// # Errors
    /// If the length of the query is larger than 64, unpredictable data will be returned.
    /// Use [`get_bits`] to avoid this.
    ///
    /// # Panics
    /// If the position or interval is larger than the length of the vector,
    /// the function will either return any valid results padded with unpredictable
    /// data or panic.
    ///
    /// [`get_bits`]: MaskedBitVec::get_bits
    #[must_use]
    #[allow(clippy::inline_always)]
    #[allow(clippy::comparison_chain)] // rust-clippy #5354
    #[inline]
    pub fn get_bits_unchecked(&self, pos: usize, len: usize) -> u64 {
        debug_assert!(len <= super::WORD_SIZE);
        let partial_word = (self.bin_op)(
            self.vec.data[pos / super::WORD_SIZE],
            self.mask.data[pos / super::WORD_SIZE],
        ) >> (pos % super::WORD_SIZE);

        if pos % super::WORD_SIZE + len == super::WORD_SIZE {
            partial_word
        } else if pos % super::WORD_SIZE + len < super::WORD_SIZE {
            partial_word & ((1 << (len % super::WORD_SIZE)) - 1)
        } else {
            let next_half = (self.bin_op)(
                self.vec.data[pos / super::WORD_SIZE + 1],
                self.mask.data[pos / super::WORD_SIZE + 1],
            ) << (super::WORD_SIZE - pos % super::WORD_SIZE);

            (partial_word | next_half) & ((1 << (len % super::WORD_SIZE)) - 1)
        }
    }

    /// Return the number of zeros in the masked bit vector.
    /// This method calls [`count_ones`].
    ///
    /// [`count_ones`]: MaskedBitVec::count_ones
    #[inline]
    #[must_use]
    pub fn count_zeros(&self) -> u64 {
        self.vec.len as u64 - self.count_ones()
    }

    /// Return the number of ones in the masked bit vector.
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> u64 {
        let mut ones = self
            .iter_limbs()
            .take(self.vec.len / super::WORD_SIZE)
            .map(|limb| limb.count_ones() as u64)
            .sum();
        if self.vec.len % super::WORD_SIZE > 0 {
            ones += ((self.bin_op)(
                *self.vec.data.last().unwrap(),
                *self.mask.data.last().unwrap(),
            ) & ((1 << (self.vec.len % super::WORD_SIZE)) - 1))
                .count_ones() as u64;
        }
        ones
    }
}
