//! This module contains a simple [bit vector][BitVec] implementation with no overhead and a fast succinct
//! bit vector implementation with [rank and select queries][fast_rs_vec::RsVec].

use crate::bit_vec::mask::MaskedBitVec;
use crate::util::impl_iterator;
use std::mem::size_of;

pub mod fast_rs_vec;

pub mod mask;

/// Size of a word in bitvectors. All vectors operate on 64-bit words.
const WORD_SIZE: usize = 64;

/// A simple bit vector that does not support rank and select queries. It stores bits densely
/// in 64 bit limbs. The last limb may be partially filled. Other than that, there is no overhead.
///
/// # Example
/// ```rust
/// use vers_vecs::{BitVec, RsVec};
///
/// let mut bit_vec = BitVec::new();
/// bit_vec.append_bit(0u64);
/// bit_vec.append_bit_u32(1u32);
/// bit_vec.append_word(0b1010_1010_1010_1010u64); // appends exactly 64 bits
///
/// assert_eq!(bit_vec.len(), 66);
/// assert_eq!(bit_vec.get(0), Some(0u64));
/// assert_eq!(bit_vec.get(1), Some(1u64));
/// ```
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitVec {
    data: Vec<u64>,
    len: usize,
}

impl BitVec {
    /// Create a new empty bit vector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new empty bit vector with the given capacity. The capacity is measured in bits.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity / WORD_SIZE + 1),
            len: 0,
        }
    }

    /// Create a new bit vector with all zeros and the given length. The length is measured in bits.
    #[must_use]
    pub fn from_zeros(len: usize) -> Self {
        let mut data = vec![0; len / WORD_SIZE];
        if len % WORD_SIZE != 0 {
            data.push(0);
        }
        Self { data, len }
    }

    /// Create a new bit vector with all ones and the given length. The length is measured in bits.
    #[must_use]
    pub fn from_ones(len: usize) -> Self {
        let mut data = vec![u64::MAX; len / WORD_SIZE];
        if len % WORD_SIZE != 0 {
            data.push((1 << (len % WORD_SIZE)) - 1);
        }
        Self { data, len }
    }

    /// Construct a bit vector from a set of bits given as distinct u8 values. The constructor will
    /// take the least significant bit from each value and append it to a bit vector. All other bits
    /// are ignored.
    ///
    /// See also: [`from_bits_u64`]
    ///
    /// [`from_bits_u64`]: BitVec::from_bits_u64
    pub fn from_bits(bits: &[u8]) -> Self {
        let mut bv = Self::with_capacity(bits.len());
        bits.iter().for_each(|&b| bv.append_bit(b.into()));
        bv
    }

    /// Construct a bit vector from a set of bits given as distinct u64 values. The constructor will
    /// take the least significant bit from each value and append it to a bit vector. All other bits
    /// are ignored.
    ///
    /// See also: [`from_bits`]
    ///
    /// [`from_bits`]: BitVec::from_bits
    pub fn from_bits_u64(bits: &[u64]) -> Self {
        let mut bv = Self::with_capacity(bits.len());
        bits.iter().for_each(|&b| bv.append_bit(b));
        bv
    }

    /// Append a bit to the bit vector. The bit is given as a boolean, where `true` means 1 and
    /// `false` means 0.
    pub fn append(&mut self, bit: bool) {
        if self.len % WORD_SIZE == 0 {
            self.data.push(0);
        }
        if bit {
            self.data[self.len / WORD_SIZE] |= 1 << (self.len % WORD_SIZE);
        } else {
            self.data[self.len / WORD_SIZE] &= !(1 << (self.len % WORD_SIZE));
        }
        self.len += 1;
    }

    /// Drop the last n bits from the bit vector. If more bits are dropped than the bit vector
    /// contains, the bit vector is cleared.
    pub fn drop_last(&mut self, n: usize) {
        if n > self.len {
            self.data.clear();
            self.len = 0;
            return;
        }

        let new_limb_count = (self.len - n + WORD_SIZE - 1) / WORD_SIZE;

        // cut off limbs that we no longer need
        if new_limb_count < self.data.len() {
            self.data.truncate(new_limb_count);
        }

        // update bit vector length
        self.len -= n;
    }

    /// Append a bit from a u64. The least significant bit is appended to the bit vector.
    /// All other bits are ignored.
    pub fn append_bit(&mut self, bit: u64) {
        if self.len % WORD_SIZE == 0 {
            self.data.push(0);
        }
        if bit % 2 == 1 {
            self.data[self.len / WORD_SIZE] |= 1 << (self.len % WORD_SIZE);
        } else {
            self.data[self.len / WORD_SIZE] &= !(1 << (self.len % WORD_SIZE));
        }

        self.len += 1;
    }

    /// Append a bit from a u32. The least significant bit is appended to the bit vector.
    /// All other bits are ignored.
    pub fn append_bit_u32(&mut self, bit: u32) {
        self.append_bit(u64::from(bit));
    }

    /// Append a bit from a u8. The least significant bit is appended to the bit vector.
    /// All other bits are ignored.
    pub fn append_bit_u8(&mut self, bit: u8) {
        self.append_bit(u64::from(bit));
    }

    /// Append a word to the bit vector. The bits are appended in little endian order (i.e. the first
    /// bit of the word is appended first).
    pub fn append_word(&mut self, word: u64) {
        if self.len % WORD_SIZE == 0 {
            self.data.push(word);
        } else {
            // zero out the unused bits before or-ing the new one, to ensure no garbage data remains
            self.data[self.len / WORD_SIZE] &= !(u64::MAX << (self.len % WORD_SIZE));
            self.data[self.len / WORD_SIZE] |= word << (self.len % WORD_SIZE);

            self.data.push(word >> (WORD_SIZE - self.len % WORD_SIZE));
        }
        self.len += WORD_SIZE;
    }

    /// Append multiple bits to the bit vector. The bits are appended in little-endian order
    /// (i.e. the least significant bit is appended first).
    /// The number of bits to append is given by `len`. The bits are taken from the least
    /// significant bits of `bits`. All other bits are ignored.
    ///
    /// # Panics
    /// Panics if `len` is larger than 64.
    pub fn append_bits(&mut self, mut bits: u64, len: usize) {
        assert!(len <= 64, "Cannot append more than 64 bits");

        // zero out garbage data
        if len < 64 {
            bits &= (1 << len) - 1;
        }

        if self.len % WORD_SIZE == 0 {
            self.data.push(bits);
        } else {
            // zero out the unused bits before or-ing the new one, to ensure no garbage data remains
            self.data[self.len / WORD_SIZE] &= !(u64::MAX << (self.len % WORD_SIZE));
            self.data[self.len / WORD_SIZE] |= bits << (self.len % WORD_SIZE);

            if self.len % WORD_SIZE + len > WORD_SIZE {
                self.data.push(bits >> (WORD_SIZE - self.len % WORD_SIZE));
            }
        }
        self.len += len;
    }

    /// Return the length of the bit vector. The length is measured in bits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the bit vector is empty (contains no bits).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Flip the bit at the given position.
    ///
    /// # Panics
    /// If the position is larger than the length of the vector, the function panics.
    pub fn flip_bit(&mut self, pos: usize) {
        assert!(pos < self.len, "Index out of bounds");
        self.flip_bit_unchecked(pos);
    }

    /// Flip the bit at the given position.
    ///
    /// # Panics
    /// If the position is larger than the length of the
    /// vector, the function will either modify unused memory or panic.
    /// This will not corrupt memory.
    pub fn flip_bit_unchecked(&mut self, pos: usize) {
        self.data[pos / WORD_SIZE] ^= 1 << (pos % WORD_SIZE);
    }

    /// Return the bit at the given position.
    /// The bit takes the least significant bit of the returned u64 word.
    /// If the position is larger than the length of the vector, None is returned.
    #[must_use]
    pub fn get(&self, pos: usize) -> Option<u64> {
        if pos >= self.len {
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
    /// [`get`]: BitVec::get
    #[must_use]
    pub fn get_unchecked(&self, pos: usize) -> u64 {
        (self.data[pos / WORD_SIZE] >> (pos % WORD_SIZE)) & 1
    }

    /// Set the bit at the given position.
    /// The bit is given as a u64 value of which only the least significant bit is used.
    /// If the position is out of range, the function will return `Err` with an error message,
    /// otherwise it will return an empty `Ok`.
    pub fn set(&mut self, pos: usize, value: u64) -> Result<(), &str> {
        if pos >= self.len {
            Err("out of range")
        } else {
            Ok(self.set_unchecked(pos, value))
        }
    }

    /// Set the bit at the given position.
    /// The bit is given as a u64 value of which only the least significant bit is used.
    ///
    /// # Panics
    /// If the position is larger than the length of the vector,
    /// the function will either do nothing, or panic.
    /// Use [`set`] to properly handle this case with a `Result`.
    ///
    /// [`set`]: BitVec::set
    pub fn set_unchecked(&mut self, pos: usize, value: u64) {
        self.data[pos / WORD_SIZE] = (self.data[pos / WORD_SIZE] & !(0x1 << (pos % WORD_SIZE)))
            | ((value & 0x1) << (pos % WORD_SIZE))
    }

    /// Return whether the bit at the given position is set.
    /// If the position is larger than the length of the vector, None is returned.
    #[must_use]
    pub fn is_bit_set(&self, pos: usize) -> Option<bool> {
        if pos >= self.len {
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
    /// [`is_bit_set`]: BitVec::is_bit_set
    #[must_use]
    pub fn is_bit_set_unchecked(&self, pos: usize) -> bool {
        self.get_unchecked(pos) != 0
    }

    /// Return multiple bits at the given position. The number of bits to return is given by `len`.
    /// At most 64 bits can be returned.
    /// If the position at the end of the query is larger than the length of the vector,
    /// None is returned (even if the query partially overlaps with the vector).
    /// If the length of the query is larger than 64, None is returned.
    #[must_use]
    pub fn get_bits(&self, pos: usize, len: usize) -> Option<u64> {
        if len > WORD_SIZE || len == 0 {
            return None;
        }
        if pos + len > self.len {
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
    /// [`get_bits`]: BitVec::get_bits
    #[must_use]
    #[allow(clippy::inline_always)]
    #[allow(clippy::comparison_chain)] // rust-clippy #5354
    #[inline(always)] // inline to gain loop optimization and pipeline advantages for elias fano
    pub fn get_bits_unchecked(&self, pos: usize, len: usize) -> u64 {
        debug_assert!(len <= WORD_SIZE);
        let partial_word = self.data[pos / WORD_SIZE] >> (pos % WORD_SIZE);
        if pos % WORD_SIZE + len == WORD_SIZE {
            partial_word
        } else if pos % WORD_SIZE + len < WORD_SIZE {
            partial_word & ((1 << (len % WORD_SIZE)) - 1)
        } else {
            (partial_word | (self.data[pos / WORD_SIZE + 1] << (WORD_SIZE - pos % WORD_SIZE)))
                & ((1 << (len % WORD_SIZE)) - 1)
        }
    }

    /// Return the number of ones in the bit vector. Since the bit vector doesn't store additional
    /// metadata, this value is calculated. Use [`RsVec`] for constant-time rank operations.
    ///
    /// [`RsVec`]: crate::RsVec
    #[must_use]
    pub fn count_ones(&self) -> u64 {
        let mut ones = self.data[0..self.len / WORD_SIZE]
            .iter()
            .map(|limb| limb.count_ones() as u64)
            .sum();
        if self.len % WORD_SIZE > 0 {
            ones += (self.data.last().unwrap() & ((1 << (self.len % WORD_SIZE)) - 1)).count_ones()
                as u64;
        }
        ones
    }

    /// Return the number of zeros in the bit vector. Since the bit vector doesn't store additional
    /// metadata, this value is calculated. Use [`RsVec`] for constant-time rank operations.
    /// This method calls [`count_ones`].
    ///
    /// [`RsVec`]: crate::RsVec
    /// [`count_ones`]: BitVec::count_ones
    #[must_use]
    pub fn count_zeros(&self) -> u64 {
        self.len as u64 - self.count_ones()
    }

    /// Mask this bit vector with another bitvector using bitwise or. The mask is applied lazily
    /// whenever an operation on the resulting vector is performed. Returns an error if the length
    /// of the vector doesn't match the mask.
    #[inline]
    pub fn mask_or<'s, 'b>(&'s self, mask: &'b BitVec) -> Result<MaskedBitVec<'s, 'b>, String> {
        MaskedBitVec::new(self, mask, |a, b| a | b)
    }

    /// Mask this bit vector with another bitvector using bitwise and. The mask is applied lazily
    /// whenever an operation on the resulting vector is performed. Returns an error if the length
    /// of the vector doesn't match the mask.
    #[inline]
    pub fn mask_and<'s, 'b>(&'s self, mask: &'b BitVec) -> Result<MaskedBitVec<'s, 'b>, String> {
        MaskedBitVec::new(self, mask, |a, b| a & b)
    }

    /// Mask this bit vector with another bitvector using bitwise xor. The mask is applied lazily
    /// whenever an operation on the resulting vector is performed. Returns an error if the length
    /// of the vector doesn't match the mask.
    #[inline]
    pub fn mask_xor<'s, 'b>(&'s self, mask: &'b BitVec) -> Result<MaskedBitVec<'s, 'b>, String> {
        MaskedBitVec::new(self, mask, |a, b| a ^ b)
    }

    /// Returns the number of bytes on the heap for this vector. Does not include allocated memory
    /// that isn't used.
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>()
    }
}

impl_iterator! { BitVec, BitVecIter, BitVecRefIter }

#[cfg(test)]
mod tests;
