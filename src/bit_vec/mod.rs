//! This module contains a simple bit vector implementation with no overhead and a fast succinct
//! [bit vector][fast_rs_vec::RsVec] implementation with rank and select queries.

use std::mem::size_of;

pub mod fast_rs_vec;

/// Size of a word in bitvectors. All vectors operate on 64-bit words.
const WORD_SIZE: usize = 64;

/// A simple bit vector that does not support rank and select queries. It has a constant memory
/// overhead of 32 bytes on the stack.
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

    /// Append a bit to the bit vector.
    pub fn append(&mut self, bit: bool) {
        if self.len % WORD_SIZE == 0 {
            self.data.push(0);
        }
        if bit {
            self.data[self.len / WORD_SIZE] |= 1 << (self.len % WORD_SIZE);
        }
        self.len += 1;
    }

    /// Drop the last n bits from the bit vector.
    pub fn truncate(&mut self, n: usize) {
        self.len -= n;
        if self.len / WORD_SIZE > 0 {
            self.data.truncate(self.len / WORD_SIZE);
        }
    }

    /// Append a bit from a quad-word. The least significant bit is appended to the bit vector.
    /// All other bits are ignored.
    pub fn append_bit(&mut self, bit: u64) {
        if self.len % WORD_SIZE == 0 {
            self.data.push(0);
        }
        self.data[self.len / WORD_SIZE] |= (bit % 2) << (self.len % WORD_SIZE);
        self.len += 1;
    }

    /// Append a word to the bit vector. The least significant bit is appended first.
    pub fn append_word(&mut self, word: u64) {
        if self.len % WORD_SIZE == 0 {
            self.data.push(word);
        } else {
            self.data[self.len / WORD_SIZE] |= word << (self.len % WORD_SIZE);
            self.data.push(word >> (WORD_SIZE - self.len % WORD_SIZE));
        }
        self.len += WORD_SIZE;
    }

    /// Append multiple bits to the bit vector. The least significant bit is appended first.
    /// The number of bits to append is given by `len`. The bits are taken from the least
    /// significant bits of `bits`. All other bits are ignored.
    pub fn append_bits(&mut self, mut bits: u64, len: usize) {
        bits &= (1 << len) - 1;

        if self.len % WORD_SIZE == 0 {
            self.data.push(bits);
        } else {
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

    /// Flip the bit at the given position. If the position is larger than the length of the
    /// vector, an error is returned. Otherwise an empty result is returned.
    pub fn flip_bit(&mut self, pos: usize) -> Result<(), &str> {
        if pos >= self.len {
            Err("Index out of bounds")
        } else {
            self.flip_bit_unchecked(pos);
            Ok(())
        }
    }

    /// Flip the bit at the given position. If the position is larger than the length of the
    /// vector, the behavior is undefined (the function will either modify unused memory or panic.
    /// This will not corrupt memory, but will affect invalid unchecked get operations).
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
    /// If the position is larger than the length of the vector,
    /// the behavior is undefined (the function will either return unpredictable data or panic).
    #[must_use]
    pub fn get_unchecked(&self, pos: usize) -> u64 {
        (self.data[pos / WORD_SIZE] >> (pos % WORD_SIZE)) & 1
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
    /// If the position is larger than the length of the vector,
    /// the behavior is undefined (the function will either return unpredictable results or panic).
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
        if len > WORD_SIZE {
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
    /// If the position is larger than the length of the vector,
    /// the behavior is undefined (the function will either return any valid results padded with unpredictable
    /// memory or panic).
    /// If the length of the query is larger than 64, the behavior is undefined.
    #[must_use]
    #[inline(always)] // inline to gain loop optimization and pipeline advantages for elias fano
    pub fn get_bits_unchecked(&self, pos: usize, len: usize) -> u64 {
        debug_assert!(len <= WORD_SIZE);
        let partial_word = self.data[pos / WORD_SIZE] >> (pos % WORD_SIZE);
        if pos % WORD_SIZE + len <= WORD_SIZE {
            partial_word & ((1 << len) - 1)
        } else {
            (partial_word | (self.data[pos / WORD_SIZE + 1] << (WORD_SIZE - pos % WORD_SIZE)))
                & ((1 << len) - 1)
        }
    }

    /// Returns the number of bytes on the heap for this vector. Does not include allocated memory
    /// that isn't used.
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>()
    }
}