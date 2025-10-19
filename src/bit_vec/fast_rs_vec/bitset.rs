//! Module that contains the bitset iterator over a `RsVec`.
//! The iterator does the same as the `iter1`/`iter0` methods of `RsVec`, but it is faster for dense vectors.
//! It only exists with the `simd` feature enabled, and since it is slower for sparse vectors,
//! it is not used as a replacement for the `iter1`/`iter0` methods.

use crate::RsVec;
use std::mem::size_of;

/// The number of bits in a RsVec that can be processed by AVX instructions at once.
const VECTOR_SIZE: u64 = 16;

// add iterator functions to RsVec
impl RsVec {
    /// Get an iterator over the 0-bits in the vector.
    /// The iterator returns the indices of the 0-bits in the vector, just as [`iter0`]
    /// and [`select0`] do.
    ///
    /// This method is faster than [`iter0`] for dense vectors, but slower for sparse vectors.
    ///
    /// See [`BitSetIter`] for more information.
    ///
    /// [`iter0`]: RsVec::iter0
    /// [`select0`]: RsVec::select0
    /// [`BitSetIter`]: BitSetIter
    #[must_use]
    pub fn bit_set_iter0(&self) -> BitSetIter<'_, true> {
        BitSetIter::new(self)
    }

    /// Get an iterator over the 1-bits in the vector.
    /// The iterator returns the indices of the 1-bits in the vector, just as [`iter1`]
    /// and [`select1`] do.
    ///
    /// This method is faster than [`iter1`] for dense vectors, but slower for sparse vectors.
    ///
    /// See [`BitSetIter`] for more information.
    ///
    /// [`iter1`]: RsVec::iter1
    /// [`select1`]: RsVec::select1
    /// [`BitSetIter`]: BitSetIter
    #[must_use]
    pub fn bit_set_iter1(&self) -> BitSetIter<'_, false> {
        BitSetIter::new(self)
    }
}

/// An iterator that iterates over 1-bits or 0-bits and returns their indices.
/// It uses AVX vector instructions to process 16 bits at once.
/// It is faster than [`SelectIter`] for dense vectors.
///
/// This is also faster than manually calling `select` on each rank,
/// because the select data structures are not parsed by this iterator.
///
/// The iterator can be constructed by calling [`bit_set_iter0`] or [`bit_set_iter1`].
///
/// # Example
/// ```rust
/// use vers_vecs::{BitVec, RsVec};
///
/// let mut bit_vec = BitVec::new();
/// bit_vec.append_word(u64::MAX);
/// bit_vec.append_word(u64::MAX);
/// bit_vec.flip_bit(4);
///
/// let rs_vec = RsVec::from_bit_vec(bit_vec);
///
/// let mut iter = rs_vec.bit_set_iter0();
///
/// assert_eq!(iter.next(), Some(4));
/// assert_eq!(iter.next(), None);
/// ```
///
/// [`bit_set_iter0`]: RsVec::bit_set_iter0
/// [`bit_set_iter1`]: RsVec::bit_set_iter1
/// [`SelectIter`]: super::SelectIter
#[allow(clippy::cast_possible_truncation)]
pub struct BitSetIter<'a, const ZERO: bool> {
    vec: &'a RsVec,
    base: u64,
    offsets: [u32; VECTOR_SIZE as usize],
    content_len: u8,
    cursor: u8,
}

impl<'a, const ZERO: bool> BitSetIter<'a, ZERO> {
    pub(super) fn new(vec: &'a RsVec) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        let mut iter = Self {
            vec,
            base: 0,
            offsets: [0; VECTOR_SIZE as usize],
            content_len: 0,
            cursor: 0,
        };

        if vec.len() > VECTOR_SIZE {
            iter.load_chunk(vec.get_bits_unchecked(0, VECTOR_SIZE) as u16);
        }

        iter
    }

    fn load_chunk(&mut self, data: u16) {
        use std::arch::x86_64::{__mmask16, _mm512_mask_compressstoreu_epi32, _mm512_setr_epi32};

        unsafe {
            let offsets = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            assert!(
                VECTOR_SIZE <= size_of::<u16>() as u64 * 8,
                "change data types"
            );
            let mut mask = __mmask16::from(data);
            if ZERO {
                mask = !mask;
            }
            _mm512_mask_compressstoreu_epi32(self.offsets.as_mut_ptr() as *mut _, mask, offsets);
            self.content_len = mask.count_ones() as u8;
            self.cursor = 0;
        }
    }

    fn load_next_chunk(&mut self) -> Option<()> {
        while self.cursor == self.content_len {
            if self.base + VECTOR_SIZE >= self.vec.len() {
                return None;
            }

            self.base += VECTOR_SIZE;
            let data = self.vec.get_bits_unchecked(self.base, VECTOR_SIZE) as u16;
            self.load_chunk(data);
        }
        Some(())
    }
}

impl<const ZERO: bool> Iterator for BitSetIter<'_, ZERO> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.base >= self.vec.len() {
            return None;
        }

        if self.cursor == self.content_len {
            if self.load_next_chunk().is_none() {
                if ZERO {
                    while self.base < self.vec.len() && self.vec.get_unchecked(self.base) != 0 {
                        self.base += 1;
                    }
                } else {
                    while self.base < self.vec.len() && self.vec.get_unchecked(self.base) != 1 {
                        self.base += 1;
                    }
                }

                return if self.base < self.vec.len() {
                    self.base += 1;
                    Some(self.base - 1)
                } else {
                    None
                };
            }
        }

        let offset = self.offsets[self.cursor as usize];
        self.cursor += 1;
        Some(self.base + offset as u64)
    }
}
