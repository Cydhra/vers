//! This module contains a simple [bit vector][BitVec] implementation with no overhead and a fast succinct
//! bit vector implementation with [rank and select queries][fast_rs_vec::RsVec].

use crate::bit_vec::mask::MaskedBitVec;
use crate::util::impl_vector_iterator;
use std::cmp::min;
use std::mem::size_of;

pub mod fast_rs_vec;

pub mod sparse;

pub mod mask;

/// Size of a word in bitvectors. All vectors operate on 64-bit words.
const WORD_SIZE: usize = 64;

/// Type alias for masked bitvectors that implement a simple bitwise binary operation.
/// The first lifetime is for the bit vector that is being masked, the second lifetime is for the
/// mask.
pub type BitMask<'s, 'b> = MaskedBitVec<'s, 'b, fn(u64, u64) -> u64>;

/// A simple bit vector that does not support rank and select queries.
/// Bits are stored in little-endian order, i.e. the least significant bit is stored first.
/// The bit vector is stored as a sequence of 64 bit limbs.
/// The last limb may be partially filled.
///
/// The bit vector has a wide range of constructors that allow for easy creation from various
/// sources.
/// Among them are constructors for creating an empty vector ([`BitVec::new`]),
/// creating one from single bits of various integer types ([`BitVec::from_bits`] and variations),
/// creating limbs from u64 values directly ([`BitVec::from_limbs`] and variations),
/// or packing a sequence of numerical values into a dense bit sequence
/// ([`BitVec::pack_sequence_u64`] and variations).
///
/// The bit vector can be modified after creation
/// (e.g. by appending [bits](BitVec::append_bits)
/// or [words](BitVec::append_word),
/// [flipping](BitVec::flip_bit),
/// or [setting](BitVec::set) bits).
/// Bits can be [accessed](BitVec::get) by position,
/// and [multiple bits](BitVec::get_bits) can be accessed at once.
/// Bits can be [dropped](BitVec::drop_last) from the end.
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
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
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

    /// Create a new empty bit vector with the given capacity.
    /// The capacity is measured in bits.
    /// The bit vector will be able to hold at least `capacity` bits without reallocating.
    /// More memory may be allocated according to the underlying allocation strategy.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity / WORD_SIZE + 1),
            len: 0,
        }
    }

    /// Create a new bit vector with all zeros and the given length.
    /// The length is measured in bits.
    #[must_use]
    pub fn from_zeros(len: usize) -> Self {
        let mut data = vec![0; len / WORD_SIZE];
        if len % WORD_SIZE != 0 {
            data.push(0);
        }
        Self { data, len }
    }

    /// Create a new bit vector with all ones and the given length.
    /// The length is measured in bits.
    #[must_use]
    pub fn from_ones(len: usize) -> Self {
        let mut data = vec![u64::MAX; len / WORD_SIZE];
        if len % WORD_SIZE != 0 {
            data.push((1 << (len % WORD_SIZE)) - 1);
        }
        Self { data, len }
    }

    /// Construct a bit vector from a set of bits given as distinct u8 values.
    /// The constructor will take the least significant bit from each value and append it to a
    /// bit vector.
    /// All other bits are ignored.
    ///
    /// See also: [`from_bits_u16`], [`from_bits_u32`], [`from_bits_u64`], [`from_bits_iter`]
    ///
    /// # Example
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let bits: &[u8] = &[1, 0, 1, 1, 1, 1];
    /// let bv = BitVec::from_bits(&bits);
    ///
    /// assert_eq!(bv.len(), 6);
    /// assert_eq!(bv.get_bits(0, 6), Some(0b111101u64));
    /// ```
    ///
    /// [`from_bits_u16`]: BitVec::from_bits_u16
    /// [`from_bits_u32`]: BitVec::from_bits_u32
    /// [`from_bits_u64`]: BitVec::from_bits_u64
    /// [`from_bits_iter`]: BitVec::from_bits_iter
    #[must_use]
    pub fn from_bits(bits: &[u8]) -> Self {
        let mut bv = Self::with_capacity(bits.len());
        bits.iter().for_each(|&b| bv.append_bit(b.into()));
        bv
    }

    /// Construct a bit vector from a set of bits given as distinct u16 values.
    /// The constructor will take the least significant bit from each value and append it to a
    /// bit vector.
    /// All other bits are ignored.
    ///
    /// See also: [`from_bits`], [`from_bits_u32`], [`from_bits_u64`], [`from_bits_iter`]
    ///
    /// [`from_bits`]: BitVec::from_bits
    /// [`from_bits_u32`]: BitVec::from_bits_u32
    /// [`from_bits_u64`]: BitVec::from_bits_u64
    /// [`from_bits_iter`]: BitVec::from_bits_iter
    #[must_use]
    pub fn from_bits_u16(bits: &[u16]) -> Self {
        let mut bv = Self::with_capacity(bits.len());
        bits.iter().for_each(|&b| bv.append_bit_u16(b));
        bv
    }

    /// Construct a bit vector from a set of bits given as distinct u32 values.
    /// The constructor will take the least significant bit from each value and append it to a
    /// bit vector.
    /// All other bits are ignored.
    ///
    /// See also: [`from_bits`], [`from_bits_u16`], [`from_bits_u64`], [`from_bits_iter`]
    ///
    /// [`from_bits`]: BitVec::from_bits
    /// [`from_bits_u16`]: BitVec::from_bits_u16
    /// [`from_bits_u64`]: BitVec::from_bits_u64
    /// [`from_bits_iter`]: BitVec::from_bits_iter
    #[must_use]
    pub fn from_bits_u32(bits: &[u32]) -> Self {
        let mut bv = Self::with_capacity(bits.len());
        bits.iter().for_each(|&b| bv.append_bit_u32(b));
        bv
    }

    /// Construct a bit vector from a set of bits given as distinct u64 values.
    /// The constructor will take the least significant bit from each value and append it to a
    /// bit vector.
    /// All other bits are ignored.
    ///
    /// See also: [`from_bits`], [`from_bits_u16`], [`from_bits_u32`], [`from_bits_iter`]
    ///
    /// [`from_bits`]: BitVec::from_bits
    /// [`from_bits_u16`]: BitVec::from_bits_u16
    /// [`from_bits_u32`]: BitVec::from_bits_u32
    /// [`from_bits_iter`]: BitVec::from_bits_iter
    #[must_use]
    pub fn from_bits_u64(bits: &[u64]) -> Self {
        let mut bv = Self::with_capacity(bits.len());
        bits.iter().for_each(|&b| bv.append_bit(b));
        bv
    }

    /// Construct a bit vector from an iterator of bits.
    /// The constructor will take the least significant bit from each value and append it to a
    /// bit vector.
    /// All other bits are ignored.
    /// The iterator must yield values that can be converted into u64 values.
    ///
    /// See also: [`from_bits`], [`from_bits_u16`], [`from_bits_u32`], [`from_bits_u64`]
    ///
    /// # Example
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let bits = [true, false, true, true, true, true];
    /// let bv = BitVec::from_bits_iter(bits.iter().copied());
    ///
    /// let bits = [0b1u8, 0b0, 0b1, 0b1, 0b1, 0b1];
    /// let bv2 = BitVec::from_bits_iter(bits.iter().copied());
    ///
    /// assert_eq!(bv.len(), 6);
    /// assert_eq!(bv.get_bits(0, 6), Some(0b111101u64));
    /// assert_eq!(bv, bv2);
    /// ```
    ///
    /// [`from_bits`]: BitVec::from_bits
    /// [`from_bits_u16`]: BitVec::from_bits_u16
    /// [`from_bits_u32`]: BitVec::from_bits_u32
    /// [`from_bits_u64`]: BitVec::from_bits_u64
    #[must_use]
    pub fn from_bits_iter<I, E>(iter: I) -> Self
    where
        E: Into<u64>,
        I: IntoIterator<Item = E>,
    {
        let iter = iter.into_iter();
        let mut bv = Self::with_capacity(iter.size_hint().0);
        for bit in iter {
            bv.append_bit(bit.into());
        }
        bv
    }

    /// Construct a bit vector from a slice of u64 quad words.
    /// The quad words are interpreted as limbs of the bit vector (i.e. each quad word contributes
    /// 64 bits to the bit vector).
    /// Since the data is only cloned without any masking or transformation,
    /// this is one of the fastest ways to create a bit vector.
    ///
    /// See also: [`from_vec`], [`from_limbs_iter`]
    ///
    /// # Example
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let words = [0, 256, u64::MAX];
    /// let bv = BitVec::from_limbs(&words);
    ///
    /// assert_eq!(bv.len(), 192);
    /// assert_eq!(bv.get_bits(0, 64), Some(0u64));
    /// assert_eq!(bv.get(72), Some(1));
    /// assert_eq!(bv.get_bits(128, 64), Some(u64::MAX));
    /// ```
    ///
    /// [`from_vec`]: BitVec::from_vec
    /// [`from_limbs_iter`]: BitVec::from_limbs_iter
    #[must_use]
    pub fn from_limbs(words: &[u64]) -> Self {
        let len = words.len() * WORD_SIZE;
        Self {
            data: words.to_vec(),
            len,
        }
    }

    /// Construct a bit vector from an iterator of u64 quad words.
    /// The quad words are interpreted as limbs of the bit vector (i.e. each quad word contributes
    /// 64 bits to the bit vector).
    /// Since the data is only cloned without any masking or transformation,
    /// this is one of the fastest ways to create a bit vector.
    ///
    /// See also: [`from_limbs`], [`from_vec`]
    ///
    /// # Example
    /// ```rust
    /// use std::iter::repeat;
    /// use vers_vecs::BitVec;
    ///
    /// let zeros = repeat(0xaaaaaaaaaaaaaaaau64).take(10);
    /// let bv = BitVec::from_limbs_iter(zeros);
    ///
    /// assert_eq!(bv.len(), 640);
    /// for i in 0..640 {
    ///    assert_eq!(bv.get(i), Some((i % 2 == 1) as u64));
    /// }
    /// ```
    ///
    /// [`from_limbs`]: BitVec::from_limbs
    /// [`from_vec`]: BitVec::from_vec
    pub fn from_limbs_iter<I, E>(iter: I) -> Self
    where
        E: Into<u64>,
        I: IntoIterator<Item = E>,
    {
        let vec = iter.into_iter().map(Into::into).collect();
        Self::from_vec(vec)
    }

    /// Construct a bit vector from a vector of u64 quad words.
    /// The quad words are interpreted as limbs of the bit vector
    /// (i.e. each quad word contributes 64 bits to the bit vector).
    /// Since the data is moved without any masking or transformation, this is one of the fastest ways
    /// to create a bit vector.
    ///
    /// See also: [`from_limbs`], [`from_limbs_iter`]
    ///
    /// # Example
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let words = vec![0, 256, u64::MAX];
    /// let bv = BitVec::from_vec(words);
    ///
    /// assert_eq!(bv.len(), 192);
    /// assert_eq!(bv.get_bits(0, 64), Some(0u64));
    /// assert_eq!(bv.get(72), Some(1));
    /// assert_eq!(bv.get_bits(128, 64), Some(u64::MAX));
    /// ```
    ///
    /// [`from_limbs`]: BitVec::from_limbs
    /// [`from_limbs_iter`]: BitVec::from_limbs_iter
    #[must_use]
    pub fn from_vec(data: Vec<u64>) -> Self {
        let len = data.len() * WORD_SIZE;
        Self { data, len }
    }

    fn pack_bits<T, const MAX_BITS: usize>(sequence: &[T], bits_per_element: usize) -> Self
    where
        T: Into<u64> + Copy,
    {
        let mut bv = Self::with_capacity(sequence.len() * bits_per_element);
        for &word in sequence {
            if bits_per_element <= MAX_BITS {
                bv.append_bits(word.into(), bits_per_element);
            } else {
                bv.append_bits(word.into(), MAX_BITS);
                let mut rest = bits_per_element - MAX_BITS;
                while rest > 0 {
                    bv.append_bits(0, min(rest, MAX_BITS));
                    rest = rest.saturating_sub(MAX_BITS);
                }
            }
        }
        bv
    }

    /// Construct a bit vector by packing a sequence of numerical values into a dense sequence.
    /// The bits are appended in little-endian order (i.e. the least significant bit is appended first).
    /// The number of bits per element is given by `bits_per_element`.
    /// The sequence is given as a slice of u64 values.
    /// If the number of bits per element is smaller than 64, the function takes the
    /// least significant bits of each element, and discards the rest.
    /// If the number of bits per element is larger than 64, the function will pad the elements
    /// with zeros.
    /// The function will append the bits of each element to the bit vector in the order they are
    /// given in the sequence (i.e. the first element takes bits `0..bits_per_element` of the vector).
    ///
    /// See also: [`pack_sequence_u32`], [`pack_sequence_u16`], [`pack_sequence_u8`]
    ///
    /// # Example
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let sequence = [0b1010u64, 0b1100u64, 0b1111u64];
    /// let bv = BitVec::pack_sequence_u64(&sequence, 4);
    ///
    /// assert_eq!(bv.len(), 12);
    /// assert_eq!(bv.get_bits(0, 4), Some(0b1010u64));
    /// assert_eq!(bv.get_bits(4, 4), Some(0b1100u64));
    /// assert_eq!(bv.get_bits(8, 4), Some(0b1111u64));
    /// ```
    ///
    /// [`pack_sequence_u32`]: BitVec::pack_sequence_u32
    /// [`pack_sequence_u16`]: BitVec::pack_sequence_u16
    /// [`pack_sequence_u8`]: BitVec::pack_sequence_u8
    #[must_use]
    pub fn pack_sequence_u64(sequence: &[u64], bits_per_element: usize) -> Self {
        Self::pack_bits::<_, 64>(sequence, bits_per_element)
    }

    /// Construct a bit vector by packing a sequence of numerical values into a dense sequence.
    /// The bits are appended in little-endian order (i.e. the least significant bit is appended first).
    /// The number of bits per element is given by `bits_per_element`.
    /// The sequence is given as a slice of u32 values.
    /// If the number of bits per element is smaller than 32, the function takes the
    /// least significant bits of each element, and discards the rest.
    /// If the number of bits per element is larger than 32, the function will pad the elements
    /// with zeros.
    /// The function will append the bits of each element to the bit vector in the order they are
    /// given in the sequence (i.e. the first element takes bits `0..bits_per_element` of the vector).
    ///
    /// See also: [`pack_sequence_u64`], [`pack_sequence_u16`], [`pack_sequence_u8`]
    ///
    /// # Example
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let sequence = [0b1010u32, 0b1100u32, 0b1111u32];
    /// let bv = BitVec::pack_sequence_u32(&sequence, 4);
    ///
    /// assert_eq!(bv.len(), 12);
    /// assert_eq!(bv.get_bits(0, 4), Some(0b1010u64));
    /// assert_eq!(bv.get_bits(4, 4), Some(0b1100u64));
    /// assert_eq!(bv.get_bits(8, 4), Some(0b1111u64));
    /// ```
    ///
    /// [`pack_sequence_u64`]: BitVec::pack_sequence_u64
    /// [`pack_sequence_u16`]: BitVec::pack_sequence_u16
    /// [`pack_sequence_u8`]: BitVec::pack_sequence_u8
    #[must_use]
    pub fn pack_sequence_u32(sequence: &[u32], bits_per_element: usize) -> Self {
        Self::pack_bits::<_, 32>(sequence, bits_per_element)
    }

    /// Construct a bit vector by packing a sequence of numerical values into a dense sequence.
    /// The bits are appended in little-endian order (i.e. the least significant bit is appended first).
    /// The number of bits per element is given by `bits_per_element`.
    /// The sequence is given as a slice of u16 values.
    /// If the number of bits per element is smaller than 16, the function takes the
    /// least significant bits of each element, and discards the rest.
    /// If the number of bits per element is larger than 16, the function will pad the elements
    /// with zeros.
    /// The function will append the bits of each element to the bit vector in the order they are
    /// given in the sequence (i.e. the first element takes bits `0..bits_per_element` of the vector).
    ///
    /// See also: [`pack_sequence_u64`], [`pack_sequence_u32`], [`pack_sequence_u8`]
    ///
    /// # Example
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let sequence = [0b1010u16, 0b1100u16, 0b1111u16];
    /// let bv = BitVec::pack_sequence_u16(&sequence, 4);
    ///
    /// assert_eq!(bv.len(), 12);
    /// assert_eq!(bv.get_bits(0, 4), Some(0b1010u64));
    /// assert_eq!(bv.get_bits(4, 4), Some(0b1100u64));
    /// assert_eq!(bv.get_bits(8, 4), Some(0b1111u64));
    /// ```
    ///
    /// [`pack_sequence_u64`]: BitVec::pack_sequence_u64
    /// [`pack_sequence_u32`]: BitVec::pack_sequence_u32
    /// [`pack_sequence_u8`]: BitVec::pack_sequence_u8
    #[must_use]
    pub fn pack_sequence_u16(sequence: &[u16], bits_per_element: usize) -> Self {
        Self::pack_bits::<_, 16>(sequence, bits_per_element)
    }

    /// Construct a bit vector by packing a sequence of numerical values into a dense sequence.
    /// The bits are appended in little-endian order (i.e. the least significant bit is appended first).
    /// The number of bits per element is given by `bits_per_element`.
    /// The sequence is given as a slice of u8 values.
    /// If the number of bits per element is smaller than 8, the function takes the
    /// least significant bits of each element, and discards the rest.
    /// If the number of bits per element is larger than 8, the function will pad the elements
    /// with zeros.
    /// The function will append the bits of each element to the bit vector in the order they are
    /// given in the sequence (i.e. the first element takes bits `0..bits_per_element` of the vector).
    ///
    /// See also: [`pack_sequence_u64`], [`pack_sequence_u32`], [`pack_sequence_u16`]
    ///
    /// # Example
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let sequence = [0b1010u8, 0b1100u8, 0b1111u8];
    /// let bv = BitVec::pack_sequence_u8(&sequence, 4);
    ///
    /// assert_eq!(bv.len(), 12);
    /// assert_eq!(bv.get_bits(0, 4), Some(0b1010u64));
    /// assert_eq!(bv.get_bits(4, 4), Some(0b1100u64));
    /// assert_eq!(bv.get_bits(8, 4), Some(0b1111u64));
    /// ```
    ///
    /// [`pack_sequence_u64`]: BitVec::pack_sequence_u64
    /// [`pack_sequence_u32`]: BitVec::pack_sequence_u32
    /// [`pack_sequence_u16`]: BitVec::pack_sequence_u16
    #[must_use]
    pub fn pack_sequence_u8(sequence: &[u8], bits_per_element: usize) -> Self {
        Self::pack_bits::<_, 8>(sequence, bits_per_element)
    }

    /// Append a bit encoded as a `bool` to the bit vector, where `true` means 1 and `false` means 0.
    ///
    /// See also: [`append_bit`], [`append_bit_u32`], [`append_bit_u16`], [`append_bit_u8`], [`append_word`]
    ///
    /// # Example
    ///
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let mut bv = BitVec::new();
    /// bv.append(true);
    ///
    /// assert_eq!(bv.len(), 1);
    /// assert_eq!(bv.get(0), Some(1));
    /// ```
    ///
    /// [`append_bit`]: BitVec::append_bit
    /// [`append_bit_u32`]: BitVec::append_bit_u32
    /// [`append_bit_u16`]: BitVec::append_bit_u16
    /// [`append_bit_u8`]: BitVec::append_bit_u8
    /// [`append_word`]: BitVec::append_word
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
    ///
    /// # Example
    ///
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let mut bv = BitVec::from_bits(&[1, 0, 1, 1, 1, 1]);
    /// bv.drop_last(3);
    ///
    /// assert_eq!(bv.len(), 3);
    /// assert_eq!(bv.get_bits(0, 3), Some(0b101u64));
    ///
    /// bv.drop_last(4);
    ///
    /// assert!(bv.is_empty());
    /// ```
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

    /// Append a bit encoded in a u64.
    /// The least significant bit is appended to the bit vector.
    /// All other bits are ignored.
    ///
    /// See also: [`append`], [`append_bit_u32`], [`append_bit_u16`], [`append_bit_u8`], [`append_word`]
    ///
    /// # Example
    ///
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let mut bv = BitVec::new();
    ///
    /// bv.append_bit(1);
    /// bv.append_bit(0);
    ///
    /// assert_eq!(bv.len(), 2);
    /// assert_eq!(bv.get(0), Some(1));
    /// assert_eq!(bv.get(1), Some(0));
    /// ```
    ///
    /// [`append`]: BitVec::append
    /// [`append_bit_u32`]: BitVec::append_bit_u32
    /// [`append_bit_u16`]: BitVec::append_bit_u16
    /// [`append_bit_u8`]: BitVec::append_bit_u8
    /// [`append_word`]: BitVec::append_word
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
    ///
    /// See also: [`append`], [`append_bit`], [`append_bit_u16`], [`append_bit_u8`], [`append_word`]
    ///
    /// [`append`]: BitVec::append
    /// [`append_bit`]: BitVec::append_bit
    /// [`append_bit_u16`]: BitVec::append_bit_u16
    /// [`append_bit_u8`]: BitVec::append_bit_u8
    /// [`append_word`]: BitVec::append_word
    pub fn append_bit_u32(&mut self, bit: u32) {
        self.append_bit(u64::from(bit));
    }

    /// Append a bit from a u16. The least significant bit is appended to the bit vector.
    /// All other bits are ignored.
    ///
    /// See also: [`append`], [`append_bit`], [`append_bit_u32`], [`append_bit_u8`], [`append_word`]
    ///
    /// [`append`]: BitVec::append
    /// [`append_bit`]: BitVec::append_bit
    /// [`append_bit_u32`]: BitVec::append_bit_u32
    /// [`append_bit_u8`]: BitVec::append_bit_u8
    /// [`append_word`]: BitVec::append_word
    pub fn append_bit_u16(&mut self, bit: u16) {
        self.append_bit(u64::from(bit));
    }

    /// Append a bit from a u8. The least significant bit is appended to the bit vector.
    /// All other bits are ignored.
    ///
    /// See also: [`append`], [`append_bit`], [`append_bit_u32`], [`append_bit_u16`], [`append_word`]
    ///
    /// [`append`]: BitVec::append
    /// [`append_bit`]: BitVec::append_bit
    /// [`append_bit_u32`]: BitVec::append_bit_u32
    /// [`append_bit_u16`]: BitVec::append_bit_u16
    /// [`append_word`]: BitVec::append_word
    pub fn append_bit_u8(&mut self, bit: u8) {
        self.append_bit(u64::from(bit));
    }

    /// Append a word to the bit vector. The bits are appended in little endian order (i.e. the first
    /// bit of the word is appended first).
    ///
    /// See also: [`append`], [`append_bit`], [`append_bit_u32`], [`append_bit_u16`], [`append_bit_u8`]
    ///
    /// # Example
    ///
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let mut bv = BitVec::new();
    /// bv.append_word(0b1010_1010_1010_1010u64);
    ///
    /// assert_eq!(bv.len(), 64);
    /// for i in 0..64 {
    ///    assert_eq!(bv.get(i), Some((0b1010_1010_1010_1010u64 >> i) & 1));
    /// }
    /// ```
    ///
    /// [`append`]: BitVec::append
    /// [`append_bit`]: BitVec::append_bit
    /// [`append_bit_u32`]: BitVec::append_bit_u32
    /// [`append_bit_u16`]: BitVec::append_bit_u16
    /// [`append_bit_u8`]: BitVec::append_bit_u8
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

    /// Append multiple bits to the bit vector.
    /// The bits are appended in little-endian order (i.e. the least significant bit is appended first).
    /// The number of bits to append is given by `len`. The bits are taken from the least
    /// significant bits of `bits`.
    /// All other bits are ignored.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let mut bv = BitVec::new();
    /// bv.append_bits(0b1010_1010_1010_1010u64, 16);
    ///
    /// assert_eq!(bv.len(), 16);
    /// assert_eq!(bv.get_bits(0, 16), Some(0b1010_1010_1010_1010u64));
    /// ```
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
    /// # Example
    ///
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let mut bv = BitVec::from_bits(&[1, 0, 1, 1, 1, 1]);
    /// bv.flip_bit(1);
    ///
    /// assert_eq!(bv.len(), 6);
    /// assert_eq!(bv.get_bits(0, 6), Some(0b111111u64));
    /// ```
    ///
    /// # Panics
    /// If the position is larger than the length of the vector, the function panics.
    pub fn flip_bit(&mut self, pos: usize) {
        assert!(pos < self.len, "Index out of bounds");
        self.flip_bit_unchecked(pos);
    }

    /// Flip the bit at the given position.
    ///
    /// See also: [`flip_bit`]
    ///
    /// # Panics
    /// If the position is larger than the length of the
    /// vector, the function will either modify unused memory or panic.
    /// This will not corrupt memory.
    ///
    /// [`flip_bit`]: BitVec::flip_bit
    pub fn flip_bit_unchecked(&mut self, pos: usize) {
        self.data[pos / WORD_SIZE] ^= 1 << (pos % WORD_SIZE);
    }

    /// Return the bit at the given position.
    /// The bit is encoded in the least significant bit of a u64 value.
    /// If the position is larger than the length of the vector, None is returned.
    ///
    /// See also: [`get_unchecked`]
    ///
    /// # Example
    ///
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let bv = BitVec::from_bits(&[1, 0, 1, 1, 1, 1]);
    ///
    /// assert_eq!(bv.get(1), Some(0));
    /// assert_eq!(bv.get(2), Some(1));
    /// ```
    #[must_use]
    pub fn get(&self, pos: usize) -> Option<u64> {
        if pos >= self.len {
            None
        } else {
            Some(self.get_unchecked(pos))
        }
    }

    /// Return the bit at the given position.
    /// The bit is encoded in the least significant bit of a u64 value.
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
    /// The bit is encoded in the least significant bit of a u64 value.
    ///
    /// See also: [`set_unchecked`]
    ///
    /// # Example
    ///
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let mut bv = BitVec::from_bits(&[1, 0, 1, 1, 1, 1]);
    /// bv.set(1, 1).unwrap();
    ///
    /// assert_eq!(bv.len(), 6);
    /// assert_eq!(bv.get_bits(0, 6), Some(0b111111u64));
    /// ```
    ///
    /// # Errors
    /// If the position is out of range, the function will return `Err` with an error message,
    /// otherwise it will return an empty `Ok`.
    ///
    /// [`set_unchecked`]: BitVec::set_unchecked
    pub fn set(&mut self, pos: usize, value: u64) -> Result<(), &str> {
        if pos >= self.len {
            Err("out of range")
        } else {
            self.set_unchecked(pos, value);
            Ok(())
        }
    }

    /// Set the bit at the given position.
    /// The bit is encoded in the least significant bit of a u64 value.
    ///
    /// # Panics
    /// If the position is larger than the length of the vector,
    /// the function will either do nothing, or panic.
    /// Use [`set`] to properly handle this case with a `Result`.
    ///
    /// [`set`]: BitVec::set
    pub fn set_unchecked(&mut self, pos: usize, value: u64) {
        self.data[pos / WORD_SIZE] = (self.data[pos / WORD_SIZE] & !(0x1 << (pos % WORD_SIZE)))
            | ((value & 0x1) << (pos % WORD_SIZE));
    }

    /// Return whether the bit at the given position is set.
    /// If the position is larger than the length of the vector, None is returned.
    ///
    /// See also: [`is_bit_set_unchecked`]
    ///
    /// # Example
    ///
    /// ```rust
    /// use vers_vecs::BitVec;
    ///
    /// let bv = BitVec::from_bits(&[1, 0, 1, 1, 1, 1]);
    ///
    /// assert!(!bv.is_bit_set(1).unwrap());
    /// assert!(bv.is_bit_set(2).unwrap());
    /// ```
    ///
    /// [`is_bit_set_unchecked`]: BitVec::is_bit_set_unchecked
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

    /// Return multiple bits at the given position.
    /// The number of bits to return is given by `len`.
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
    #[allow(clippy::comparison_chain)] // readability
    #[inline(always)] // inline to gain loop optimization and pipeline advantages for elias fano
    #[allow(clippy::cast_possible_truncation)] // parameter must be out of scope for this to happen
    pub fn get_bits_unchecked(&self, pos: usize, len: usize) -> u64 {
        debug_assert!(len <= WORD_SIZE);
        let partial_word = self.data[pos / WORD_SIZE] >> (pos % WORD_SIZE);
        if pos % WORD_SIZE + len == WORD_SIZE {
            partial_word
        } else if pos % WORD_SIZE + len < WORD_SIZE {
            partial_word & ((1 << (len % WORD_SIZE)) - 1)
        } else {
            (partial_word | (self.data[pos / WORD_SIZE + 1] << (WORD_SIZE - pos % WORD_SIZE)))
                & 1u64.checked_shl(len as u32).unwrap_or(0).wrapping_sub(1)
        }
    }

    /// Return the number of ones in the bit vector. Since the bit vector doesn't store additional
    /// metadata, this value is calculated. Use [`RsVec`] for constant-time rank operations.
    ///
    /// [`RsVec`]: crate::RsVec
    #[must_use]
    #[allow(clippy::missing_panics_doc)] // can't panic because of manual bounds check
    pub fn count_ones(&self) -> u64 {
        let mut ones: u64 = self.data[0..self.len / WORD_SIZE]
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
    /// whenever an operation on the resulting vector is performed.
    ///
    /// # Errors
    /// Returns an error if the length of the vector doesn't match the mask length.
    #[inline]
    pub fn mask_or<'s, 'b>(&'s self, mask: &'b BitVec) -> Result<BitMask<'s, 'b>, String> {
        MaskedBitVec::new(self, mask, |a, b| a | b)
    }

    /// Mask this bit vector with another bitvector using bitwise or.
    /// The mask is applied immediately, unlike in [`mask_or`].
    ///
    /// # Errors
    /// Returns an error if the length of the vector doesn't match the mask length.
    ///
    /// [`mask_or`]: BitVec::mask_or
    pub fn apply_mask_or(&mut self, mask: &BitVec) -> Result<(), String> {
        if self.len != mask.len {
            return Err(String::from(
                "mask cannot have different length than vector",
            ));
        }

        for i in 0..self.data.len() {
            self.data[i] |= mask.data[i];
        }

        Ok(())
    }

    /// Mask this bit vector with another bitvector using bitwise and. The mask is applied lazily
    /// whenever an operation on the resulting vector is performed.
    ///
    /// # Errors
    /// Returns an error if the length of the vector doesn't match the mask length.
    #[inline]
    pub fn mask_and<'s, 'b>(&'s self, mask: &'b BitVec) -> Result<BitMask<'s, 'b>, String> {
        MaskedBitVec::new(self, mask, |a, b| a & b)
    }

    /// Mask this bit vector with another bitvector using bitwise and.
    /// The mask is applied immediately, unlike in [`mask_and`].
    ///
    /// # Errors
    /// Returns an error if the length of the vector doesn't match the mask length.
    ///
    /// [`mask_and`]: BitVec::mask_and
    pub fn apply_mask_and(&mut self, mask: &BitVec) -> Result<(), String> {
        if self.len != mask.len {
            return Err(String::from(
                "mask cannot have different length than vector",
            ));
        }

        for i in 0..self.data.len() {
            self.data[i] &= mask.data[i];
        }

        Ok(())
    }

    /// Mask this bit vector with another bitvector using bitwise xor. The mask is applied lazily
    /// whenever an operation on the resulting vector is performed.
    ///
    /// # Errors
    /// Returns an error if the length of the vector doesn't match the mask length.
    #[inline]
    pub fn mask_xor<'s, 'b>(&'s self, mask: &'b BitVec) -> Result<BitMask<'s, 'b>, String> {
        MaskedBitVec::new(self, mask, |a, b| a ^ b)
    }

    /// Mask this bit vector with another bitvector using bitwise xor.
    /// The mask is applied immediately, unlike in [`mask_xor`].
    ///
    /// # Errors
    /// Returns an error if the length of the vector doesn't match the mask length.
    ///
    /// [`mask_xor`]: BitVec::mask_xor
    pub fn apply_mask_xor(&mut self, mask: &BitVec) -> Result<(), String> {
        if self.len != mask.len {
            return Err(String::from(
                "mask cannot have different length than vector",
            ));
        }

        for i in 0..self.data.len() {
            self.data[i] ^= mask.data[i];
        }

        Ok(())
    }

    /// Mask this bit vector with another bitvector using a custom masking operation. The mask is
    /// applied lazily whenever an operation on the resulting vector is performed.
    ///
    /// The masking operation takes two 64 bit values which contain blocks of 64 bits each.
    /// The last block of a bit vector might contain fewer bits, and will be padded with
    /// unpredictable data. Implementations may choose to modify those padding bits without
    /// repercussions. Implementations shouldn't use operations like bit shift, because the bit order
    /// within the vector is unspecified.
    ///
    /// # Errors
    /// Returns an error if the length of the vector doesn't match the mask length.
    #[inline]
    pub fn mask_custom<'s, 'b, F>(
        &'s self,
        mask: &'b BitVec,
        mask_op: F,
    ) -> Result<MaskedBitVec<'s, 'b, F>, String>
    where
        F: Fn(u64, u64) -> u64,
    {
        MaskedBitVec::new(self, mask, mask_op)
    }

    /// Mask this bit vector with another bitvector using a custom masking operation.
    /// The mask is applied immediately, unlike in [`mask_custom`].
    ///
    /// The masking operation takes two 64 bit values which contain blocks of 64 bits each.
    /// The last block of a bit vector might contain fewer bits, and will be padded with
    /// unpredictable data. Implementations may choose to modify those padding bits without
    /// repercussions. Implementations shouldn't use operations like bit shift, because the bit order
    /// within the vector is unspecified.
    ///
    /// # Errors
    /// Returns an error if the length of the vector doesn't match the mask length.
    ///
    /// [`mask_custom`]: BitVec::mask_custom
    #[inline]
    pub fn apply_mask_custom(
        &mut self,
        mask: &BitVec,
        mask_op: fn(u64, u64) -> u64,
    ) -> Result<(), String> {
        if self.len != mask.len {
            return Err(String::from(
                "mask cannot have different length than vector",
            ));
        }

        for i in 0..self.data.len() {
            self.data[i] = mask_op(self.data[i], mask.data[i]);
        }

        Ok(())
    }

    /// Returns the number of bytes on the heap for this vector.
    /// Does not include allocated memory that isn't used.
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>()
    }
}

impl_vector_iterator! { BitVec, BitVecIter, BitVecRefIter }

/// Create a new bit vector from a slice of u64 values.
/// The bits are appended in little-endian order (i.e. the least significant bit is appended first).
/// The function will append the bits of each element to the bit vector in the order they are
/// given in the slice (i.e. the first element takes bits `0..64` of the vector).
impl From<&[u64]> for BitVec {
    fn from(data: &[u64]) -> Self {
        BitVec::from_limbs(data)
    }
}

/// Create a new bit vector from a slice of u64 values.
/// The bits are appended in little-endian order (i.e. the least significant bit is appended first).
/// The function will append the bits of each element to the bit vector in the order they are
/// given in the slice (i.e. the first element takes bits `0..64` of the vector).
impl From<Vec<u64>> for BitVec {
    fn from(data: Vec<u64>) -> Self {
        BitVec::from_limbs(&data)
    }
}

/// Create a new bit vector from u64 values.
/// The bits are appended in little-endian order (i.e. the least significant bit is appended first).
/// The function will append the bits of each element to the bit vector in the order they are
/// given in the iterator (i.e. the first element takes bits `0..64` of the vector).
impl FromIterator<u64> for BitVec {
    fn from_iter<T: IntoIterator<Item = u64>>(iter: T) -> Self {
        BitVec::from_limbs_iter(iter)
    }
}

#[cfg(test)]
mod tests;
