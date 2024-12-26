//! This module provides the lookup table and lookup functionality to obtain excess values for
//! checking blocks of bits in the tree's bitvector, instead of checking each bit individually.

/// How big the lookup blocks are. We store this in a constant so we can switch out this module
/// using a crate feature against one where this constant is redefined to 16, but reuse the actual
/// scanning code for operations on the tree.
#[cfg(feature = "u16_lookup")]
pub(crate) const LOOKUP_BLOCK_SIZE: u64 = 16;
#[cfg(not(feature = "u16_lookup"))]
pub(crate) const LOOKUP_BLOCK_SIZE: u64 = 8;

/// Integer type holding the blocks of the parenthesis expression we look up at once
#[cfg(feature = "u16_lookup")]
type LookupBlockType = u16;
#[cfg(not(feature = "u16_lookup"))]
type LookupBlockType = u8;

/// Signed version of `LookupBlockType`
#[cfg(feature = "u16_lookup")]
type SignedLookupBlockType = i16;
#[cfg(not(feature = "u16_lookup"))]
type SignedLookupBlockType = i8;

/// Data type we use in the lookup table to store excess values for lookup. Needs to be one size larger
/// than `LookupBlockType`
#[cfg(feature = "u16_lookup")]
type EncodedTableType = u32;
#[cfg(not(feature = "u16_lookup"))]
type EncodedTableType = u16;

/// Maximum value that `LookupBlockType` can hold, stored in one size larger because we need to
/// iterate up to and including it
#[cfg(feature = "u16_lookup")]
const LOOKUP_MAX_VALUE: u32 = u16::MAX as u32;
#[cfg(not(feature = "u16_lookup"))]
const LOOKUP_MAX_VALUE: u32 = u8::MAX as u32;

/// The lookup entry is indexed by the numerical value of a parenthesis expression block. The table
/// contains the minimum, maximum, and total excess encoded in a single integer.
///
/// The encoding scheme is simple:
/// The least significant 5 (6) bits encode maximum excess (which is between -8 (-16) and 8 (16),
/// which we store with an offset of 8 (16), so we don't have to deal with dual encoding),
/// the next 5 (6) bits are the minimum excess encoded analogously,
/// and the following 5 (6) bits are the total excess.
///
/// The rest of the bits are zero.
#[allow(long_running_const_eval)]
const PAREN_BLOCK_LOOKUP: [EncodedTableType; 1 << LOOKUP_BLOCK_SIZE] = calculate_lookup_table();

/// Offset to add to encoded excess values, so negative numbers are stored as positive integers, reducing
/// encoding complexity
const ENCODING_OFFSET: i32 = LOOKUP_BLOCK_SIZE as i32;

/// Bitmask for one of the lookup values.
#[cfg(feature = "u16_lookup")]
const ENCODING_MASK: u32 = 0b111111;
#[cfg(not(feature = "u16_lookup"))]
const ENCODING_MASK: u16 = 0b11111;

/// Where in the encoded bit pattern to store total excess
#[cfg(feature = "u16_lookup")]
const TOTAL_EXCESS_POSITION: usize = 12;
#[cfg(not(feature = "u16_lookup"))]
const TOTAL_EXCESS_POSITION: usize = 10;

/// Where in the encoded bit pattern to store minimum excess
#[cfg(feature = "u16_lookup")]
const MINIMUM_EXCESS_POSITION: usize = 6;
#[cfg(not(feature = "u16_lookup"))]
const MINIMUM_EXCESS_POSITION: usize = 5;

const fn calculate_lookup_table() -> [EncodedTableType; 1 << LOOKUP_BLOCK_SIZE] {
    // initial sentinel values during excess computation
    const MORE_THAN_MAX: SignedLookupBlockType = (LOOKUP_BLOCK_SIZE + 1) as SignedLookupBlockType;
    const LESS_THAN_MIN: SignedLookupBlockType = -(LOOKUP_BLOCK_SIZE as SignedLookupBlockType) - 1;

    let mut lookup = [0; 1 << LOOKUP_BLOCK_SIZE];
    let mut v: u32 = 0;
    while v <= LOOKUP_MAX_VALUE {
        let mut minimum_excess = MORE_THAN_MAX;
        let mut maximum_excess = LESS_THAN_MIN;
        let mut total_excess = 0;

        let mut i = 0;
        while i < LOOKUP_BLOCK_SIZE {
            if ((v >> i) & 1) == 1 {
                total_excess += 1;
            } else {
                total_excess -= 1;
            }

            minimum_excess = min(minimum_excess, total_excess);
            maximum_excess = max(maximum_excess, total_excess);
            i += 1;
        }

        let mut encoded: EncodedTableType = ((total_excess as i32 + ENCODING_OFFSET) as EncodedTableType & ENCODING_MASK) << TOTAL_EXCESS_POSITION;
        encoded |= ((minimum_excess as i32 + ENCODING_OFFSET) as EncodedTableType & ENCODING_MASK) << MINIMUM_EXCESS_POSITION;
        encoded |= (maximum_excess as i32 + ENCODING_OFFSET) as EncodedTableType & ENCODING_MASK;
        lookup[v as usize] = encoded;

        v += 1;
    }

    lookup
}

/// Obtain the total excess from an encoded 16 bit value from the lookup table
const fn get_total_excess(value: EncodedTableType) -> i64 {
    (value >> TOTAL_EXCESS_POSITION) as i64 - ENCODING_OFFSET as i64
}

/// Obtain the minimum excess from an encoded 16 bit value from the lookup table
const fn get_minimum_excess(value: EncodedTableType) -> i64 {
    ((value >> MINIMUM_EXCESS_POSITION) & ENCODING_MASK) as i64 - ENCODING_OFFSET as i64
}

/// Obtain the minimum excess from an encoded 16 bit value from the lookup table
const fn get_maximum_excess(value: EncodedTableType) -> i64 {
    (value & ENCODING_MASK) as i64 - ENCODING_OFFSET as i64
}

/// Branchless const minimum computation for values that cannot overflow
const fn min(a: SignedLookupBlockType, b: SignedLookupBlockType) -> SignedLookupBlockType {
    b + ((a - b) & -(((a - b) as LookupBlockType >> (LOOKUP_BLOCK_SIZE - 1)) as SignedLookupBlockType))
}

/// Branchless const maximum computation for values that cannot overflow
const fn max(a: SignedLookupBlockType, b: SignedLookupBlockType) -> SignedLookupBlockType {
    a - ((a - b) & -(((a - b) as LookupBlockType >> (LOOKUP_BLOCK_SIZE - 1)) as SignedLookupBlockType))
}

/// Get the total excess of a block of eight parenthesis
#[inline(always)]
fn lookup_total_excess(block: LookupBlockType) -> i64 {
    get_total_excess(PAREN_BLOCK_LOOKUP[block as usize])
}

/// Get the maximum excess of a block of eight parenthesis
#[inline(always)]
fn lookup_maximum_excess(block: LookupBlockType) -> i64 {
    get_maximum_excess(PAREN_BLOCK_LOOKUP[block as usize])
}

/// Get the minimum excess of a block of eight parenthesis
#[inline(always)]
fn lookup_minimum_excess(block: LookupBlockType) -> i64 {
    get_minimum_excess(PAREN_BLOCK_LOOKUP[block as usize])
}

#[inline(always)]
pub(crate) fn process_block_fwd(block: LookupBlockType, relative_excess: &mut i64) -> Result<u64, ()> {
    if *relative_excess <= lookup_maximum_excess(block) && lookup_minimum_excess(block) <= *relative_excess {
        for i in 0..LOOKUP_BLOCK_SIZE {
            let bit = (block >> i) & 0x1;
            *relative_excess -= if bit == 1 { 1 } else { -1 };

            if *relative_excess == 0 {
                return Ok(i);
            }
        }

        unreachable!()
    } else {
        *relative_excess -= lookup_total_excess(block);
        Err(())
    }
}

#[inline(always)]
pub(crate) fn process_block_bwd(block: LookupBlockType, relative_excess: &mut i64) -> Result<u64, ()> {
    let total_excess = lookup_total_excess(block);
    if (*relative_excess + total_excess == 0) || (lookup_minimum_excess(block)
        <= *relative_excess + total_excess
        && *relative_excess + total_excess <= lookup_maximum_excess(block)) {
        for i in (0..LOOKUP_BLOCK_SIZE).rev() {
            let bit = (block >> i) & 0x1;
            *relative_excess -= if bit == 1 { -1 } else { 1 };

            if *relative_excess == 0 {
                return Ok(i);
            }
        }

        unreachable!()
    } else {
        *relative_excess += total_excess;
        Err(())
    }
}