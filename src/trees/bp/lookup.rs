//! This module provides the lookup table and lookup functionality to obtain excess values for
//! checking blocks of bits in the tree's bitvector, instead of checking each bit individually.

/// How big the lookup blocks are. We store this in a constant so we can switch out this module
/// using a crate feature against one where this constant is redefined to 16, but reuse the actual
/// scanning code for operations on the tree.
pub(crate) const LOOKUP_BLOCK_SIZE: u64 = 8;

/// The lookup entry is indexed by the numerical value of an 8-bit parenthesis expression. The table
/// contains the minimum, maximum, and total excess encoded in 16 bit.
///
/// The encoding scheme is simple:
/// The least significant 5 bits encode maximum excess (which is between -8 and 8, which we store
/// with an offset of 8, so we don't have to deal with dual encoding), the next 5 bits are the
/// minimum excess encoded analogously, and the following 5 bits are the total excess.
///
/// The rest of the bits are zero.
const PAREN_BLOCK_LOOKUP: [u16; 256] = calculate_lookup_table();

/// Offset encoded numbers so negative numbers are stored as positive integers, reducing
/// encoding complexity
const ENCODING_OFFSET: i16 = 8;

const ENCODING_MASK: u16 = 0b11111;

/// Where in the encoded bit pattern to store total excess
const TOTAL_EXCESS_POSITION: usize = 10;

/// Where in the encoded bit pattern to store minimum excess
const MINIMUM_EXCESS_POSITION: usize = 5;

const fn calculate_lookup_table() -> [u16; 256] {
    // initial sentinel values during excess computation
    const MORE_THAN_MAX: i8 = 9;
    const LESS_THAN_MIN: i8 = -9;

    let mut lookup = [0; 256];
    let mut v: u16 = 0;
    while v <= u8::MAX as u16 {
        let mut minimum_excess = MORE_THAN_MAX;
        let mut maximum_excess = LESS_THAN_MIN;
        let mut total_excess = 0;

        let mut i = 0;
        while i < 8 {
            if ((v >> i) & 1) == 1 {
                total_excess += 1;
            } else {
                total_excess -= 1;
            }

            minimum_excess = min(minimum_excess, total_excess);
            maximum_excess = max(maximum_excess, total_excess);
            i += 1;
        }

        let mut encoded: u16 = ((total_excess as i16 + ENCODING_OFFSET) as u16 & ENCODING_MASK) << TOTAL_EXCESS_POSITION;
        encoded |= ((minimum_excess as i16 + ENCODING_OFFSET) as u16 & ENCODING_MASK) << MINIMUM_EXCESS_POSITION;
        encoded |= (maximum_excess as i16 + ENCODING_OFFSET) as u16 & ENCODING_MASK;
        lookup[v as usize] = encoded;

        v += 1;
    }

    lookup
}

/// Obtain the total excess from an encoded 16 bit value from the lookup table
const fn get_total_excess(value: u16) -> i64 {
    (value >> TOTAL_EXCESS_POSITION) as i64 - ENCODING_OFFSET as i64
}

/// Obtain the minimum excess from an encoded 16 bit value from the lookup table
const fn get_minimum_excess(value: u16) -> i64 {
    ((value >> MINIMUM_EXCESS_POSITION) & ENCODING_MASK) as i64 - ENCODING_OFFSET as i64
}

/// Obtain the minimum excess from an encoded 16 bit value from the lookup table
const fn get_maximum_excess(value: u16) -> i64 {
    (value & ENCODING_MASK) as i64 - ENCODING_OFFSET as i64
}

/// Branchless const minimum computation for values that cannot overflow
const fn min(a: i8, b: i8) -> i8 {
    b + ((a - b) & -(((a - b) as u8 >> 7) as i8))
}

/// Branchless const maximum computation for values that cannot overflow
const fn max(a: i8, b: i8) -> i8 {
    a - ((a - b) & -(((a - b) as u8 >> 7) as i8))
}

/// Get the total excess of a block of eight parenthesis
#[inline(always)]
pub(crate) fn lookup_total_excess(block: u8) -> i64 {
    get_total_excess(PAREN_BLOCK_LOOKUP[block as usize])
}

/// Get the maximum excess of a block of eight parenthesis
#[inline(always)]
pub(crate) fn lookup_maximum_excess(block: u8) -> i64 {
    get_maximum_excess(PAREN_BLOCK_LOOKUP[block as usize])
}

/// Get the minimum excess of a block of eight parenthesis
#[inline(always)]
pub(crate) fn lookup_minimum_excess(block: u8) -> i64 {
    get_minimum_excess(PAREN_BLOCK_LOOKUP[block as usize])
}

#[inline(always)]
pub(crate) fn process_block_fwd(block: u8, relative_excess: i64) -> Result<u64, i64> {
    if relative_excess <= lookup_maximum_excess(block) && lookup_minimum_excess(block) <= relative_excess {
        let mut current_relative_excess = 0;
        for i in 0..LOOKUP_BLOCK_SIZE {
            let bit = (block >> i) & 0x1;
            current_relative_excess += if bit == 1 { 1 } else { -1 };

            if current_relative_excess == relative_excess {
                return Ok(i);
            }
        }

        unreachable!()
    } else {
        Err(lookup_total_excess(block))
    }
}