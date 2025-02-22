//! This module provides a lookup table for 8-bit blocks of parenthesis, answering
//! excess queries. The table contains the minimum and maximum excess of every possible
//! block, and the answer to every possible relative excess query (-8 to 8).
//! This module only works for 8 bit blocks, since 16 bit blocks are too large to
//! efficiently store all 33 excess queries for every possible block.

/// How big the lookup blocks are.
pub(crate) const LOOKUP_BLOCK_SIZE: u64 = 8;

/// Integer type holding the blocks of the parenthesis expression we look up at once
type LookupBlockType = u8;

/// Signed version of `LookupBlockType`
type SignedLookupBlockType = i8;

/// Maximum value that `LookupBlockType` can hold, stored in one size larger because we need to
/// iterate up to and including it
const LOOKUP_MAX_VALUE: u32 = u8::MAX as u32;

/// Encoded fwd query results for all possible 8-bit blocks.
/// The encoding reserves 10 bits for minimum and maximum excess (shifted by 8 bits so we don't have
/// to dual-encode negative excess), and another 51 bits for all 17 queries that may end in this block
/// (-8 to 8 relative excess).
#[allow(long_running_const_eval)]
const PAREN_BLOCK_LOOKUP_FWD: [u64; 1 << LOOKUP_BLOCK_SIZE] = calculate_lookup_table(true);

/// Encoded bwd query results for all possible 8-bit blocks.
/// The encoding reserves 10 bits for minimum and maximum excess (shifted by 8 bits so we don't have
/// to dual-encode negative excess), and another 51 bits for all 17 queries that may end in this block
/// (-8 to 8 relative excess).
#[allow(long_running_const_eval)]
const PAREN_BLOCK_LOOKUP_BWD: [u64; 1 << LOOKUP_BLOCK_SIZE] = calculate_lookup_table(false);

/// Bitmask for one of the lookup values.
const ENCODING_MASK: u64 = 0b11111;

/// Where in the encoded bit pattern to store minimum excess
const MINIMUM_EXCESS_POSITION: usize = 5;

/// Where the encoded queries are stored in the encoded bit pattern
const QUERY_BASE_POSITION: usize = 10;

#[allow(clippy::cast_possible_truncation)] // we know that the values are within bounds
#[allow(clippy::cast_sign_loss)] // we know that the values are within bounds
const fn calculate_lookup_table(fwd: bool) -> [u64; 1 << LOOKUP_BLOCK_SIZE] {
    // initial sentinel values during excess computation
    const MORE_THAN_MAX: SignedLookupBlockType = (LOOKUP_BLOCK_SIZE + 1) as SignedLookupBlockType;
    const LESS_THAN_MIN: SignedLookupBlockType = -(LOOKUP_BLOCK_SIZE as SignedLookupBlockType) - 1;

    let mut lookup = [0; 1 << LOOKUP_BLOCK_SIZE];
    let mut query_map = [-1i8; 17];
    let mut v: u32 = 0;
    while v <= LOOKUP_MAX_VALUE {
        let mut minimum_excess = MORE_THAN_MAX;
        let mut maximum_excess = LESS_THAN_MIN;

        if fwd {
            calculate_values_fwd(v, &mut minimum_excess, &mut maximum_excess, &mut query_map);
        } else {
            calculate_values_bwd(v, &mut minimum_excess, &mut maximum_excess, &mut query_map);
        }

        let mut encoded: u64 = ((minimum_excess as i32 + LOOKUP_BLOCK_SIZE as i32) as u64
            & ENCODING_MASK)
            << MINIMUM_EXCESS_POSITION;
        encoded |= (maximum_excess as i32 + LOOKUP_BLOCK_SIZE as i32) as u64 & ENCODING_MASK;

        let mut relative_off = 0;
        while relative_off <= (LOOKUP_BLOCK_SIZE * 2) as usize {
            encoded |= ((query_map[relative_off] & 0b111) as u64)
                << (QUERY_BASE_POSITION + (relative_off * 3)) as u64;
            // reset query map to -1, so next block knows which queries are already answered
            query_map[relative_off] = -1;
            relative_off += 1;
        }

        lookup[v as usize] = encoded;
        v += 1;
    }

    lookup
}

#[allow(clippy::cast_possible_truncation)] // we know that the values are within bounds
#[allow(clippy::cast_sign_loss)] // we know that the values are within bounds
const fn calculate_values_fwd(
    v: u32,
    minimum_excess: &mut SignedLookupBlockType,
    maximum_excess: &mut SignedLookupBlockType,
    query_map: &mut [i8; 17],
) {
    let mut total_excess = 0;
    let mut i = 0;
    while i < LOOKUP_BLOCK_SIZE {
        if ((v >> i) & 1) == 1 {
            total_excess += 1;
        } else {
            total_excess -= 1;
        }

        *minimum_excess = min(*minimum_excess, total_excess);
        *maximum_excess = max(*maximum_excess, total_excess);

        if query_map[(total_excess + LOOKUP_BLOCK_SIZE as i8) as usize] == -1 {
            query_map[(total_excess + LOOKUP_BLOCK_SIZE as i8) as usize] = i as i8;
        }
        i += 1;
    }
}

#[allow(clippy::cast_possible_truncation)] // we know that the values are within bounds
#[allow(clippy::cast_sign_loss)] // we know that the values are within bounds
#[allow(clippy::cast_possible_wrap)] // we know that the values are within bounds
const fn calculate_values_bwd(
    v: u32,
    minimum_excess: &mut SignedLookupBlockType,
    maximum_excess: &mut SignedLookupBlockType,
    query_map: &mut [i8; 17],
) {
    let mut total_excess = 0;
    let mut i = LOOKUP_BLOCK_SIZE as i64 - 1;
    while i >= 0 {
        if ((v >> i) & 1) == 1 {
            total_excess -= 1;
        } else {
            total_excess += 1;
        }

        *minimum_excess = min(*minimum_excess, total_excess);
        *maximum_excess = max(*maximum_excess, total_excess);

        if query_map[(total_excess + LOOKUP_BLOCK_SIZE as i8) as usize] == -1 {
            query_map[(total_excess + LOOKUP_BLOCK_SIZE as i8) as usize] = i as i8;
        }
        i -= 1;
    }
}

#[allow(clippy::cast_possible_truncation)] // we know that the table values are within bounds
#[allow(clippy::cast_sign_loss)] // we know that the table values are within bounds
#[allow(clippy::cast_possible_wrap)] // we know that the table values are within bounds
const fn answer_query(value: u64, relative_excess: i64) -> u64 {
    debug_assert!(relative_excess.abs() <= LOOKUP_BLOCK_SIZE as i64);
    (value >> (QUERY_BASE_POSITION + ((relative_excess + 8) as usize * 3))) & 0b111
}

/// Obtain the minimum excess from an encoded 16 bit value from the lookup table
#[allow(clippy::cast_possible_truncation)] // we know that the table values are within bounds
#[allow(clippy::cast_sign_loss)] // we know that the table values are within bounds
#[allow(clippy::cast_possible_wrap)] // we know that the table values are within bounds
const fn get_minimum_excess(value: u64) -> i64 {
    ((value >> MINIMUM_EXCESS_POSITION) & ENCODING_MASK) as i64 - LOOKUP_BLOCK_SIZE as i64
}

/// Obtain the minimum excess from an encoded 16 bit value from the lookup table
#[allow(clippy::cast_possible_truncation)] // we know that the table values are within bounds
#[allow(clippy::cast_sign_loss)] // we know that the table values are within bounds
#[allow(clippy::cast_possible_wrap)] // we know that the table values are within bounds
const fn get_maximum_excess(value: u64) -> i64 {
    (value & ENCODING_MASK) as i64 - LOOKUP_BLOCK_SIZE as i64
}

/// Branchless const minimum computation for values that cannot overflow
#[allow(clippy::cast_possible_truncation)] // we only call this with values that are within bounds
#[allow(clippy::cast_sign_loss)] // we only call this with values that are within bounds
#[allow(clippy::cast_possible_wrap)] // we only call this with values that are within bounds
const fn min(a: SignedLookupBlockType, b: SignedLookupBlockType) -> SignedLookupBlockType {
    b + ((a - b)
        & -(((a - b) as LookupBlockType >> (LOOKUP_BLOCK_SIZE - 1)) as SignedLookupBlockType))
}

/// Branchless const maximum computation for values that cannot overflow
#[allow(clippy::cast_possible_truncation)] // we only call this with values that are within bounds
#[allow(clippy::cast_sign_loss)] // we only call this with values that are within bounds
#[allow(clippy::cast_possible_wrap)] // we only call this with values that are within bounds
const fn max(a: SignedLookupBlockType, b: SignedLookupBlockType) -> SignedLookupBlockType {
    a - ((a - b)
        & -(((a - b) as LookupBlockType >> (LOOKUP_BLOCK_SIZE - 1)) as SignedLookupBlockType))
}

/// Get the total excess of a block of eight parenthesis
#[inline(always)]
fn lookup_total_excess(block: LookupBlockType) -> i64 {
    i64::from(block.count_ones()) - i64::from(block.count_zeros())
}

/// Get the maximum excess of a block of eight parenthesis
#[inline(always)]
fn lookup_maximum_excess(block: LookupBlockType) -> i64 {
    get_maximum_excess(PAREN_BLOCK_LOOKUP_FWD[block as usize])
}

/// Get the minimum excess of a block of eight parenthesis
#[inline(always)]
fn lookup_minimum_excess(block: LookupBlockType) -> i64 {
    get_minimum_excess(PAREN_BLOCK_LOOKUP_FWD[block as usize])
}

#[inline(always)]
pub(crate) fn process_block_fwd(
    block: LookupBlockType,
    relative_excess: &mut i64,
) -> Result<u64, ()> {
    if *relative_excess <= lookup_maximum_excess(block)
        && lookup_minimum_excess(block) <= *relative_excess
    {
        Ok(answer_query(
            PAREN_BLOCK_LOOKUP_FWD[block as usize],
            *relative_excess,
        ))
    } else {
        *relative_excess -= lookup_total_excess(block);
        Err(())
    }
}

#[inline(always)]
pub(crate) fn process_block_bwd(
    block: LookupBlockType,
    relative_excess: &mut i64,
) -> Result<u64, ()> {
    let total_excess = lookup_total_excess(block);
    if (*relative_excess + total_excess == 0)
        || (lookup_minimum_excess(block) <= *relative_excess + total_excess
            && *relative_excess + total_excess <= lookup_maximum_excess(block))
    {
        Ok(answer_query(
            PAREN_BLOCK_LOOKUP_BWD[block as usize],
            *relative_excess,
        ))
    } else {
        *relative_excess += total_excess;
        Err(())
    }
}
