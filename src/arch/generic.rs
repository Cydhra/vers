//! Generic fallback implementations for unsupported architectures

use super::BitOps;

/// Generic bit manipulation operations
#[allow(dead_code)]
pub struct GenericBitOps;

impl BitOps for GenericBitOps {
    #[inline(always)]
    fn popcount_u64(value: u64) -> u32 {
        value.count_ones()
    }

    #[inline(always)]
    fn leading_zeros_u64(value: u64) -> u32 {
        value.leading_zeros()
    }

    #[inline(always)]
    fn trailing_zeros_u64(value: u64) -> u32 {
        value.trailing_zeros()
    }

    #[inline(always)]
    fn pext_u64(value: u64, mask: u64) -> u64 {
        let mut result = 0u64;
        let mut bb = 1u64;
        let mut m = mask;
        
        while m != 0 {
            if value & bb != 0 {
                result |= m & m.wrapping_neg();
            }
            m &= m - 1;
            bb <<= 1;
        }
        
        result
    }

    #[inline(always)]
    fn pdep_u64(value: u64, mut mask: u64) -> u64 {
        let mut res = 0;
        let mut bb: u64 = 1;
        loop {
            if mask == 0 {
                break;
            }
            if (value & bb) != 0 {
                res |= mask & mask.wrapping_neg();
            }
            mask &= mask - 1;
            bb = bb.wrapping_add(bb);
        }
        res
    }
}
