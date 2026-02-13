//! Parallel bits deposit intrinsics for all platforms.
//! Uses the `PDEP` instruction on `x86`/`x86_64` platforms with the `bmi2` feature enabled.
//! Uses NEON-accelerated ARM64 implementations on `aarch64` platforms.

// bit manipulation generally doesn't care about sign, so the caller is aware of the consequences
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

/// Parallel bits deposit
pub trait Pdep {
    /// Parallel bits deposit.
    ///
    /// Scatter contiguous low order bits of `x` to the result at the positions
    /// specified by the `mask`.
    ///
    /// All other bits (bits not set in the `mask`) of the result are set to
    /// zero.
    fn pdep(self, mask: Self) -> Self;
}

/// Parallel bits extract (complementary to PDEP)
pub trait Pext {
    /// Parallel bits extract.
    ///
    /// Extract bits from `self` at the positions specified by `mask` to 
    /// contiguous low order bits of the result.
    fn pext(self, mask: Self) -> Self;
}

// x86_64 implementations
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
use std::arch::x86_64::{_pdep_u64, _pext_u64};

// ARM64: Import sophisticated NEON implementations
#[cfg(target_arch = "aarch64")]
use crate::arch::aarch64::{Arm64BitOps, BitOps};

// Generic implementations for other architectures
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod generic_impl {
    #[inline(always)]
    pub fn pext_u64_generic(value: u64, mask: u64) -> u64 {
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
    pub fn pdep_u64_generic(value: u64, mut mask: u64) -> u64 {
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

// Implement for u64 with direct dispatch (no trait overhead)
impl Pdep for u64 {
    #[inline(always)]
    fn pdep(self, mask: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        unsafe {
            _pdep_u64(self, mask)
        }
        
        #[cfg(all(target_arch = "aarch64", not(all(target_arch = "x86_64", target_feature = "bmi2"))))]
        {
            // Use sophisticated NEON-accelerated implementations from arch/aarch64
            Arm64BitOps::pdep_u64(self, mask)
        }
        
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "bmi2"),
            target_arch = "aarch64"
        )))]
        {
            generic_impl::pdep_u64_generic(self, mask)
        }
    }
}

impl Pext for u64 {
    #[inline(always)]
    fn pext(self, mask: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        unsafe {
            _pext_u64(self, mask)
        }
        
        #[cfg(all(target_arch = "aarch64", not(all(target_arch = "x86_64", target_feature = "bmi2"))))]
        {
            // Use sophisticated NEON-accelerated implementations from arch/aarch64
            Arm64BitOps::pext_u64(self, mask)
        }
        
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "bmi2"),
            target_arch = "aarch64"
        )))]
        {
            generic_impl::pext_u64_generic(self, mask)
        }
    }
}

// Implement for other integer types by casting
macro_rules! impl_pdep_pext {
    ($($t:ty),*) => {
        $(
            impl Pdep for $t {
                #[inline(always)]
                fn pdep(self, mask: Self) -> Self {
                    (self as u64).pdep(mask as u64) as Self
                }
            }
            
            impl Pext for $t {
                #[inline(always)]
                fn pext(self, mask: Self) -> Self {
                    (self as u64).pext(mask as u64) as Self
                }
            }
        )*
    };
}

impl_pdep_pext!(u8, u16, u32, i8, i16, i32, i64);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pdep_basic() {
        let value: u64 = 0b1111;
        let mask: u64 = 0b10101010;
        assert_eq!(value.pdep(mask), 0b10101010);
    }
    
    #[test]
    fn test_pext_basic() {
        let value: u64 = 0b11111111;
        let mask: u64 = 0b10101010;
        assert_eq!(value.pext(mask), 0b1111);
    }
    
    #[test]
    fn test_pdep_pext_inverse() {
        let original: u64 = 0b1010;
        let mask: u64 = 0b11110000;
        let deposited = original.pdep(mask);
        let extracted = deposited.pext(mask);
        assert_eq!(original, extracted);
    }
}
