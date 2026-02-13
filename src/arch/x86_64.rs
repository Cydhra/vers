//! x86_64 optimized implementations
//!
//! This module contains the existing x86_64 optimizations using BMI2 and other
//! x86-specific instructions. It serves as the reference implementation.

use super::BitOps;

/// x86_64-optimized bit manipulation operations
pub struct X86BitOps;

impl BitOps for X86BitOps {
    #[inline(always)]
    fn popcount_u64(value: u64) -> u32 {
        // Use native popcnt instruction when available
        value.count_ones()
    }

    #[inline(always)]
    fn leading_zeros_u64(value: u64) -> u32 {
        // Use lzcnt instruction when available
        value.leading_zeros()
    }

    #[inline(always)]
    fn trailing_zeros_u64(value: u64) -> u32 {
        // Use tzcnt instruction when available
        value.trailing_zeros()
    }

    #[inline(always)]
    fn pext_u64(value: u64, mask: u64) -> u64 {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        unsafe {
            std::arch::x86_64::_pext_u64(value, mask)
        }
        
        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        {
            // Fallback to generic implementation
            super::generic::GenericBitOps::pext_u64(value, mask)
        }
    }

    #[inline(always)]
    fn pdep_u64(value: u64, mask: u64) -> u64 {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        unsafe {
            std::arch::x86_64::_pdep_u64(value, mask)
        }
        
        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        {
            // Fallback to generic implementation
            super::generic::GenericBitOps::pdep_u64(value, mask)
        }
    }
}
