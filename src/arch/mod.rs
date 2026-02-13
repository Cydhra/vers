//! Architecture-specific optimizations
//!
//! This module contains platform-specific implementations of performance-critical
//! operations. Currently supports x86_64 (with BMI2) and ARM64 (aarch64).

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

/// Generic fallback implementations for unsupported architectures
pub mod generic;

/// Trait for architecture-specific bit manipulation operations
#[allow(dead_code)]
pub trait BitOps {
    /// Count the number of set bits (population count)
    fn popcount_u64(value: u64) -> u32;

    /// Count leading zeros
    fn leading_zeros_u64(value: u64) -> u32;

    /// Count trailing zeros
    fn trailing_zeros_u64(value: u64) -> u32;

    /// Parallel bit extract (PEXT equivalent)
    fn pext_u64(value: u64, mask: u64) -> u64;

    /// Parallel bit deposit (PDEP equivalent)
    fn pdep_u64(value: u64, mask: u64) -> u64;
}

/// Select the appropriate implementation based on the target architecture
#[cfg(target_arch = "x86_64")]
/// Architecture-specific bit operations implementation.
pub type ArchBitOps = x86_64::X86BitOps;

#[cfg(target_arch = "aarch64")]
/// Architecture-specific bit operations implementation.
pub type ArchBitOps = aarch64::Arm64BitOps;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
/// Architecture-specific bit operations implementation.
pub type ArchBitOps = generic::GenericBitOps;

/// Re-export architecture-specific implementations
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86_64::*;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use aarch64::*;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use generic::*;
