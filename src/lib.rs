#![cfg_attr(
    all(
        feature = "simd",
        target_arch = "x86_64",
        target_feature = "avx",
        target_feature = "avx2",
        target_feature = "avx512f",
        target_feature = "avx512bw",
    ),
    feature(stdarch_x86_avx512)
)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::assertions_on_constants)] // for asserts warning about incompatible constant values
#![allow(clippy::inline_always)] // we actually measure performance increases with most of these
#![cfg_attr(docsrs, feature(doc_cfg), feature(doc_auto_cfg))] // for conditional compilation in docs

//! This crate provides a collection of data structures supported by fast implementations of
//! rank and select queries. The data structures are static, meaning that they cannot be modified
//! after they have been created.
//!
//! # Data structures
//!  - [Bit-Vector][bit_vec::BitVec] with no overhead. The only data structure that can be modified after creation.
//!  - [Succinct Bit-Vector][bit_vec::fast_rs_vec::RsVec] supporting fast rank and select queries.
//!  - [Elias-Fano][elias_fano::EliasFanoVec] encoding of monotone sequences supporting constant-time predecessor queries.
//!  - Two [Range Minimum Query][rmq] structures for constant-time range minimum queries.
//!  - [Wavelet Matrix][wavelet::WaveletMatrix] encoding `k`-bit symbols, supporting rank, select, statistical, and predecessor/successor queries in `O(k)`.
//!
//! # Performance
//! Performance was benchmarked against publicly available implementations of the same (or similar)
//! data structures on crates.io.
//! Vers is among the fastest for all benchmarked operations.
//! The benchmark results can be found
//! in the [Benchmark repository](https://github.com/Cydhra/vers_benchmarks).
//! Some tradeoffs between average time, worst-case time, and available API features should be taken
//! into consideration when selecting among the fastest libraries
//! (see the GitHub repository for a discussion).
//!
//! # Intrinsics
//! This crate uses compiler intrinsics for bit-manipulation. The intrinsics are supported by
//! all modern ``x86_64`` CPUs, but not by other architectures. The crate will compile on other
//! architectures using fallback implementations,
//! but the performance will be significantly worse. It is strongly recommended to
//! enable the ``BMI2`` and ``popcnt`` target features when using this crate.
//!
//! The intrinsics in question are `popcnt` (supported since ``SSE4.2`` resp. ``SSE4a`` on AMD, 2007-2008),
//! `pdep` (supported with ``BMI2`` since Intel Haswell resp. AMD Excavator, in hardware since AMD Zen 3, 2011-2013),
//! and `tzcnt` (supported with ``BMI1`` since Intel Haswell resp. AMD Jaguar, ca. 2013).
//!
//! # Safety
//! When the `simd` crate feature is not enabled (default),
//! this crate uses no unsafe code, with the only exception being compiler intrinsics for
//! bit-manipulation. The intrinsics cannot fail with their inputs (provided they are
//! supported by the target machine), so even if they were to be implemented incorrectly,
//! no memory safety issues would arise.
//!
//! # Crate Features
//! - `simd` (disabled by default): Enables the use of SIMD instructions in the `RsVec`
//!   implementation, and an additional iterator for the `RsVec` data structure.
//! - `serde` (disabled by default): Enables serialization and deserialization support for all
//!   data structures in this crate using the `serde` crate.

pub use bit_vec::fast_rs_vec::RsVec;
pub use bit_vec::BitVec;
pub use elias_fano::EliasFanoVec;
pub use rmq::binary_rmq::BinaryRmq;
pub use rmq::fast_rmq::FastRmq;
pub use trees::bp::{BpBuilder, BpTree};
pub use trees::{IsAncestor, LevelTree, SubtreeSize, Tree, TreeBuilder};
pub use wavelet::WaveletMatrix;

pub mod bit_vec;

#[forbid(unsafe_code)]
pub mod elias_fano;

#[forbid(unsafe_code)]
pub mod rmq;

#[forbid(unsafe_code)]
pub mod trees;

#[forbid(unsafe_code)]
pub mod wavelet;

pub(crate) mod util;
