#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::assertions_on_constants)] // for asserts warning about incompatible constant values

//! This crate provides a collection of data structures supported by fast implementations of
//! rank and select queries. The data structures are static, meaning that they cannot be modified
//! after they have been created.
//!
//! # Data structures
//!  - [Bit-Vector][bit_vec::BitVec] with no overhead. The only data structure that can be modified after creation.
//!  - [Succinct Bit-Vector][bit_vec::fast_rs_vec::RsVec] supporting fast rank and select queries.
//!  - [Elias-Fano][elias_fano::EliasFanoVec] encoding of monotone sequences supporting constant time predecessor queries.
//!  - Two [Range Minimum Query][rmq] structures for constant time range minimum queries.
//!
//! # Performance
//! Performance was benchmarked against publicly available implementations of the same (or similar)
//! data structures on crates.io at the time of writing. The benchmark results can be found
//! in the [Github repository](https://github.com/Cydhra/vers). At the time of writing,
//! this crate is among the fastest implementations of all data structures, and the most efficient
//! with regard to memory usage.
//! Some tradeoffs between average time, worst-case time, and available API features should be taken
//! into consideration when selecting among the fastest libraries
//! (see the Github repository for a discussion).
//!
//! # Intrinsics
//! This crate uses compiler intrinsics for bit-manipulation. The intrinsics are supported by
//! all modern ``x86_64`` CPUs, but not by other architectures. The crate will compile on other
//! architectures, but the performance will be significantly worse. It is strongly recommended to
//! enable the ``BMI2`` and ``popcnt`` target features when using this crate.
//!
//! The intrinsics in question are `popcnt` (supported since ``SSE4.2`` resp. ``SSE4a`` on AMD, 2007-2008),
//! `pdep` (supported with ``BMI2`` since Intel Haswell resp. AMD Excavator, in hardware since AMD Zen 3, 2011-2013),
//! and `tzcnt` (supported with ``BMI1`` since Intel Haswell resp. AMD Jaguar, ca. 2013).
//!
//! # Safety
//! This crate uses no unsafe code, with the only exception being compiler intrinsics for
//! bit-manipulation. The intrinsics cannot fail with the provided inputs (provided they are
//! supported by the target machine), so even if they were to be implemented incorrectly, no
//! memory unsafety can occur (only incorrect results).

pub use crate::elias_fano::EliasFanoVec;
pub use bit_vec::fast_rs_vec::RsVec;
pub use bit_vec::BitVec;
pub use rmq::binary_rmq::BinaryRmq;
pub use rmq::fast_rmq::FastRmq;

pub mod bit_vec;
pub mod elias_fano;
pub mod rmq;

pub(crate) mod util;
