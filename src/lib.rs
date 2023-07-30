#![warn(missing_docs)]

//! This crate provides a collection of data structures supported by fast implementations of
//! rank and select queries. The data structures are static, meaning that they cannot be modified
//! after they have been created.
//!
//! # Data structures
//!  - [BitVector][bit_vec::BitVec] with no overhead.
//!  - [Succinct Bit-Vector][bit_vec::fast_rs_vec::RsVec] supporting fast rank and select queries.
//!  - [Elias-Fano][elias_fano::EliasFanoVec] encoding of monotone sequences supporting constant time predecessor queries.
//!  - Two [Range Minimum Query][rmq] structures for constant time range minimum queries.
//!
//! # Performance
//! Performance was benchmarked against publicly available implementations of the same (or similar)
//! data structures on crates.io at the time of writing. The benchmark results can be found
//! in the [Github repository](https://github.com/Cydhra/vers). At the time of writing,
//! this crate is the fastest implementation of all data structures except for one implementation
//! of rank on bit-vectors (which pays for its speed with a missing select implementation).

pub use crate::elias_fano::EliasFanoVec;
pub use bit_vec::fast_rs_vec::{RsVec, RsVectorBuilder};
pub use bit_vec::BitVec;
pub use rmq::fast_rmq::FastRmq;

pub mod bit_vec;
pub mod elias_fano;
pub mod rmq;

pub(crate) mod util;
