//! Range minimum query data structures. These data structures allow to calculate the index of the
//! minimum element in a range of a static array in constant time. The implementations are located
//! in the [`binary_rmq`] and [`fast_rmq`] modules.

pub mod fast_rmq;

pub mod binary_rmq;
