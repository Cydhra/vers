//! Range minimum query data structures. These data structures allow for the calculation of the index of the
//! minimum element in a range of a static array in constant-time. The implementations are located
//! in the [`sparse`] and [`small`] modules.

pub mod small;

pub mod sparse;
