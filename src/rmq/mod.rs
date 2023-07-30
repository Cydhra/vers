//! Range minimum query data structures. These data structures allow to calculate the index of the
//! minimum element in a range of a static array in constant time. The implementations are located
//! in the [binary_rmq][binary_rmq] and [fast_rmq][fast_rmq] modules.

pub mod fast_rmq;

pub mod binary_rmq;

/// A common trait for range minimum query data structures to allow generic implementations.
trait RangeMinimumVec {
    fn new(data: &[u64]) -> Self;

    /// Returns the index of the minimum element in the range [i, j].
    fn range_min(&self, i: usize, j: usize) -> usize;

    /// Returns the number of bytes used by the data structure.
    fn heap_size(&self) -> usize;

    /// Returns the number of elements in the vector.
    fn len(&self) -> usize;
}
