pub mod fast_rmq;
pub mod binary_rmq;

trait RangeMinimumVec {
    fn new(data: &[u64]) -> Self;

    /// Returns the index of the minimum element in the range [i, j].
    fn range_min(&self, i: usize, j: usize) -> usize;

    /// Returns the number of bytes used by the data structure.
    fn heap_size(&self) -> usize;

    /// Returns the number of elements in the vector.
    fn len(&self) -> usize;
}
