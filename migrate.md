# Migration Guide from 1.X to 2.0
The following guide explains the changes from versions 1.X to the 2.0 release and points out what changes are necessary
to downstream crates.

## Renamed Members
The following structures and functions were renamed
- `BitVec::from_bit_vector` to `BitVec::from_bit_vec`
- `SparseRSVec` to `SparseRsVec`
- `FastRmq` to `SmallRmq`
- `BinaryRmq` to `SparseRmq`
- `BitVec::from_bits` to `BitVec::from_bits_u8`
- module `fast_rs_vec` to `rs`
- module `elias_fano` to `ef`
- module `fast_rmq` to `small`
- module `binary_rmq` to `sparse`

## Changed Index Type
All vector types that operate on bits or sub-byte words are now indexed by `u64` instead of `usize`, 
allowing full utilization of the memory in 32-bit architectures.
This affects `BitVec`, `RsVec`, `EliasFano`, `SparseRsVec`, `BpTree`, and `WaveletMatrix`
This changes the parameter and return types of various functions on the affected types from `usize` to `u64`.
The only adverse effect is that `len()` and `count()` of iterators over these data structures may panic if the
iterator has more than `usize::MAX` elements.

## Changed Backing Structures
`RsVec`, `SparseRmq`, and `FastRmq` now use `Box<[_]>` instead of `Vec<_>` as backing structs, which reduces the stack 
footprint.
This breaks the serde-compatibility with already serialized data.
It also changes the `Deref` implementation of the RMQ structs, which previously returned `Vec<_>`.