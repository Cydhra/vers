# Migration Guide from 1.X to 2.0
The following guide explains the changes from versions 1.X to the 2.0 release and points out what changes are necessary
to downstream crates.

## Renamed Members
The following structures and functions were renamed
- `BitVec::from_bit_vector` to `BitVec::from_bit_vec`
- `SparseRSVec` to `SparseRsVec`
- `FastRmq` to `SmallRmq`