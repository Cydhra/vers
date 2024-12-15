//! Tree data structures. Currently only the [BP][bp] tree is exposed.
//! The trees are succinct, approaching the information-theoretic lower bound for the space complexity:
//! They need O(n) bits to store a tree with n nodes, and o(n) extra bits to support queries.
//!
//! For details, see the submodules.

pub mod bp;

pub(crate) mod mmt;
