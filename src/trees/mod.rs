mod bp;

/// Min-Max tree implementation as described by Cordova and Navarro in
/// [Simple and efficient fully-functional succinct trees](https://doi.org/10.1016/j.tcs.2016.04.031).
///
/// The Min-Max tree is a complete binary tree that stores the minimum and maximum relative
/// excess values of parenthesis expressions in its nodes. Since the tree is complete, it can be
/// stored linearly.
pub(crate) mod mmt;
