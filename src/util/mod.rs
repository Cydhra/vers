pub(crate) mod elias_fano_iter;
pub(crate) mod general_iter;
pub(crate) mod pdep;
pub(crate) mod unroll;

// reexport all macros at toplevel for convenience
pub(crate) use elias_fano_iter::gen_ef_iter_impl;
pub(crate) use elias_fano_iter::impl_ef_iterator;
pub(crate) use general_iter::gen_bv_iter_impl;
pub(crate) use general_iter::impl_bv_iterator;
pub(crate) use general_iter::impl_into_iterator_impls;
pub(crate) use unroll::unroll;
