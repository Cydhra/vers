//! Unroll a loop a fixed number of times. This is a macro that performs manual loop unrolling,
//! because LLVM is sometimes too conservative to do it itself.
//! We only use it in hyper-optimized code paths like ``rank``.

macro_rules! unroll {
    (1, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { let mut $i: usize = $e; $s };
    (2, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(1, |$i = {$e}| $s, $inc); $inc; $s };
    (3, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(2, |$i = {$e}| $s, $inc); $inc; $s };
    (4, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(3, |$i = {$e}| $s, $inc); $inc; $s };
    (5, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(4, |$i = {$e}| $s, $inc); $inc; $s };
    (6, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(5, |$i = {$e}| $s, $inc); $inc; $s };
    (7, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(6, |$i = {$e}| $s, $inc); $inc; $s };
    (8, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(7, |$i = {$e}| $s, $inc); $inc; $s };
}

// export the macro to the crate
pub(crate) use unroll;
