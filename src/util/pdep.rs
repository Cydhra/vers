//! Parallel bits deposit intrinsics for all platforms.
//! Uses the `PDEP` instruction on `x86`/`x86_64` platforms with the `bmi2` feature enabled.

// This file is part of the `bitintr` crate and is licensed under the terms of the MIT license.
// Since this crate is dual-licensed, you may choose to use this file under either the MIT license
// or the Apache License, Version 2.0, at your option (in compliance with the terms of the MIT license).
//
// Since the `bitintr` crate is abandoned, and the version on crates.io is outdated,
// the contents of this file are copied from the `bitintr` crate from the files
// `src/pdep.rs`, `src/macros.rs` and `src/lib.rs` at commit `6c49e01`.
// The code is functionally identical to the original code, with only minor edits to make it
// self-contained and update some documentation.
// None of the utils here are publicly exposed.

mod arch {
    #[cfg(all(target_arch = "x86", target_feature = "bmi2"))]
    pub use core::arch::x86::*;

    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    pub use core::arch::x86_64::*;
}

/// Parallel bits deposit
pub trait Pdep {
    /// Parallel bits deposit.
    ///
    /// Scatter contiguous low order bits of `x` to the result at the positions
    /// specified by the `mask`.
    ///
    /// All other bits (bits not set in the `mask`) of the result are set to
    /// zero.
    ///
    /// **Keywords**: Parallel bits deposit, scatter bits.
    ///
    /// # Instructions
    ///
    /// - [`PDEP`](http://www.felixcloutier.com/x86/PDEP.html):
    ///   - Description: Parallel bits deposit.
    ///   - Architecture: x86.
    ///   - Instruction set: BMI2.
    ///   - Registers: 32/64 bit.
    /// ```
    fn pdep(self, mask: Self) -> Self;
}

macro_rules! impl_all {
    ($impl_macro:ident: $($id:ident),*) => {
        $(
            $impl_macro!($id);
        )*
    }
}

macro_rules! cfg_if {
    // match if/else chains with a final `else`
    ($(
        if #[cfg($($meta:meta),*)] { $($it:item)* }
    ) else * else {
        $($it2:item)*
    }) => {
        cfg_if! {
            @__items
            () ;
            $( ( ($($meta),*) ($($it)*) ), )*
            ( () ($($it2)*) ),
        }
    };

    // match if/else chains lacking a final `else`
    (
        if #[cfg($($i_met:meta),*)] { $($i_it:item)* }
        $(
            else if #[cfg($($e_met:meta),*)] { $($e_it:item)* }
        )*
    ) => {
        cfg_if! {
            @__items
            () ;
            ( ($($i_met),*) ($($i_it)*) ),
            $( ( ($($e_met),*) ($($e_it)*) ), )*
            ( () () ),
        }
    };

    // Internal and recursive macro to emit all the items
    //
    // Collects all the negated cfgs in a list at the beginning and after the
    // semicolon is all the remaining items
    (@__items ($($not:meta,)*) ; ) => {};
    (@__items ($($not:meta,)*) ; ( ($($m:meta),*) ($($it:item)*) ), $($rest:tt)*) => {
        // Emit all items within one block, applying an approprate #[cfg]. The
        // #[cfg] will require all `$m` matchers specified and must also negate
        // all previous matchers.
        cfg_if! { @__apply cfg(all($($m,)* not(any($($not),*)))), $($it)* }

        // Recurse to emit all other items in `$rest`, and when we do so add all
        // our `$m` matchers to the list of `$not` matchers as future emissions
        // will have to negate everything we just matched as well.
        cfg_if! { @__items ($($not,)* $($m,)*) ; $($rest)* }
    };

    // Internal macro to Apply a cfg attribute to a list of items
    (@__apply $m:meta, $($it:item)*) => {
        $(#[$m] $it)*
    };
}

macro_rules! pdep_impl {
    ($ty:ty) => {
        #[inline]
        fn pdep_(value: $ty, mut mask: $ty) -> $ty {
            let mut res = 0;
            let mut bb: $ty = 1;
            loop {
                if mask == 0 {
                    break;
                }
                if (value & bb) != 0 {
                    res |= mask & mask.wrapping_neg();
                }
                mask &= mask - 1;
                bb = bb.wrapping_add(bb);
            }
            res
        }
    };
    ($ty:ty, $intr:ident) => {
        cfg_if! {
            if  #[cfg(all(
                  any(target_arch = "x86", target_arch = "x86_64"),
                  target_feature = "bmi2"
            ))] {
                #[inline]
                #[target_feature(enable = "bmi2")]
                unsafe fn pdep_(value: $ty, mask: $ty) -> $ty {
                    crate::util::pdep::arch::$intr(
                        value as _,
                        mask as _,
                    ) as _
                }
            } else {
                pdep_impl!($ty);
            }

        }
    };
}

macro_rules! impl_pdep {
    ($id:ident $(,$args:ident)*) => {
        impl Pdep for $id {
            #[inline]
            #[allow(unused_unsafe)]
            fn pdep(self, mask: Self) -> Self {
                pdep_impl!($id $(,$args)*);
                // UNSAFETY: this is always safe, because
                // the unsafe `#[target_feature]` function
                // is only generated when the feature is
                // statically-enabled at compile-time.
                unsafe { pdep_(self, mask) }
           }
        }
    }
}

impl_all!(impl_pdep: u8, u16, i8, i16);

cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        impl_pdep!(u32, _pdep_u32);
        impl_pdep!(i32, _pdep_u32);
        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                impl_pdep!(u64, _pdep_u64);
                impl_pdep!(i64, _pdep_u64);
            } else {
                impl_all!(impl_pdep: i64, u64);
            }
        }
    } else {
        impl_all!(impl_pdep: u32, i32, i64, u64);
    }
}
