macro_rules! unroll {
    (1, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { let mut $i: usize = $e; $s };
    (2, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(1, |$i = {$e}| $s, $inc); $inc; #[allow(unused_assignments)] $s };
    (3, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(2, |$i = {$e}| $s, $inc); $inc; #[allow(unused_assignments)] $s };
    (4, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(3, |$i = {$e}| $s, $inc); $inc; #[allow(unused_assignments)] $s };
    (5, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(4, |$i = {$e}| $s, $inc); $inc; #[allow(unused_assignments)] $s };
    (6, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(5, |$i = {$e}| $s, $inc); $inc; #[allow(unused_assignments)] $s };
    (7, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(6, |$i = {$e}| $s, $inc); $inc; #[allow(unused_assignments)] $s };
    (8, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(7, |$i = {$e}| $s, $inc); $inc; #[allow(unused_assignments)] $s };
}

pub(crate) use unroll;
