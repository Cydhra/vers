macro_rules! gen_ef_iter_impl {
    ($($life:lifetime, )? $name:ident, $converter:ident) => {
        impl $(<$life>)? $name $(<$life>)? {
            /// Advances the iterator by `n` elements. Returns an error if the iterator does not have
            /// enough elements left. Does not call `next` internally.
            /// This method is currently being added to the iterator trait, see
            /// [this issue](https://github.com/rust-lang/rust/issues/77404).
            /// As soon as it is stabilized, this method will be removed and replaced with a custom
            /// implementation in the iterator impl.
            fn advance_by(&mut self, n: usize) -> Result<(), std::num::NonZeroUsize> {
                if n == 0 {
                    return Ok(());
                }

                if Some(self.index + n - 1) > self.back_index {
                    if Some(self.index) > self.back_index {
                        Err(std::num::NonZeroUsize::new(n).unwrap())
                    } else {
                        Err(std::num::NonZeroUsize::new(n - (self.back_index.as_ref().unwrap_or(&usize::MAX).wrapping_sub(self.index).wrapping_add(1))).unwrap())
                    }
                } else {
                    self.index += n;
                    if n > 0 {
                        // since advance_by is not stable yet, we need to call nth - 1.
                        self.upper_iter.nth(n - 1).expect("upper iterator should not be exhausted");
                    }
                    Ok(())
                }
            }

            /// Advances the back iterator by `n` elements. Returns an error if the iterator does not have
            /// enough elements left. Does not call `next` internally.
            /// This method is currently being added to the iterator trait, see
            /// [this issue](https://github.com/rust-lang/rust/issues/77404).
            /// As soon as it is stabilized, this method will be removed and replaced with a custom
            /// implementation in the iterator impl.
            fn advance_back_by(&mut self, n: usize) -> Result<(), std::num::NonZeroUsize> {
                if n == 0 {
                    return Ok(());
                }

                // special case this, because otherwise back_index might be None and we would panic
                if self.is_iter_empty() {
                    return Err(std::num::NonZeroUsize::new(n).unwrap());
                }

                // since the cursors point to unconsumed items, we need to add 1
                let remaining = *self.back_index.as_ref().unwrap() - self.index + 1;
                if remaining < n {
                    return Err(std::num::NonZeroUsize::new(n - remaining).unwrap());
                }
                self.back_index = if self.back_index >= Some(n) { self.back_index.map(|b| b - n) } else { None };
                if n > 0 {
                    // since advance_by is not stable yet, we need to call nth - 1.
                    self.upper_iter.nth_back(n - 1).expect("upper iterator should not be exhausted");
                }
                Ok(())
            }

            fn is_iter_empty(&self) -> bool {
                // this is legal because Ord is behaving as expected on Option
                Some(self.index) > self.back_index
            }
        }

        impl $(<$life>)? Iterator for $name $(<$life>)? {
            type Item = u64;

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(upper) = self.upper_iter.next() {
                    let upper = upper - self.index - 1;
                    let lower = self
                        .vec
                        .get_bits_unchecked(self.index * self.lower_len, self.lower_len);
                    self.index += 1;
                    Some((((upper as u64) << self.lower_len) | lower) + self.universe_zero)
                } else {
                    None
                }
            }

            /// Returns the number of elements that this iterator will iterate over. The size is
            /// precise.
            #[must_use]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.len(), Some(self.len()))
            }

            /// Returns the exact number of elements that this iterator would iterate over. Does not
            /// call `next` internally.
            #[must_use]
            fn count(self) -> usize
            where
                Self: Sized,
            {
                self.len()
            }

            /// Returns the last element of the iterator. Does not call `next` internally.
            #[must_use]
            fn last(self) -> Option<Self::Item>
            where
                Self: Sized,
            {
                if self.is_iter_empty() {
                    return None;
                }

                let upper = self.upper_iter.last().unwrap() - self.back_index.unwrap() - 1;
                let lower = self
                    .vec
                    .get_bits_unchecked(self.back_index.unwrap() * self.lower_len, self.lower_len);
                Some(((upper as u64) << self.lower_len) | lower)
            }

            /// Returns the nth element of the iterator. Does not call `next` internally, but advances
            /// the iterator by `n` elements.
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                self.advance_by(n).ok()?;
                self.next()
            }

            /// Returns the minimum remaining element of the iterator.
            /// Operates in constant time, because Elias-Fano vectors are sorted.
            fn min(mut self) -> Option<Self::Item>
            where
                Self: Sized,
                Self::Item: Ord,
            {
                self.next()
            }

            /// Returns the maximum remaining element of the iterator. Operates in constant time,
            /// because Elias-Fano vectors are sorted.
            fn max(self) -> Option<Self::Item>
            where
                Self: Sized,
                Self::Item: Ord,
            {
                self.last()
            }
        }

        impl $(<$life>)? std::iter::ExactSizeIterator for $name $(<$life>)? {
            fn len(&self) -> usize {
                // intentionally overflowing calculations to avoid branches on empty iterator
                (*self.back_index.as_ref().unwrap_or(&usize::MAX)).wrapping_sub(self.index).wrapping_add(1)
            }
        }

        impl $(<$life>)? std::iter::FusedIterator for $name $(<$life>)? {}

        impl $(<$life>)? std::iter::DoubleEndedIterator for $name $(<$life>)? {
            fn next_back(&mut self) -> Option<Self::Item> {
                if let Some(upper) = self.upper_iter.next_back() {
                    let index_back = self.back_index.unwrap();
                    let upper = upper - index_back - 1;
                    let lower = self
                        .vec
                        .get_bits_unchecked(index_back * self.lower_len, self.lower_len);
                    if *self.back_index.as_ref().unwrap() == 0 {
                        self.back_index = None;
                    } else {
                        self.back_index = Some(self.back_index.unwrap() - 1);
                    }
                    Some((((upper as u64) << self.lower_len) | lower) + self.universe_zero)
                } else {
                    None
                }
            }

            fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                self.advance_back_by(n).ok()?;
                self.next_back()
            }
        }
    };
}

macro_rules! impl_ef_iterator {
    ($own:ident, $bor:ident) => {
        #[doc = concat!("An owning iterator for `", stringify!($type), "`.")]
        #[doc = concat!("This struct is created by the `into_iter` trait implementation of `", stringify!($type), "`.")]
        #[derive(Clone, Debug)]
        pub struct $own {
            upper_iter: crate::bit_vec::fast_rs_vec::SelectIntoIter<false>,
            vec: crate::bit_vec::BitVec,
            index: usize,
            // back index is none, iff it points to element -1 (i.e. element 0 has been consumed by
            // a call to next_back()). It can be Some(..) even if the iterator is empty
            back_index: Option<usize>,
            lower_len: usize,
            universe_zero: u64,
        }

        impl $own {
            #[must_use]
            fn new(vec: crate::elias_fano::EliasFanoVec) -> Self {
                if vec.is_empty() {
                    return Self {
                        upper_iter: vec.upper_vec.into_iter1(),
                        vec: vec.lower_vec,
                        index: 0,
                        back_index: None,
                        lower_len: vec.lower_len,
                        universe_zero: vec.universe_zero,
                    };
                }

                let last = vec.len - 1;
                Self {
                    upper_iter: vec.upper_vec.into_iter1(),
                    vec: vec.lower_vec,
                    index: 0,
                    back_index: Some(last),
                    lower_len: vec.lower_len,
                    universe_zero: vec.universe_zero,
                }
            }
        }

        impl EliasFanoVec {
            #[doc = concat!("Returns an iterator over the elements of `", stringify!($type), "`.")]
            #[must_use]
            pub fn iter(&self) -> $bor<'_> {
                $bor::new(self)
            }
        }

        #[doc = concat!("A borrowing iterator for `", stringify!($type), "`.")]
        #[doc = concat!("This struct is created by the `iter` method of `", stringify!($type), "`.")]
        #[derive(Clone, Debug)]
        pub struct $bor<'a> {
            upper_iter: crate::bit_vec::fast_rs_vec::SelectIter<'a, false>,
            vec: &'a crate::bit_vec::BitVec,
            index: usize,
            // back index is none, iff it points to element -1 (i.e. element 0 has been consumed by
            // a call to next_back()). It can be Some(..) even if the iterator is empty
            back_index: Option<usize>,
            lower_len: usize,
            universe_zero: u64,
        }

        impl<'a> $bor<'a> {
            #[must_use]
            fn new(vec: &'a crate::elias_fano::EliasFanoVec) -> Self {
                if vec.is_empty() {
                    return Self {
                        upper_iter: vec.upper_vec.iter1(),
                        vec: &vec.lower_vec,
                        index: 0,
                        back_index: None,
                        lower_len: vec.lower_len,
                        universe_zero: vec.universe_zero,
                    };
                }

                let last = vec.len - 1;
                Self {
                    upper_iter: vec.upper_vec.iter1(),
                    vec: &vec.lower_vec,
                    index: 0,
                    back_index: Some(last),
                    lower_len: vec.lower_len,
                    universe_zero: vec.universe_zero,
                }
            }
        }

        crate::util::impl_into_iterator_impls!(EliasFanoVec, $own, $bor);

        crate::util::gen_ef_iter_impl!($own, into_iter1);

         crate::util::gen_ef_iter_impl!('a, $bor, iter1);
    };
}

pub(crate) use gen_ef_iter_impl;
pub(crate) use impl_ef_iterator;
