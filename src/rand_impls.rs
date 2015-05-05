// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The implementations of `Rand` for the built-in types.

use std::char;
use std::mem;
use std::marker::PhantomData;
use std::ops::RangeFull;

use {Rand,RandStream,Rng};

pub struct NumStream<T> {
    _x: PhantomData<T>
}

macro_rules! rand {
    ($($ty: ty),*) => {
        $(
            impl Rand<RangeFull> for $ty {
                type Stream = NumStream<$ty>;

                fn rand(_: RangeFull) -> Self::Stream {
                    NumStream {
                        _x: PhantomData
                    }
                }
            }
            )*
    }
}

rand! {
    isize, i8, i16, i32, i64,
    usize, u8, u16, u32, u64
}

impl RandStream for NumStream<isize> {
    type Output = isize;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> isize {
        if mem::size_of::<isize>() == 4 {
            rng.gen::<i32, _>(..) as isize
        } else {
            rng.gen::<i64, _>(..) as isize
        }
    }
}

impl RandStream for NumStream<i8> {
    type Output = i8;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> i8 {
        rng.next_u32() as i8
    }
}

impl RandStream for NumStream<i16> {
    type Output = i16;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> i16 {
        rng.next_u32() as i16
    }
}

impl RandStream for NumStream<i32> {
    type Output = i32;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> i32 {
        rng.next_u32() as i32
    }
}

impl RandStream for NumStream<i64> {
    type Output = i64;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> i64 {
        rng.next_u64() as i64
    }
}

impl RandStream for NumStream<usize> {
    type Output = usize;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> usize {
        if mem::size_of::<usize>() == 4 {
            rng.gen::<u32, _>(..) as usize
        } else {
            rng.gen::<u64, _>(..) as usize
        }
    }
}

impl RandStream for NumStream<u8> {
    type Output = u8;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> u8 {
        rng.next_u32() as u8
    }
}

impl RandStream for NumStream<u16> {
    type Output = u16;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> u16 {
        rng.next_u32() as u16
    }
}

impl RandStream for NumStream<u32> {
    type Output = u32;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> u32 {
        rng.next_u32()
    }
}

impl RandStream for NumStream<u64> {
    type Output = u64;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> u64 {
        rng.next_u64()
    }
}

macro_rules! float_impls {
    ($mod_name:ident, $ty:ty, $mantissa_bits:expr, $method_name:ident) => {

        mod $mod_name {
            use {Rng, Rand, RandStream, Open01, Closed01};
            use super::NumStream;
            use std::ops::RangeFull;
            use std::marker::PhantomData;

            const SCALE: $ty = (1u64 << $mantissa_bits) as $ty;

            impl Rand<RangeFull> for $ty {
                type Stream = NumStream<$ty>;

                fn rand(_: RangeFull) -> Self::Stream {
                    NumStream { _x: PhantomData }
                }
            }
            impl RandStream for NumStream<$ty> {
                type Output = $ty;
                /// Generate a floating point number in the half-open
                /// interval `[0,1)`.
                ///
                /// See `Closed01` for the closed interval `[0,1]`,
                /// and `Open01` for the open interval `(0,1)`.
                #[inline]
                fn next<R: Rng>(&self, rng: &mut R) -> $ty {
                    rng.$method_name()
                }
            }
            pub struct Open01Stream;
            impl Rand<Open01> for $ty {
                type Stream = Open01Stream;
                fn rand(_: Open01) -> Self::Stream {
                    Open01Stream
                }
            }
            impl RandStream for Open01Stream {
                type Output = $ty;
                fn next<R: Rng>(&self, rng: &mut R) -> $ty {
                    rng.$method_name() + 0.25 / SCALE
                }
            }
            pub struct Closed01Stream;
            impl Rand<Closed01> for $ty {
                type Stream = Closed01Stream;
                fn rand(_: Closed01) -> Self::Stream {
                    Closed01Stream
                }
            }
            impl RandStream for Closed01Stream {
                type Output = $ty;
                fn next<R: Rng>(&self, rng: &mut R) -> $ty {
                    rng.$method_name() * SCALE / (SCALE - 1.0)
                }
            }
        }
    }
}
float_impls! { f64_rand_impls, f64, 53, next_f64 }
float_impls! { f32_rand_impls, f32, 24, next_f32 }

pub struct CharStream;
impl Rand<RangeFull> for char {
    type Stream = CharStream;
    fn rand(_: RangeFull) -> CharStream { CharStream }
}
impl RandStream for CharStream {
    type Output = char;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> char {
        // a char is 21 bits
        const CHAR_MASK: u32 = 0x001f_ffff;
        loop {
            // Rejection sampling. About 0.2% of numbers with at most
            // 21-bits are invalid codepoints (surrogates), so this
            // will succeed first go almost every time.
            match char::from_u32(rng.next_u32() & CHAR_MASK) {
                Some(c) => return c,
                None => {}
            }
        }
    }
}

pub struct BoolStream;
impl Rand<RangeFull> for bool {
    type Stream = BoolStream;
    fn rand(_: RangeFull) -> BoolStream { BoolStream }
}
impl RandStream for BoolStream {
    type Output = bool;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> bool {
        rng.gen::<u8, _>(..) & 1 == 1
    }
}

macro_rules! tuple_impl {
    // use variables to indicate the arity of the tuple
    ($mod_: ident, $($tyvar:ident),* ) => {
        mod $mod_ {
            use std::marker::PhantomData;
            use {Rng, Rand, RandStream};
            #[allow(non_snake_case)]
            pub struct Stream<Dist, $($tyvar: Rand<Dist>),*> {
                _x: PhantomData<Dist>,
                $($tyvar: $tyvar::Stream),*
            }
            // the trailing commas are for the 1 tuple
            impl<
                $( $tyvar : Rand<Dist>, )*
            Dist: Clone
                > Rand<Dist> for ( $( $tyvar, )*) {
                    type Stream = Stream<Dist, $($tyvar),*>;

                    fn rand(_dist: Dist) -> Stream<Dist, $($tyvar),*> {
                        Stream {
                            _x: PhantomData,
                            $($tyvar: $tyvar::rand(_dist.clone())),*
                        }
                    }
                }
            impl<
                $( $tyvar : Rand<Dist>, )*
            Dist,
            > RandStream for Stream<Dist, $($tyvar),*> {
                type Output = ($($tyvar, )*);
                #[inline]
                fn next<R: Rng>(&self, _rng: &mut R) -> Self::Output {
                    (
                        // use the $tyvar's to get the appropriate number of
                        // repeats (they're not actually needed)
                        $(
                            self.$tyvar.next(_rng),
                                )*

                        )
                }
            }
        }
    }
}

tuple_impl!{zero, }
tuple_impl!{i,   A}
tuple_impl!{ii,  A, B}
tuple_impl!{iii, A, B, C}
tuple_impl!{iv,  A, B, C, D}
tuple_impl!{v,   A, B, C, D, E}
tuple_impl!{vi,  A, B, C, D, E, F}
tuple_impl!{vii, A, B, C, D, E, F, G}
tuple_impl!{viii,A, B, C, D, E, F, G, H}
tuple_impl!{ix,  A, B, C, D, E, F, G, H, I}
tuple_impl!{x,   A, B, C, D, E, F, G, H, I, J}
tuple_impl!{xi,  A, B, C, D, E, F, G, H, I, J, K}
tuple_impl!{xii, A, B, C, D, E, F, G, H, I, J, K, L}

pub struct OptionStream<BDist, TDist> {
    bdist: BDist,
    tdist: TDist,
}

impl<T: Rand<TDist>, TDist> Rand<TDist> for Option<T> {
    type Stream = OptionStream<BoolStream, T::Stream>;
    fn rand(dist: TDist) -> Self::Stream {
        OptionStream {
            bdist: BoolStream,
            tdist: T::rand(dist)
        }
    }
}
impl<BDist: RandStream<Output = bool>, TDist: RandStream> RandStream for OptionStream<BDist, TDist> {
    type Output = Option<TDist::Output>;
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> Self::Output {
        if self.bdist.next(rng) {
            Some(self.tdist.next(rng))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use {Rng, thread_rng, Open01, Closed01};

    struct ConstantRng(u64);
    impl Rng for ConstantRng {
        fn next_u32(&mut self) -> u32 {
            let ConstantRng(v) = *self;
            v as u32
        }
        fn next_u64(&mut self) -> u64 {
            let ConstantRng(v) = *self;
            v
        }
    }

    #[test]
    fn floating_point_edge_cases() {
        // the test for exact equality is correct here.
        assert!(ConstantRng(0xffff_ffff).gen::<f32, _>(..) != 1.0);
        assert!(ConstantRng(0xffff_ffff_ffff_ffff).gen::<f64, _>(..) != 1.0);
    }

    #[test]
    fn rand_open() {
        // this is unlikely to catch an incorrect implementation that
        // generates exactly 0 or 1, but it keeps it sane.
        let mut rng = thread_rng();
        for _ in 0..1_000 {
            // strict inequalities
            let f = rng.gen::<f64, _>(Open01);
            assert!(0.0 < f && f < 1.0);

            let f = rng.gen::<f32, _>(Open01);
            assert!(0.0 < f && f < 1.0);
        }
    }

    #[test]
    fn rand_closed() {
        let mut rng = thread_rng();
        for _ in 0..1_000 {
            // strict inequalities
            let f = rng.gen::<f64, _>(Closed01);
            assert!(0.0 <= f && f <= 1.0);

            let f = rng.gen::<f32, _>(Closed01);
            assert!(0.0 <= f && f <= 1.0);
        }
    }
}
