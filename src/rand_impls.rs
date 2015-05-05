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
use std::ops::RangeFull;

use {RandStream,Rng};

impl RandStream<isize> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> isize {
        if mem::size_of::<isize>() == 4 {
            rng.gen::<i32, _>(..) as isize
        } else {
            rng.gen::<i64, _>(..) as isize
        }
    }
}

impl RandStream<i8> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> i8 {
        rng.next_u32() as i8
    }
}

impl RandStream<i16> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> i16 {
        rng.next_u32() as i16
    }
}

impl RandStream<i32> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> i32 {
        rng.next_u32() as i32
    }
}

impl RandStream<i64> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> i64 {
        rng.next_u64() as i64
    }
}

impl RandStream<usize> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> usize {
        if mem::size_of::<usize>() == 4 {
            rng.gen::<u32, _>(..) as usize
        } else {
            rng.gen::<u64, _>(..) as usize
        }
    }
}

impl RandStream<u8> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> u8 {
        rng.next_u32() as u8
    }
}

impl RandStream<u16> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> u16 {
        rng.next_u32() as u16
    }
}

impl RandStream<u32> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> u32 {
        rng.next_u32()
    }
}

impl RandStream<u64> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> u64 {
        rng.next_u64()
    }
}

macro_rules! float_impls {
    ($mod_name:ident, $ty:ty, $mantissa_bits:expr, $method_name:ident) => {

        mod $mod_name {
            use {Rng, RandStream, Open01, Closed01};
            use std::ops::RangeFull;

            const SCALE: $ty = (1u64 << $mantissa_bits) as $ty;

            impl RandStream<$ty> for RangeFull {
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
            impl RandStream<$ty> for Open01 {
                fn next<R: Rng>(&self, rng: &mut R) -> $ty {
                    rng.$method_name() + 0.25 / SCALE
                }
            }
            impl RandStream<$ty> for Closed01 {
                fn next<R: Rng>(&self, rng: &mut R) -> $ty {
                    rng.$method_name() * SCALE / (SCALE - 1.0)
                }
            }
        }
    }
}
float_impls! { f64_rand_impls, f64, 53, next_f64 }
float_impls! { f32_rand_impls, f32, 24, next_f32 }

impl RandStream<char> for RangeFull {
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

impl RandStream<bool> for RangeFull {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> bool {
        rng.gen::<u8, _>(..) & 1 == 1
    }
}

macro_rules! tuple_impl {
    // use variables to indicate the arity of the tuple
    ($mod_: ident, $($tyvar:ident, $distvar: ident),* ) => {
        mod $mod_ {
            use std::marker::PhantomData;
            use {Rng, Rand, RandStream, Splat};
            #[allow(non_snake_case)]
            pub struct Stream<$($tyvar: Rand<$distvar>, $distvar),*> {
                _x: PhantomData<($($distvar,)*)>,
                $($tyvar: $tyvar::Stream),*
            }
            // the trailing commas are for the 1 tuple
            #[allow(non_snake_case)]
            impl<
                $( $tyvar : Rand<$distvar>, $distvar, )*
                > Rand<($($distvar,)*)> for ( $( $tyvar, )*) {
                    type Stream = Stream<$($tyvar, $distvar),*>;

                    fn rand(($($distvar,)*): ($($distvar,)*)) -> Stream<$($tyvar, $distvar),*> {
                        Stream {
                            _x: PhantomData,
                            $($tyvar: $tyvar::rand($distvar)),*
                        }
                    }
                }
            impl<
                $( $tyvar : Rand<$distvar>, $distvar, )*
            > RandStream<($($tyvar, )*)> for Stream<$($tyvar, $distvar),*> {
                #[inline]
                fn next<R: Rng>(&self, _rng: &mut R) -> ($($tyvar,)*) {
                    (
                        $(
                            self.$tyvar.next(_rng),
                                )*

                        )
                }
            }
            impl<Dist: Clone, $($tyvar: Rand<Dist>),*>
                Rand<Splat<Dist>> for ($($tyvar, )*) {
                    type Stream = Stream<$($tyvar,Dist,)*>;

                    fn rand(_x: Splat<Dist>) -> Stream<$($tyvar,Dist,)*> {
                        Stream {
                            _x: PhantomData,
                            $($tyvar: $tyvar::rand(_x.dist.clone()),)*
                        }
                    }
                }
        }
    }
}

tuple_impl!{zero, }
tuple_impl!{i,   A, A_}
tuple_impl!{ii,  A, A_, B, B_}
tuple_impl!{iii, A, A_, B, B_, C, C_}
tuple_impl!{iv,  A, A_, B, B_, C, C_, D, D_}
tuple_impl!{v,   A, A_, B, B_, C, C_, D, D_, E, E_}
tuple_impl!{vi,  A, A_, B, B_, C, C_, D, D_, E, E_, F, F_}
tuple_impl!{vii, A, A_, B, B_, C, C_, D, D_, E, E_, F, F_, G, G_}
tuple_impl!{viii,A, A_, B, B_, C, C_, D, D_, E, E_, F, F_, G, G_, H, H_}
tuple_impl!{ix,  A, A_, B, B_, C, C_, D, D_, E, E_, F, F_, G, G_, H, H_, I, I_}
tuple_impl!{x,   A, A_, B, B_, C, C_, D, D_, E, E_, F, F_, G, G_, H, H_, I, I_, J, J_}
tuple_impl!{xi,  A, A_, B, B_, C, C_, D, D_, E, E_, F, F_, G, G_, H, H_, I, I_, J, J_, K, K_}
tuple_impl!{xii, A, A_, B, B_, C, C_, D, D_, E, E_, F, F_, G, G_, H, H_, I, I_, J, J_, K, K_, L, L_}

/*
pub struct OptionStream<BDist, TDist> {
    bdist: BDist,
    tdist: TDist,
}

impl<T: Rand<TDist>, TDist> Rand<TDist> for Option<T> {
    fn rand(dist: TDist) -> Self::Stream {
        OptionStream<OptionStream<RangeFull, T::Stream>> {
            bdist: RangeFull,
            tdist: T::rand(dist)
        }
    }
}
impl<BDist: RandStream<bool>, TDist: RandStream<T>, T> RandStream<Option<T>> for OptionStream<BDist, TDist> {
    #[inline]
    fn next<R: Rng>(&self, rng: &mut R) -> Option<T> {
        if self.bdist.next(rng) {
            Some(self.tdist.next(rng))
        } else {
            None
        }
    }
}
*/

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
