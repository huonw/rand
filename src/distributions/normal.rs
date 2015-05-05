// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The normal and derived distributions.

use {Rng, Rand, RandStream, Open01};
use distributions::{ziggurat, ziggurat_tables};

/// A wrapper around an `f64` to generate N(0, 1) random numbers
/// (a.k.a.  a standard normal, or Gaussian).
///
/// See `Normal` for the general normal distribution. That this has to
/// be unwrapped before use as an `f64` (using either `*` or
/// `mem::transmute` is safe).
///
/// Implemented via the ZIGNOR variant[1] of the Ziggurat method.
///
/// [1]: Jurgen A. Doornik (2005). [*An Improved Ziggurat Method to
/// Generate Normal Random
/// Samples*](http://www.doornik.com/research/ziggurat.pdf). Nuffield
/// College, Oxford
#[derive(Clone, Copy)]
pub struct StandardNormal;

impl Rand<StandardNormal> for f64 {
    type Stream = StandardNormal;
    fn rand(x: StandardNormal) -> StandardNormal {
        x
    }
}
impl RandStream for StandardNormal {
    type Output = f64;

    fn next<R: Rng>(&self, rng: &mut R) -> f64 {
        #[inline]
        fn pdf(x: f64) -> f64 {
            (-x*x/2.0).exp()
        }
        #[inline]
        fn zero_case<R:Rng>(rng: &mut R, u: f64) -> f64 {
            // compute a random number in the tail by hand

            // strange initial conditions, because the loop is not
            // do-while, so the condition should be true on the first
            // run, they get overwritten anyway (0 < 1, so these are
            // good).
            let mut x = 1.0f64;
            let mut y = 0.0f64;

            while -2.0 * y < x * x {
                let x_ = rng.gen::<f64, _>(Open01);
                let y_ = rng.gen::<f64, _>(Open01);

                x = x_.ln() / ziggurat_tables::ZIG_NORM_R;
                y = y_.ln();
            }

            if u < 0.0 { x - ziggurat_tables::ZIG_NORM_R } else { ziggurat_tables::ZIG_NORM_R - x }
        }

        ziggurat(
            rng,
            true, // this is symmetric
            &ziggurat_tables::ZIG_NORM_X,
            &ziggurat_tables::ZIG_NORM_F,
            pdf, zero_case)
    }
}

/// The normal distribution `N(mean, std_dev**2)`.
///
/// This uses the ZIGNOR variant of the Ziggurat method, see
/// `StandardNormal` for more details.
///
/// # Example
///
/// ```rust
/// use rand::Rng;
/// use rand::distributions::Normal;
///
/// // mean 2, standard deviation 3
/// let normal = Normal::new(2.0, 3.0);
/// let v: f64 = rand::thread_rng().gen(normal);
/// println!("{} is from a N(2, 9) distribution", v)
/// ```
#[derive(Clone, Copy)]
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Normal {
    /// Construct a new `Normal` distribution with the given mean and
    /// standard deviation.
    ///
    /// # Panics
    ///
    /// Panics if `std_dev < 0`.
    pub fn new(mean: f64, std_dev: f64) -> Normal {
        assert!(std_dev >= 0.0, "Normal::new called with `std_dev` < 0");
        Normal {
            mean: mean,
            std_dev: std_dev
        }
    }
}
impl Rand<Normal> for f64 {
    type Stream = Normal;

    fn rand(s: Normal) -> Normal { s }
}
impl RandStream for Normal {
    type Output = f64;

    fn next<R: Rng>(&self, rng: &mut R) -> f64 {
        let n = rng.gen::<f64, _>(StandardNormal);
        self.mean + self.std_dev * n
    }
}


/// The log-normal distribution `ln N(mean, std_dev**2)`.
///
/// If `X` is log-normal distributed, then `ln(X)` is `N(mean,
/// std_dev**2)` distributed.
///
/// # Example
///
/// ```rust
/// use rand::Rng;
/// use rand::distributions::LogNormal;
///
/// // mean 2, standard deviation 3
/// let log_normal = LogNormal::new(2.0, 3.0);
/// let v: f64 = rand::thread_rng().gen(log_normal);
/// println!("{} is from an ln N(2, 9) distribution", v)
/// ```
#[derive(Clone, Copy)]
pub struct LogNormal {
    norm: Normal
}

impl LogNormal {
    /// Construct a new `LogNormal` distribution with the given mean
    /// and standard deviation.
    ///
    /// # Panics
    ///
    /// Panics if `std_dev < 0`.
    pub fn new(mean: f64, std_dev: f64) -> LogNormal {
        assert!(std_dev >= 0.0, "LogNormal::new called with `std_dev` < 0");
        LogNormal { norm: Normal::new(mean, std_dev) }
    }
}
impl Rand<LogNormal> for f64 {
    type Stream = LogNormal;

    fn rand(s: LogNormal) -> LogNormal { s }
}
impl RandStream for LogNormal {
    type Output = f64;

    fn next<R: Rng>(&self, rng: &mut R) -> f64 {
        self.norm.next(rng).exp()
    }
}

#[cfg(test)]
mod tests {
    use RandStream;
    use super::{Normal, LogNormal};

    #[test]
    fn test_normal() {
        let norm = Normal::new(10.0, 10.0);
        let mut rng = ::test::rng();
        for _ in 0..1000 {
            norm.next(&mut rng);
        }
    }
    #[test]
    #[should_panic]
    fn test_normal_invalid_sd() {
        Normal::new(10.0, -1.0);
    }


    #[test]
    fn test_log_normal() {
        let lnorm = LogNormal::new(10.0, 10.0);
        let mut rng = ::test::rng();
        for _ in 0..1000 {
            lnorm.next(&mut rng);
        }
    }
    #[test]
    #[should_panic]
    fn test_log_normal_invalid_sd() {
        LogNormal::new(10.0, -1.0);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;
    use std::mem::size_of;
    use super::Normal;
    use RandStream;

    #[bench]
    fn rand_normal(b: &mut Bencher) {
        let mut rng = ::test::weak_rng();
        let normal = Normal::new(-2.71828, 3.14159);

        b.iter(|| {
            for _ in 0..::RAND_BENCH_N {
                normal.next(&mut rng);
            }
        });
        b.bytes = size_of::<f64>() as u64 * ::RAND_BENCH_N;
    }
}
