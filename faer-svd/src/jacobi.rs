// Algorithm ported from Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2013-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use assert2::assert;
use faer_core::{permutation::swap_cols, zipped, MatMut, RealField};
use reborrow::*;

#[derive(Copy, Clone, Debug)]
pub struct JacobiRotation<T> {
    pub c: T,
    pub s: T,
}

impl<E: RealField> JacobiRotation<E> {
    pub fn from_triplet(x: E, y: E, z: E) -> Self {
        let abs_y = y.abs();
        let two_abs_y = abs_y.add(&abs_y);
        if two_abs_y == E::zero() {
            Self {
                c: E::one(),
                s: E::zero(),
            }
        } else {
            let tau = (x.sub(&z)).mul(&two_abs_y.inv());
            let w = ((tau.mul(&tau)).add(&E::one())).sqrt();
            let t = if tau > E::zero() {
                (tau.add(&w)).inv()
            } else {
                (tau.sub(&w)).inv()
            };

            let neg_sign_y = if y > E::zero() {
                E::one().neg()
            } else {
                E::one()
            };
            let n = (t.mul(&t).add(&E::one())).sqrt().inv();

            Self {
                c: n.clone(),
                s: neg_sign_y.mul(&t).mul(&n),
            }
        }
    }

    pub fn apply_on_the_left_2x2(&self, m00: E, m01: E, m10: E, m11: E) -> (E, E, E, E) {
        let Self { c, s } = self;
        (
            m00.mul(c).add(&m10.mul(s)),
            m01.mul(c).add(&m11.mul(s)),
            s.neg().mul(&m00).add(&c.mul(&m10)),
            s.neg().mul(&m01).add(&c.mul(&m11)),
        )
    }

    pub fn apply_on_the_right_2x2(&self, m00: E, m01: E, m10: E, m11: E) -> (E, E, E, E) {
        let (r00, r01, r10, r11) = self.transpose().apply_on_the_left_2x2(m00, m10, m01, m11);
        (r00, r10, r01, r11)
    }

    pub fn apply_on_the_left_in_place(&self, x: MatMut<'_, E>, y: MatMut<'_, E>) {
        pulp::Arch::new().dispatch(
            #[inline(always)]
            move || {
                assert!(x.nrows() == 1);

                let Self { c, s } = self;
                if *c == E::one() && *s == E::zero() {
                    return;
                }

                zipped!(x, y).for_each(move |mut x, mut y| {
                    let x_ = x.read();
                    let y_ = y.read();
                    x.write(c.mul(&x_).add(&s.mul(&y_)));
                    y.write(s.neg().mul(&x_).add(&c.mul(&y_)));
                });
            },
        )
    }

    pub fn apply_on_the_right_in_place(&self, x: MatMut<'_, E>, y: MatMut<'_, E>) {
        self.transpose()
            .apply_on_the_left_in_place(x.transpose(), y.transpose());
    }

    pub fn transpose(&self) -> Self {
        Self {
            c: self.c.clone(),
            s: self.s.neg(),
        }
    }
}

impl<E: RealField> core::ops::Mul for JacobiRotation<E> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            c: self.c.mul(&rhs.c).sub(&self.s.mul(&rhs.s)),
            s: self.c.mul(&rhs.s).add(&self.s.mul(&rhs.c)),
        }
    }
}

fn compute_2x2<E: RealField>(
    m00: E,
    m01: E,
    m10: E,
    m11: E,
) -> (JacobiRotation<E>, JacobiRotation<E>) {
    let t = m00.add(&m11);
    let d = m10.sub(&m01);

    let rot1 = if d == E::zero() {
        JacobiRotation {
            c: E::one(),
            s: E::zero(),
        }
    } else {
        let u = t.mul(&d.inv());
        let tmp = (E::one().add(&u.mul(&u))).sqrt().inv();
        let tmp = if tmp == E::zero() { u.abs().inv() } else { tmp };
        JacobiRotation {
            c: u.mul(&tmp),
            s: tmp,
        }
    };
    let j_right = {
        let (m00, m01, _, m11) = rot1.apply_on_the_left_2x2(m00, m01, m10, m11);
        JacobiRotation::from_triplet(m00, m01, m11)
    };
    let j_left = rot1 * j_right.transpose();

    (j_left, j_right)
}

pub enum Skip {
    None,
    First,
    Last,
}

pub fn jacobi_svd<E: RealField>(
    matrix: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    v: Option<MatMut<'_, E>>,
    skip: Skip,
    epsilon: E,
    consider_zero_threshold: E,
) -> usize {
    assert!(matrix.nrows() == matrix.ncols());
    let n = matrix.nrows();

    if let Some(u) = u.rb() {
        assert!(n == u.nrows());
        assert!(n == u.ncols());
    };
    if let Some(v) = v.rb() {
        assert!(n == v.ncols());
    }

    let mut matrix = matrix;
    let mut u = u;
    let mut v = v;

    if let Some(mut u) = u.rb_mut() {
        for j in 0..n {
            for i in 0..j {
                u.rb_mut().write(i, j, E::zero());
            }
            u.rb_mut().write(j, j, E::one());
            for i in j + 1..n {
                u.rb_mut().write(i, j, E::zero());
            }
        }
    }

    if let Some(mut v) = v.rb_mut() {
        if matches!(skip, Skip::First) {
            for i in 0..n - 1 {
                v.rb_mut().write(i, 0, E::zero());
            }
            v = v.submatrix(0, 1, n - 1, n - 1);
        }

        let m = v.nrows();
        let n = v.ncols();
        for j in 0..n {
            for i in 0..j {
                v.rb_mut().write(i, j, E::zero());
            }
            if j == m {
                break;
            }
            v.rb_mut().write(j, j, E::one());
            for i in j + 1..m {
                v.rb_mut().write(i, j, E::zero());
            }
        }
    }

    let mut max_diag = E::zero();
    {
        let diag = matrix.rb().diagonal();
        for idx in 0..diag.nrows() {
            let d = diag.read(idx, 0).abs();
            max_diag = if d > max_diag { d } else { max_diag };
        }
    }

    let precision = epsilon.scale_power_of_two(&E::one().add(&E::one()));
    loop {
        let mut failed = false;
        for p in 1..n {
            for q in 0..p {
                let threshold = precision.mul(&max_diag);
                let threshold = if threshold > consider_zero_threshold {
                    threshold
                } else {
                    consider_zero_threshold.clone()
                };

                if (matrix.read(p, q).abs() > threshold) || (matrix.read(q, p).abs() > threshold) {
                    failed = true;
                    let (j_left, j_right) = compute_2x2(
                        matrix.read(p, p),
                        matrix.read(p, q),
                        matrix.read(q, p),
                        matrix.read(q, q),
                    );

                    let [top, bottom] = matrix.rb_mut().split_at_row(p);
                    j_left.apply_on_the_left_in_place(bottom.row(0), top.row(q));
                    let [left, right] = matrix.rb_mut().split_at_col(p);
                    j_right.apply_on_the_right_in_place(right.col(0), left.col(q));

                    if let Some(u) = u.rb_mut() {
                        let [left, right] = u.split_at_col(p);
                        j_left
                            .transpose()
                            .apply_on_the_right_in_place(right.col(0), left.col(q))
                    }
                    if let Some(v) = v.rb_mut() {
                        let [left, right] = v.split_at_col(p);
                        j_right.apply_on_the_right_in_place(right.col(0), left.col(q))
                    }

                    for idx in [p, q] {
                        let d = matrix.read(idx, idx).abs();
                        max_diag = if d > max_diag { d } else { max_diag };
                    }
                }
            }
        }
        if !failed {
            break;
        }
    }

    // make diagonal elements positive
    for j in 0..n {
        let d = matrix.read(j, j);
        if d < E::zero() {
            matrix.write(j, j, d.neg());
            if let Some(mut u) = u.rb_mut() {
                for i in 0..n {
                    u.write(i, j, u.read(i, j).neg());
                }
            }
        }
    }

    // sort singular values and count nonzero ones
    let (start, new_n) = match skip {
        Skip::None => (0, n),
        Skip::First => (1, n - 1),
        Skip::Last => (0, n - 1),
    };

    let mut matrix = matrix.submatrix(start, start, new_n, new_n);
    let mut u = u.map(|u| u.submatrix(0, start, n, new_n));
    let mut v = v.map(|v| {
        let vn = v.nrows();
        v.submatrix(0, start, vn, new_n)
    });

    let n = new_n;
    let mut nnz_count = n;
    for i in 0..n {
        let mut largest_elem = E::zero();
        let mut largest_pos = i;

        for j in i..n {
            let mjj = matrix.read(j, j);
            (largest_elem, largest_pos) = if mjj > largest_elem {
                (mjj, j)
            } else {
                (largest_elem, largest_pos)
            };
        }

        if largest_elem == E::zero() {
            nnz_count = i;
        }

        if largest_pos > i {
            let mii = matrix.read(i, i);
            matrix.write(i, i, largest_elem);
            matrix.write(largest_pos, largest_pos, mii);
            if let Some(u) = u.rb_mut() {
                swap_cols(u, i, largest_pos);
            }
            if let Some(v) = v.rb_mut() {
                swap_cols(v, i, largest_pos);
            }
        }
    }
    nnz_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{Mat, MatRef};

    #[track_caller]
    fn check_svd(mat: MatRef<'_, f64>, u: MatRef<'_, f64>, v: MatRef<'_, f64>, s: MatRef<'_, f64>) {
        let m = mat.nrows();
        let n = mat.ncols();
        let reconstructed = u * s * v.transpose();

        for i in 0..m {
            for j in 0..n {
                if i == j {
                    assert!(s.read(i, j) >= 0.0);
                } else {
                    assert_approx_eq!(s.read(i, j), 0.0);
                }
            }
        }

        for o in [u * u.transpose(), v * v.transpose()] {
            let m = o.nrows();
            for i in 0..m {
                for j in 0..m {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(o.read(i, j), target);
                }
            }
        }
        for i in 0..m {
            for j in 0..n {
                assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j));
            }
        }

        let size = m.min(n);
        if size > 1 {
            for i in 0..size - 1 {
                assert!(s.read(i, i) >= s.read(i + 1, i + 1));
            }
        }
    }

    #[test]
    fn test_jacobi() {
        for n in [0, 1, 2, 4, 8, 15, 16, 31, 32] {
            let mat = Mat::<f64>::with_dims(n, n, |_, _| rand::random::<f64>());

            let mut s = mat.clone();
            let mut u = Mat::<f64>::zeros(n, n);
            let mut v = Mat::<f64>::zeros(n, n);

            jacobi_svd(
                s.as_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                Skip::None,
                f64::EPSILON,
                f64::MIN_POSITIVE,
            );
            check_svd(mat.as_ref(), u.as_ref(), v.as_ref(), s.as_ref());
        }
    }

    #[test]
    fn test_skip_first() {
        for n in [2, 4, 8, 15, 16, 31, 32] {
            let mat = Mat::<f64>::with_dims(
                n,
                n,
                |_, j| if j == 0 { 0.0 } else { rand::random::<f64>() },
            );

            let mut s = mat.clone();
            let mut u = Mat::<f64>::zeros(n, n);
            let mut v = Mat::<f64>::zeros(n, n);

            jacobi_svd(
                s.as_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                Skip::First,
                f64::EPSILON,
                f64::MIN_POSITIVE,
            );
            let mut u_shifted = Mat::<f64>::zeros(n, n);
            for j in 1..n {
                for i in 0..n {
                    u_shifted.write(i, j - 1, u.read(i, j));
                }

                s.write(j - 1, j, s.read(j, j));
                s.write(j, j, 0.0);
            }
            for i in 0..n {
                u_shifted.write(i, n - 1, u.read(i, 0));
            }
            check_svd(
                mat.as_ref().submatrix(0, 1, n, n - 1),
                u_shifted.as_ref(),
                v.as_ref().submatrix(0, 1, n - 1, n - 1),
                s.as_ref().submatrix(0, 1, n, n - 1),
            );
        }
    }

    #[test]
    fn test_skip_last() {
        for n in [2, 4, 8, 15, 16, 31, 32] {
            let mat = Mat::<f64>::with_dims(n, n, |_, j| {
                if j == n - 1 {
                    0.0
                } else {
                    rand::random::<f64>()
                }
            });

            let mut s = mat.clone();
            let mut u = Mat::<f64>::zeros(n, n);
            let mut v = Mat::<f64>::zeros(n, n);

            jacobi_svd(
                s.as_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                Skip::Last,
                f64::EPSILON,
                f64::MIN_POSITIVE,
            );
            assert!(v.read(n - 1, n - 1) == 1.0);
            for j in 0..n - 1 {
                assert_approx_eq!(v.read(n - 1, j), 0.0);
                assert_approx_eq!(v.read(j, n - 1), 0.0);
            }
            check_svd(
                mat.as_ref().submatrix(0, 0, n, n - 1),
                u.as_ref(),
                v.as_ref().submatrix(0, 0, n - 1, n - 1),
                s.as_ref().submatrix(0, 0, n, n - 1),
            );
        }
    }

    #[test]
    fn eigen_286() {
        let mat = faer_core::mat![[-7.90884e-313, -4.94e-324], [0.0, 5.60844e-313]];
        let n = 2;
        let mut s = mat.clone();
        let mut u = Mat::<f64>::zeros(n, n);
        let mut v = Mat::<f64>::zeros(n, n);
        jacobi_svd(
            s.as_mut(),
            Some(u.as_mut()),
            Some(v.as_mut()),
            Skip::None,
            f64::EPSILON,
            f64::MIN_POSITIVE,
        );
        check_svd(mat.as_ref(), u.as_ref(), v.as_ref(), s.as_ref());
    }
}
