// Algorithm ported from Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use faer_core::{jacobi::JacobiRotation, permutation::swap_cols, zipped, MatMut, RealField};
use reborrow::*;

pub fn compute_tridiag_real_evd_qr_algorithm<E: RealField>(
    diag: &mut [E],
    offdiag: &mut [E],
    u: Option<MatMut<'_, E>>,
    epsilon: E,
    consider_zero_threshold: E,
) {
    let n = diag.len();
    if n <= 1 {
        return;
    }

    let mut end = n - 1;
    let mut start = 0;
    let mut iter = 0;
    // TODO: abort after too many iterations
    let _ = &iter;

    let mut u = u;

    if let Some(mut u) = u.rb_mut() {
        zipped!(u.rb_mut()).for_each(|mut u| u.write(E::faer_zero()));
        zipped!(u.rb_mut().diagonal()).for_each(|mut u| u.write(E::faer_one()));
    }

    let arch = E::Simd::default();

    while end > 0 {
        for i in start..end {
            if (offdiag[i].faer_abs() < consider_zero_threshold)
                || (offdiag[i].faer_abs()
                    <= epsilon.faer_mul(
                        E::faer_add(diag[i].faer_abs(), diag[i + 1].faer_abs()).faer_sqrt(),
                    ))
            {
                offdiag[i] = E::faer_zero();
            }
        }

        while end > 0 && offdiag[end - 1] == E::faer_zero() {
            end -= 1;
        }

        if end == 0 {
            break;
        }

        iter += 1;

        start = end - 1;
        while start > 0 && offdiag[start - 1] != E::faer_zero() {
            start -= 1;
        }

        {
            // Wilkinson Shift.
            let td = diag[end - 1];
            let e = offdiag[end - 1];

            let mut mu = diag[end];

            if td == E::faer_zero() {
                mu = mu.faer_sub(e.faer_abs());
            } else if e != E::faer_zero() {
                let e2 = e.faer_abs2();
                let h = (td.faer_abs2().faer_add(e.faer_abs2())).faer_sqrt();

                let h = if td > E::faer_zero() { h } else { h.faer_neg() };
                if e2 == E::faer_zero() {
                    mu = mu.faer_sub(e.faer_div(td.faer_add(h).faer_div(e)));
                } else {
                    mu = mu.faer_sub(e2.faer_div(td.faer_add(h)));
                }
            }

            let mut x = diag[start].faer_sub(mu);
            let mut z = offdiag[start];

            let mut k = start;
            while k < end && z != E::faer_zero() {
                let rot = JacobiRotation::make_givens(x, z);
                // do T = G' T G
                let sdk = rot.s.faer_mul(diag[k]).faer_add(rot.c.faer_mul(offdiag[k]));
                let dkp1 = rot
                    .s
                    .faer_mul(offdiag[k])
                    .faer_add(rot.c.faer_mul(diag[k + 1]));

                diag[k] = rot
                    .c
                    .faer_mul(rot.c.faer_mul(diag[k]).faer_sub(rot.s.faer_mul(offdiag[k])))
                    .faer_sub(
                        rot.s.faer_mul(
                            rot.c
                                .faer_mul(offdiag[k])
                                .faer_sub(rot.s.faer_mul(diag[k + 1])),
                        ),
                    );
                diag[k + 1] = rot.s.faer_mul(sdk).faer_add(rot.c.faer_mul(dkp1));
                offdiag[k] = rot.c.faer_mul(sdk).faer_sub(rot.s.faer_mul(dkp1));

                if k > start {
                    offdiag[k - 1] = rot.c.faer_mul(offdiag[k - 1]).faer_sub(rot.s.faer_mul(z));
                }

                x = offdiag[k];
                if k < end - 1 {
                    z = rot.s.faer_neg().faer_mul(offdiag[k + 1]);
                    offdiag[k + 1] = rot.c.faer_mul(offdiag[k + 1]);
                }

                // apply the givens rotation to the unit matrix Q = Q * G
                if let Some(u) = u.rb_mut() {
                    unsafe {
                        let x = u.rb().col(k).const_cast();
                        let y = u.rb().col(k + 1).const_cast();
                        rot.apply_on_the_right_in_place_arch(arch, x, y);
                    }
                }
                k += 1;
            }
        }
    }

    for i in 0..n - 1 {
        let mut min_idx = i;
        let mut min_val = diag[i];

        for (k, diag) in diag[i + 1..n].iter().enumerate() {
            let k = k + i + 1;
            if *diag < min_val {
                min_idx = k;
                min_val = *diag;
            }
        }
        if min_idx > i {
            diag.swap(i, min_idx);
            if let Some(u) = u.rb_mut() {
                swap_cols(u, min_idx, i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::Mat;

    #[track_caller]
    fn test_evd(diag: &[f64], offdiag: &[f64]) {
        let n = diag.len();
        let mut u = Mat::from_fn(n, n, |_, _| f64::NAN);

        let s = {
            let mut diag = diag.to_vec();
            let mut offdiag = offdiag.to_vec();

            compute_tridiag_real_evd_qr_algorithm(
                &mut diag,
                &mut offdiag,
                Some(u.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
            );

            Mat::from_fn(n, n, |i, j| if i == j { diag[i] } else { 0.0 })
        };

        let reconstructed = &u * &s * u.transpose();
        for j in 0..n {
            for i in 0..n {
                let target = if i == j {
                    diag[j]
                } else if i == j + 1 {
                    offdiag[j]
                } else if j == i + 1 {
                    offdiag[i]
                } else {
                    0.0
                };

                assert_approx_eq!(reconstructed.read(i, j), target, 1e-14);
            }
        }
    }

    #[test]
    fn test_evd_2_0() {
        let diag = [1.0, 1.0];
        let offdiag = [0.0];
        test_evd(&diag, &offdiag);
    }

    #[test]
    fn test_evd_2_1() {
        let diag = [1.0, 1.0];
        let offdiag = [0.5213289];
        test_evd(&diag, &offdiag);
    }

    #[test]
    fn test_evd_3() {
        let diag = [1.79069356, 1.20930644, 1.0];
        let offdiag = [-4.06813537e-01, 0.0];

        test_evd(&diag, &offdiag);
    }

    #[test]
    fn test_evd_5() {
        let diag = [1.95069537, 2.44845332, 2.56957029, 3.03128102, 1.0];
        let offdiag = [-7.02200909e-01, -1.11661820e+00, -6.81418803e-01, 0.0];
        test_evd(&diag, &offdiag);
    }

    #[test]
    fn test_evd_wilkinson() {
        let diag = [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let offdiag = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        test_evd(&diag, &offdiag);
    }

    #[test]
    fn test_glued_wilkinson() {
        let diag = [
            3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0,
        ];
        let x = 1e-6;
        let offdiag = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, x, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        test_evd(&diag, &offdiag);
    }
}
