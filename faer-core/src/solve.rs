//! Triangular solve module.

use crate::{join_raw, ComplexField, Conj, Conjugate, MatMut, MatRef, Parallelism};
use assert2::{assert, debug_assert};
use reborrow::*;

#[inline(always)]
fn identity<E: Clone>(x: E) -> E {
    x.clone()
}

#[inline(always)]
fn conj<E: ComplexField>(x: E) -> E {
    x.clone().conj()
}

#[inline(always)]
unsafe fn solve_unit_lower_triangular_in_place_base_case_generic_unchecked<E: ComplexField>(
    tril: MatRef<'_, E>,
    rhs: MatMut<'_, E>,
    maybe_conj_lhs: impl Fn(E) -> E,
) {
    let n = tril.nrows();
    match n {
        0 | 1 => (),
        2 => {
            let nl10_div_l11 = maybe_conj_lhs(tril.read_unchecked(1, 0)).neg();

            let [_, x0, _, x1] = rhs.split_at(1, 0);
            let x0 = x0.subrows(0, 1);
            let x1 = x1.subrows(0, 1);

            x0.cwise().zip_unchecked(x1).for_each(|x0, mut x1| {
                x1.write(x1.read().add(&nl10_div_l11.mul(&x0.read())));
            });
        }
        3 => {
            let nl10_div_l11 = maybe_conj_lhs(tril.read_unchecked(1, 0)).neg();
            let nl20_div_l22 = maybe_conj_lhs(tril.read_unchecked(2, 0)).neg();
            let nl21_div_l22 = maybe_conj_lhs(tril.read_unchecked(2, 1)).neg();

            let [_, x0, _, x1_2] = rhs.split_at(1, 0);
            let [_, x1, _, x2] = x1_2.split_at(1, 0);
            let x0 = x0.subrows(0, 1);
            let x1 = x1.subrows(0, 1);
            let x2 = x2.subrows(0, 1);

            x0.cwise()
                .zip_unchecked(x1)
                .zip_unchecked(x2)
                .for_each(|mut x0, mut x1, mut x2| {
                    let y0 = x0.read();
                    let mut y1 = x1.read();
                    let mut y2 = x2.read();
                    y1 = y1.add(&nl10_div_l11.mul(&y0));
                    y2 = y2.add(&nl20_div_l22.mul(&y0)).add(&nl21_div_l22.mul(&y1));
                    x0.write(y0);
                    x1.write(y1);
                    x2.write(y2);
                });
        }
        4 => {
            let nl10_div_l11 = maybe_conj_lhs(tril.read_unchecked(1, 0)).neg();
            let nl20_div_l22 = maybe_conj_lhs(tril.read_unchecked(2, 0)).neg();
            let nl21_div_l22 = maybe_conj_lhs(tril.read_unchecked(2, 1)).neg();
            let nl30_div_l33 = maybe_conj_lhs(tril.read_unchecked(3, 0)).neg();
            let nl31_div_l33 = maybe_conj_lhs(tril.read_unchecked(3, 1)).neg();
            let nl32_div_l33 = maybe_conj_lhs(tril.read_unchecked(3, 2)).neg();

            let [_, x0, _, x1_2_3] = rhs.split_at(1, 0);
            let [_, x1, _, x2_3] = x1_2_3.split_at(1, 0);
            let [_, x2, _, x3] = x2_3.split_at(1, 0);
            let x0 = x0.subrows(0, 1);
            let x1 = x1.subrows(0, 1);
            let x2 = x2.subrows(0, 1);
            let x3 = x3.subrows(0, 1);

            x0.cwise()
                .zip_unchecked(x1)
                .zip_unchecked(x2)
                .zip_unchecked(x3)
                .for_each(|mut x0, mut x1, mut x2, mut x3| {
                    let y0 = x0.read();
                    let mut y1 = x1.read();
                    let mut y2 = x2.read();
                    let mut y3 = x3.read();
                    y1 = y1.add(&nl10_div_l11.mul(&y0));
                    y2 = y2.add(&nl20_div_l22.mul(&y0).add(&nl21_div_l22.mul(&y1)));
                    y3 = (y3.add(&nl30_div_l33.mul(&y0)))
                        .add(&nl31_div_l33.mul(&y1).add(&nl32_div_l33.mul(&y2)));
                    x0.write(y0);
                    x1.write(y1);
                    x2.write(y2);
                    x3.write(y3);
                });
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
unsafe fn solve_lower_triangular_in_place_base_case_generic_unchecked<E: ComplexField>(
    tril: MatRef<'_, E>,
    rhs: MatMut<'_, E>,
    maybe_conj_lhs: impl Fn(E) -> E,
) {
    let n = tril.nrows();
    match n {
        0 => (),
        1 => {
            let inv = maybe_conj_lhs(tril.read_unchecked(0, 0)).inv();
            let x0 = rhs.subrows(0, 1);
            x0.cwise().for_each(|mut x0| x0.write(x0.read().mul(&inv)));
        }
        2 => {
            let l00_inv = maybe_conj_lhs(tril.read_unchecked(0, 0)).inv();
            let l11_inv = maybe_conj_lhs(tril.read_unchecked(1, 1)).inv();
            let nl10_div_l11 = (maybe_conj_lhs(tril.read_unchecked(1, 0)).mul(&l11_inv)).neg();

            let [_, x0, _, x1] = rhs.split_at(1, 0);
            let x0 = x0.subrows(0, 1);
            let x1 = x1.subrows(0, 1);

            x0.cwise().zip_unchecked(x1).for_each(|mut x0, mut x1| {
                x0.write(x0.read().mul(&l00_inv));
                x1.write(x1.read().mul(&l11_inv).add(&nl10_div_l11.mul(&x0.read())));
            });
        }
        3 => {
            let l00_inv = maybe_conj_lhs(tril.read_unchecked(0, 0)).inv();
            let l11_inv = maybe_conj_lhs(tril.read_unchecked(1, 1)).inv();
            let l22_inv = maybe_conj_lhs(tril.read_unchecked(2, 2)).inv();
            let nl10_div_l11 = (maybe_conj_lhs(tril.read_unchecked(1, 0)).mul(&l11_inv)).neg();
            let nl20_div_l22 = (maybe_conj_lhs(tril.read_unchecked(2, 0)).mul(&l22_inv)).neg();
            let nl21_div_l22 = (maybe_conj_lhs(tril.read_unchecked(2, 1)).mul(&l22_inv)).neg();

            let [_, x0, _, x1_2] = rhs.split_at(1, 0);
            let [_, x1, _, x2] = x1_2.split_at(1, 0);
            let x0 = x0.subrows(0, 1);
            let x1 = x1.subrows(0, 1);
            let x2 = x2.subrows(0, 1);

            x0.cwise()
                .zip_unchecked(x1)
                .zip_unchecked(x2)
                .for_each(|mut x0, mut x1, mut x2| {
                    let mut y0 = x0.read();
                    let mut y1 = x1.read();
                    let mut y2 = x2.read();
                    y0 = y0.mul(&l00_inv);
                    y1 = y1.mul(&l11_inv).add(&nl10_div_l11.mul(&y0));
                    y2 = y2
                        .mul(&l22_inv)
                        .add(&nl20_div_l22.mul(&y0))
                        .add(&nl21_div_l22.mul(&y1));
                    x0.write(y0);
                    x1.write(y1);
                    x2.write(y2);
                });
        }
        4 => {
            let l00_inv = maybe_conj_lhs(tril.read_unchecked(0, 0)).inv();
            let l11_inv = maybe_conj_lhs(tril.read_unchecked(1, 1)).inv();
            let l22_inv = maybe_conj_lhs(tril.read_unchecked(2, 2)).inv();
            let l33_inv = maybe_conj_lhs(tril.read_unchecked(3, 3)).inv();
            let nl10_div_l11 = (maybe_conj_lhs(tril.read_unchecked(1, 0)).mul(&l11_inv)).neg();
            let nl20_div_l22 = (maybe_conj_lhs(tril.read_unchecked(2, 0)).mul(&l22_inv)).neg();
            let nl21_div_l22 = (maybe_conj_lhs(tril.read_unchecked(2, 1)).mul(&l22_inv)).neg();
            let nl30_div_l33 = (maybe_conj_lhs(tril.read_unchecked(3, 0)).mul(&l33_inv)).neg();
            let nl31_div_l33 = (maybe_conj_lhs(tril.read_unchecked(3, 1)).mul(&l33_inv)).neg();
            let nl32_div_l33 = (maybe_conj_lhs(tril.read_unchecked(3, 2)).mul(&l33_inv)).neg();

            let [_, x0, _, x1_2_3] = rhs.split_at(1, 0);
            let [_, x1, _, x2_3] = x1_2_3.split_at(1, 0);
            let [_, x2, _, x3] = x2_3.split_at(1, 0);
            let x0 = x0.subrows(0, 1);
            let x1 = x1.subrows(0, 1);
            let x2 = x2.subrows(0, 1);
            let x3 = x3.subrows(0, 1);

            x0.cwise()
                .zip_unchecked(x1)
                .zip_unchecked(x2)
                .zip_unchecked(x3)
                .for_each(|mut x0, mut x1, mut x2, mut x3| {
                    let mut y0 = x0.read();
                    let mut y1 = x1.read();
                    let mut y2 = x2.read();
                    let mut y3 = x3.read();
                    y0 = y0.mul(&l00_inv);
                    y1 = y1.mul(&l11_inv).add(&nl10_div_l11.mul(&y0));
                    y2 = y2
                        .mul(&l22_inv)
                        .add(&nl20_div_l22.mul(&y0).add(&nl21_div_l22.mul(&y1)));
                    y3 = (y3.mul(&l33_inv).add(&nl30_div_l33.mul(&y0)))
                        .add(&nl31_div_l33.mul(&y1).add(&nl32_div_l33.mul(&y2)));
                    x0.write(y0);
                    x1.write(y1);
                    x2.write(y2);
                    x3.write(y3);
                });
        }
        _ => unreachable!(),
    }
}

#[inline]
fn blocksize<E: 'static>(n: usize) -> usize {
    // we want remainder to be a multiple of register size

    let base_rem = n / 2;
    n - if n >= 32 {
        (base_rem + 15) / 16 * 16
    } else if n >= 16 {
        (base_rem + 7) / 8 * 8
    } else if n >= 8 {
        (base_rem + 3) / 4 * 4
    } else {
        base_rem
    }
}

#[inline]
fn recursion_threshold<E: 'static>() -> usize {
    4
}

/// Computes the solution of `Op_lhs(triangular_lower)×X = rhs`, and stores the result in
/// `rhs`.
///
/// `triangular_lower` is interpreted as a lower triangular matrix (diagonal included).
/// Its strictly upper triangular part is not accessed.
///
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
///
/// # Panics
///
///  - Panics if `triangular_lower` is not a square matrix.
///  - Panics if `rhs.nrows() != triangular_lower.ncols()`
///
/// # Example
///
/// ```
/// use faer_core::{
///     mat,
///     mul::triangular::{matmul, BlockStructure},
///     solve::solve_lower_triangular_in_place_with_conj,
///     zipped, Conj, Mat, Parallelism,
/// };
///
/// let m = mat![[1.0, 0.0], [2.0, 3.0]];
/// let rhs = mat![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let mut sol = rhs.clone();
/// solve_lower_triangular_in_place_with_conj(
///     m.as_ref(),
///     Conj::No,
///     sol.as_mut(),
///     Parallelism::None,
/// );
///
/// let mut m_times_sol = Mat::<f64>::zeros(2, 3);
/// matmul(
///     m_times_sol.as_mut(),
///     BlockStructure::Rectangular,
///     m.as_ref(),
///     BlockStructure::TriangularLower,
///     sol.as_ref(),
///     BlockStructure::Rectangular,
///     None,
///     1.0,
///     Parallelism::None,
/// );
///
/// zipped!(m_times_sol.as_ref(), rhs.as_ref())
///     .for_each(|x, target| assert!((x.read() - target.read()).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn solve_lower_triangular_in_place_with_conj<E: ComplexField>(
    triangular_lower: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    assert!(triangular_lower.nrows() == triangular_lower.ncols());
    assert!(rhs.nrows() == triangular_lower.ncols());

    unsafe {
        solve_lower_triangular_in_place_unchecked(triangular_lower, conj_lhs, rhs, parallelism);
    }
}

#[track_caller]
#[inline]
pub fn solve_lower_triangular_in_place<E: ComplexField, TriE: Conjugate<Canonical = E>>(
    triangular_lower: MatRef<'_, TriE>,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let (tri, conj) = triangular_lower.canonicalize();
    solve_lower_triangular_in_place_with_conj(tri, conj, rhs, parallelism)
}

/// Computes the solution of `Op_lhs(triangular_upper)×X = rhs`, and stores the result in
/// `rhs`.
///
/// `triangular_upper` is interpreted as a upper triangular matrix (diagonal included).
/// Its strictly lower triangular part is not accessed.
///
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
///
/// # Panics
///
///  - Panics if `triangular_upper` is not a square matrix.
///  - Panics if `rhs.nrows() != triangular_lower.ncols()`
///
/// # Example
///
/// ```
/// use faer_core::{
///     mat,
///     mul::triangular::{matmul, BlockStructure},
///     solve::solve_upper_triangular_in_place_with_conj,
///     zipped, Conj, Mat, Parallelism,
/// };
///
/// let m = mat![[1.0, 2.0], [0.0, 3.0]];
/// let rhs = mat![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let mut sol = rhs.clone();
/// solve_upper_triangular_in_place_with_conj(
///     m.as_ref(),
///     Conj::No,
///     sol.as_mut(),
///     Parallelism::None,
/// );
///
/// let mut m_times_sol = Mat::<f64>::zeros(2, 3);
/// matmul(
///     m_times_sol.as_mut(),
///     BlockStructure::Rectangular,
///     m.as_ref(),
///     BlockStructure::TriangularUpper,
///     sol.as_ref(),
///     BlockStructure::Rectangular,
///     None,
///     1.0,
///     Parallelism::None,
/// );
///
/// zipped!(m_times_sol.as_ref(), rhs.as_ref())
///     .for_each(|x, target| assert!((x.read() - target.read()).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn solve_upper_triangular_in_place_with_conj<E: ComplexField>(
    triangular_upper: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    assert!(triangular_upper.nrows() == triangular_upper.ncols());
    assert!(rhs.nrows() == triangular_upper.ncols());

    unsafe {
        solve_upper_triangular_in_place_unchecked(triangular_upper, conj_lhs, rhs, parallelism);
    }
}

#[track_caller]
#[inline]
pub fn solve_upper_triangular_in_place<E: ComplexField, TriE: Conjugate<Canonical = E>>(
    triangular_upper: MatRef<'_, TriE>,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let (tri, conj) = triangular_upper.canonicalize();
    solve_upper_triangular_in_place_with_conj(tri, conj, rhs, parallelism)
}

/// Computes the solution of `Op_lhs(triangular_lower)×X = rhs`, and stores the result in
/// `rhs`.
///
/// `triangular_lower` is interpreted as a lower triangular matrix, and its diagonal elements are
/// implicitly considered to be `1.0`. Its upper triangular part is not accessed.
///
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
///
/// # Panics
///
///  - Panics if `triangular_lower` is not a square matrix.
///  - Panics if `rhs.nrows() != triangular_lower.ncols()`
///
/// # Example
///
/// ```
/// use faer_core::{
///     mat,
///     mul::triangular::{matmul, BlockStructure},
///     solve::solve_unit_lower_triangular_in_place_with_conj,
///     zipped, Conj, Mat, Parallelism,
/// };
///
/// let m = mat![[0.0, 0.0], [2.0, 0.0]];
/// let rhs = mat![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let mut sol = rhs.clone();
/// solve_unit_lower_triangular_in_place_with_conj(
///     m.as_ref(),
///     Conj::No,
///     sol.as_mut(),
///     Parallelism::None,
/// );
///
/// let mut m_times_sol = Mat::<f64>::zeros(2, 3);
/// matmul(
///     m_times_sol.as_mut(),
///     BlockStructure::Rectangular,
///     m.as_ref(),
///     BlockStructure::UnitTriangularLower,
///     sol.as_ref(),
///     BlockStructure::Rectangular,
///     None,
///     1.0,
///     Parallelism::None,
/// );
///
/// zipped!(m_times_sol.as_ref(), rhs.as_ref())
///     .for_each(|x, target| assert!((x.read() - target.read()).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn solve_unit_lower_triangular_in_place_with_conj<E: ComplexField>(
    triangular_lower: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    assert!(triangular_lower.nrows() == triangular_lower.ncols());
    assert!(rhs.nrows() == triangular_lower.ncols());

    unsafe {
        solve_unit_lower_triangular_in_place_unchecked(
            triangular_lower,
            conj_lhs,
            rhs,
            parallelism,
        );
    }
}

#[track_caller]
#[inline]
pub fn solve_unit_lower_triangular_in_place<E: ComplexField, TriE: Conjugate<Canonical = E>>(
    triangular_lower: MatRef<'_, TriE>,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let (tri, conj) = triangular_lower.canonicalize();
    solve_unit_lower_triangular_in_place_with_conj(tri, conj, rhs, parallelism)
}

/// Computes the solution of `Op_lhs(triangular_upper)×X = rhs`, and stores the result in
/// `rhs`.
///
/// `triangular_upper` is interpreted as a upper triangular matrix, and its diagonal elements are
/// implicitly considered to be `1.0`. Its lower triangular part is not accessed.
///
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
///
/// # Panics
///
///  - Panics if `triangular_upper` is not a square matrix.
///  - Panics if `rhs.nrows() != triangular_lower.ncols()`
///
/// ```
/// use faer_core::{
///     mat,
///     mul::triangular::{matmul, BlockStructure},
///     solve::solve_unit_upper_triangular_in_place_with_conj,
///     zipped, Conj, Mat, Parallelism,
/// };
///
/// let m = mat![[0.0, 2.0], [0.0, 0.0]];
/// let rhs = mat![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let mut sol = rhs.clone();
/// solve_unit_upper_triangular_in_place_with_conj(
///     m.as_ref(),
///     Conj::No,
///     sol.as_mut(),
///     Parallelism::None,
/// );
///
/// let mut m_times_sol = Mat::<f64>::zeros(2, 3);
/// matmul(
///     m_times_sol.as_mut(),
///     BlockStructure::Rectangular,
///     m.as_ref(),
///     BlockStructure::UnitTriangularUpper,
///     sol.as_ref(),
///     BlockStructure::Rectangular,
///     None,
///     1.0,
///     Parallelism::None,
/// );
///
/// zipped!(m_times_sol.as_ref(), rhs.as_ref())
///     .for_each(|x, target| assert!((x.read() - target.read()).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn solve_unit_upper_triangular_in_place_with_conj<E: ComplexField>(
    triangular_upper: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    assert!(triangular_upper.nrows() == triangular_upper.ncols());
    assert!(rhs.nrows() == triangular_upper.ncols());

    unsafe {
        solve_unit_upper_triangular_in_place_unchecked(
            triangular_upper,
            conj_lhs,
            rhs,
            parallelism,
        );
    }
}

#[track_caller]
#[inline]
pub fn solve_unit_upper_triangular_in_place<E: ComplexField, TriE: Conjugate<Canonical = E>>(
    triangular_upper: MatRef<'_, TriE>,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let (tri, conj) = triangular_upper.canonicalize();
    solve_unit_upper_triangular_in_place_with_conj(tri, conj, rhs, parallelism)
}

/// # Safety
///
/// Same as [`solve_unit_lower_triangular_in_place`], except that panics become undefined behavior.
///
/// # Example
///
/// See [`solve_unit_lower_triangular_in_place`].
unsafe fn solve_unit_lower_triangular_in_place_unchecked<E: ComplexField>(
    tril: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let n = tril.nrows();
    let k = rhs.ncols();

    if k > 64 && n <= 128 {
        let [_, _, rhs_left, rhs_right] = rhs.split_at(0, k / 2);
        join_raw(
            |_| {
                solve_unit_lower_triangular_in_place_unchecked(
                    tril,
                    conj_lhs,
                    rhs_left,
                    parallelism,
                )
            },
            |_| {
                solve_unit_lower_triangular_in_place_unchecked(
                    tril,
                    conj_lhs,
                    rhs_right,
                    parallelism,
                )
            },
            parallelism,
        );
        return;
    }

    debug_assert!(tril.nrows() == tril.ncols());
    debug_assert!(rhs.nrows() == tril.ncols());

    if n <= recursion_threshold::<E>() {
        pulp::Arch::new().dispatch(
            #[inline(always)]
            || match conj_lhs {
                Conj::Yes => solve_unit_lower_triangular_in_place_base_case_generic_unchecked(
                    tril, rhs, conj,
                ),
                Conj::No => solve_unit_lower_triangular_in_place_base_case_generic_unchecked(
                    tril, rhs, identity,
                ),
            },
        );
        return;
    }

    let bs = blocksize::<E>(n);

    let [tril_top_left, _, tril_bot_left, tril_bot_right] = tril.split_at(bs, bs);
    let [_, mut rhs_top, _, mut rhs_bot] = rhs.split_at(bs, 0);

    //       (A00    )   X0         (B0)
    // ConjA?(A10 A11)   X1 = ConjB?(B1)
    //
    //
    // 1. ConjA?(A00) X0 = ConjB?(B0)
    //
    // 2. ConjA?(A10) X0 + ConjA?(A11) X1 = ConjB?(B1)
    // => ConjA?(A11) X1 = ConjB?(B1) - ConjA?(A10) X0

    solve_unit_lower_triangular_in_place_unchecked(
        tril_top_left,
        conj_lhs,
        rhs_top.rb_mut(),
        parallelism,
    );

    crate::mul::matmul_with_conj(
        rhs_bot.rb_mut(),
        tril_bot_left,
        conj_lhs,
        rhs_top.into_const(),
        Conj::No,
        Some(E::one()),
        E::one().neg(),
        parallelism,
    );

    solve_unit_lower_triangular_in_place_unchecked(tril_bot_right, conj_lhs, rhs_bot, parallelism);
}

/// # Safety
///
/// Same as [`solve_unit_upper_triangular_in_place`], except that panics become undefined behavior.
///
/// # Example
///
/// See [`solve_unit_upper_triangular_in_place`].
#[inline]
unsafe fn solve_unit_upper_triangular_in_place_unchecked<E: ComplexField>(
    triu: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    solve_unit_lower_triangular_in_place_unchecked(
        triu.reverse_rows_and_cols(),
        conj_lhs,
        rhs.reverse_rows(),
        parallelism,
    );
}

/// # Safety
///
/// Same as [`solve_lower_triangular_in_place`], except that panics become undefined behavior.
///
/// # Example
///
/// See [`solve_lower_triangular_in_place`].
unsafe fn solve_lower_triangular_in_place_unchecked<E: ComplexField>(
    tril: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let n = tril.nrows();
    let k = rhs.ncols();

    if k > 64 && n <= 128 {
        let [_, _, rhs_left, rhs_right] = rhs.split_at(0, k / 2);
        join_raw(
            |_| solve_lower_triangular_in_place_unchecked(tril, conj_lhs, rhs_left, parallelism),
            |_| solve_lower_triangular_in_place_unchecked(tril, conj_lhs, rhs_right, parallelism),
            parallelism,
        );
        return;
    }

    debug_assert!(tril.nrows() == tril.ncols());
    debug_assert!(rhs.nrows() == tril.ncols());

    let n = tril.nrows();

    if n <= recursion_threshold::<E>() {
        pulp::Arch::new().dispatch(
            #[inline(always)]
            || match conj_lhs {
                Conj::Yes => {
                    solve_lower_triangular_in_place_base_case_generic_unchecked(tril, rhs, conj)
                }
                Conj::No => {
                    solve_lower_triangular_in_place_base_case_generic_unchecked(tril, rhs, identity)
                }
            },
        );
        return;
    }

    let bs = blocksize::<E>(n);

    let [tril_top_left, _, tril_bot_left, tril_bot_right] = tril.split_at(bs, bs);
    let [_, mut rhs_top, _, mut rhs_bot] = rhs.split_at(bs, 0);

    solve_lower_triangular_in_place_unchecked(
        tril_top_left,
        conj_lhs,
        rhs_top.rb_mut(),
        parallelism,
    );

    crate::mul::matmul_with_conj(
        rhs_bot.rb_mut(),
        tril_bot_left,
        conj_lhs,
        rhs_top.into_const(),
        Conj::No,
        Some(E::one()),
        E::one().neg(),
        parallelism,
    );

    solve_lower_triangular_in_place_unchecked(tril_bot_right, conj_lhs, rhs_bot, parallelism);
}

/// # Safety
///
/// Same as [`solve_upper_triangular_in_place`], except that panics become undefined behavior.
///
/// # Example
///
/// See [`solve_upper_triangular_in_place`].
#[inline]
unsafe fn solve_upper_triangular_in_place_unchecked<E: ComplexField>(
    triu: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    solve_lower_triangular_in_place_unchecked(
        triu.reverse_rows_and_cols(),
        conj_lhs,
        rhs.reverse_rows(),
        parallelism,
    );
}
