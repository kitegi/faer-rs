use super::CholeskyError;
use crate::ldlt_diagonal::compute::RankUpdate;
#[cfg(feature = "std")]
use assert2::{assert, debug_assert};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    mul::triangular::BlockStructure, parallelism_degree, solve, zipped, ComplexField, Entity,
    MatMut, Parallelism, SimdCtx,
};
use reborrow::*;

fn cholesky_in_place_left_looking_impl<E: ComplexField>(
    matrix: MatMut<'_, E>,
    regularization: LltRegularization<E>,
    parallelism: Parallelism,
    params: LltParams,
) -> Result<usize, CholeskyError> {
    let mut matrix = matrix;
    let _ = params;
    let _ = parallelism;
    assert_eq!(matrix.ncols(), matrix.nrows());

    let n = matrix.nrows();

    if n == 0 {
        return Ok(0);
    }

    let mut idx = 0;
    let arch = E::Simd::default();
    let eps = regularization
        .dynamic_regularization_epsilon
        .faer_real()
        .faer_abs();
    let delta = regularization
        .dynamic_regularization_delta
        .faer_real()
        .faer_abs();
    let has_eps = delta > E::Real::faer_zero();
    let mut dynamic_regularization_count = 0usize;
    loop {
        let block_size = 1;

        let [_, _, bottom_left, bottom_right] = matrix.rb_mut().split_at(idx, idx);
        let [_, l10, _, l20] = bottom_left.into_const().split_at(block_size, 0);
        let [mut a11, _, a21, _] = bottom_right.split_at(block_size, block_size);

        let l10 = l10.row(0);
        let mut a21 = a21.col(0);

        //
        //      L00
        // A =  L10  A11
        //      L20  A21  A22
        //
        // the first column block is already computed
        // we now compute A11 and A21
        //
        // L00           L00^H L10^H L20^H
        // L10 L11             L11^H L21^H
        // L20 L21 L22 ×             L22^H
        //
        //
        // L00×L00^H
        // L10×L00^H  L10×L10^H + L11×L11^H
        // L20×L00^H  L20×L10^H + L21×L11^H  L20×L20^H + L21×L21^H + L22×L22^H

        // A11 -= L10 × L10^H
        let mut dot = E::Real::faer_zero();
        for j in 0..idx {
            dot = dot.faer_add(l10.read(0, j).faer_abs2());
        }
        a11.write(
            0,
            0,
            E::faer_from_real(a11.read(0, 0).faer_real().faer_sub(dot)),
        );

        let mut real = a11.read(0, 0).faer_real();
        if has_eps && real >= E::Real::faer_zero() && real <= eps {
            real = delta;
            dynamic_regularization_count += 1;
        }

        if real > E::Real::faer_zero() {
            a11.write(0, 0, E::faer_from_real(real.faer_sqrt()));
        } else {
            return Err(CholeskyError);
        };

        if idx + block_size == n {
            break;
        }

        let l11 = a11.read(0, 0);

        // A21 -= L20 × L10^H
        if a21.row_stride() == 1 {
            arch.dispatch(RankUpdate {
                a21: a21.rb_mut(),
                l20,
                l10,
            });
        } else {
            for j in 0..idx {
                let l20_col = l20.col(j);
                let l10_conj = l10.read(0, j).faer_conj();

                zipped!(a21.rb_mut(), l20_col).for_each(|mut dst, src| {
                    dst.write(dst.read().faer_sub(src.read().faer_mul(l10_conj)))
                });
            }
        }

        // A21 is now L21×L11^H
        // find L21
        //
        // conj(L11) L21^T = A21^T

        let r = l11.faer_real().faer_inv();
        zipped!(a21.rb_mut()).for_each(|mut x| x.write(x.read().faer_scale_real(r)));

        idx += block_size;
    }
    Ok(dynamic_regularization_count)
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct LltParams {}

#[derive(Copy, Clone, Debug)]
pub struct LltRegularization<E: ComplexField> {
    pub dynamic_regularization_delta: E::Real,
    pub dynamic_regularization_epsilon: E::Real,
}

impl<E: ComplexField> Default for LltRegularization<E> {
    fn default() -> Self {
        Self {
            dynamic_regularization_delta: E::Real::faer_zero(),
            dynamic_regularization_epsilon: E::Real::faer_zero(),
        }
    }
}

/// Computes the size and alignment of required workspace for performing a Cholesky
/// decomposition with partial pivoting.
pub fn cholesky_in_place_req<E: Entity>(
    dim: usize,
    parallelism: Parallelism,
    params: LltParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = dim;
    let _ = parallelism;
    let _ = params;
    Ok(StackReq::default())
}

// uses an out parameter for tail recursion
fn cholesky_in_place_impl<E: ComplexField>(
    count: &mut usize,
    matrix: MatMut<'_, E>,
    regularization: LltRegularization<E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: LltParams,
) -> Result<(), CholeskyError> {
    // right looking cholesky

    debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    let n = matrix.nrows();
    if n < 32 {
        *count += cholesky_in_place_left_looking_impl(matrix, regularization, parallelism, params)?;
        Ok(())
    } else {
        let block_size = Ord::min(n / 2, 128 * parallelism_degree(parallelism));
        let [mut l00, _, mut a10, mut a11] = matrix.rb_mut().split_at(block_size, block_size);

        cholesky_in_place_impl(
            count,
            l00.rb_mut(),
            regularization,
            parallelism,
            stack.rb_mut(),
            params,
        )?;

        let l00 = l00.into_const();

        solve::solve_lower_triangular_in_place(
            l00.conjugate(),
            a10.rb_mut().transpose(),
            parallelism,
        );

        faer_core::mul::triangular::matmul(
            a11.rb_mut(),
            BlockStructure::TriangularLower,
            a10.rb(),
            BlockStructure::Rectangular,
            a10.rb().adjoint(),
            BlockStructure::Rectangular,
            Some(E::faer_one()),
            E::faer_one().faer_neg(),
            parallelism,
        );

        cholesky_in_place_impl(count, a11, regularization, parallelism, stack, params)
    }
}

/// Computes the Cholesky factor $L$ of a hermitian positive definite input matrix $A$ such that
/// $L$ is lower triangular, and
/// $$LL^H == A.$$
///
/// The result is stored back in the lower half of the same matrix, or an error is returned if the
/// matrix is not positive definite.
///
/// The input matrix is interpreted as symmetric and only the lower triangular part is read.
///
/// The strictly upper triangular part of the matrix is clobbered and may be filled with garbage
/// values.
///
/// # Panics
///
/// Panics if the input matrix is not square.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`cholesky_in_place_req`]).
#[track_caller]
#[inline]
pub fn cholesky_in_place<E: ComplexField>(
    matrix: MatMut<'_, E>,
    regularization: LltRegularization<E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: LltParams,
) -> Result<usize, CholeskyError> {
    let _ = params;
    assert!(matrix.ncols() == matrix.nrows());
    #[cfg(feature = "perf-warn")]
    if matrix.row_stride().unsigned_abs() != 1 && faer_core::__perf_warn!(CHOLESKY_WARN) {
        if matrix.col_stride().unsigned_abs() == 1 {
            log::warn!(target: "faer_perf", "LLT prefers column-major matrix. Found row-major matrix.");
        } else {
            log::warn!(target: "faer_perf", "LLT prefers column-major matrix. Found matrix with generic strides.");
        }
    }

    let mut count = 0;
    cholesky_in_place_impl(
        &mut count,
        matrix,
        regularization,
        parallelism,
        stack,
        params,
    )?;
    Ok(count)
}
