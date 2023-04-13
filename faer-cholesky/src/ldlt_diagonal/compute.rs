use assert2::{assert, debug_assert};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    mul::triangular::BlockStructure, solve, temp_mat_req, temp_mat_uninit, zipped, ComplexField,
    Conj, Entity, MatMut, Parallelism,
};
use reborrow::*;

fn cholesky_in_place_left_looking_impl<E: ComplexField>(
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let mut matrix = matrix;
    let _ = parallelism;

    debug_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );

    let n = matrix.nrows();

    match n {
        0 | 1 => return,
        _ => (),
    };

    let mut idx = 0;
    loop {
        let block_size = 1;

        // we split L/D rows/cols into 3 sections each
        //     ┌             ┐
        //     | L00         |
        // L = | L10 A11     |
        //     | L20 A21 A22 |
        //     └             ┘
        //     ┌          ┐
        //     | D0       |
        // D = |    D1    |
        //     |       D2 |
        //     └          ┘
        //
        // we already computed L00, L10, L20, and D0. we now compute L11, L21, and D1

        let [top_left, top_right, bottom_left, bottom_right] = matrix.rb_mut().split_at(idx, idx);
        let l00 = top_left.into_const();
        let d0 = l00.diagonal();
        let [_, l10, _, l20] = bottom_left.into_const().split_at(block_size, 0);
        let [mut a11, _, a21, _] = bottom_right.split_at(block_size, block_size);

        // reserve space for L10×D0
        let mut l10xd0 = top_right.submatrix(0, 0, idx, block_size).transpose();

        zipped!(l10xd0.rb_mut(), l10, d0.transpose())
            .for_each(|mut dst, src, factor| dst.write(src.read().mul(&factor.read())));

        let l10xd0 = l10xd0.into_const();

        a11.write(
            0,
            0,
            a11.read(0, 0)
                .sub(&faer_core::mul::inner_prod::inner_prod_with_conj(
                    l10xd0.row(0).transpose(),
                    Conj::Yes,
                    l10.row(0).transpose(),
                    Conj::No,
                )),
        );

        if idx + block_size == n {
            break;
        }

        let ld11 = a11.into_const();
        let l11 = ld11;

        let mut a21 = a21.col(0);

        for j in 0..idx {
            let l20_col = l20.col(j);
            let l10_conj = l10xd0.read(0, j).conj();
            zipped!(a21.rb_mut(), l20_col)
                .for_each(|mut dst, src| dst.write(dst.read().sub(&src.read().mul(&l10_conj))));
        }

        let r = l11.read(0, 0).real().inv();
        zipped!(a21.rb_mut()).for_each(|mut x| x.write(x.read().scale_real(&r)));

        idx += block_size;
    }
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct LdltDiagParams {}

/// Computes the size and alignment of required workspace for performing a Cholesky
/// decomposition with partial pivoting.
pub fn raw_cholesky_in_place_req<E: Entity>(
    dim: usize,
    parallelism: Parallelism,
    params: LdltDiagParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    let _ = params;
    temp_mat_req::<E>(dim, dim)
}

fn cholesky_in_place_impl<E: ComplexField>(
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    // right looking cholesky

    debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    let n = matrix.nrows();
    if n < 32 {
        cholesky_in_place_left_looking_impl(matrix, parallelism);
    } else {
        let block_size = <usize as Ord>::min(n / 2, 128);
        let rem = n - block_size;
        let [mut l00, _, mut a10, mut a11] = matrix.rb_mut().split_at(block_size, block_size);

        cholesky_in_place_impl(l00.rb_mut(), parallelism, stack.rb_mut());

        let l00 = l00.into_const();
        let d0 = l00.diagonal();

        solve::solve_unit_lower_triangular_in_place(
            l00.conjugate(),
            a10.rb_mut().transpose(),
            parallelism,
        );

        {
            // reserve space for L10×D0
            let (mut l10xd0, _) = unsafe { temp_mat_uninit(rem, block_size, stack.rb_mut()) };
            let mut l10xd0 = l10xd0.as_mut();

            for j in 0..block_size {
                let l10xd0_col = l10xd0.rb_mut().col(j);
                let a10_col = a10.rb_mut().col(j);
                let d0_elem = d0.read(j, 0);

                let d0_elem_inv = d0_elem.inv();

                zipped!(l10xd0_col, a10_col).for_each(|mut l10xd0_elem, mut a10_elem| {
                    let a10_elem_read = a10_elem.read();
                    a10_elem.write(a10_elem_read.mul(&d0_elem_inv));
                    l10xd0_elem.write(a10_elem_read);
                });
            }

            faer_core::mul::triangular::matmul(
                a11.rb_mut(),
                BlockStructure::TriangularLower,
                a10.into_const(),
                BlockStructure::Rectangular,
                l10xd0.adjoint().into_const(),
                BlockStructure::Rectangular,
                Some(E::one()),
                E::one().neg(),
                parallelism,
            );
        }

        cholesky_in_place_impl(a11, parallelism, stack);
    }
}

/// Computes the Cholesky factors $L$ and $D$ of the input matrix such that $L$ is strictly lower
/// triangular, $D$ is real-valued diagonal, and
/// $$LDL^H = A.$$
///
/// The result is stored back in the same matrix.
///
/// The input matrix is interpreted as symmetric and only the lower triangular part is read.
///
/// The matrix $L$ is stored in the strictly lower triangular part of the input matrix, and the
/// diagonal elements of $D$ are stored on the diagonal.
///
/// The strictly upper triangular part of the matrix is clobbered and may be filled with garbage
/// values.
///
/// # Warning
///
/// The Cholesky decomposition may have poor numerical stability properties when used with non
/// positive definite matrices. In the general case, it is recommended to first permute the matrix
/// using [`crate::compute_cholesky_permutation`] and
/// [`permute_rows_and_cols_symmetric`](faer_core::permutation::permute_rows_and_cols_symmetric_lower).
///
/// # Panics
///
/// Panics if the input matrix is not square.
#[track_caller]
#[inline]
pub fn raw_cholesky_in_place<E: ComplexField>(
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: LdltDiagParams,
) {
    let _ = params;
    assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );
    cholesky_in_place_impl(matrix, parallelism, stack)
}
