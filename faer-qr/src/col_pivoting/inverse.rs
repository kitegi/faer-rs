use assert2::assert as fancy_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    householder::apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj,
    inverse::invert_upper_triangular,
    permutation::{permute_cols_in_place_req, permute_rows_in_place, PermutationRef},
    temp_mat_req, temp_mat_uninit, zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism,
};
use reborrow::*;

/// Computes the inverse of a matrix, given its QR decomposition with column pivoting,
/// and stores the result in `dst`.
///
/// # Panics
///
/// - Panics if `qr_factors` is not a square matrix.
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `col_perm` doesn't have the same dimension as `qr_factors`.
/// - Panics if `dst` doesn't have the same shape as `qr_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see [`invert_req`]).
#[track_caller]
pub fn invert<E: ComplexField>(
    dst: MatMut<'_, E>,
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    col_perm: PermutationRef<'_>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    fancy_assert!(qr_factors.nrows() == qr_factors.ncols());
    fancy_assert!((dst.nrows(), dst.ncols()) == (qr_factors.nrows(), qr_factors.ncols()));
    fancy_assert!(
        householder_factor.ncols() == <usize as Ord>::min(qr_factors.nrows(), qr_factors.ncols())
    );
    fancy_assert!(householder_factor.nrows() > 0);

    let mut dst = dst;
    let mut stack = stack;

    // invert R
    invert_upper_triangular(dst.rb_mut(), qr_factors, parallelism);

    // zero bottom part
    zipped!(dst.rb_mut())
        .for_each_triangular_lower(faer_core::zip::Diag::Skip, |mut dst| dst.write(E::zero()));

    apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj(
        qr_factors,
        householder_factor,
        Conj::Yes,
        dst.rb_mut(),
        parallelism,
        stack.rb_mut(),
    );

    permute_rows_in_place(dst.rb_mut(), col_perm.inverse(), stack)
}

/// Computes the inverse of a matrix, given its QR decomposition with column pivoting,
/// and stores the result in `qr_factors`.
///
/// # Panics
///
/// - Panics if `qr_factors` is not a square matrix.
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `col_perm` doesn't have the same dimension as `qr_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see [`invert_in_place_req`]).
#[track_caller]
pub fn invert_in_place<E: ComplexField>(
    qr_factors: MatMut<'_, E>,
    householder_factor: MatRef<'_, E>,
    col_perm: PermutationRef<'_>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let (mut dst, stack) =
        unsafe { temp_mat_uninit::<E>(qr_factors.nrows(), qr_factors.ncols(), stack) };
    let mut dst = dst.as_mut();

    invert(
        dst.rb_mut(),
        qr_factors.rb(),
        householder_factor,
        col_perm,
        parallelism,
        stack,
    );

    zipped!(qr_factors, dst.rb()).for_each(|mut dst, src| dst.write(src.read()));
}

/// Computes the size and alignment of required workspace for computing the inverse of a
/// matrix out of place, given its QR decomposition with column pivoting.
pub fn invert_req<E: Entity>(
    qr_nrows: usize,
    qr_ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = qr_nrows;
    let _ = parallelism;
    StackReq::try_any_of([
        temp_mat_req::<E>(blocksize, qr_ncols)?,
        permute_cols_in_place_req::<E>(qr_nrows, qr_ncols)?,
    ])
}

/// Computes the size and alignment of required workspace for computing the inverse of a
/// matrix in place, given its QR decomposition with column pivoting.
pub fn invert_in_place_req<E: Entity>(
    qr_nrows: usize,
    qr_ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        temp_mat_req::<E>(qr_nrows, qr_ncols)?,
        invert_req::<E>(qr_nrows, qr_ncols, blocksize, parallelism)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::col_pivoting::compute::{qr_in_place, qr_in_place_req, recommended_blocksize};
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{c64, mul::matmul, Mat};
    use rand::prelude::*;
    use std::cell::RefCell;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req.unwrap()))
        };
    }

    type E = c64;

    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
    }

    fn random_value() -> E {
        RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let rng = &mut *rng;
            E::new(rng.gen(), rng.gen())
        })
    }

    #[test]
    fn test_invert() {
        for n in [31, 32, 48, 65] {
            let mat = Mat::with_dims(n, n, |_, _| random_value());
            let blocksize = recommended_blocksize::<E>(n, n);
            let mut qr = mat.clone();
            let mut householder_factor = Mat::zeros(blocksize, n);

            let parallelism = faer_core::Parallelism::Rayon(0);
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            let (_, perm) = qr_in_place(
                qr.as_mut(),
                householder_factor.as_mut(),
                &mut perm,
                &mut perm_inv,
                parallelism,
                make_stack!(qr_in_place_req::<E>(
                    n,
                    n,
                    blocksize,
                    parallelism,
                    Default::default()
                )),
                Default::default(),
            );

            let mut inv = Mat::zeros(n, n);
            invert(
                inv.as_mut(),
                qr.as_ref(),
                householder_factor.as_ref(),
                perm.rb(),
                parallelism,
                make_stack!(invert_req::<E>(n, n, blocksize, parallelism)),
            );

            let mut eye = Mat::zeros(n, n);
            matmul(
                eye.as_mut(),
                inv.as_ref(),
                mat.as_ref(),
                None,
                E::one(),
                Parallelism::None,
            );

            for i in 0..n {
                for j in 0..n {
                    let target = if i == j { E::one() } else { E::zero() };
                    assert_approx_eq!(eye.read(i, j), target);
                }
            }
        }
    }
}
