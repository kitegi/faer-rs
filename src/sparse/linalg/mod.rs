use crate::{
    perm::PermRef,
    sparse::{windows2, *},
    Entity, Side,
};
use bytemuck::Pod;
use dyn_stack::PodStack;

const CHOLESKY_SUPERNODAL_RATIO_FACTOR: f64 = 40.0;
const QR_SUPERNODAL_RATIO_FACTOR: f64 = 40.0;
const LU_SUPERNODAL_RATIO_FACTOR: f64 = 40.0;

/// Tuning parameters for the supernodal factorizations.
#[derive(Copy, Clone, Debug)]
pub struct SymbolicSupernodalParams<'a> {
    /// Supernode relaxation thresholds.
    ///
    /// Let `n` be the total number of columns in two adjacent supernodes.
    /// Let `z` be the fraction of zero entries in the two supernodes if they
    /// are merged (z includes zero entries from prior amalgamations). The
    /// two supernodes are merged if:
    ///
    /// `(n <= relax[0].0 && z < relax[0].1) || (n <= relax[1].0 && z < relax[1].1) || ...`
    pub relax: Option<&'a [(usize, f64)]>,
}

impl Default for SymbolicSupernodalParams<'_> {
    #[inline]
    fn default() -> Self {
        Self {
            relax: Some(&[(4, 1.0), (16, 0.8), (48, 0.1), (usize::MAX, 0.05)]),
        }
    }
}

/// Nonnegative threshold controlling when the supernodal factorization is used.
///
/// Increasing it makes it more likely for the simplicial factorization to be used,
/// while decreasing it makes it more likely for the supernodal factorization to be used.
///
/// A value of `1.0` is the default.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SupernodalThreshold(pub f64);

impl Default for SupernodalThreshold {
    #[inline]
    fn default() -> Self {
        Self(1.0)
    }
}

impl SupernodalThreshold {
    /// Simplicial factorization is always selected.
    pub const FORCE_SIMPLICIAL: Self = Self(f64::INFINITY);
    /// Supernodal factorization is always selected.
    pub const FORCE_SUPERNODAL: Self = Self(0.0);
    /// Determine automatically which variant to select.
    pub const AUTO: Self = Self(1.0);
}

use super::utils::{ghost_permute_hermitian_unsorted, ghost_permute_hermitian_unsorted_symbolic};

#[deprecated = "moved to faer::sparse::FaerError"]
pub use crate::sparse::FaerError;
#[deprecated = "moved to faer::Index"]
pub use crate::Index;
#[deprecated = "moved to faer::SignedIndex"]
pub use crate::SignedIndex;

#[allow(unused_macros)]
macro_rules! shadow {
    ($id: ident) => {
        let $id = {
            struct Shadowed;
            impl ::core::ops::Drop for Shadowed {
                fn drop(&mut self) {}
            }
            Shadowed
        };
        ::core::mem::drop($id);
    };
}

macro_rules! impl_copy {
    (< $($lt_param: lifetime),* >< $($ty_param: ident $(: $tt: tt)?),* > <$ty: ty>) => {
        impl<$($lt_param,)* $($ty_param $(: $tt)?,)*> Copy for $ty {}
        impl<$($lt_param,)* $($ty_param $(: $tt)?,)*> Clone for $ty {
            #[inline(always)]
            fn clone(&self) -> Self {
                *self
            }
        }
    };
}

#[inline]
#[track_caller]
fn try_zeroed<I: Pod>(n: usize) -> Result<alloc::vec::Vec<I>, FaerError> {
    let mut v = alloc::vec::Vec::new();
    v.try_reserve_exact(n).map_err(nomem)?;
    unsafe {
        core::ptr::write_bytes::<I>(v.as_mut_ptr(), 0u8, n);
        v.set_len(n);
    }
    Ok(v)
}

#[inline]
#[track_caller]
fn try_collect<I: IntoIterator>(iter: I) -> Result<alloc::vec::Vec<I::Item>, FaerError> {
    let iter = iter.into_iter();
    let mut v = alloc::vec::Vec::new();
    v.try_reserve_exact(iter.size_hint().0).map_err(nomem)?;
    v.extend(iter);
    Ok(v)
}

#[inline]
fn nomem<T>(_: T) -> FaerError {
    FaerError::OutOfMemory
}

fn make_raw_req<E: Entity>(size: usize) -> Result<StackReq, SizeOverflow> {
    let req = Ok(StackReq::empty());
    let additional = StackReq::try_new::<E::Unit>(size)?;
    let (req, _) = E::faer_map_with_context(req, E::UNIT, &mut {
        #[inline(always)]
        |req, ()| {
            let req = match req {
                Ok(req) => req.try_and(additional),
                _ => Err(SizeOverflow),
            };
            (req, ())
        }
    });
    req
}

fn make_raw<E: Entity>(size: usize, stack: &mut PodStack) -> (SliceGroupMut<'_, E>, &mut PodStack) {
    let (stack, array) = E::faer_map_with_context(stack, E::UNIT, &mut {
        #[inline(always)]
        |stack, ()| {
            let (alloc, stack) = stack.make_raw::<E::Unit>(size);
            (stack, alloc)
        }
    });
    (SliceGroupMut::new(array), stack)
}

#[cfg(test)]
macro_rules! monomorphize_test {
    ($name: ident) => {
        monomorphize_test!($name, u32);
        monomorphize_test!($name, u64);
    };

    ($name: ident, $ty: ident) => {
        paste::paste! {
            #[test]
            fn [<$name _ $ty>]() {
                $name::<$ty>();
            }
        }
    };
}

/// Solving sparse triangular linear systems with a dense right-hand-side.
pub mod triangular_solve;

pub mod amd;
pub mod colamd;

pub mod cholesky;
pub mod lu;
pub mod qr;

/// Sparse LU error.
#[derive(Copy, Clone, Debug)]
pub enum LuError {
    /// Generic sparse error.
    Generic(FaerError),
    /// Rank deficient symbolic structure.
    ///
    /// Contains the iteration at which a pivot could not be found.
    SymbolicSingular(usize),
}

impl core::fmt::Display for LuError {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl core::error::Error for LuError {}

impl From<FaerError> for LuError {
    #[inline]
    fn from(value: FaerError) -> Self {
        Self::Generic(value)
    }
}

/// Sparse Cholesky error.
#[derive(Copy, Clone, Debug)]
pub enum CholeskyError {
    /// Generic sparse error.
    Generic(FaerError),
    /// Rank deficient symbolic structure.
    SymbolicSingular,
    /// Matrix is not positive definite.
    NotPositiveDefinite,
}

impl core::fmt::Display for CholeskyError {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl core::error::Error for CholeskyError {}

impl From<FaerError> for CholeskyError {
    #[inline]
    fn from(value: FaerError) -> Self {
        Self::Generic(value)
    }
}

impl From<crate::linalg::solvers::CholeskyError> for CholeskyError {
    #[inline]
    fn from(_: crate::linalg::solvers::CholeskyError) -> Self {
        Self::NotPositiveDefinite
    }
}

/// High level sparse solvers.
pub mod solvers;

mod ghost {
    pub use crate::utils::bound::*;
    use crate::Index;

    pub const NONE_BYTE: u8 = u8::MAX;

    #[inline]
    pub fn fill_zero<'n, 'a, I: Index>(slice: &'a mut [I], size: Dim<'n>) -> &'a mut [Idx<'n, I>] {
        let len = slice.len();
        if len > 0 {
            assert!(*size > 0);
        }
        unsafe {
            core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len);
            &mut *(slice as *mut _ as *mut _)
        }
    }

    #[inline]
    pub fn fill_none<'n, 'a, I: Index>(
        slice: &'a mut [I::Signed],
        size: Dim<'n>,
    ) -> &'a mut [MaybeIdx<'n, I>] {
        let _ = size;
        let len = slice.len();
        unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), NONE_BYTE, len) };
        unsafe { &mut *(slice as *mut _ as *mut _) }
    }

    #[inline]
    pub fn copy_slice<'n, 'a, I: Index>(
        dst: &'a mut [I],
        src: &[Idx<'n, I>],
    ) -> &'a mut [Idx<'n, I>] {
        let dst: &mut [Idx<'_, I>] = unsafe { &mut *(dst as *mut _ as *mut _) };
        dst.copy_from_slice(src);
        dst
    }
}

mod mem {
    use crate::SignedIndex;

    pub const NONE_BYTE: u8 = 0xFF;
    pub const NONE: usize = crate::sparse::repeat_byte(NONE_BYTE);

    #[inline]
    pub fn fill_none<I: SignedIndex>(slice: &mut [I]) {
        let len = slice.len();
        unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), NONE_BYTE, len) }
    }
    #[inline]
    pub fn fill_zero<I: bytemuck::Zeroable>(slice: &mut [I]) {
        let len = slice.len();
        unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len) }
    }
}

/// Sparse matrix multiplication.
pub mod matmul;

#[cfg(test)]
pub(crate) mod qd {
    // https://web.mit.edu/tabbott/Public/quaddouble-debian/qd-2.3.4-old/docs/qd.pdf
    // https://gitlab.com/hodge_star/mantis

    use bytemuck::{Pod, Zeroable};
    use faer_entity::*;
    use pulp::{Scalar, Simd};

    /// Value representing the implicit sum of two floating point terms, such that the absolute
    /// value of the second term is less half a ULP of the first term.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(C)]
    pub struct Double<T>(pub T, pub T);

    unsafe impl<T: Zeroable> Zeroable for Double<T> {}
    unsafe impl<T: Pod> Pod for Double<T> {}

    impl<I: Iterator> Iterator for Double<I> {
        type Item = Double<I::Item>;

        #[inline(always)]
        fn next(&mut self) -> Option<Self::Item> {
            let x0 = self.0.next()?;
            let x1 = self.1.next()?;
            Some(Double(x0, x1))
        }
    }

    #[inline(always)]
    fn quick_two_sum<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
        let s = simd.f64s_add(a, b);
        let err = simd.f64s_sub(b, simd.f64s_sub(s, a));
        (s, err)
    }

    #[inline(always)]
    fn two_sum<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
        let s = simd.f64s_add(a, b);
        let bb = simd.f64s_sub(s, a);

        // (a - (s - bb)) + (b - bb)
        let err = simd.f64s_add(simd.f64s_sub(a, simd.f64s_sub(s, bb)), simd.f64s_sub(b, bb));
        (s, err)
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn quick_two_diff<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
        let s = simd.f64s_sub(a, b);
        let err = simd.f64s_sub(simd.f64s_sub(a, s), b);
        (s, err)
    }

    #[inline(always)]
    fn two_diff<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
        let s = simd.f64s_sub(a, b);
        let bb = simd.f64s_sub(s, a);

        // (a - (s - bb)) - (b + bb)
        let err = simd.f64s_sub(simd.f64s_sub(a, simd.f64s_sub(s, bb)), simd.f64s_add(b, bb));
        (s, err)
    }

    #[inline(always)]
    fn two_prod<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
        let p = simd.f64s_mul(a, b);
        let err = simd.f64s_mul_add(a, b, simd.f64s_neg(p));

        (p, err)
    }

    pub mod double {
        use super::*;

        #[inline(always)]
        pub fn simd_add<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> Double<S::f64s> {
            let (s, e) = two_sum(simd, a.0, b.0);
            let e = simd.f64s_add(e, simd.f64s_add(a.1, b.1));
            let (s, e) = quick_two_sum(simd, s, e);
            Double(s, e)
        }

        #[inline(always)]
        pub fn simd_sub<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> Double<S::f64s> {
            let (s, e) = two_diff(simd, a.0, b.0);
            let e = simd.f64s_add(e, a.1);
            let e = simd.f64s_sub(e, b.1);
            let (s, e) = quick_two_sum(simd, s, e);
            Double(s, e)
        }

        #[inline(always)]
        pub fn simd_neg<S: Simd>(simd: S, a: Double<S::f64s>) -> Double<S::f64s> {
            Double(simd.f64s_neg(a.0), simd.f64s_neg(a.1))
        }

        #[inline(always)]
        pub fn simd_mul<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> Double<S::f64s> {
            let (p1, p2) = two_prod(simd, a.0, b.0);
            let p2 = simd.f64s_add(
                p2,
                simd.f64s_add(simd.f64s_mul(a.0, b.1), simd.f64s_mul(a.1, b.0)),
            );
            let (p1, p2) = quick_two_sum(simd, p1, p2);
            Double(p1, p2)
        }

        #[inline(always)]
        fn simd_mul_f64<S: Simd>(simd: S, a: Double<S::f64s>, b: S::f64s) -> Double<S::f64s> {
            let (p1, p2) = two_prod(simd, a.0, b);
            let p2 = simd.f64s_add(p2, simd.f64s_mul(a.1, b));
            let (p1, p2) = quick_two_sum(simd, p1, p2);
            Double(p1, p2)
        }

        pub fn simd_select<S: Simd>(
            simd: S,
            mask: S::m64s,
            if_true: Double<S::f64s>,
            if_false: Double<S::f64s>,
        ) -> Double<S::f64s> {
            Double(
                simd.m64s_select_f64s(mask, if_true.0, if_false.0),
                simd.m64s_select_f64s(mask, if_true.1, if_false.1),
            )
        }

        #[inline]
        pub fn simd_div<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> Double<S::f64s> {
            simd.vectorize(
                #[inline(always)]
                || {
                    let pos_zero = simd.f64s_splat(0.0);
                    let pos_infty = simd.f64s_splat(f64::INFINITY);
                    let sign_bit = simd.f64s_splat(-0.0);

                    let a_sign = simd.f64s_and(a.0, sign_bit);
                    let b_sign = simd.f64s_and(b.0, sign_bit);

                    let combined_sign = simd.f64s_xor(a_sign, b_sign);

                    let a_is_zero = simd_eq(simd, a, Double(pos_zero, pos_zero));
                    let b_is_zero = simd_eq(simd, b, Double(pos_zero, pos_zero));
                    let a_is_infty = simd_eq(
                        simd,
                        Double(simd.f64s_abs(a.0), simd.f64s_abs(a.1)),
                        Double(pos_infty, pos_infty),
                    );
                    let b_is_infty = simd_eq(
                        simd,
                        Double(simd.f64s_abs(b.0), simd.f64s_abs(b.1)),
                        Double(pos_infty, pos_infty),
                    );

                    let q1 = simd.f64s_div(a.0, b.0);
                    let r = simd_mul_f64(simd, b, q1);

                    let (s1, s2) = two_diff(simd, a.0, r.0);
                    let s2 = simd.f64s_sub(s2, r.1);
                    let s2 = simd.f64s_add(s2, a.1);

                    let q2 = simd.f64s_div(simd.f64s_add(s1, s2), b.0);
                    let (q0, q1) = quick_two_sum(simd, q1, q2);

                    simd_select(
                        simd,
                        simd.m64s_and(b_is_zero, simd.m64s_not(a_is_zero)),
                        Double(
                            simd.f64s_or(combined_sign, pos_infty),
                            simd.f64s_or(combined_sign, pos_infty),
                        ),
                        simd_select(
                            simd,
                            simd.m64s_and(b_is_infty, simd.m64s_not(a_is_infty)),
                            Double(
                                simd.f64s_or(combined_sign, pos_zero),
                                simd.f64s_or(combined_sign, pos_zero),
                            ),
                            Double(q0, q1),
                        ),
                    )
                },
            )
        }

        #[inline(always)]
        pub fn simd_abs<S: Simd>(simd: S, a: Double<S::f64s>) -> Double<S::f64s> {
            let is_negative = simd.f64s_less_than(a.0, simd.f64s_splat(0.0));
            Double(
                simd.m64s_select_f64s(is_negative, simd.f64s_neg(a.0), a.0),
                simd.m64s_select_f64s(is_negative, simd.f64s_neg(a.1), a.1),
            )
        }

        #[inline(always)]
        pub fn simd_less_than<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> S::m64s {
            let lt0 = simd.f64s_less_than(a.0, b.0);
            let eq0 = simd.f64s_equal(a.0, b.0);
            let lt1 = simd.f64s_less_than(a.1, b.1);
            simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
        }

        #[inline(always)]
        pub fn simd_less_than_or_equal<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> S::m64s {
            let lt0 = simd.f64s_less_than(a.0, b.0);
            let eq0 = simd.f64s_equal(a.0, b.0);
            let lt1 = simd.f64s_less_than_or_equal(a.1, b.1);
            simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
        }

        #[inline(always)]
        pub fn simd_greater_than<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> S::m64s {
            let lt0 = simd.f64s_greater_than(a.0, b.0);
            let eq0 = simd.f64s_equal(a.0, b.0);
            let lt1 = simd.f64s_greater_than(a.1, b.1);
            simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
        }

        #[inline(always)]
        pub fn simd_greater_than_or_equal<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> S::m64s {
            let lt0 = simd.f64s_greater_than(a.0, b.0);
            let eq0 = simd.f64s_equal(a.0, b.0);
            let lt1 = simd.f64s_greater_than_or_equal(a.1, b.1);
            simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
        }

        #[inline(always)]
        pub fn simd_eq<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> S::m64s {
            let eq0 = simd.f64s_equal(a.0, b.0);
            let eq1 = simd.f64s_equal(a.1, b.1);
            simd.m64s_and(eq0, eq1)
        }
    }

    impl core::ops::Add for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn add(self, rhs: Self) -> Self::Output {
            double::simd_add(Scalar::new(), self, rhs)
        }
    }

    impl core::ops::Sub for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn sub(self, rhs: Self) -> Self::Output {
            double::simd_sub(Scalar::new(), self, rhs)
        }
    }

    impl core::ops::Mul for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn mul(self, rhs: Self) -> Self::Output {
            double::simd_mul(Scalar::new(), self, rhs)
        }
    }

    impl core::ops::Rem for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn rem(self, _: Self) -> Self::Output {
            todo!()
        }
    }
    impl core::ops::Div for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn div(self, rhs: Self) -> Self::Output {
            double::simd_div(Scalar::new(), self, rhs)
        }
    }

    impl core::ops::AddAssign for Double<f64> {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for Double<f64> {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for Double<f64> {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for Double<f64> {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }
    impl core::ops::RemAssign for Double<f64> {
        #[inline(always)]
        fn rem_assign(&mut self, _: Self) {
            todo!()
        }
    }

    impl core::ops::Neg for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn neg(self) -> Self::Output {
            Self(-self.0, -self.1)
        }
    }

    impl num_traits::Num for Double<f64> {
        type FromStrRadixErr = ();
        fn from_str_radix(_: &str, _: u32) -> Result<Self, Self::FromStrRadixErr> {
            Err(())
        }
    }

    impl num_traits::Zero for Double<f64> {
        fn zero() -> Self {
            Self(0.0, 0.0)
        }
        fn is_zero(&self) -> bool {
            *self == Self::zero()
        }
    }
    impl num_traits::One for Double<f64> {
        fn one() -> Self {
            Self(1.0, 0.0)
        }
    }

    impl Double<f64> {
        /// 2.0^{-100}
        pub const EPSILON: Self = Self(7.888609052210118e-31, 0.0);
        /// 2.0^{-970}: precision below this value begins to degrade.
        pub const MIN_POSITIVE: Self = Self(1.0020841800044864e-292, 0.0);

        pub const ZERO: Self = Self(0.0, 0.0);
        pub const NAN: Self = Self(f64::NAN, f64::NAN);
        pub const INFINITY: Self = Self(f64::INFINITY, f64::INFINITY);

        #[inline(always)]
        pub fn abs(self) -> Self {
            double::simd_abs(Scalar::new(), self)
        }

        #[inline(always)]
        pub fn recip(self) -> Self {
            double::simd_div(Scalar::new(), Self(1.0, 0.0), self)
        }

        #[inline]
        pub fn sqrt(self) -> Self {
            if self == Self::ZERO {
                Self::ZERO
            } else if self < Self::ZERO {
                Self::NAN
            } else if self == Self::INFINITY {
                Self::INFINITY
            } else {
                let a = self;
                let x = a.0.sqrt().recip();
                let ax = Self(a.0 * x, 0.0);

                ax + (a - ax * ax) * Double(x * 0.5, 0.0)
            }
        }
    }

    pub struct DoubleGroup {
        __private: (),
    }

    impl ForType for DoubleGroup {
        type FaerOf<T> = Double<T>;
    }
    impl ForCopyType for DoubleGroup {
        type FaerOfCopy<T: Copy> = Double<T>;
    }
    impl ForDebugType for DoubleGroup {
        type FaerOfDebug<T: core::fmt::Debug> = Double<T>;
    }

    mod faer_impl {
        use super::*;

        unsafe impl Entity for Double<f64> {
            type Unit = f64;
            type Index = u64;

            type SimdUnit<S: Simd> = S::f64s;
            type SimdMask<S: Simd> = S::m64s;
            type SimdIndex<S: Simd> = S::u64s;

            type Group = DoubleGroup;
            type Iter<I: Iterator> = Double<I>;

            type PrefixUnit<'a, S: Simd> = pulp::Prefix<'a, f64, S, S::m64s>;
            type SuffixUnit<'a, S: Simd> = pulp::Suffix<'a, f64, S, S::m64s>;
            type PrefixMutUnit<'a, S: Simd> = pulp::PrefixMut<'a, f64, S, S::m64s>;
            type SuffixMutUnit<'a, S: Simd> = pulp::SuffixMut<'a, f64, S, S::m64s>;

            const N_COMPONENTS: usize = 2;
            const UNIT: GroupCopyFor<Self, ()> = Double((), ());

            #[inline(always)]
            fn faer_first<T>(group: GroupFor<Self, T>) -> T {
                group.0
            }

            #[inline(always)]
            fn faer_from_units(group: GroupFor<Self, Self::Unit>) -> Self {
                group
            }

            #[inline(always)]
            fn faer_into_units(self) -> GroupFor<Self, Self::Unit> {
                self
            }

            #[inline(always)]
            fn faer_as_ref<T>(group: &GroupFor<Self, T>) -> GroupFor<Self, &T> {
                Double(&group.0, &group.1)
            }

            #[inline(always)]
            fn faer_as_mut<T>(group: &mut GroupFor<Self, T>) -> GroupFor<Self, &mut T> {
                Double(&mut group.0, &mut group.1)
            }

            #[inline(always)]
            fn faer_as_ptr<T>(group: *mut GroupFor<Self, T>) -> GroupFor<Self, *mut T> {
                unsafe {
                    Double(
                        core::ptr::addr_of_mut!((*group).0),
                        core::ptr::addr_of_mut!((*group).1),
                    )
                }
            }

            #[inline(always)]
            fn faer_map_impl<T, U>(
                group: GroupFor<Self, T>,
                f: &mut impl FnMut(T) -> U,
            ) -> GroupFor<Self, U> {
                Double((*f)(group.0), (*f)(group.1))
            }

            #[inline(always)]
            fn faer_zip<T, U>(
                first: GroupFor<Self, T>,
                second: GroupFor<Self, U>,
            ) -> GroupFor<Self, (T, U)> {
                Double((first.0, second.0), (first.1, second.1))
            }

            #[inline(always)]
            fn faer_unzip<T, U>(
                zipped: GroupFor<Self, (T, U)>,
            ) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
                (
                    Double(zipped.0 .0, zipped.1 .0),
                    Double(zipped.0 .1, zipped.1 .1),
                )
            }

            #[inline(always)]
            fn faer_map_with_context<Ctx, T, U>(
                ctx: Ctx,
                group: GroupFor<Self, T>,
                f: &mut impl FnMut(Ctx, T) -> (Ctx, U),
            ) -> (Ctx, GroupFor<Self, U>) {
                let (ctx, x0) = (*f)(ctx, group.0);
                let (ctx, x1) = (*f)(ctx, group.1);
                (ctx, Double(x0, x1))
            }

            #[inline(always)]
            fn faer_into_iter<I: IntoIterator>(iter: GroupFor<Self, I>) -> Self::Iter<I::IntoIter> {
                Double(iter.0.into_iter(), iter.1.into_iter())
            }
        }

        unsafe impl Conjugate for Double<f64> {
            type Conj = Double<f64>;
            type Canonical = Double<f64>;
            #[inline(always)]
            fn canonicalize(self) -> Self::Canonical {
                self
            }
        }

        impl RealField for Double<f64> {
            #[inline(always)]
            fn faer_epsilon() -> Self {
                Self::EPSILON
            }
            #[inline(always)]
            fn faer_zero_threshold() -> Self {
                Self::MIN_POSITIVE
            }

            #[inline(always)]
            fn faer_usize_to_index(a: usize) -> Self::Index {
                a as _
            }

            #[inline(always)]
            fn faer_index_to_usize(a: Self::Index) -> usize {
                a as _
            }

            #[inline(always)]
            fn faer_max_index() -> Self::Index {
                Self::Index::MAX
            }

            #[inline(always)]
            fn faer_simd_less_than<S: Simd>(
                simd: S,
                a: SimdGroupFor<Self, S>,
                b: SimdGroupFor<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_less_than(simd, a, b)
            }

            #[inline(always)]
            fn faer_simd_less_than_or_equal<S: Simd>(
                simd: S,
                a: SimdGroupFor<Self, S>,
                b: SimdGroupFor<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_less_than_or_equal(simd, a, b)
            }

            #[inline(always)]
            fn faer_simd_greater_than<S: Simd>(
                simd: S,
                a: SimdGroupFor<Self, S>,
                b: SimdGroupFor<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_greater_than(simd, a, b)
            }

            #[inline(always)]
            fn faer_simd_greater_than_or_equal<S: Simd>(
                simd: S,
                a: SimdGroupFor<Self, S>,
                b: SimdGroupFor<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_greater_than_or_equal(simd, a, b)
            }

            #[inline(always)]
            fn faer_simd_select<S: Simd>(
                simd: S,
                mask: Self::SimdMask<S>,
                if_true: SimdGroupFor<Self, S>,
                if_false: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                double::simd_select(simd, mask, if_true, if_false)
            }

            #[inline(always)]
            fn faer_simd_index_select<S: Simd>(
                simd: S,
                mask: Self::SimdMask<S>,
                if_true: Self::SimdIndex<S>,
                if_false: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                simd.m64s_select_u64s(mask, if_true, if_false)
            }

            #[inline(always)]
            fn faer_simd_index_seq<S: Simd>(simd: S) -> Self::SimdIndex<S> {
                let _ = simd;
                pulp::cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u64])
            }

            #[inline(always)]
            fn faer_simd_index_splat<S: Simd>(simd: S, value: Self::Index) -> Self::SimdIndex<S> {
                simd.u64s_splat(value)
            }

            #[inline(always)]
            fn faer_simd_index_add<S: Simd>(
                simd: S,
                a: Self::SimdIndex<S>,
                b: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                simd.u64s_add(a, b)
            }

            #[inline(always)]
            fn faer_min_positive() -> Self {
                Self::MIN_POSITIVE
            }

            #[inline(always)]
            fn faer_min_positive_inv() -> Self {
                Self::MIN_POSITIVE.recip()
            }

            #[inline(always)]
            fn faer_min_positive_sqrt() -> Self {
                Self::MIN_POSITIVE.sqrt()
            }

            #[inline(always)]
            fn faer_min_positive_sqrt_inv() -> Self {
                Self::MIN_POSITIVE.sqrt().recip()
            }

            #[inline(always)]
            fn faer_simd_index_rotate_left<S: Simd>(
                simd: S,
                values: SimdIndexFor<Self, S>,
                amount: usize,
            ) -> SimdIndexFor<Self, S> {
                simd.u64s_rotate_left(values, amount)
            }

            #[inline(always)]
            fn faer_simd_abs<S: Simd>(
                simd: S,
                values: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                double::simd_abs(simd, values)
            }
        }

        impl ComplexField for Double<f64> {
            type Real = Double<f64>;
            type Simd = pulp::Arch;
            type ScalarSimd = pulp::Arch;
            type PortableSimd = pulp::Arch;

            #[inline(always)]
            fn faer_sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline(always)]
            fn faer_from_f64(value: f64) -> Self {
                Self(value, 0.0)
            }

            #[inline(always)]
            fn faer_add(self, rhs: Self) -> Self {
                self + rhs
            }

            #[inline(always)]
            fn faer_sub(self, rhs: Self) -> Self {
                self - rhs
            }

            #[inline(always)]
            fn faer_mul(self, rhs: Self) -> Self {
                self * rhs
            }
            #[inline(always)]
            fn faer_div(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn faer_neg(self) -> Self {
                -self
            }

            #[inline(always)]
            fn faer_inv(self) -> Self {
                self.recip()
            }

            #[inline(always)]
            fn faer_conj(self) -> Self {
                self
            }

            #[inline(always)]
            fn faer_scale_real(self, rhs: Self::Real) -> Self {
                self * rhs
            }

            #[inline(always)]
            fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self {
                Self(self.0 * rhs.0, self.1 * rhs.0)
            }

            #[inline(always)]
            fn faer_score(self) -> Self::Real {
                self.abs()
            }

            #[inline(always)]
            fn faer_abs(self) -> Self::Real {
                self.abs()
            }

            #[inline(always)]
            fn faer_abs2(self) -> Self::Real {
                self * self
            }

            #[inline(always)]
            fn faer_nan() -> Self {
                Self::NAN
            }

            #[inline(always)]
            fn faer_from_real(real: Self::Real) -> Self {
                real
            }

            #[inline(always)]
            fn faer_real(self) -> Self::Real {
                self
            }

            #[inline(always)]
            fn faer_imag(self) -> Self::Real {
                Self::ZERO
            }

            #[inline(always)]
            fn faer_zero() -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn faer_one() -> Self {
                Self(1.0, 0.0)
            }

            #[inline(always)]
            fn faer_slice_as_simd<S: Simd>(
                slice: &[Self::Unit],
            ) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
                S::f64s_as_simd(slice)
            }

            #[inline(always)]
            fn faer_slice_as_simd_mut<S: Simd>(
                slice: &mut [Self::Unit],
            ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
                S::f64s_as_mut_simd(slice)
            }

            #[inline(always)]
            fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
                simd.f64s_partial_load(slice)
            }

            #[inline(always)]
            fn faer_partial_store_unit<S: Simd>(
                simd: S,
                slice: &mut [Self::Unit],
                values: Self::SimdUnit<S>,
            ) {
                simd.f64s_partial_store(slice, values)
            }

            #[inline(always)]
            fn faer_partial_load_last_unit<S: Simd>(
                simd: S,
                slice: &[Self::Unit],
            ) -> Self::SimdUnit<S> {
                simd.f64s_partial_load_last(slice)
            }

            #[inline(always)]
            fn faer_partial_store_last_unit<S: Simd>(
                simd: S,
                slice: &mut [Self::Unit],
                values: Self::SimdUnit<S>,
            ) {
                simd.f64s_partial_store_last(slice, values)
            }

            #[inline(always)]
            fn faer_simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
                simd.f64s_splat(unit)
            }

            #[inline(always)]
            fn faer_simd_neg<S: Simd>(
                simd: S,
                values: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                double::simd_neg(simd, values)
            }

            #[inline(always)]
            fn faer_simd_conj<S: Simd>(
                simd: S,
                values: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                let _ = simd;
                values
            }

            #[inline(always)]
            fn faer_simd_add<S: Simd>(
                simd: S,
                lhs: SimdGroupFor<Self, S>,
                rhs: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                double::simd_add(simd, lhs, rhs)
            }

            #[inline(always)]
            fn faer_simd_sub<S: Simd>(
                simd: S,
                lhs: SimdGroupFor<Self, S>,
                rhs: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                double::simd_sub(simd, lhs, rhs)
            }

            #[inline(always)]
            fn faer_simd_mul<S: Simd>(
                simd: S,
                lhs: SimdGroupFor<Self, S>,
                rhs: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                double::simd_mul(simd, lhs, rhs)
            }

            #[inline(always)]
            fn faer_simd_scale_real<S: Simd>(
                simd: S,
                lhs: SimdGroupFor<Self, S>,
                rhs: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                double::simd_mul(simd, lhs, rhs)
            }

            #[inline(always)]
            fn faer_simd_conj_mul<S: Simd>(
                simd: S,
                lhs: SimdGroupFor<Self, S>,
                rhs: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                double::simd_mul(simd, lhs, rhs)
            }

            #[inline(always)]
            fn faer_simd_mul_adde<S: Simd>(
                simd: S,
                lhs: SimdGroupFor<Self, S>,
                rhs: SimdGroupFor<Self, S>,
                acc: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                double::simd_add(simd, acc, double::simd_mul(simd, lhs, rhs))
            }

            #[inline(always)]
            fn faer_simd_conj_mul_adde<S: Simd>(
                simd: S,
                lhs: SimdGroupFor<Self, S>,
                rhs: SimdGroupFor<Self, S>,
                acc: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self, S> {
                double::simd_add(simd, acc, double::simd_mul(simd, lhs, rhs))
            }

            #[inline(always)]
            fn faer_simd_score<S: Simd>(
                simd: S,
                values: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self::Real, S> {
                double::simd_abs(simd, values)
            }

            #[inline(always)]
            fn faer_simd_abs2_adde<S: Simd>(
                simd: S,
                values: SimdGroupFor<Self, S>,
                acc: SimdGroupFor<Self::Real, S>,
            ) -> SimdGroupFor<Self::Real, S> {
                Self::faer_simd_add(simd, acc, Self::faer_simd_mul(simd, values, values))
            }

            #[inline(always)]
            fn faer_simd_abs2<S: Simd>(
                simd: S,
                values: SimdGroupFor<Self, S>,
            ) -> SimdGroupFor<Self::Real, S> {
                Self::faer_simd_mul(simd, values, values)
            }

            #[inline(always)]
            fn faer_simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
                let _ = simd;
                lhs * rhs
            }

            #[inline(always)]
            fn faer_simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
                let _ = simd;
                lhs * rhs
            }

            #[inline(always)]
            fn faer_simd_scalar_mul_adde<S: Simd>(
                simd: S,
                lhs: Self,
                rhs: Self,
                acc: Self,
            ) -> Self {
                let _ = simd;
                lhs * rhs + acc
            }

            #[inline(always)]
            fn faer_simd_scalar_conj_mul_adde<S: Simd>(
                simd: S,
                lhs: Self,
                rhs: Self,
                acc: Self,
            ) -> Self {
                let _ = simd;
                lhs * rhs + acc
            }

            #[inline(always)]
            fn faer_slice_as_aligned_simd<S: Simd>(
                simd: S,
                slice: &[UnitFor<Self>],
                offset: pulp::Offset<SimdMaskFor<Self, S>>,
            ) -> (
                pulp::Prefix<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
                &[SimdUnitFor<Self, S>],
                pulp::Suffix<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
            ) {
                simd.f64s_as_aligned_simd(slice, offset)
            }
            #[inline(always)]
            fn faer_slice_as_aligned_simd_mut<S: Simd>(
                simd: S,
                slice: &mut [UnitFor<Self>],
                offset: pulp::Offset<SimdMaskFor<Self, S>>,
            ) -> (
                pulp::PrefixMut<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
                &mut [SimdUnitFor<Self, S>],
                pulp::SuffixMut<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
            ) {
                simd.f64s_as_aligned_mut_simd(slice, offset)
            }

            #[inline(always)]
            fn faer_simd_rotate_left<S: Simd>(
                simd: S,
                values: SimdGroupFor<Self, S>,
                amount: usize,
            ) -> SimdGroupFor<Self, S> {
                Double(
                    simd.f64s_rotate_left(values.0, amount),
                    simd.f64s_rotate_left(values.1, amount),
                )
            }

            #[inline(always)]
            fn faer_align_offset<S: Simd>(
                simd: S,
                ptr: *const UnitFor<Self>,
                len: usize,
            ) -> pulp::Offset<SimdMaskFor<Self, S>> {
                simd.f64s_align_offset(ptr, len)
            }
        }
    }
}
