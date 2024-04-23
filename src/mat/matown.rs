use super::*;
use crate::{
    assert, debug_assert,
    diag::{DiagMut, DiagRef},
    mat::matalloc::{align_for, is_vectorizable, MatUnit, RawMat, RawMatUnit},
    utils::DivCeil,
};
use core::mem::ManuallyDrop;

/// Heap allocated resizable matrix, similar to a 2D [`Vec`].
///
/// # Note
///
/// The memory layout of `Mat` is guaranteed to be column-major, meaning that it has a row stride
/// of `1`, and an unspecified column stride that can be queried with [`Mat::col_stride`].
///
/// This implies that while each individual column is stored contiguously in memory, the matrix as
/// a whole may not necessarily be contiguous. The implementation may add padding at the end of
/// each column when overaligning each column can provide a performance gain.
///
/// Let us consider a 3×4 matrix
///
/// ```notcode
///  0 │ 3 │ 6 │  9
/// ───┼───┼───┼───
///  1 │ 4 │ 7 │ 10
/// ───┼───┼───┼───
///  2 │ 5 │ 8 │ 11
/// ```
/// The memory representation of the data held by such a matrix could look like the following:
///
/// ```notcode
/// 0 1 2 X 3 4 5 X 6 7 8 X 9 10 11 X
/// ```
///
/// where X represents padding elements.
#[repr(C)]
pub struct Mat<E: Entity> {
    inner: MatOwnImpl<E>,
    row_capacity: usize,
    col_capacity: usize,
    __marker: PhantomData<E>,
}

impl<E: Entity> Drop for Mat<E> {
    #[inline]
    fn drop(&mut self) {
        drop(RawMat::<E> {
            ptr: self.inner.ptr,
            row_capacity: self.row_capacity,
            col_capacity: self.col_capacity,
        });
    }
}

impl<E: Entity> Mat<E> {
    /// Returns an empty matrix of dimension `0×0`.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: MatOwnImpl {
                ptr: into_copy::<E, _>(E::faer_map(E::UNIT, |()| NonNull::<E::Unit>::dangling())),
                nrows: 0,
                ncols: 0,
            },
            row_capacity: 0,
            col_capacity: 0,
            __marker: PhantomData,
        }
    }

    /// Returns a new matrix with dimensions `(0, 0)`, with enough capacity to hold a maximum of
    /// `row_capacity` rows and `col_capacity` columns without reallocating. If either is `0`,
    /// the matrix will not allocate.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_capacity(row_capacity: usize, col_capacity: usize) -> Self {
        let raw = ManuallyDrop::new(RawMat::<E>::new(row_capacity, col_capacity));
        Self {
            inner: MatOwnImpl {
                ptr: raw.ptr,
                nrows: 0,
                ncols: 0,
            },
            row_capacity: raw.row_capacity,
            col_capacity: raw.col_capacity,
            __marker: PhantomData,
        }
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with the provided function.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> E) -> Self {
        let mut this = Self::new();
        this.resize_with(nrows, ncols, f);
        this
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with zeros.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self::from_fn(nrows, ncols, |_, _| unsafe { core::mem::zeroed() })
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with ones.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn ones(nrows: usize, ncols: usize) -> Self
    where
        E: ComplexField,
    {
        Self::from_fn(nrows, ncols, |_, _| E::faer_one())
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with a constant value.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn full(nrows: usize, ncols: usize, constant: E) -> Self
    where
        E: ComplexField,
    {
        Self::from_fn(nrows, ncols, |_, _| constant)
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with zeros, except the main
    /// diagonal which is filled with ones.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    #[doc(alias = "eye")]
    pub fn identity(nrows: usize, ncols: usize) -> Self
    where
        E: ComplexField,
    {
        let mut matrix = Self::zeros(nrows, ncols);
        matrix
            .as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .fill(E::faer_one());
        matrix
    }

    /// Returns the number of rows of the matrix.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.inner.nrows
    }
    /// Returns the number of columns of the matrix.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.inner.ncols
    }

    /// Set the dimensions of the matrix.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `nrows < self.row_capacity()`.
    /// * `ncols < self.col_capacity()`.
    /// * The elements that were previously out of bounds but are now in bounds must be
    /// initialized.
    #[inline]
    pub unsafe fn set_dims(&mut self, nrows: usize, ncols: usize) {
        self.inner.nrows = nrows;
        self.inner.ncols = ncols;
    }

    /// Returns a pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr(&self) -> GroupFor<E, *const E::Unit> {
        E::faer_map(from_copy::<E, _>(self.inner.ptr), |ptr| {
            ptr.as_ptr() as *const E::Unit
        })
    }

    /// Returns a mutable pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr_mut(&mut self) -> GroupFor<E, *mut E::Unit> {
        E::faer_map(from_copy::<E, _>(self.inner.ptr), |ptr| ptr.as_ptr())
    }

    /// Returns the row capacity, that is, the number of rows that the matrix is able to hold
    /// without needing to reallocate, excluding column insertions.
    #[inline]
    pub fn row_capacity(&self) -> usize {
        self.row_capacity
    }

    /// Returns the column capacity, that is, the number of columns that the matrix is able to hold
    /// without needing to reallocate, excluding row insertions.
    #[inline]
    pub fn col_capacity(&self) -> usize {
        self.col_capacity
    }

    /// Returns the offset between the first elements of two successive rows in the matrix.
    /// Always returns `1` since the matrix is column major.
    #[inline]
    pub fn row_stride(&self) -> isize {
        1
    }

    /// Returns the offset between the first elements of two successive columns in the matrix.
    #[inline]
    pub fn col_stride(&self) -> isize {
        self.row_capacity() as isize
    }

    /// Returns raw pointers to the element at the given indices.
    #[inline(always)]
    pub fn ptr_at(&self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().ptr_at(row, col)
    }

    /// Returns raw pointers to the element at the given indices.
    #[inline(always)]
    pub fn ptr_at_mut(&mut self, row: usize, col: usize) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().ptr_at_mut(row, col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(&self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().ptr_at_unchecked(row, col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_mut_unchecked(
        &mut self,
        row: usize,
        col: usize,
    ) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().ptr_at_mut_unchecked(row, col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(&self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().overflowing_ptr_at(row, col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at_mut(
        &mut self,
        row: usize,
        col: usize,
    ) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().overflowing_ptr_at_mut(row, col)
    }

    /// Returns raw pointers to the element at the given indices, assuming the provided indices
    /// are within the matrix dimensions.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().ptr_inbounds_at(row, col)
    }

    /// Returns raw pointers to the element at the given indices, assuming the provided indices
    /// are within the matrix dimensions.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(
        &mut self,
        row: usize,
        col: usize,
    ) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().ptr_inbounds_at_mut(row, col)
    }

    #[cold]
    fn do_reserve_exact(&mut self, mut new_row_capacity: usize, new_col_capacity: usize) {
        if is_vectorizable::<E::Unit>() {
            let align_factor = align_for::<E::Unit>() / core::mem::size_of::<E::Unit>();
            new_row_capacity = new_row_capacity
                .msrv_checked_next_multiple_of(align_factor)
                .unwrap();
        }

        let nrows = self.inner.nrows;
        let ncols = self.inner.ncols;
        let old_row_capacity = self.row_capacity;
        let old_col_capacity = self.col_capacity;

        let mut this = ManuallyDrop::new(core::mem::take(self));
        {
            let mut this_group = E::faer_map(from_copy::<E, _>(this.inner.ptr), |ptr| MatUnit {
                raw: RawMatUnit {
                    ptr,
                    row_capacity: old_row_capacity,
                    col_capacity: old_col_capacity,
                },
                nrows,
                ncols,
            });

            E::faer_map(E::faer_as_mut(&mut this_group), |mat_unit| {
                mat_unit.do_reserve_exact(new_row_capacity, new_col_capacity);
            });

            let this_group = E::faer_map(this_group, ManuallyDrop::new);
            this.inner.ptr =
                into_copy::<E, _>(E::faer_map(this_group, |mat_unit| mat_unit.raw.ptr));
            this.row_capacity = new_row_capacity;
            this.col_capacity = new_col_capacity;
        }
        *self = ManuallyDrop::into_inner(this);
    }

    /// Reserves the minimum capacity for `row_capacity` rows and `col_capacity`
    /// columns without reallocating. Does nothing if the capacity is already sufficient.
    ///
    /// # Panics
    /// The function panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, row_capacity: usize, col_capacity: usize) {
        if self.row_capacity() >= row_capacity && self.col_capacity() >= col_capacity {
            // do nothing
        } else if core::mem::size_of::<E::Unit>() == 0 {
            self.row_capacity = self.row_capacity().max(row_capacity);
            self.col_capacity = self.col_capacity().max(col_capacity);
        } else {
            self.do_reserve_exact(row_capacity, col_capacity);
        }
    }

    unsafe fn insert_block_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) {
        debug_assert!(all(row_start <= row_end, col_start <= col_end));

        let ptr = self.as_ptr_mut();

        for j in col_start..col_end {
            let ptr_j = E::faer_map(E::faer_copy(&ptr), |ptr| {
                ptr.wrapping_offset(j as isize * self.col_stride())
            });

            for i in row_start..row_end {
                // SAFETY:
                // * pointer to element at index (i, j), which is within the
                // allocation since we reserved enough space
                // * writing to this memory region is sound since it is properly
                // aligned and valid for writes
                let ptr_ij = E::faer_map(E::faer_copy(&ptr_j), |ptr_j| ptr_j.add(i));
                let value = E::faer_into_units(f(i, j));

                E::faer_map(E::faer_zip(ptr_ij, value), |(ptr_ij, value)| {
                    core::ptr::write(ptr_ij, value)
                });
            }
        }
    }

    fn erase_last_cols(&mut self, new_ncols: usize) {
        let old_ncols = self.ncols();
        debug_assert!(new_ncols <= old_ncols);
        self.inner.ncols = new_ncols;
    }

    fn erase_last_rows(&mut self, new_nrows: usize) {
        let old_nrows = self.nrows();
        debug_assert!(new_nrows <= old_nrows);
        self.inner.nrows = new_nrows;
    }

    unsafe fn insert_last_cols_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        new_ncols: usize,
    ) {
        let old_ncols = self.ncols();

        debug_assert!(new_ncols > old_ncols);

        self.insert_block_with(f, 0, self.nrows(), old_ncols, new_ncols);
        self.inner.ncols = new_ncols;
    }

    unsafe fn insert_last_rows_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        new_nrows: usize,
    ) {
        let old_nrows = self.nrows();

        debug_assert!(new_nrows > old_nrows);

        self.insert_block_with(f, old_nrows, new_nrows, 0, self.ncols());
        self.inner.nrows = new_nrows;
    }

    /// Resizes the matrix in-place so that the new dimensions are `(new_nrows, new_ncols)`.
    /// New elements are created with the given function `f`, so that elements at indices `(i, j)`
    /// are created by calling `f(i, j)`.
    pub fn resize_with(
        &mut self,
        new_nrows: usize,
        new_ncols: usize,
        f: impl FnMut(usize, usize) -> E,
    ) {
        let mut f = f;
        let old_nrows = self.nrows();
        let old_ncols = self.ncols();

        if new_ncols <= old_ncols {
            self.erase_last_cols(new_ncols);
            if new_nrows <= old_nrows {
                self.erase_last_rows(new_nrows);
            } else {
                self.reserve_exact(new_nrows, new_ncols);
                unsafe {
                    self.insert_last_rows_with(&mut f, new_nrows);
                }
            }
        } else {
            if new_nrows <= old_nrows {
                self.erase_last_rows(new_nrows);
            } else {
                self.reserve_exact(new_nrows, new_ncols);
                unsafe {
                    self.insert_last_rows_with(&mut f, new_nrows);
                }
            }
            self.reserve_exact(new_nrows, new_ncols);
            unsafe {
                self.insert_last_cols_with(&mut f, new_ncols);
            }
        }
    }

    /// Truncates the matrix so that its new dimensions are `new_nrows` and `new_ncols`.  
    /// Both of the new dimensions must be smaller than or equal to the current dimensions.
    ///
    /// # Panics
    /// - Panics if `new_nrows > self.nrows()`.
    /// - Panics if `new_ncols > self.ncols()`.
    #[inline]
    pub fn truncate(&mut self, new_nrows: usize, new_ncols: usize) {
        assert!(all(new_nrows <= self.nrows(), new_ncols <= self.ncols()));
        self.resize_with(new_nrows, new_ncols, |_, _| unreachable!());
    }

    /// Returns a reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    pub fn col_as_slice(&self, col: usize) -> GroupFor<E, &[E::Unit]> {
        assert!(col < self.ncols());
        let nrows = self.nrows();
        let ptr = self.as_ref().ptr_at(0, col);
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, nrows) },
        )
    }

    /// Returns a mutable reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    pub fn col_as_slice_mut(&mut self, col: usize) -> GroupFor<E, &mut [E::Unit]> {
        assert!(col < self.ncols());
        let nrows = self.nrows();
        let ptr = self.as_mut().ptr_at_mut(0, col);
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, nrows) },
        )
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, E> {
        unsafe {
            super::from_raw_parts(
                self.as_ptr(),
                self.nrows(),
                self.ncols(),
                1,
                self.col_stride(),
            )
        }
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, E> {
        unsafe {
            super::from_raw_parts_mut(
                self.as_ptr_mut(),
                self.nrows(),
                self.ncols(),
                1,
                self.col_stride(),
            )
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(&self) -> MatMut<'_, E> {
        self.as_ref().const_cast()
    }

    /// Splits the matrix horizontally and vertically at the given indices into four corners and
    /// returns an array of each submatrix, in the following order:
    /// * top left.
    /// * top right.
    /// * bottom left.
    /// * bottom right.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row <= self.nrows()`.
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_unchecked(
        &self,
        row: usize,
        col: usize,
    ) -> (MatRef<'_, E>, MatRef<'_, E>, MatRef<'_, E>, MatRef<'_, E>) {
        self.as_ref().split_at_unchecked(row, col)
    }

    /// Splits the matrix horizontally and vertically at the given indices into four corners and
    /// returns an array of each submatrix, in the following order:
    /// * top left.
    /// * top right.
    /// * bottom left.
    /// * bottom right.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row <= self.nrows()`.
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at(
        &self,
        row: usize,
        col: usize,
    ) -> (MatRef<'_, E>, MatRef<'_, E>, MatRef<'_, E>, MatRef<'_, E>) {
        self.as_ref().split_at(row, col)
    }

    /// Splits the matrix horizontally and vertically at the given indices into four corners and
    /// returns an array of each submatrix, in the following order:
    /// * top left.
    /// * top right.
    /// * bottom left.
    /// * bottom right.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row <= self.nrows()`.
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_mut_unchecked(
        &mut self,
        row: usize,
        col: usize,
    ) -> (MatMut<'_, E>, MatMut<'_, E>, MatMut<'_, E>, MatMut<'_, E>) {
        self.as_mut().split_at_mut_unchecked(row, col)
    }

    /// Splits the matrix horizontally and vertically at the given indices into four corners and
    /// returns an array of each submatrix, in the following order:
    /// * top left.
    /// * top right.
    /// * bottom left.
    /// * bottom right.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row <= self.nrows()`.
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_mut(
        &mut self,
        row: usize,
        col: usize,
    ) -> (MatMut<'_, E>, MatMut<'_, E>, MatMut<'_, E>, MatMut<'_, E>) {
        self.as_mut().split_at_mut(row, col)
    }

    /// Splits the matrix horizontally at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Safety
    /// The behavior is undefined if the following condition is violated:
    /// * `row <= self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_row_unchecked(&self, row: usize) -> (MatRef<'_, E>, MatRef<'_, E>) {
        self.as_ref().split_at_row_unchecked(row)
    }

    /// Splits the matrix horizontally at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Panics
    /// The function panics if the following condition is violated:
    /// * `row <= self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_row(&self, row: usize) -> (MatRef<'_, E>, MatRef<'_, E>) {
        self.as_ref().split_at_row(row)
    }

    /// Splits the matrix horizontally at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Safety
    /// The behavior is undefined if the following condition is violated:
    /// * `row <= self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_row_mut_unchecked(
        &mut self,
        row: usize,
    ) -> (MatMut<'_, E>, MatMut<'_, E>) {
        self.as_mut().split_at_row_mut_unchecked(row)
    }

    /// Splits the matrix horizontally at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Panics
    /// The function panics if the following condition is violated:
    /// * `row <= self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_row_mut(&mut self, row: usize) -> (MatMut<'_, E>, MatMut<'_, E>) {
        self.as_mut().split_at_row_mut(row)
    }

    /// Splits the matrix vertically at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * left.
    /// * right.
    ///
    /// # Safety
    /// The behavior is undefined if the following condition is violated:
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_col_unchecked(&self, col: usize) -> (MatRef<'_, E>, MatRef<'_, E>) {
        self.as_ref().split_at_col_unchecked(col)
    }

    /// Splits the matrix vertically at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * left.
    /// * right.
    ///
    /// # Panics
    /// The function panics if the following condition is violated:
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_col(&self, col: usize) -> (MatRef<'_, E>, MatRef<'_, E>) {
        self.as_ref().split_at_col(col)
    }

    /// Splits the matrix vertically at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * left.
    /// * right.
    ///
    /// # Safety
    /// The behavior is undefined if the following condition is violated:
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_col_mut_unchecked(
        &mut self,
        col: usize,
    ) -> (MatMut<'_, E>, MatMut<'_, E>) {
        self.as_mut().split_at_col_mut_unchecked(col)
    }

    /// Splits the matrix vertically at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * left.
    /// * right.
    ///
    /// # Panics
    /// The function panics if the following condition is violated:
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_col_mut(&mut self, col: usize) -> (MatMut<'_, E>, MatMut<'_, E>) {
        self.as_mut().split_at_col_mut(col)
    }

    /// Returns references to the element at the given indices, or submatrices if either `row` or
    /// `col` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    #[track_caller]
    pub unsafe fn get_unchecked<RowRange, ColRange>(
        &self,
        row: RowRange,
        col: ColRange,
    ) -> <MatRef<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatRef<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_ref().get_unchecked(row, col)
    }

    /// Returns references to the element at the given indices, or submatrices if either `row` or
    /// `col` is a range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    #[track_caller]
    pub fn get<RowRange, ColRange>(
        &self,
        row: RowRange,
        col: ColRange,
    ) -> <MatRef<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatRef<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_ref().get(row, col)
    }

    /// Returns mutable references to the element at the given indices, or submatrices if either
    /// `row` or `col` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    #[track_caller]
    pub unsafe fn get_mut_unchecked<RowRange, ColRange>(
        &mut self,
        row: RowRange,
        col: ColRange,
    ) -> <MatMut<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatMut<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_mut().get_mut_unchecked(row, col)
    }

    /// Returns mutable references to the element at the given indices, or submatrices if either
    /// `row` or `col` is a range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    #[track_caller]
    pub fn get_mut<RowRange, ColRange>(
        &mut self,
        row: RowRange,
        col: ColRange,
    ) -> <MatMut<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatMut<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_mut().get_mut(row, col)
    }

    /// Reads the value of the element at the given indices.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, row: usize, col: usize) -> E {
        self.as_ref().read_unchecked(row, col)
    }

    /// Reads the value of the element at the given indices, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, row: usize, col: usize) -> E {
        self.as_ref().read(row, col)
    }

    /// Writes the value to the element at the given indices.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, row: usize, col: usize, value: E) {
        self.as_mut().write_unchecked(row, col, value);
    }

    /// Writes the value to the element at the given indices, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, row: usize, col: usize, value: E) {
        self.as_mut().write(row, col, value);
    }

    /// Copies the values from the lower triangular part of `other` into the lower triangular
    /// part of `self`. The diagonal part is included.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    /// * `self.nrows() == self.ncols()`.
    #[track_caller]
    pub fn copy_from_triangular_lower<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsMatRef<ViewE>,
    ) {
        self.as_mut().copy_from_triangular_lower(other)
    }

    /// Copies the values from the lower triangular part of `other` into the lower triangular
    /// part of `self`. The diagonal part is excluded.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    /// * `self.nrows() == self.ncols()`.
    #[track_caller]
    pub fn copy_from_strict_triangular_lower<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsMatRef<ViewE>,
    ) {
        self.as_mut().copy_from_strict_triangular_lower(other)
    }

    /// Copies the values from the upper triangular part of `other` into the upper triangular
    /// part of `self`. The diagonal part is included.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    /// * `self.nrows() == self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn copy_from_triangular_upper<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsMatRef<ViewE>,
    ) {
        self.as_mut().copy_from_triangular_upper(other)
    }

    /// Copies the values from the upper triangular part of `other` into the upper triangular
    /// part of `self`. The diagonal part is excluded.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    /// * `self.nrows() == self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn copy_from_strict_triangular_upper<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsMatRef<ViewE>,
    ) {
        self.as_mut().copy_from_strict_triangular_upper(other)
    }

    /// Copies the values from `other` into `self`.
    #[inline(always)]
    #[track_caller]
    pub fn copy_from<ViewE: Conjugate<Canonical = E>>(&mut self, other: impl AsMatRef<ViewE>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: &mut Mat<E>,
            other: MatRef<'_, ViewE>,
        ) {
            let mut mat = Mat::<E>::new();
            mat.resize_with(
                other.nrows(),
                other.ncols(),
                #[inline(always)]
                |row, col| unsafe { other.read_unchecked(row, col).canonicalize() },
            );
            *this = mat;
        }
        implementation(self, other.as_mat_ref());
    }

    /// Fills the elements of `self` with zeros.
    #[inline(always)]
    #[track_caller]
    pub fn fill_zero(&mut self)
    where
        E: ComplexField,
    {
        self.as_mut().fill_zero()
    }

    /// Fills the elements of `self` with copies of `constant`.
    #[inline(always)]
    #[track_caller]
    pub fn fill(&mut self, constant: E) {
        self.as_mut().fill(constant)
    }

    /// Returns a view over the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> MatRef<'_, E> {
        self.as_ref().transpose()
    }

    /// Returns a view over the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn transpose_mut(&mut self) -> MatMut<'_, E> {
        self.as_mut().transpose_mut()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    #[must_use]
    pub fn conjugate(&self) -> MatRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    #[must_use]
    pub fn conjugate_mut(&mut self) -> MatMut<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_mut().conjugate_mut()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    #[must_use]
    pub fn adjoint(&self) -> MatRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    #[must_use]
    pub fn adjoint_mut(&mut self) -> MatMut<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_mut().adjoint_mut()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    #[must_use]
    pub fn canonicalize(&self) -> (MatRef<'_, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        self.as_ref().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    #[must_use]
    pub fn canonicalize_mut(&mut self) -> (MatMut<'_, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        self.as_mut().canonicalize_mut()
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_ref();
    /// let reversed_rows = view.reverse_rows();
    ///
    /// let expected = mat![[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]];
    /// assert_eq!(expected.as_ref(), reversed_rows);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows(&self) -> MatRef<'_, E> {
        self.as_ref().reverse_rows()
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_mut();
    /// let reversed_rows = view.reverse_rows_mut();
    ///
    /// let mut expected = mat![[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]];
    /// assert_eq!(expected.as_mut(), reversed_rows);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows_mut(&mut self) -> MatMut<'_, E> {
        self.as_mut().reverse_rows_mut()
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_ref();
    /// let reversed_cols = view.reverse_cols();
    ///
    /// let expected = mat![[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]];
    /// assert_eq!(expected.as_ref(), reversed_cols);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols(&self) -> MatRef<'_, E> {
        self.as_ref().reverse_cols()
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_mut();
    /// let reversed_cols = view.reverse_cols_mut();
    ///
    /// let mut expected = mat![[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]];
    /// assert_eq!(expected.as_mut(), reversed_cols);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols_mut(&mut self) -> MatMut<'_, E> {
        self.as_mut().reverse_cols_mut()
    }

    /// Returns a view over the `self`, with the rows and the columns in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_ref();
    /// let reversed = view.reverse_rows_and_cols();
    ///
    /// let expected = mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
    /// assert_eq!(expected.as_ref(), reversed);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows_and_cols(&self) -> MatRef<'_, E> {
        self.as_ref().reverse_rows_and_cols()
    }

    /// Returns a view over the `self`, with the rows and the columns in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_mut();
    /// let reversed = view.reverse_rows_and_cols_mut();
    ///
    /// let mut expected = mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
    /// assert_eq!(expected.as_mut(), reversed);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows_and_cols_mut(&mut self) -> MatMut<'_, E> {
        self.as_mut().reverse_rows_and_cols_mut()
    }

    /// Returns a view over the submatrix starting at indices `(row_start, col_start)`, and with
    /// dimensions `(nrows, ncols)`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `col_start <= self.ncols()`.
    /// * `nrows <= self.nrows() - row_start`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn submatrix_unchecked(
        &self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'_, E> {
        self.as_ref()
            .submatrix_unchecked(row_start, col_start, nrows, ncols)
    }

    /// Returns a view over the submatrix starting at indices `(row_start, col_start)`, and with
    /// dimensions `(nrows, ncols)`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `col_start <= self.ncols()`.
    /// * `nrows <= self.nrows() - row_start`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn submatrix_mut_unchecked(
        &mut self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatMut<'_, E> {
        self.as_mut()
            .submatrix_mut_unchecked(row_start, col_start, nrows, ncols)
    }

    /// Returns a view over the submatrix starting at indices `(row_start, col_start)`, and with
    /// dimensions `(nrows, ncols)`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `col_start <= self.ncols()`.
    /// * `nrows <= self.nrows() - row_start`.
    /// * `ncols <= self.ncols() - col_start`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_ref();
    /// let submatrix = view.submatrix(2, 1, 2, 2);
    ///
    /// let expected = mat![[7.0, 11.0], [8.0, 12.0f64]];
    /// assert_eq!(expected.as_ref(), submatrix);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn submatrix(
        &self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'_, E> {
        self.as_ref().submatrix(row_start, col_start, nrows, ncols)
    }

    /// Returns a view over the submatrix starting at indices `(row_start, col_start)`, and with
    /// dimensions `(nrows, ncols)`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `col_start <= self.ncols()`.
    /// * `nrows <= self.nrows() - row_start`.
    /// * `ncols <= self.ncols() - col_start`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_mut();
    /// let submatrix = view.submatrix_mut(2, 1, 2, 2);
    ///
    /// let mut expected = mat![[7.0, 11.0], [8.0, 12.0f64]];
    /// assert_eq!(expected.as_mut(), submatrix);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn submatrix_mut(
        &mut self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatMut<'_, E> {
        self.as_mut()
            .submatrix_mut(row_start, col_start, nrows, ncols)
    }

    /// Returns a view over the submatrix starting at row `row_start`, and with number of rows
    /// `nrows`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `nrows <= self.nrows() - row_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn subrows_unchecked(&self, row_start: usize, nrows: usize) -> MatRef<'_, E> {
        self.as_ref().subrows_unchecked(row_start, nrows)
    }

    /// Returns a view over the submatrix starting at row `row_start`, and with number of rows
    /// `nrows`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `nrows <= self.nrows() - row_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn subrows_mut_unchecked(
        &mut self,
        row_start: usize,
        nrows: usize,
    ) -> MatMut<'_, E> {
        self.as_mut().subrows_mut_unchecked(row_start, nrows)
    }

    /// Returns a view over the submatrix starting at row `row_start`, and with number of rows
    /// `nrows`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `nrows <= self.nrows() - row_start`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_ref();
    /// let subrows = view.subrows(1, 2);
    ///
    /// let expected = mat![[2.0, 6.0, 10.0], [3.0, 7.0, 11.0],];
    /// assert_eq!(expected.as_ref(), subrows);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn subrows(&self, row_start: usize, nrows: usize) -> MatRef<'_, E> {
        self.as_ref().subrows(row_start, nrows)
    }

    /// Returns a view over the submatrix starting at row `row_start`, and with number of rows
    /// `nrows`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `nrows <= self.nrows() - row_start`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_mut();
    /// let subrows = view.subrows_mut(1, 2);
    ///
    /// let mut expected = mat![[2.0, 6.0, 10.0], [3.0, 7.0, 11.0],];
    /// assert_eq!(expected.as_mut(), subrows);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn subrows_mut(&mut self, row_start: usize, nrows: usize) -> MatMut<'_, E> {
        self.as_mut().subrows_mut(row_start, nrows)
    }

    /// Returns a view over the submatrix starting at column `col_start`, and with number of
    /// columns `ncols`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_start <= self.ncols()`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn subcols_unchecked(&self, col_start: usize, ncols: usize) -> MatRef<'_, E> {
        self.as_ref().subcols_unchecked(col_start, ncols)
    }

    /// Returns a view over the submatrix starting at column `col_start`, and with number of
    /// columns `ncols`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_start <= self.ncols()`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn subcols_mut_unchecked(
        &mut self,
        col_start: usize,
        ncols: usize,
    ) -> MatMut<'_, E> {
        self.as_mut().subcols_mut_unchecked(col_start, ncols)
    }

    /// Returns a view over the submatrix starting at column `col_start`, and with number of
    /// columns `ncols`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_start <= self.ncols()`.
    /// * `ncols <= self.ncols() - col_start`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_ref();
    /// let subcols = view.subcols(2, 1);
    ///
    /// let expected = mat![[9.0], [10.0], [11.0], [12.0f64]];
    /// assert_eq!(expected.as_ref(), subcols);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn subcols(&self, col_start: usize, ncols: usize) -> MatRef<'_, E> {
        self.as_ref().subcols(col_start, ncols)
    }

    /// Returns a view over the submatrix starting at column `col_start`, and with number of
    /// columns `ncols`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_start <= self.ncols()`.
    /// * `ncols <= self.ncols() - col_start`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_mut();
    /// let subcols = view.subcols_mut(2, 1);
    ///
    /// let mut expected = mat![[9.0], [10.0], [11.0], [12.0f64]];
    /// assert_eq!(expected.as_mut(), subcols);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn subcols_mut(&mut self, col_start: usize, ncols: usize) -> MatMut<'_, E> {
        self.as_mut().subcols_mut(col_start, ncols)
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Safety
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn row_unchecked(&self, row_idx: usize) -> RowRef<'_, E> {
        self.as_ref().row_unchecked(row_idx)
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Safety
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn row_mut_unchecked(&mut self, row_idx: usize) -> RowMut<'_, E> {
        self.as_mut().row_mut_unchecked(row_idx)
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub fn row(&self, row_idx: usize) -> RowRef<'_, E> {
        self.as_ref().row(row_idx)
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub fn row_mut(&mut self, row_idx: usize) -> RowMut<'_, E> {
        self.as_mut().row_mut(row_idx)
    }

    /// Returns views over the rows at the given indices.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx0 < self.nrows()`.
    /// * `row_idx1 < self.nrows()`.
    /// * `row_idx0 == row_idx1`.
    #[track_caller]
    #[inline(always)]
    pub fn two_rows_mut(
        &mut self,
        row_idx0: usize,
        row_idx1: usize,
    ) -> (RowMut<'_, E>, RowMut<'_, E>) {
        self.as_mut().two_rows_mut(row_idx0, row_idx1)
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn col_unchecked(&self, col_idx: usize) -> ColRef<'_, E> {
        self.as_ref().col_unchecked(col_idx)
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn col_mut_unchecked(&mut self, col_idx: usize) -> ColMut<'_, E> {
        self.as_mut().col_mut_unchecked(col_idx)
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn col(&self, col_idx: usize) -> ColRef<'_, E> {
        self.as_ref().col(col_idx)
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn col_mut(&mut self, col_idx: usize) -> ColMut<'_, E> {
        self.as_mut().col_mut(col_idx)
    }

    /// Returns views over the columns at the given indices.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_idx0 < self.ncols()`.
    /// * `col_idx1 < self.ncols()`.
    /// * `col_idx0 == col_idx1`.
    #[track_caller]
    #[inline(always)]
    pub fn two_cols_mut(
        &mut self,
        col_idx0: usize,
        col_idx1: usize,
    ) -> (ColMut<'_, E>, ColMut<'_, E>) {
        self.as_mut().two_cols_mut(col_idx0, col_idx1)
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal(&self) -> DiagRef<'_, E> {
        self.as_ref().column_vector_as_diagonal()
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal_mut(&mut self) -> DiagMut<'_, E> {
        self.as_mut().column_vector_as_diagonal_mut()
    }

    /// Returns a view over the diagonal of the matrix.
    #[inline]
    pub fn diagonal(&self) -> DiagRef<'_, E> {
        self.as_ref().diagonal()
    }

    /// Returns a view over the diagonal of the matrix.
    #[inline]
    pub fn diagonal_mut(&mut self) -> DiagMut<'_, E> {
        self.as_mut().diagonal_mut()
    }

    /// Returns an owning [`Mat`] of the data
    #[inline]
    pub fn to_owned(&self) -> Mat<E::Canonical>
    where
        E: Conjugate,
    {
        self.as_ref().to_owned()
    }

    /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
    #[inline]
    pub fn has_nan(&self) -> bool
    where
        E: ComplexField,
    {
        self.as_ref().has_nan()
    }

    /// Returns `true` if all of the elements are finite, otherwise returns `false`.
    #[inline]
    pub fn is_all_finite(&self) -> bool
    where
        E: ComplexField,
    {
        self.as_ref().is_all_finite()
    }

    /// Returns the maximum norm of `self`.
    #[inline]
    pub fn norm_max(&self) -> E::Real
    where
        E: ComplexField,
    {
        crate::linalg::reductions::norm_max::norm_max((*self).as_ref())
    }

    /// Returns the L1 norm of `self`.
    #[inline]
    pub fn norm_l1(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.as_ref().norm_l1()
    }

    /// Returns the L2 norm of `self`.
    #[inline]
    pub fn norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.as_ref().norm_l2()
    }

    /// Returns the squared L2 norm of `self`.
    #[inline]
    pub fn squared_norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.as_ref().squared_norm_l2()
    }

    /// Returns the sum of `self`.
    #[inline]
    pub fn sum(&self) -> E
    where
        E: ComplexField,
    {
        crate::linalg::reductions::sum::sum((*self).as_ref())
    }

    /// Kroneckor product of `self` and `rhs`.
    ///
    /// This is an allocating operation; see [`faer::linalg::kron`](crate::linalg::kron) for the
    /// allocation-free version or more info in general.
    #[inline]
    #[track_caller]
    pub fn kron(&self, rhs: impl As2D<E>) -> Mat<E>
    where
        E: ComplexField,
    {
        self.as_2d_ref().kron(rhs)
    }

    /// Returns an iterator that provides successive chunks of the columns of a view over this
    /// matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    #[inline]
    #[track_caller]
    pub fn col_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatRef<'_, E>> {
        self.as_ref().col_chunks(chunk_size)
    }

    /// Returns an iterator that provides successive chunks of the columns of a mutable view over
    /// this matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    #[inline]
    #[track_caller]
    pub fn col_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatMut<'_, E>> {
        self.as_mut().col_chunks_mut(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the columns of a view over
    /// this matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_col_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, E>> {
        self.as_ref().par_col_chunks(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the columns of a mutable view
    /// over this matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_col_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, E>> {
        self.as_mut().par_col_chunks_mut(chunk_size)
    }

    /// Returns an iterator that provides successive chunks of the rows of a view over this
    /// matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn row_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatRef<'_, E>> {
        self.as_ref().row_chunks(chunk_size)
    }

    /// Returns an iterator that provides successive chunks of the rows of a mutable view over
    /// this matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn row_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatMut<'_, E>> {
        self.as_mut().row_chunks_mut(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the rows of a view over this
    /// matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_row_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, E>> {
        self.as_ref().par_row_chunks(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the rows of a mutable view
    /// over this matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_row_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, E>> {
        self.as_mut().par_row_chunks_mut(chunk_size)
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col(&self, j: usize) -> GroupFor<E, &[E::Unit]> {
        self.as_ref().try_get_contiguous_col(j)
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col_mut(&mut self, j: usize) -> GroupFor<E, &mut [E::Unit]> {
        self.as_mut().try_get_contiguous_col_mut(j)
    }
}

impl<E: RealField> Mat<num_complex::Complex<E>> {
    /// Returns the real and imaginary components of `self`.
    #[inline(always)]
    pub fn real_imag(&self) -> num_complex::Complex<MatRef<'_, E>> {
        self.as_ref().real_imag()
    }

    /// Returns the real and imaginary components of `self`.
    #[inline(always)]
    pub fn real_imag_mut(&mut self) -> num_complex::Complex<MatMut<'_, E>> {
        self.as_mut().real_imag_mut()
    }
}

impl<E: Entity> Default for Mat<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Entity> Clone for Mat<E> {
    fn clone(&self) -> Self {
        let this = self.as_ref();
        unsafe {
            Self::from_fn(self.nrows(), self.ncols(), |i, j| {
                E::faer_from_units(E::faer_deref(this.get_unchecked(i, j)))
            })
        }
    }
}

impl<E: Entity> AsMatRef<E> for Mat<E> {
    #[inline]
    fn as_mat_ref(&self) -> MatRef<'_, E> {
        (*self).as_ref()
    }
}

impl<E: Entity> AsMatMut<E> for Mat<E> {
    #[inline]
    fn as_mat_mut(&mut self) -> MatMut<'_, E> {
        (*self).as_mut()
    }
}

impl<E: Entity> As2D<E> for Mat<E> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).as_ref()
    }
}

impl<E: Entity> As2DMut<E> for Mat<E> {
    #[inline]
    fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        (*self).as_mut()
    }
}

impl<E: Entity> core::fmt::Debug for Mat<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for Mat<E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &E {
        self.as_ref().get(row, col)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for Mat<E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut E {
        self.as_mut().get_mut(row, col)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::Matrix<E> for Mat<E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::DenseAccess<E> for Mat<E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

impl<E: Conjugate> ColBatch<E> for Mat<E> {
    type Owned = Mat<E::Canonical>;

    #[inline]
    #[track_caller]
    fn new_owned_zeros(nrows: usize, ncols: usize) -> Self::Owned {
        Mat::zeros(nrows, ncols)
    }

    #[inline]
    fn new_owned_copied(src: &Self) -> Self::Owned {
        src.to_owned()
    }

    #[inline]
    #[track_caller]
    fn resize_owned(owned: &mut Self::Owned, nrows: usize, ncols: usize) {
        owned.resize_with(nrows, ncols, |_, _| unsafe { core::mem::zeroed() });
    }
}

impl<E: Conjugate> RowBatch<E> for Mat<E> {
    type Owned = Mat<E::Canonical>;

    #[inline]
    #[track_caller]
    fn new_owned_zeros(nrows: usize, ncols: usize) -> Self::Owned {
        Mat::zeros(nrows, ncols)
    }

    #[inline]
    fn new_owned_copied(src: &Self) -> Self::Owned {
        src.to_owned()
    }

    #[inline]
    #[track_caller]
    fn resize_owned(owned: &mut Self::Owned, nrows: usize, ncols: usize) {
        owned.resize_with(nrows, ncols, |_, _| unsafe { core::mem::zeroed() });
    }
}

impl<E: Conjugate> ColBatchMut<E> for Mat<E> {}
impl<E: Conjugate> RowBatchMut<E> for Mat<E> {}
