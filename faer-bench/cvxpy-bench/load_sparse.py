import scipy.sparse as sp

testName = "unconstrainedQP"
testType = "sparse-mm"
LHS = sp.load_npz(f"{testName}/{testType}-lhs.npz")
RHS = sp.load_npz(f"{testName}/{testType}-rhs.npz")
out = sp.load_npz(f"{testName}/{testType}-out.npz")
assert ((LHS@RHS).toarray() == out).all()
print(out.shape)
print(out.nnz)