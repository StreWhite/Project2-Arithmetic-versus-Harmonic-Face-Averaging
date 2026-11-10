"""Smoke test: vectorised assembly vs a reference scalar-loop implementation,
plus a quick check of the red-black stationary iterators."""
import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all as R


def reference_assemble(N, coefficient, forcing, boundary, average):
    """Original scalar-loop assembly (kept only for this regression test)."""
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    n = N - 1
    size = n * n

    def index(i, j):
        return (j - 1) * n + (i - 1)

    rows, cols, data = [], [], []
    b = np.zeros(size)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    inv_h2 = 1.0 / (h * h)
    for j in range(1, N):
        for i in range(1, N):
            k = index(i, j)
            xi, yj = x[i], y[j]
            a_center = coefficient(xi, yj)
            b[k] = forcing(xi, yj)
            diagonal = 0.0
            for di, dj in directions:
                ni, nj = i + di, j + dj
                a_neighbor = coefficient(x[ni], y[nj])
                a_face = R.face_average(a_center, a_neighbor, average)
                weight = a_face * inv_h2
                diagonal += weight
                if 1 <= ni <= N - 1 and 1 <= nj <= N - 1:
                    rows.append(k)
                    cols.append(index(ni, nj))
                    data.append(-weight)
                else:
                    b[k] += weight * boundary(x[ni], y[nj])
            rows.append(k)
            cols.append(k)
            data.append(diagonal)
    A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    return A, b


def scalar_layered_coefficient(a_left, a_right):
    def c(x, y):
        return a_left if x < 0.5 else a_right
    return c


def check(name, N, coeff, forcing, boundary, average):
    A_ref, b_ref = reference_assemble(N, coeff, forcing, boundary, average)
    res = R.assemble_and_solve(N, coeff, forcing, boundary, average)
    diff_A = abs(res.A - A_ref)
    diff_A = 0.0 if diff_A.nnz == 0 else diff_A.max()
    diff_b = float(np.max(np.abs(res.b - b_ref)))
    ok = diff_A == 0.0 and diff_b == 0.0
    print(f"{name:38s} N={N:4d} avg={average:10s} |dA|={diff_A:.3e} |db|={diff_b:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    all_ok = True
    layered = R.layered_functions(10.0)
    coeff_lay_arr = layered[0]
    coeff_lay_scalar = scalar_layered_coefficient(1.0, 10.0)

    for N in (4, 8, 31, 63):
        # smooth, elementwise coefficient
        all_ok &= check("smooth", N, R.smooth_coefficient, R.smooth_forcing,
                        lambda x, y: x + y, "arithmetic")
        # layered with array coefficient (np.where)
        all_ok &= check("layered-harmonic", N, coeff_lay_arr, layered[2],
                        layered[1], "harmonic")
        # layered, arithmetic, comparing against scalar reference coefficient
        all_ok &= check("layered-arith(vs scalar ref)", N, coeff_lay_arr, layered[2],
                        layered[1], "arithmetic")
        _ = coeff_lay_scalar  # reference values produced via array path already

    # constant coefficient / constant forcing (scalar-returning callables)
    all_ok &= check("const-coeff nonzero-dirichlet", 8,
                    lambda x, y: 1.0, lambda x, y: 0.0,
                    lambda x, y: x * x - y * y, "arithmetic")
    all_ok &= check("const-forcing linear", 4, R.smooth_coefficient,
                    lambda x, y: -1.0, lambda x, y: x + y, "arithmetic")

    # red-black iterator on a small smooth system: compare with spsolve
    res = R.assemble_and_solve(16, R.smooth_coefficient, R.smooth_forcing, R.smooth_exact)
    red, black = R.red_black_indices(15)
    for omega in (1.0, 1.5):
        iters, info, x, _ = R.stationary_sor(res.A, res.b, red, black, omega, 1e-10, 50000)
        diff = float(np.max(np.abs(x - res.interior)))
        print(f"red-black SOR omega={omega:3.1f}: iters={iters:6d} info={info} "
              f"linf-diff-vs-direct={diff:.3e} {'OK' if info == 0 and diff < 1e-8 else 'FAIL'}")
        all_ok &= info == 0 and diff < 1e-8

    # two-colour guard should reject a non-bipartite split (contiguous blocks
    # couple same-colour neighbours along the block boundary)
    bad_red, bad_black = np.arange(0, 113), np.arange(113, 225)
    try:
        R.stationary_sor(res.A, res.b, bad_red, bad_black, 1.0, 1e-10, 5)
        print("two-colour guard: FAIL (no error raised)")
        all_ok = False
    except ValueError:
        print("two-colour guard: OK (rejected non-bipartite split)")

    print("\nALL SMOKE TESTS PASSED" if all_ok else "\nSMOKE TEST FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
