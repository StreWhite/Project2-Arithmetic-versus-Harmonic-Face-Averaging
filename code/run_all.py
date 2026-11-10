"""Reproduce every numerical result for MTH321 Project 2.

The script solves

    -div(a grad u) = f  in (0, 1)^2,
    u = g              on the boundary,

with a conservative five-point finite-difference operator.  It runs a smooth
manufactured-solution verification, a two-material interface comparison, a
coefficient-contrast study, a sparse-direct versus stationary-iterator
(Gauss-Seidel/SOR) depth comparison, and a matched direct/CG cost study.
Tables and LaTeX macros are generated from the recorded results so that the
report cannot silently disagree with the code.

Implementation notes
--------------------
* Assembly is fully vectorised (NumPy array operations, no per-node Python
  loops), per the Project 2 brief's vectorisation requirement.  The floating
  point accumulation order (E, W, N, S) matches the original scalar loop so
  that all recorded results reproduce bit-for-bit.
* ``coefficient``, ``forcing``, and ``boundary`` callables must be elementwise:
  they receive ndarrays and return ndarrays (scalars broadcast to full grids).
* ``--report-only`` regenerates the LaTeX tables/macros from the recorded CSV
  files without rerunning experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg, eigsh, spsolve
from scipy.sparse.linalg import norm as sparse_norm


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

Average = str  # "arithmetic" | "harmonic"

INT_COLUMNS = {"N", "unknowns", "nnz", "cg_iterations", "cg_info", "iterations", "info"}


@dataclass
class SolveResult:
    N: int
    h: float
    x: np.ndarray
    y: np.ndarray
    U: np.ndarray
    A: csr_matrix
    b: np.ndarray
    interior: np.ndarray
    residual: float
    symmetry: float
    solve_time: float


def face_average(left, right, average: Average):
    """Return one shared positive coefficient for a grid face (scalars or arrays)."""
    if average == "arithmetic":
        return 0.5 * (left + right)
    if average == "harmonic":
        return 2.0 * left * right / (left + right)
    raise ValueError(f"Unknown face average: {average}")


def _as_grid(values, shape: tuple[int, ...]) -> np.ndarray:
    """Coerce a callable's return value to a float array of the given shape.

    Elementwise callables return an array of the right shape; scalar-returning
    callables (constant data) are broadcast to the full grid.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr))
    if arr.shape != shape:
        raise ValueError(f"expected array of shape {shape}, got {arr.shape}")
    return arr


def assemble_and_solve(
    N: int,
    coefficient: Callable[[np.ndarray, np.ndarray], np.ndarray],
    forcing: Callable[[np.ndarray, np.ndarray], np.ndarray],
    boundary: Callable[[np.ndarray, np.ndarray], np.ndarray],
    average: Average = "arithmetic",
) -> SolveResult:
    """Assemble the conservative operator and solve for interior nodal values.

    N is the number of intervals in each coordinate direction, so h=1/N and
    there are (N-1)^2 interior unknowns. Dirichlet data are eliminated and
    moved to the right-hand side with their actual face coefficients.

    The assembly is vectorised: nodal coefficient values are evaluated on the
    whole grid at once, the four directional face-weight arrays are formed by
    array slicing, and the sparse matrix is built from index arrays in COO
    format.  The index map k(i, j) = (j-1)*(N-1) + (i-1) equals the F-order
    ravel index of an (N-1, N-1) array with axes [i-1, j-1], so Fortran-order
    ravel throughout replaces the explicit double loop.
    """
    if N < 2:
        raise ValueError("N must be at least 2")

    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    n = N - 1
    size = n * n

    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    a_grid = _as_grid(coefficient(Xg, Yg), (N + 1, N + 1))
    if np.any(a_grid <= 0):
        raise ValueError("The diffusion coefficient must be positive")

    # Face weights a_face / h^2 in the four directions, shape (n, n), axes
    # [i-1, j-1].  East face of (i, j) lies between nodes (i, j) and (i+1, j).
    a_c = a_grid[1:-1, 1:-1]
    inv_h2 = 1.0 / (h * h)
    wE = face_average(a_c, a_grid[2:, 1:-1], average) * inv_h2
    wW = face_average(a_c, a_grid[:-2, 1:-1], average) * inv_h2
    wN = face_average(a_c, a_grid[1:-1, 2:], average) * inv_h2
    wS = face_average(a_c, a_grid[1:-1, :-2], average) * inv_h2

    # K[i-1, j-1] = k(i, j); Fortran-order ravel recovers the linear index.
    K = np.arange(size).reshape((n, n), order="F")

    def F(a: np.ndarray) -> np.ndarray:
        return np.ravel(a, order="F")

    # Interior couplings.  East couples (i, j)-(i+1, j) for i <= N-2, i.e. all
    # but the last i-slice; similarly for the other three directions.
    rows = np.concatenate([F(K), F(K[:-1, :]), F(K[1:, :]), F(K[:, :-1]), F(K[:, 1:])])
    cols = np.concatenate(
        [F(K), F(K[:-1, :]) + 1, F(K[1:, :]) - 1, F(K[:, :-1]) + n, F(K[:, 1:]) - n]
    )
    data = np.concatenate(
        [
            F(wE + wW + wN + wS),
            -F(wE[:-1, :]),
            -F(wW[1:, :]),
            -F(wN[:, :-1]),
            -F(wS[:, 1:]),
        ]
    )
    A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()

    # Right-hand side: forcing on interior nodes plus the Dirichlet edge
    # contributions of the four boundary strips, each with its actual face
    # coefficient (never g/h^2).
    Xc, Yc = Xg[1:-1, 1:-1], Yg[1:-1, 1:-1]
    b = F(_as_grid(forcing(Xc, Yc), (n, n)))
    b[F(K[-1, :])] += F(wE[-1, :]) * _as_grid(boundary(np.full(n, x[N]), y[1:-1]), (n,))
    b[F(K[0, :])] += F(wW[0, :]) * _as_grid(boundary(np.full(n, x[0]), y[1:-1]), (n,))
    b[F(K[:, -1])] += F(wN[:, -1]) * _as_grid(boundary(x[1:-1], np.full(n, y[N])), (n,))
    b[F(K[:, 0])] += F(wS[:, 0]) * _as_grid(boundary(x[1:-1], np.full(n, y[0])), (n,))

    start = time.perf_counter()
    interior = spsolve(A, b)
    elapsed = time.perf_counter() - start

    U = _as_grid(boundary(Xg, Yg), (N + 1, N + 1)).copy()
    U[1:-1, 1:-1] = interior.reshape((n, n), order="F")

    residual = np.linalg.norm(A @ interior - b) / max(np.linalg.norm(b), 1.0)
    asym = A - A.T
    symmetry = sparse_norm(asym) / sparse_norm(A)
    return SolveResult(
        N=N,
        h=h,
        x=x,
        y=y,
        U=U,
        A=A,
        b=b,
        interior=interior,
        residual=float(residual),
        symmetry=float(symmetry),
        solve_time=elapsed,
    )


# ---------------------------------------------------------------------------
# Smooth benchmark (a = 1 + x).  Several studies reuse the same linear
# systems, so they are assembled once and cached.
# ---------------------------------------------------------------------------

_SMOOTH_SYSTEMS: dict[int, SolveResult] = {}


def smooth_system(N: int) -> SolveResult:
    """Return the (cached) smooth-benchmark system with N intervals."""
    if N not in _SMOOTH_SYSTEMS:
        _SMOOTH_SYSTEMS[N] = assemble_and_solve(
            N, smooth_coefficient, smooth_forcing, smooth_exact, "arithmetic"
        )
    return _SMOOTH_SYSTEMS[N]


def error_norms(result: SolveResult, exact: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> tuple[float, float]:
    """Discrete L2 norm h*||e||_2 and nodal maximum norm on interior nodes."""
    X, Y = np.meshgrid(result.x[1:-1], result.y[1:-1], indexing="ij")
    error = result.U[1:-1, 1:-1] - exact(X, Y)
    l2 = result.h * np.linalg.norm(error.ravel())
    linf = np.max(np.abs(error))
    return float(l2), float(linf)


def fitted_order(h: list[float], error: list[float]) -> float:
    return float(np.polyfit(np.log(h), np.log(error), 1)[0])


def pairwise_orders(h: list[float], error: list[float]) -> list[float]:
    orders = [math.nan]
    for k in range(1, len(h)):
        orders.append(math.log(error[k - 1] / error[k]) / math.log(h[k - 1] / h[k]))
    return orders


def smooth_coefficient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    del y
    return 1.0 + x


def smooth_exact(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def smooth_forcing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        2.0 * np.pi**2 * (1.0 + x) * np.sin(np.pi * x) * np.sin(np.pi * y)
        - np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    )


def layered_functions(contrast: float):
    """Return a, exact u, f, and exact x-flux for a two-layer medium.

    The boundary values are u(0,y)=0 and u(1,y)=1.  The interface is x=1/2.
    """
    if contrast <= 0:
        raise ValueError("contrast must be positive")
    a_left, a_right = 1.0, float(contrast)
    slope_flux = 2.0 * a_left * a_right / (a_left + a_right)
    u_interface = a_right / (a_left + a_right)

    def coefficient(x, y):
        del y
        return np.where(np.asarray(x, dtype=float) < 0.5, a_left, a_right)

    def exact(x, y):
        del y
        xx = np.asarray(x, dtype=float)
        values = np.where(
            xx <= 0.5,
            (slope_flux / a_left) * xx,
            u_interface + (slope_flux / a_right) * (xx - 0.5),
        )
        return float(values) if values.ndim == 0 else values

    def forcing(x, y):
        del x, y
        return 0.0

    return coefficient, exact, forcing, -slope_flux


def face_flux_profile(result: SolveResult, coefficient, average: Average, row: int | None = None):
    """Return numerical x-flux on every vertical face along one grid row."""
    j = result.N // 2 if row is None else row
    mids = 0.5 * (result.x[:-1] + result.x[1:])
    yj = np.full(result.N, result.y[j])
    a_left = _as_grid(coefficient(result.x[:-1], yj), (result.N,))
    a_right = _as_grid(coefficient(result.x[1:], yj), (result.N,))
    a_face = face_average(a_left, a_right, average)
    flux = -a_face * (result.U[1:, j] - result.U[:-1, j]) / result.h
    return mids, flux


def interface_diagnostics(result: SolveResult, coefficient, average: Average, exact_flux: float):
    mids, flux = face_flux_profile(result, coefficient, average)
    interface_index = int(np.argmin(np.abs(mids - 0.5)))
    interface_flux = float(flux[interface_index])
    left_flux = float(flux[max(interface_index - 1, 0)])
    right_flux = float(flux[min(interface_index + 1, len(flux) - 1)])
    return {
        "interface_flux": interface_flux,
        "flux_error": abs(interface_flux - exact_flux),
        "flux_jump": abs(left_flux - right_flux),
        "x_faces": mids,
        "flux_profile": flux,
    }


def cg_iterations(A: csr_matrix, b: np.ndarray, rtol: float = 1e-10) -> tuple[int, int]:
    count = 0

    def callback(_):
        nonlocal count
        count += 1

    try:
        _, info = cg(A, b, rtol=rtol, atol=0.0, callback=callback, maxiter=20000)
    except TypeError:  # SciPy < 1.12
        _, info = cg(A, b, tol=rtol, callback=callback, maxiter=20000)
    return count, int(info)


def smallest_eigenvalue(A: csr_matrix) -> float:
    """Smallest eigenvalue estimate with eigsh (SPD corroboration).

    Returns NaN instead of crashing if ARPACK fails to converge; the value is
    numerical corroboration, not the primary positive-definiteness evidence.
    """
    try:
        v0 = np.ones(A.shape[0]) / math.sqrt(A.shape[0])
        return float(eigsh(A, k=1, which="SA", return_eigenvectors=False, v0=v0)[0])
    except Exception:
        return math.nan


def condition_estimate(A: csr_matrix) -> float:
    try:
        largest = float(eigsh(A, k=1, which="LA", return_eigenvectors=False)[0])
        smallest = float(eigsh(A, k=1, which="SA", return_eigenvectors=False)[0])
        return largest / smallest
    except Exception:
        return math.nan


def verify_forcing_independently() -> float:
    """Check the coded forcing against separately differentiated flux terms."""
    points = ((0.17, 0.23), (0.41, 0.72), (0.83, 0.38))
    discrepancies = []
    for x, y in points:
        ux = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        uxx = -np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        uyy = uxx
        independently_derived = -(ux + (1.0 + x) * (uxx + uyy))
        discrepancies.append(abs(smooth_forcing(x, y) - independently_derived))
    return float(max(discrepancies))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_mesh_conditioning():
    """Hold material contrast fixed while refining the aligned interface grid."""
    coefficient, exact, forcing, _ = layered_functions(10.0)
    rows = []
    for N in (31, 63, 127, 255):
        result = assemble_and_solve(N, coefficient, forcing, exact, "harmonic")
        iterations, info = cg_iterations(result.A, result.b)
        rows.append({"N": N, "h": result.h, "unknowns": (N - 1)**2,
                     "cg_iterations": iterations, "cg_info": info,
                     "scaled_residual": result.residual})
    write_csv(RESULTS / "mesh_conditioning.csv", rows)
    return rows


def run_algebraic_crosscheck():
    """Independent CG solve and SPD corroboration on a representative mesh."""
    N = 64
    result = smooth_system(N)
    try:
        other, info = cg(result.A, result.b, rtol=1e-12, atol=0.0,
                         maxiter=20000)
    except TypeError:
        other, info = cg(result.A, result.b, tol=1e-12, maxiter=20000)
    if info != 0:
        raise RuntimeError(f"Independent CG cross-check failed: info={info}")
    discretisation_error = error_norms(result, smooth_exact)[1]
    rows = [{"N": N, "direct_scaled_residual": result.residual,
             "cg_scaled_residual": float(np.linalg.norm(result.A @ other - result.b)
                                         / max(np.linalg.norm(result.b), 1.0)),
             "solution_Linf_difference": float(np.max(np.abs(result.interior - other))),
             "discretisation_Linf_error": discretisation_error,
             "lambda_min_eigsh": smallest_eigenvalue(result.A),
             "forcing_check_max_difference": verify_forcing_independently()}]
    write_csv(RESULTS / "independent_checks.csv", rows)
    return rows


def run_boundary_and_row_audit():
    """Independent exact polynomial and hand-calculated variable-a boundary row."""
    # With a=1+x and u=x+y, the exact source is -d_x(a*1)=-1.
    linear = assemble_and_solve(4, smooth_coefficient, lambda x, y: -1.0,
                                lambda x, y: x + y)
    # P=(1,1), h=1/4: face weights E,W,N,S = 22,18,20,20.
    # West and south neighbours are prescribed; E and N are interior.
    expected = np.zeros(9)
    expected[0], expected[1], expected[3] = 80.0, -22.0, -20.0
    expected_rhs = -1.0 + 18.0 * 0.25 + 20.0 * 0.25
    row_error = float(np.max(np.abs(linear.A.getrow(0).toarray().ravel() - expected)))
    rhs_error = float(abs(linear.b[0] - expected_rhs))
    linear_error = error_norms(linear, lambda x, y: x + y)[1]
    if max(row_error, rhs_error, linear_error) > 1e-11:
        raise AssertionError("Hand-checked variable-coefficient boundary row failed")
    audit = [{"N": 4, "row_max_difference": row_error,
              "rhs_difference": rhs_error, "exact_solution_Linf_error": linear_error,
              "expected_rhs": expected_rhs, "assembled_rhs": float(linear.b[0]),
              "nonzeros": linear.A.nnz}]
    write_csv(RESULTS / "hand_checked_row.csv", audit)

    # Course bank P2: -Delta(x^2-y^2)=0 and nonzero Dirichlet data.
    # This is exact for centred second differences, so do not fit a rate.
    exact = lambda x, y: x * x - y * y
    rows = []
    for N in (4, 8, 16, 32):
        result = assemble_and_solve(N, lambda x, y: 1.0,
                                    lambda x, y: 0.0, exact)
        l2, linf = error_norms(result, exact)
        rows.append({"N": N, "h": result.h, "L2_error": l2,
                     "Linf_error": linf, "scaled_residual": result.residual})
        if linf > 1e-11:
            raise AssertionError(f"Nonzero-Dirichlet polynomial check failed N={N}")
    write_csv(RESULTS / "nonzero_dirichlet_check.csv", rows)
    return audit, rows


def run_tolerance_sweep():
    """Show when tightening CG stops changing the quantity of interest."""
    result = smooth_system(64)
    X, Y = np.meshgrid(result.x[1:-1], result.y[1:-1], indexing="ij")
    truth = smooth_exact(X, Y).ravel(order="F")
    rows = []
    for rtol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        iterations = 0

        def count(_):
            nonlocal iterations
            iterations += 1

        try:
            candidate, info = cg(result.A, result.b, rtol=rtol, atol=0.0,
                                 maxiter=20000, callback=count)
        except TypeError:
            candidate, info = cg(result.A, result.b, tol=rtol,
                                 maxiter=20000, callback=count)
        if info != 0:
            raise RuntimeError(f"Tolerance sweep failed rtol={rtol}: {info}")
        rows.append({"N": 64, "rtol": rtol, "atol": 0.0,
                     "cg_iterations": iterations, "cg_info": info,
                     "scaled_residual": float(np.linalg.norm(result.A @ candidate - result.b)
                                              / max(np.linalg.norm(result.b), 1.0)),
                     "Linf_error": float(np.max(np.abs(candidate - truth))),
                     "Linf_difference_from_direct": float(np.max(np.abs(candidate - result.interior)))})
    write_csv(RESULTS / "tolerance_sweep.csv", rows)
    return rows


def run_solver_cost_comparison():
    """Matched direct/CG comparison on the smooth benchmark, with wall times."""
    rows = []
    for N in (16, 32, 64, 128):
        result = smooth_system(N)
        direct_times = []
        cg_times = []
        cg_counts = []
        for _ in range(3):
            start = time.perf_counter()
            direct = spsolve(result.A, result.b)
            direct_times.append(time.perf_counter() - start)
            iterations = 0

            def count(_):
                nonlocal iterations
                iterations += 1

            start = time.perf_counter()
            try:
                iterative, info = cg(result.A, result.b, rtol=1e-12,
                                     atol=0.0, maxiter=20000, callback=count)
            except TypeError:
                iterative, info = cg(result.A, result.b, tol=1e-12,
                                     maxiter=20000, callback=count)
            cg_times.append(time.perf_counter() - start)
            cg_counts.append(iterations)
            if info != 0:
                raise RuntimeError(f"CG failed in cost study N={N}: {info}")
        X, Y = np.meshgrid(result.x[1:-1], result.y[1:-1], indexing="ij")
        truth = smooth_exact(X, Y).ravel(order="F")
        direct_error = float(np.max(np.abs(direct - truth)))
        cg_error = float(np.max(np.abs(iterative - truth)))
        rows.append({"N": N, "h": result.h, "unknowns": result.A.shape[0],
                     "nnz": result.A.nnz, "direct_Linf_error": direct_error,
                     "cg_Linf_error": cg_error,
                     "solution_Linf_difference": float(np.max(np.abs(direct - iterative))),
                     "cg_iterations": cg_counts[-1], "cg_info": info,
                     "direct_median_s": float(np.median(direct_times)),
                     "cg_median_s": float(np.median(cg_times)),
                     "direct_scaled_residual": float(np.linalg.norm(result.A @ direct - result.b)
                                                    / max(np.linalg.norm(result.b), 1.0)),
                     "cg_scaled_residual": float(np.linalg.norm(result.A @ iterative - result.b)
                                                / max(np.linalg.norm(result.b), 1.0))})
    write_csv(RESULTS / "solver_cost_comparison.csv", rows)
    return rows


def red_black_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Split interior unknowns into the two colours of the grid checkerboard.

    With k(i, j) = (j-1)*n + (i-1), the parity of i+j is (k % n + k // n) % 2;
    neighbours k+/-1 and k+/-n always change that parity, so the stencil graph
    is two-colourable and a red-black sweep is well defined.
    """
    k = np.arange(n * n)
    parity = ((k % n) + (k // n)) % 2
    return np.flatnonzero(parity == 0), np.flatnonzero(parity == 1)


def stationary_sor(A: csr_matrix, b: np.ndarray, red: np.ndarray, black: np.ndarray,
                   omega: float, rtol: float, maxiter: int) -> tuple[int, int, np.ndarray, float]:
    """Red-black SOR (omega = 1 gives Gauss-Seidel) with a residual stopping rule.

    Each sweep updates all red unknowns from the black ones, then all black
    unknowns from the updated red ones.  The iteration stops when the scaled
    algebraic residual ||Ax - b|| / max(||b||, 1) drops below rtol; the
    stopping criterion is the residual, not the iterate difference.
    """
    # The two-colour split is only valid if same-colour nodes are uncoupled.
    for same in (A[red][:, red], A[black][:, black]):
        coo = same.tocoo()
        if np.any(coo.row != coo.col):
            raise ValueError("stencil graph is not two-colourable under this split")

    AB = A[red][:, black]
    BA = A[black][:, red]
    d = A.diagonal()
    b_red, b_black = b[red], b[black]
    d_red, d_black = d[red], d[black]
    x_red = np.zeros(len(red))
    x_black = np.zeros(len(black))
    b_norm = max(np.linalg.norm(b), 1.0)
    x = np.zeros_like(b)

    start = time.perf_counter()
    iterations = 0
    info = 1
    while iterations < maxiter:
        iterations += 1
        x_red = (1.0 - omega) * x_red + omega * (b_red - AB @ x_black) / d_red
        x_black = (1.0 - omega) * x_black + omega * (b_black - BA @ x_red) / d_black
        x[red] = x_red
        x[black] = x_black
        if np.linalg.norm(A @ x - b) / b_norm < rtol:
            info = 0
            break
    elapsed = time.perf_counter() - start
    return iterations, info, x, elapsed


def run_stationary_iterator_comparison():
    """Depth item: sparse direct versus Gauss-Seidel/SOR at matched accuracy.

    Red-black Gauss-Seidel (omega = 1) and SOR with the Poisson-model optimal
    relaxation parameter omega = 2 / (1 + sqrt(1 - cos^2(pi h))) are driven to
    the same scaled residual target (1e-10) on the smooth benchmark, with the
    same meshes, norm, and tolerance as the direct/CG cost study.  If the
    stationary iteration does not reach the target within maxiter sweeps, the
    row records info = 1 rather than silently reporting a failure as success.
    """
    rows = []
    for N in (16, 32, 64, 128):
        result = smooth_system(N)
        red, black = red_black_indices(N - 1)
        rho_jacobi = math.cos(math.pi * result.h)
        omega_opt = 2.0 / (1.0 + math.sqrt(1.0 - rho_jacobi**2))
        rows.append({
            "N": N, "method": "spsolve", "omega": math.nan,
            "iterations": math.nan, "info": 0,
            "target_rtol": 1e-10,
            "scaled_residual": result.residual,
            "linf_diff_direct": 0.0,
            "wall_time_s": result.solve_time,
        })
        for method, omega, maxiter in (
            ("gauss-seidel", 1.0, 80000),
            ("sor", omega_opt, 20000),
        ):
            iterations, info, x, elapsed = stationary_sor(
                result.A, result.b, red, black, omega, 1e-10, maxiter
            )
            rows.append({
                "N": N, "method": method, "omega": omega,
                "iterations": iterations, "info": info,
                "target_rtol": 1e-10,
                "scaled_residual": float(np.linalg.norm(result.A @ x - result.b)
                                         / max(np.linalg.norm(result.b), 1.0)),
                "linf_diff_direct": (float(np.max(np.abs(x - result.interior)))
                                     if info == 0 else math.nan),
                "wall_time_s": elapsed,
            })
    write_csv(RESULTS / "stationary_iterator_comparison.csv", rows)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table to {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    """Read a generated CSV and recover numeric columns where possible."""
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row: dict = {}
            for key, value in raw.items():
                if key == "average" or key == "method":
                    row[key] = value
                    continue
                try:
                    number = float(value)
                    row[key] = int(number) if key in INT_COLUMNS and math.isfinite(number) else number
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


def require_csv(name: str) -> list[dict]:
    """Load a results CSV, with a clear message if it has not been generated."""
    path = RESULTS / name
    if not path.exists():
        available = sorted(p.name for p in RESULTS.glob("*.csv"))
        raise SystemExit(
            f"Missing {path}.\n"
            f"'--report-only' needs the CSV records of a full run; available files: {available}.\n"
            f"Run 'python code/run_all.py' first."
        )
    return read_csv(path)


def run_smooth_study():
    Ns = [16, 32, 64, 128]
    rows = []
    final = None
    for N in Ns:
        result = smooth_system(N)
        l2, linf = error_norms(result, smooth_exact)
        rows.append(
            {
                "N": N,
                "h": result.h,
                "unknowns": (N - 1) ** 2,
                "L2_error": l2,
                "Linf_error": linf,
                "scaled_residual": result.residual,
                "symmetry_ratio": result.symmetry,
                "solve_time_s": result.solve_time,
            }
        )
        final = result

    h = [row["h"] for row in rows]
    l2s = [row["L2_error"] for row in rows]
    linfs = [row["Linf_error"] for row in rows]
    for row, p2, pi in zip(rows, pairwise_orders(h, l2s), pairwise_orders(h, linfs)):
        row["L2_pair_order"] = p2
        row["Linf_pair_order"] = pi
    write_csv(RESULTS / "smooth_convergence.csv", rows)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.loglog(h, l2s, "o-", label=r"discrete $L^2$")
    ax.loglog(h, linfs, "s-", label=r"nodal $L^\infty$")
    reference = l2s[0] * (np.asarray(h) / h[0]) ** 2
    ax.loglog(h, reference, "--", color="0.35", label=r"$O(h^2)$ reference")
    ax.invert_xaxis()
    ax.set_xlabel("mesh spacing $h$")
    ax.set_ylabel("error")
    ax.set_title("Smooth-coefficient refinement study")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01_smooth_convergence.png", dpi=220)
    plt.close(fig)

    assert final is not None
    X, Y = np.meshgrid(final.x, final.y, indexing="ij")
    exact_grid = smooth_exact(X, Y)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    im0 = axes[0].contourf(X, Y, final.U, levels=25, cmap="viridis")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title(f"Numerical solution, N={final.N}")
    im1 = axes[1].contourf(X, Y, np.abs(final.U - exact_grid), levels=25, cmap="magma")
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("Absolute nodal error")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02_smooth_solution_error.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.spy(final.A, markersize=0.25)
    ax.set_title(f"Sparsity pattern, N={final.N}")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03_matrix_spy.png", dpi=220)
    plt.close(fig)

    return rows, fitted_order(h, l2s), fitted_order(h, linfs)


def run_layered_study():
    Ns = [31, 63, 127, 255]
    coefficient, exact, forcing, exact_flux = layered_functions(10.0)
    all_rows: list[dict] = []
    final_results: dict[str, SolveResult] = {}
    final_diags: dict[str, dict] = {}

    for average in ("arithmetic", "harmonic"):
        for N in Ns:
            result = assemble_and_solve(N, coefficient, forcing, exact, average)
            l2, linf = error_norms(result, exact)
            diag = interface_diagnostics(result, coefficient, average, exact_flux)
            bounds_defect = max(0.0, -float(result.U.min()), float(result.U.max()) - 1.0)
            all_rows.append(
                {
                    "average": average,
                    "N": N,
                    "h": result.h,
                    "unknowns": (N - 1) ** 2,
                    "L2_error": l2,
                    "Linf_error": linf,
                    "interface_flux": diag["interface_flux"],
                    "flux_error": diag["flux_error"],
                    "flux_jump": diag["flux_jump"],
                    "bounds_defect": bounds_defect,
                    "scaled_residual": result.residual,
                    "symmetry_ratio": result.symmetry,
                    "solve_time_s": result.solve_time,
                }
            )
            if N == Ns[-1]:
                final_results[average] = result
                final_diags[average] = diag

    for average in ("arithmetic", "harmonic"):
        subset = [r for r in all_rows if r["average"] == average]
        h = [r["h"] for r in subset]
        l2 = [r["L2_error"] for r in subset]
        linf = [r["Linf_error"] for r in subset]
        for row, p2, pi in zip(subset, pairwise_orders(h, l2), pairwise_orders(h, linf)):
            row["L2_pair_order"] = p2
            row["Linf_pair_order"] = pi
    write_csv(RESULTS / "layered_comparison.csv", all_rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xx = np.linspace(0.0, 1.0, 1001)
    ax.plot(xx, exact(xx, np.zeros_like(xx)), "k-", lw=2.2, label="exact")
    j = Ns[-1] // 2
    for average, style in (("arithmetic", "--"), ("harmonic", ":")):
        result = final_results[average]
        ax.plot(result.x, result.U[:, j], style, lw=1.8, label=average)
    ax.axvline(0.5, color="0.5", lw=1, label="material interface")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, 0.5)")
    ax.set_title("Layered-medium centreline solution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04_layered_solution.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for average, style in (("arithmetic", "--"), ("harmonic", ":")):
        diag = final_diags[average]
        ax.plot(diag["x_faces"], diag["flux_profile"], style, lw=1.8, label=average)
    ax.axhline(exact_flux, color="k", lw=2.0, label="exact flux")
    ax.axvline(0.5, color="0.5", lw=1)
    ax.set_xlabel("vertical-face midpoint x")
    ax.set_ylabel(r"numerical flux $-a u_x$")
    ax.set_title("Normal flux across the layered medium")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_layered_flux.png", dpi=220)
    plt.close(fig)

    return all_rows


def run_contrast_study():
    contrasts = [1.0, 10.0, 100.0, 1000.0]
    N = 63
    rows = []
    for contrast in contrasts:
        coefficient, exact, forcing, exact_flux = layered_functions(contrast)
        for average in ("arithmetic", "harmonic"):
            result = assemble_and_solve(N, coefficient, forcing, exact, average)
            l2, linf = error_norms(result, exact)
            diag = interface_diagnostics(result, coefficient, average, exact_flux)
            iterations, info = cg_iterations(result.A, result.b)
            rows.append(
                {
                    "contrast": contrast,
                    "average": average,
                    "N": N,
                    "L2_error": l2,
                    "Linf_error": linf,
                    "flux_error": diag["flux_error"],
                    "cg_iterations": iterations,
                    "cg_info": info,
                    "condition_estimate": condition_estimate(result.A),
                }
            )
    write_csv(RESULTS / "contrast_study.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for average, marker in (("arithmetic", "o"), ("harmonic", "s")):
        subset = [r for r in rows if r["average"] == average]
        c = [r["contrast"] for r in subset]
        e = [max(r["Linf_error"], 1e-18) for r in subset]
        it = [r["cg_iterations"] for r in subset]
        axes[0].loglog(c, e, marker + "-", label=average)
        axes[1].semilogx(c, it, marker + "-", label=average)
    axes[0].set_xlabel(r"coefficient contrast $a_R/a_L$")
    axes[0].set_ylabel(r"nodal $L^\infty$ error")
    axes[1].set_xlabel(r"coefficient contrast $a_R/a_L$")
    axes[1].set_ylabel("unpreconditioned CG iterations")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"Coefficient-contrast study, N={N}")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06_contrast_study.png", dpi=220)
    plt.close(fig)
    return rows


def latex_float(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}e}"


METHOD_LABELS = {
    "spsolve": "spsolve",
    "gauss-seidel": "Gauss--Seidel",
    "sor": "SOR",
}


def write_latex_outputs(smooth_rows, smooth_l2_order, smooth_linf_order,
                        layered_rows, contrast_rows, mesh_rows, check_rows,
                        cost_rows, tol_rows, iterator_rows):
    finest_smooth = smooth_rows[-1]
    harmonic_finest = [r for r in layered_rows if r["average"] == "harmonic"][-1]
    arithmetic_finest = [r for r in layered_rows if r["average"] == "arithmetic"][-1]
    check = check_rows[0]
    # tolerance_sweep rows are ordered loosest (1e-4) to tightest (1e-12);
    # index 0 is the loose and index 2 the mid tolerance quoted in the report.
    tol_loose = tol_rows[0]
    tol_mid = tol_rows[2]
    direct_cg_max_diff = max(r["solution_Linf_difference"] for r in cost_rows)

    macros = (
        "% Generated by code/run_all.py. Do not edit numerical values by hand.\n"
        f"\\newcommand{{\\SmoothLTwoOrder}}{{{smooth_l2_order:.3f}}}\n"
        f"\\newcommand{{\\SmoothLInfOrder}}{{{smooth_linf_order:.3f}}}\n"
        f"\\newcommand{{\\SmoothFinestLInf}}{{{latex_float(finest_smooth['Linf_error'])}}}\n"
        f"\\newcommand{{\\SmoothFinestResidual}}{{{latex_float(finest_smooth['scaled_residual'])}}}\n"
        f"\\newcommand{{\\HarmonicFinestLInf}}{{{latex_float(harmonic_finest['Linf_error'])}}}\n"
        f"\\newcommand{{\\ArithmeticFinestLInf}}{{{latex_float(arithmetic_finest['Linf_error'])}}}\n"
        f"\\newcommand{{\\HarmonicFluxError}}{{{latex_float(harmonic_finest['flux_error'])}}}\n"
        f"\\newcommand{{\\ArithmeticFluxError}}{{{latex_float(arithmetic_finest['flux_error'])}}}\n"
        f"\\newcommand{{\\SmoothLambdaMin}}{{{latex_float(check.get('lambda_min_eigsh', math.nan))}}}\n"
        f"\\newcommand{{\\ForcingCheckDiff}}{{{latex_float(check.get('forcing_check_max_difference', math.nan))}}}\n"
        f"\\newcommand{{\\CrossCheckSolverDiff}}{{{latex_float(check.get('solution_Linf_difference', math.nan))}}}\n"
        f"\\newcommand{{\\CrossCheckLinf}}{{{latex_float(check.get('discretisation_Linf_error', math.nan))}}}\n"
        f"\\newcommand{{\\DirectCGMaxDiff}}{{{latex_float(direct_cg_max_diff)}}}\n"
        f"\\newcommand{{\\SmoothCGItersCoarse}}{{{cost_rows[0]['cg_iterations']}}}\n"
        f"\\newcommand{{\\SmoothCGItersFine}}{{{cost_rows[-1]['cg_iterations']}}}\n"
        f"\\newcommand{{\\MeshCondItersCoarse}}{{{mesh_rows[0]['cg_iterations']}}}\n"
        f"\\newcommand{{\\MeshCondItersFine}}{{{mesh_rows[-1]['cg_iterations']}}}\n"
        f"\\newcommand{{\\MeshCondUnknownsCoarse}}{{{mesh_rows[0]['unknowns']}}}\n"
        f"\\newcommand{{\\MeshCondUnknownsFine}}{{{mesh_rows[-1]['unknowns']}}}\n"
        f"\\newcommand{{\\TolSweepLooseDiff}}{{{latex_float(tol_loose['Linf_difference_from_direct'], 2)}}}\n"
        f"\\newcommand{{\\TolSweepMidDiff}}{{{latex_float(tol_mid['Linf_difference_from_direct'], 2)}}}\n"
    )
    (RESULTS / "report_macros.tex").write_text(macros, encoding="utf-8")

    def table_lines(rows, columns, formats):
        lines = []
        for row in rows:
            values = []
            for key, fmt in zip(columns, formats):
                value = row[key]
                if isinstance(value, float) and not np.isfinite(value):
                    values.append("--")
                else:
                    values.append(fmt(value))
            lines.append(" & ".join(values) + r" \\")
        # Keep the closing rule in the included file. A rule placed immediately
        # after an \input boundary is not accepted by every LaTeX installation.
        return "\n".join(lines) + "\n\\bottomrule"

    smooth_table = table_lines(
        smooth_rows,
        ["N", "h", "L2_error", "L2_pair_order", "Linf_error", "Linf_pair_order", "scaled_residual"],
        [lambda v: str(v), lambda v: f"{v:.5f}", lambda v: f"{v:.3e}", lambda v: f"{v:.3f}", lambda v: f"{v:.3e}", lambda v: f"{v:.3f}", lambda v: f"{v:.2e}"],
    )
    (RESULTS / "smooth_table.tex").write_text(smooth_table + "\n", encoding="utf-8")

    layered_table = table_lines(
        layered_rows,
        ["average", "N", "Linf_error", "Linf_pair_order", "flux_error", "flux_jump", "bounds_defect", "scaled_residual"],
        [lambda v: str(v).title(), lambda v: str(v), lambda v: f"{v:.3e}", lambda v: f"{v:.3f}", lambda v: f"{v:.3e}", lambda v: f"{v:.3e}", lambda v: f"{v:.1e}", lambda v: f"{v:.2e}"],
    )
    (RESULTS / "layered_table.tex").write_text(layered_table + "\n", encoding="utf-8")

    contrast_table = table_lines(
        contrast_rows,
        ["contrast", "average", "Linf_error", "flux_error", "cg_iterations", "condition_estimate"],
        [lambda v: f"{v:.0f}", lambda v: str(v).title(), lambda v: f"{v:.3e}", lambda v: f"{v:.3e}", lambda v: str(v), lambda v: f"{v:.3e}"],
    )
    (RESULTS / "contrast_table.tex").write_text(contrast_table + "\n", encoding="utf-8")

    mesh_table = table_lines(
        mesh_rows, ["N", "h", "unknowns", "cg_iterations", "scaled_residual"],
        [lambda v: str(v), lambda v: f"{v:.5f}", lambda v: str(v),
         lambda v: str(v), lambda v: f"{v:.2e}"],
    )
    (RESULTS / "mesh_conditioning_table.tex").write_text(mesh_table + "\n", encoding="utf-8")

    cost_table = table_lines(
        cost_rows,
        ["N", "unknowns", "nnz", "direct_Linf_error", "cg_Linf_error",
         "cg_iterations", "direct_median_s", "cg_median_s"],
        [lambda v: str(v), lambda v: str(v), lambda v: str(v),
         lambda v: f"{v:.3e}", lambda v: f"{v:.3e}", lambda v: str(v),
         lambda v: f"{v:.3e}", lambda v: f"{v:.3e}"],
    )
    (RESULTS / "solver_cost_table.tex").write_text(cost_table + "\n", encoding="utf-8")

    iterator_table = table_lines(
        iterator_rows,
        ["N", "method", "omega", "iterations", "scaled_residual",
         "linf_diff_direct", "wall_time_s"],
        [lambda v: str(v), lambda v: METHOD_LABELS.get(v, str(v)),
         lambda v: f"{v:.3f}", lambda v: str(v), lambda v: f"{v:.2e}",
         lambda v: f"{v:.2e}", lambda v: f"{v:.3e}"],
    )
    (RESULTS / "stationary_iterator_table.tex").write_text(iterator_table + "\n", encoding="utf-8")

    boundary_rows = read_csv(RESULTS / "nonzero_dirichlet_check.csv")
    boundary_table = table_lines(
        boundary_rows, ["N", "L2_error", "Linf_error", "scaled_residual"],
        [lambda v: str(v), lambda v: f"{v:.2e}",
         lambda v: f"{v:.2e}", lambda v: f"{v:.2e}"],
    )
    (RESULTS / "boundary_check_table.tex").write_text(boundary_table + "\n", encoding="utf-8")

    tolerance_rows = read_csv(RESULTS / "tolerance_sweep.csv")
    tolerance_table = table_lines(
        tolerance_rows,
        ["rtol", "cg_iterations", "scaled_residual", "Linf_error",
         "Linf_difference_from_direct"],
        [lambda v: f"{v:.0e}", lambda v: str(v), lambda v: f"{v:.2e}",
         lambda v: f"{v:.3e}", lambda v: f"{v:.2e}"],
    )
    (RESULTS / "tolerance_table.tex").write_text(tolerance_table + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate LaTeX tables/macros from existing CSV files without rerunning experiments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report_only:
        smooth_rows = require_csv("smooth_convergence.csv")
        layered_rows = require_csv("layered_comparison.csv")
        contrast_rows = require_csv("contrast_study.csv")
        mesh_rows = require_csv("mesh_conditioning.csv")
        check_rows = require_csv("independent_checks.csv")
        cost_rows = require_csv("solver_cost_comparison.csv")
        tol_rows = require_csv("tolerance_sweep.csv")
        iterator_rows = require_csv("stationary_iterator_comparison.csv")
        p_l2 = fitted_order(
            [row["h"] for row in smooth_rows],
            [row["L2_error"] for row in smooth_rows],
        )
        p_linf = fitted_order(
            [row["h"] for row in smooth_rows],
            [row["Linf_error"] for row in smooth_rows],
        )
    else:
        smooth_rows, p_l2, p_linf = run_smooth_study()
        layered_rows = run_layered_study()
        contrast_rows = run_contrast_study()
        mesh_rows = run_mesh_conditioning()
        check_rows = run_algebraic_crosscheck()
        cost_rows = run_solver_cost_comparison()
        iterator_rows = run_stationary_iterator_comparison()
        run_boundary_and_row_audit()
        tol_rows = run_tolerance_sweep()
    write_latex_outputs(smooth_rows, p_l2, p_linf, layered_rows, contrast_rows,
                        mesh_rows, check_rows, cost_rows, tol_rows, iterator_rows)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "smooth_meshes": [16, 32, 64, 128],
        "layered_meshes": [31, 63, 127, 255],
        "contrast_mesh": 63,
        "contrasts": [1, 10, 100, 1000],
        "face_averages": ["arithmetic", "harmonic"],
        "depth_iterators": ["gauss-seidel", "sor"],
        "depth_target_scaled_residual": 1e-10,
        "report_only": args.report_only,
    }
    if not args.report_only:
        (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    print("Project 2 report outputs refreshed." if args.report_only else "Project 2 experiments completed.")
    print(f"Smooth fitted orders: L2={p_l2:.4f}, Linf={p_linf:.4f}")
    print(f"Figures: {FIGURES}")
    print(f"Tables:  {RESULTS}")


if __name__ == "__main__":
    main()
