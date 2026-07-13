import numpy as np
from time import perf_counter_ns


def conjugate_gradient(
    A,
    b,
    x0=None,
    tol=1e-8,
    max_iter=None,
    verbose=False,
):
    """
    Giải hệ Ax = b bằng phương pháp Conjugate Gradient.

    Điều kiện:
    - A là ma trận đối xứng xác định dương.
    - b là vector có kích thước phù hợp với A.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)

    if A.shape != (n, n):
        raise ValueError("Kích thước A và b không phù hợp.")

    if not np.allclose(A, A.T):
        raise ValueError("Ma trận A phải đối xứng.")

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    if max_iter is None:
        max_iter = n

    r = b - A @ x
    p = r.copy()
    rs_old = r @ r

    for iteration in range(1, max_iter + 1):
        Ap = A @ p
        denominator = p @ Ap

        if denominator <= 0:
            raise ValueError(
                "Ma trận A có thể không xác định dương."
            )

        alpha = rs_old / denominator

        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = r @ r
        residual = np.sqrt(rs_new)

        if verbose:
            print(
                f"Iteration {iteration}: "
                f"x = {x}, residual = {residual:.6e}"
            )

        if residual < tol:
            return x, iteration

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter


# =========================================================
# THAY MA TRẬN A VÀ VECTOR b TẠI ĐÂY
# =========================================================

MATRIX_DIMENSION = 16
A = (
    np.diag(np.ones(MATRIX_DIMENSION))
    + np.diag(-0.25 * np.ones(MATRIX_DIMENSION - 1), k=1)
    + np.diag(-0.25 * np.ones(MATRIX_DIMENSION - 1), k=-1)
)

b = np.zeros(MATRIX_DIMENSION)
b[0] = 1.0

# Nghiệm ban đầu, có thể để None để tự động dùng vector 0
x0 = None

# Sai số cho phép
tolerance = 1e-8

# Số vòng lặp tối đa
max_iterations = 100


# =========================================================
# CHẠY THUẬT TOÁN
# =========================================================

start_time_ns = perf_counter_ns()
x, iterations = conjugate_gradient(
    A=A,
    b=b,
    x0=x0,
    tol=tolerance,
    max_iter=max_iterations,
    verbose=False,
)
elapsed_time_ns = perf_counter_ns() - start_time_ns
elapsed_time_seconds = elapsed_time_ns / 1e9

print("\nKết quả Conjugate Gradient")
print("x =", x)
print("Số vòng lặp =", iterations)
print(f"Thời gian CG = {elapsed_time_seconds:.12e} giây")
print(f"Thời gian CG = {elapsed_time_ns / 1e3:.3f} us")

# So sánh với nghiệm trực tiếp của NumPy
x_exact = np.linalg.solve(
    np.array(A, dtype=float),
    np.array(b, dtype=float)
)

print("\nNghiệm từ NumPy")
print("x_exact =", x_exact)

print("\nSai số")
print("||x - x_exact|| =", np.linalg.norm(x - x_exact))