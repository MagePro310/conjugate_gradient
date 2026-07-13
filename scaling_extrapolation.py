"""Benchmark and extrapolate CG and HHL circuit execution-time scaling.

The quantum measurement is the scheduled duration of one circuit shot on
FakeFez. It excludes transpilation, queueing, state readout/tomography, and
classical post-processing. The CG measurement returns the complete solution
vector and is restricted to one BLAS thread.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter_ns

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
from threadpoolctl import threadpool_limits

from hhl_circuit_time import build_hhl_circuit, estimate_duration


DEFAULT_CG_SIZES = [16, 32, 64, 128, 256, 512, 1024]
DEFAULT_QUANTUM_SIZES = [2, 4, 8, 16]
DEFAULT_PREDICTION_SIZES = [32, 64, 128, 256, 512, 1024]


def make_linear_system(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the same positive-definite tridiagonal system used by HHL."""

    if dimension < 2:
        raise ValueError("dimension must be at least 2")

    matrix = (
        np.diag(np.ones(dimension))
        + np.diag(-0.25 * np.ones(dimension - 1), k=1)
        + np.diag(-0.25 * np.ones(dimension - 1), k=-1)
    )
    vector = np.zeros(dimension)
    vector[0] = 1.0
    return matrix, vector


def conjugate_gradient(
    matrix: np.ndarray,
    vector: np.ndarray,
    tolerance: float = 1e-8,
    max_iterations: int | None = None,
) -> tuple[np.ndarray, int]:
    """Solve Ax=b without printing so terminal I/O does not affect timing."""

    dimension = vector.size
    if max_iterations is None:
        max_iterations = dimension

    solution = np.zeros(dimension)
    residual = vector - matrix @ solution
    direction = residual.copy()
    residual_squared = residual @ residual

    for iteration in range(1, max_iterations + 1):
        matrix_direction = matrix @ direction
        denominator = direction @ matrix_direction
        if denominator <= 0:
            raise ValueError("The matrix is not positive definite.")

        alpha = residual_squared / denominator
        solution += alpha * direction
        residual -= alpha * matrix_direction
        new_residual_squared = residual @ residual

        if np.sqrt(new_residual_squared) < tolerance:
            return solution, iteration

        beta = new_residual_squared / residual_squared
        direction = residual + beta * direction
        residual_squared = new_residual_squared

    return solution, max_iterations


def benchmark_cg(
    dimension: int,
    repeats: int,
    warmups: int,
    tolerance: float,
) -> tuple[float, int]:
    """Return median CG execution time in seconds and iteration count."""

    matrix, vector = make_linear_system(dimension)
    samples_ns: list[int] = []
    iterations = 0

    with threadpool_limits(limits=1, user_api="blas"):
        for _ in range(warmups):
            conjugate_gradient(matrix, vector, tolerance)

        for _ in range(repeats):
            start_ns = perf_counter_ns()
            _, iterations = conjugate_gradient(matrix, vector, tolerance)
            samples_ns.append(perf_counter_ns() - start_ns)

    return float(np.median(samples_ns)) / 1e9, iterations


def benchmark_quantum_one_shot(
    dimension: int,
    backend: FakeFez,
    phase_qubits: int,
) -> tuple[float, float, int]:
    """Return scheduled seconds, dt count, and transpiled circuit depth."""

    matrix, vector = make_linear_system(dimension)
    circuit, _ = build_hhl_circuit(
        A=matrix,
        b=vector,
        phase_qubit_count=phase_qubits,
        phase_target=0.25,
        C=0.05,
        include_measurement=True,
    )
    scheduled = transpile(
        circuit,
        backend=backend,
        scheduling_method="alap",
        layout_method="trivial",
        optimization_level=1,
        seed_transpiler=42,
    )
    duration = estimate_duration(scheduled, backend)

    if duration.duration_seconds is None or duration.duration_dt is None:
        raise RuntimeError("The backend did not provide a circuit duration.")

    return duration.duration_seconds, duration.duration_dt, scheduled.depth()


def fit_power_law(
    dimensions: list[int],
    times: list[float],
) -> tuple[float, float, float]:
    """Fit time = coefficient * dimension**exponent in log space."""

    if len(dimensions) < 2:
        raise ValueError("At least two measurements are required for fitting.")

    log_dimensions = np.log(np.asarray(dimensions, dtype=float))
    log_times = np.log(np.asarray(times, dtype=float))
    exponent, log_coefficient = np.polyfit(log_dimensions, log_times, 1)
    fitted = exponent * log_dimensions + log_coefficient
    residual_sum = float(np.sum((log_times - fitted) ** 2))
    total_sum = float(np.sum((log_times - np.mean(log_times)) ** 2))
    r_squared = 1.0 if total_sum == 0 else 1.0 - residual_sum / total_sum
    return float(np.exp(log_coefficient)), float(exponent), r_squared


def predict(
    dimensions: np.ndarray,
    coefficient: float,
    exponent: float,
) -> np.ndarray:
    return coefficient * dimensions.astype(float) ** exponent


def save_csv(
    path: Path,
    dimensions: np.ndarray,
    cg_prediction: np.ndarray,
    quantum_prediction: np.ndarray,
    shots: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "dimension",
                "cg_fitted_seconds",
                "quantum_fitted_one_shot_seconds",
                f"quantum_fitted_{shots}_shots_seconds",
            ]
        )
        for dimension, cg_time, quantum_time in zip(
            dimensions,
            cg_prediction,
            quantum_prediction,
        ):
            writer.writerow(
                [dimension, cg_time, quantum_time, quantum_time * shots]
            )


def save_plot(
    path: Path,
    cg_sizes: list[int],
    cg_times: list[float],
    quantum_sizes: list[int],
    quantum_times: list[float],
    curve_dimensions: np.ndarray,
    cg_curve: np.ndarray,
    quantum_curve: np.ndarray,
    shots: int,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(cg_sizes, cg_times, label="CG measured (1 CPU thread)")
    axis.scatter(
        quantum_sizes,
        quantum_times,
        label="HHL measured (one-shot gate time)",
    )
    axis.plot(curve_dimensions, cg_curve, label="CG power-law fit")
    axis.plot(
        curve_dimensions,
        quantum_curve,
        label="HHL one-shot power-law fit",
    )
    if shots > 1:
        axis.plot(
            curve_dimensions,
            quantum_curve * shots,
            linestyle="--",
            label=f"HHL fit x {shots} shots",
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Matrix dimension N")
    axis.set_ylabel("Time (seconds)")
    axis.set_title("Measured scaling and empirical extrapolation")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure and extrapolate CG and HHL time scaling."
    )
    parser.add_argument("--cg-sizes", nargs="+", type=int, default=DEFAULT_CG_SIZES)
    parser.add_argument(
        "--quantum-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_QUANTUM_SIZES,
    )
    parser.add_argument(
        "--prediction-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_PREDICTION_SIZES,
    )
    parser.add_argument("--cg-repeats", type=int, default=30)
    parser.add_argument("--cg-warmups", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--phase-qubits", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("scaling_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cg_repeats < 1 or args.cg_warmups < 0 or args.shots < 1:
        raise ValueError("repeats and shots must be positive; warmups cannot be negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cg_times: list[float] = []
    print("\nCG measurements (one BLAS thread)")
    for dimension in args.cg_sizes:
        elapsed, iterations = benchmark_cg(
            dimension,
            args.cg_repeats,
            args.cg_warmups,
            args.tolerance,
        )
        cg_times.append(elapsed)
        print(f"N={dimension:5d}: {elapsed:.6e} s, iterations={iterations}")

    backend = FakeFez()
    quantum_times: list[float] = []
    print("\nHHL scheduled one-shot measurements")
    for dimension in args.quantum_sizes:
        elapsed, duration_dt, depth = benchmark_quantum_one_shot(
            dimension,
            backend,
            args.phase_qubits,
        )
        quantum_times.append(elapsed)
        print(
            f"N={dimension:5d}: {elapsed:.6e} s, "
            f"duration={duration_dt:.0f} dt, depth={depth}"
        )

    cg_coefficient, cg_exponent, cg_r_squared = fit_power_law(
        args.cg_sizes,
        cg_times,
    )
    quantum_coefficient, quantum_exponent, quantum_r_squared = fit_power_law(
        args.quantum_sizes,
        quantum_times,
    )

    prediction_dimensions = np.asarray(
        sorted(set(args.prediction_sizes)),
        dtype=int,
    )
    cg_prediction = predict(
        prediction_dimensions,
        cg_coefficient,
        cg_exponent,
    )
    quantum_prediction = predict(
        prediction_dimensions,
        quantum_coefficient,
        quantum_exponent,
    )

    print("\nEmpirical fitted models")
    print(
        f"CG:      T(N) = {cg_coefficient:.6e} * N^{cg_exponent:.4f} "
        f"(log-space R^2={cg_r_squared:.4f})"
    )
    print(
        f"HHL:     T(N) = {quantum_coefficient:.6e} * N^{quantum_exponent:.4f} "
        f"(log-space R^2={quantum_r_squared:.4f}, one shot)"
    )

    print("\nExtrapolated times")
    print("N       CG (s)       HHL 1 shot (s)    HHL total (s)")
    for dimension, cg_time, quantum_time in zip(
        prediction_dimensions,
        cg_prediction,
        quantum_prediction,
    ):
        print(
            f"{dimension:<7d} {cg_time:<12.6e} "
            f"{quantum_time:<17.6e} {quantum_time * args.shots:.6e}"
        )

    csv_path = args.output_dir / "scaling_predictions.csv"
    plot_path = args.output_dir / "scaling_extrapolation.png"
    save_csv(
        csv_path,
        prediction_dimensions,
        cg_prediction,
        quantum_prediction,
        args.shots,
    )

    curve_minimum = min(min(args.cg_sizes), min(args.quantum_sizes))
    curve_maximum = max(
        max(args.cg_sizes),
        max(args.quantum_sizes),
        int(np.max(prediction_dimensions)),
    )
    curve_dimensions = np.geomspace(curve_minimum, curve_maximum, 300)
    save_plot(
        plot_path,
        args.cg_sizes,
        cg_times,
        args.quantum_sizes,
        quantum_times,
        curve_dimensions,
        predict(curve_dimensions, cg_coefficient, cg_exponent),
        predict(curve_dimensions, quantum_coefficient, quantum_exponent),
        args.shots,
    )

    print(f"\nCSV:  {csv_path.resolve()}")
    print(f"Plot: {plot_path.resolve()}")
    print(
        "\nWarning: this is an empirical extrapolation. The HHL curve is based "
        "only on small circuits and is not a fault-tolerant asymptotic guarantee."
    )


if __name__ == "__main__":
    main()
