"""
Estimate the scheduled execution time of the HHL circuit used in quantum_solvers.py.

The reported time is the hardware schedule duration of ONE circuit shot on a
selected fake IBM backend. It does not include queue time, transpilation time,
classical post-processing, repeated-shot overhead, postselection, or tomography.

Recommended packages:
    pip install "qiskit[visualization]>=1.4" qiskit-ibm-runtime scipy matplotlib

Run:
    python hhl_circuit_time.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFT, RYGate
from qiskit.visualization.timeline import draw as timeline_draw
from qiskit_ibm_runtime.fake_provider import FakeFez


# ============================================================
# USER PARAMETERS
# ============================================================

# Replace A and b here. A must be square and Hermitian for standard HHL.
A_MATRIX = np.array(
    [
        [1.0, -0.25],
        [-0.25, 1.0],
    ],
    dtype=complex,
)

B_VECTOR = np.array([1.0, 0.0], dtype=complex)

# These defaults can be replaced by values from your config.py.
HHL_PHASE_QUBITS = 3
HHL_PHASE_TARGET = 0.25
HHL_C = 0.05

SCHEDULING_METHOD = "alap"       # "alap" or "asap"
LAYOUT_METHOD = "trivial"        # use the first physical qubits
OPTIMIZATION_LEVEL = 1            # 0, 1, 2, or 3
INCLUDE_MEASUREMENT = True        # include readout time in one-shot duration
SAVE_CIRCUIT_IMAGE = True
SAVE_TIMELINE_IMAGE = True
OUTPUT_DIRECTORY = Path("hhl_timing_output")


# ============================================================
# HHL CIRCUIT CONSTRUCTION
# Copied from the structure used by quantum_solvers.py
# ============================================================


def pad_to_power_of_two(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad A and b so that the linear-system dimension is a power of two."""

    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex).reshape(-1)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    original_dimension = A.shape[0]

    if b.size != original_dimension:
        raise ValueError("The size of b must match the dimension of A.")

    padded_dimension = 1
    while padded_dimension < original_dimension:
        padded_dimension *= 2

    # The solver uses at least one target qubit.
    padded_dimension = max(2, padded_dimension)

    A_padded = np.eye(padded_dimension, dtype=complex)
    b_padded = np.zeros(padded_dimension, dtype=complex)
    A_padded[:original_dimension, :original_dimension] = A
    b_padded[:original_dimension] = b

    return A_padded, b_padded, original_dimension


def gershgorin_lambda_bound(A: np.ndarray) -> float:
    """Return the Gershgorin upper bound used to scale the HHL phase."""

    A = np.asarray(A, dtype=complex)
    diagonal = np.abs(np.diag(A))
    row_sum = np.sum(np.abs(A), axis=1)
    radii = row_sum - diagonal
    bound = float(np.max(diagonal + radii))

    if bound <= 1e-12:
        raise ValueError("The Gershgorin bound is too small for HHL scaling.")

    return bound


def controlled_unitary_power(
    circuit: QuantumCircuit,
    unitary: np.ndarray,
    power: int,
    control_qubit: int,
    target_qubits: list[int],
) -> None:
    """Append controlled-U**power exactly as in the supplied HHL solver."""

    unitary_power = np.linalg.matrix_power(unitary, power)

    unitary_circuit = QuantumCircuit(
        len(target_qubits),
        name=f"U^{power}",
    )
    unitary_circuit.unitary(
        unitary_power,
        list(range(len(target_qubits))),
    )

    controlled_gate = unitary_circuit.to_gate().control(1)
    circuit.append(
        controlled_gate,
        [control_qubit] + list(target_qubits),
    )


def append_qpe(
    circuit: QuantumCircuit,
    phase_qubits: list[int],
    target_qubits: list[int],
    unitary: np.ndarray,
) -> None:
    """Append H gates, controlled powers of U, and inverse QFT."""

    for qubit in phase_qubits:
        circuit.h(qubit)

    for index, control in enumerate(reversed(phase_qubits)):
        controlled_unitary_power(
            circuit=circuit,
            unitary=unitary,
            power=2**index,
            control_qubit=control,
            target_qubits=target_qubits,
        )

    inverse_qft = QFT(
        num_qubits=len(phase_qubits),
        inverse=True,
        do_swaps=False,
    )
    circuit.append(inverse_qft, phase_qubits)


def append_control_rotation(
    circuit: QuantumCircuit,
    control_qubits: list[int],
    ancilla_qubit: int,
    evolution_time: float,
    C: float,
) -> None:
    """Append the multi-controlled RY rotations used by the solver."""

    number_of_controls = len(control_qubits)

    for decimal_value in range(1, 2**number_of_controls):
        bit_string = f"{decimal_value:0{number_of_controls}b}"

        phase = sum(
            int(bit) * 2 ** (-(position + 1))
            for position, bit in enumerate(bit_string)
        )

        if phase > 0.5:
            phase -= 1.0

        eigenvalue = (2 * np.pi / evolution_time) * phase

        if abs(eigenvalue) < 1e-12:
            continue

        amplitude = C / eigenvalue

        if abs(amplitude) > 1:
            continue

        theta = 2 * np.arcsin(amplitude)
        controlled_ry = RYGate(theta).control(
            number_of_controls,
            ctrl_state=bit_string,
        )
        circuit.append(
            controlled_ry,
            control_qubits + [ancilla_qubit],
        )


def build_hhl_circuit(
    A: np.ndarray,
    b: np.ndarray,
    phase_qubit_count: int,
    phase_target: float,
    C: float,
    include_measurement: bool = False,
) -> tuple[QuantumCircuit, dict[str, float | int]]:
    """Build the same HHL circuit structure used by hhl_solve()."""

    if phase_qubit_count < 1:
        raise ValueError("phase_qubit_count must be at least 1.")

    A_padded, b_padded, original_dimension = pad_to_power_of_two(A, b)

    if not np.allclose(A_padded, A_padded.conj().T, atol=1e-10):
        raise ValueError("A must be Hermitian for this HHL construction.")

    b_norm_value = np.linalg.norm(b_padded)
    if b_norm_value < 1e-14:
        raise ValueError("b must not be the zero vector.")

    padded_dimension = A_padded.shape[0]
    target_qubit_count = int(np.log2(padded_dimension))
    ancilla_qubit_count = 1
    total_qubits = (
        phase_qubit_count
        + target_qubit_count
        + ancilla_qubit_count
    )

    normalized_b = b_padded / b_norm_value
    lambda_bound = gershgorin_lambda_bound(A_padded)
    evolution_time = phase_target * 2 * np.pi / lambda_bound
    unitary = expm(1j * A_padded * evolution_time)

    if not np.allclose(
        unitary.conj().T @ unitary,
        np.eye(padded_dimension),
        atol=1e-8,
    ):
        raise ValueError("exp(iAt) is not unitary within numerical tolerance.")

    phase_register = QuantumRegister(
        phase_qubit_count,
        name="phase",
    )
    target_register = QuantumRegister(
        target_qubit_count,
        name="target",
    )
    ancilla_register = QuantumRegister(1, name="ancilla")

    circuit = QuantumCircuit(
        phase_register,
        target_register,
        ancilla_register,
        name="HHL",
    )

    phase_indices = list(range(phase_qubit_count))
    target_indices = list(
        range(
            phase_qubit_count,
            phase_qubit_count + target_qubit_count,
        )
    )
    ancilla_index = phase_qubit_count + target_qubit_count

    # Step 1: prepare |b>.
    circuit.initialize(normalized_b, target_indices)

    # Step 2: QPE.
    append_qpe(
        circuit,
        phase_indices,
        target_indices,
        unitary,
    )

    # Step 3: controlled reciprocal rotation.
    append_control_rotation(
        circuit,
        phase_indices,
        ancilla_index,
        evolution_time,
        C,
    )

    # Step 4: inverse QPE, matching the original solver implementation.
    qpe_only = QuantumCircuit(
        phase_qubit_count + target_qubit_count,
        name="QPE",
    )
    append_qpe(
        qpe_only,
        list(range(phase_qubit_count)),
        list(
            range(
                phase_qubit_count,
                phase_qubit_count + target_qubit_count,
            )
        ),
        unitary,
    )

    inverse_qpe = qpe_only.inverse()
    inverse_qpe.name = "QPE_dagger"
    circuit.append(
        inverse_qpe,
        phase_indices + target_indices,
    )

    if include_measurement:
        circuit.measure_all()

    metadata: dict[str, float | int] = {
        "original_dimension": original_dimension,
        "padded_dimension": padded_dimension,
        "phase_qubits": phase_qubit_count,
        "target_qubits": target_qubit_count,
        "ancilla_qubits": ancilla_qubit_count,
        "total_qubits": total_qubits,
        "lambda_bound": lambda_bound,
        "evolution_time": evolution_time,
    }

    return circuit, metadata


# ============================================================
# TIMING
# ============================================================


@dataclass(frozen=True)
class DurationResult:
    duration_dt: Optional[float]
    duration_seconds: Optional[float]
    backend_dt_seconds: Optional[float]


def estimate_duration(
    scheduled_circuit: QuantumCircuit,
    backend: FakeFez,
) -> DurationResult:
    """Support both newer and older Qiskit duration APIs."""

    backend_dt = getattr(backend, "dt", None)
    duration_dt: Optional[float] = None
    duration_seconds: Optional[float] = None

    # Qiskit >= 1.4 and Qiskit 2.x.
    if hasattr(scheduled_circuit, "estimate_duration"):
        duration_dt = float(
            scheduled_circuit.estimate_duration(
                backend.target,
                unit="dt",
            )
        )
        duration_seconds = float(
            scheduled_circuit.estimate_duration(
                backend.target,
                unit="s",
            )
        )
    else:
        # Compatibility fallback for older Qiskit versions.
        legacy_duration = getattr(scheduled_circuit, "duration", None)
        if legacy_duration is None:
            legacy_duration = getattr(scheduled_circuit, "_duration", None)

        if legacy_duration is not None:
            duration_dt = float(legacy_duration)
            if backend_dt is not None:
                duration_seconds = duration_dt * float(backend_dt)

    return DurationResult(
        duration_dt=duration_dt,
        duration_seconds=duration_seconds,
        backend_dt_seconds=(
            float(backend_dt) if backend_dt is not None else None
        ),
    )


def human_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unavailable"
    if seconds < 1e-6:
        return f"{seconds * 1e9:.3f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.6f} s"


def save_figures(
    original_circuit: QuantumCircuit,
    scheduled_circuit: QuantumCircuit,
    backend: FakeFez,
) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if SAVE_CIRCUIT_IMAGE:
        circuit_figure = original_circuit.draw(
            output="mpl",
            fold=-1,
            style="iqp",
        )
        circuit_path = OUTPUT_DIRECTORY / "hhl_circuit.png"
        circuit_figure.savefig(circuit_path, dpi=180, bbox_inches="tight")
        print(f"Circuit image:       {circuit_path.resolve()}")

    if SAVE_TIMELINE_IMAGE:
        timeline_figure = timeline_draw(
            scheduled_circuit,
            target=backend.target,
        )
        timeline_path = OUTPUT_DIRECTORY / "hhl_timeline.png"
        timeline_figure.savefig(timeline_path, dpi=180, bbox_inches="tight")
        print(f"Timeline image:      {timeline_path.resolve()}")


def main() -> None:
    backend = FakeFez()

    hhl_circuit, metadata = build_hhl_circuit(
        A=A_MATRIX,
        b=B_VECTOR,
        phase_qubit_count=HHL_PHASE_QUBITS,
        phase_target=HHL_PHASE_TARGET,
        C=HHL_C,
        include_measurement=INCLUDE_MEASUREMENT,
    )

    if hhl_circuit.num_qubits > backend.num_qubits:
        raise ValueError(
            f"The HHL circuit needs {hhl_circuit.num_qubits} qubits, "
            f"but {backend.name} has only {backend.num_qubits}. "
            "Reduce HHL_PHASE_QUBITS, reduce the matrix dimension, "
            "or select a larger backend."
        )

    scheduled_circuit = transpile(
        hhl_circuit,
        backend=backend,
        scheduling_method=SCHEDULING_METHOD,
        layout_method=LAYOUT_METHOD,
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=42,
    )

    duration = estimate_duration(scheduled_circuit, backend)

    print("\n================ HHL CIRCUIT TIMING ================")
    print(f"Backend:             {backend.name}")
    print(f"Original dimension:  {metadata['original_dimension']}")
    print(f"Padded dimension:    {metadata['padded_dimension']}")
    print(f"Phase qubits:        {metadata['phase_qubits']}")
    print(f"Target qubits:       {metadata['target_qubits']}")
    print(f"Total logical qubits:{metadata['total_qubits']:>8}")
    print(f"Measurement included:{str(INCLUDE_MEASUREMENT):>8}")
    print(f"Scheduling method:   {SCHEDULING_METHOD}")
    print(f"Optimization level:  {OPTIMIZATION_LEVEL}")
    print(f"Transpiled depth:    {scheduled_circuit.depth()}")
    print(f"Transpiled size:     {scheduled_circuit.size()}")
    print(f"Gate counts:         {dict(scheduled_circuit.count_ops())}")

    if duration.backend_dt_seconds is not None:
        print(
            "Backend dt:          "
            f"{duration.backend_dt_seconds:.12e} s"
        )
    else:
        print("Backend dt:          unavailable")

    if duration.duration_dt is not None:
        print(f"Duration:            {duration.duration_dt:.0f} dt")
    else:
        print("Duration in dt:      unavailable")

    if duration.backend_dt_seconds is not None:
        print(
            "dt conversion:       "
            f"1 dt = {duration.backend_dt_seconds:.12e} s "
            "(from backend configuration)"
        )
    else:
        print("dt conversion:       unavailable in backend configuration")

    print(
        "One-shot gate time:   "
        f"{human_time(duration.duration_seconds)}"
    )
    print("====================================================\n")

    save_figures(
        original_circuit=hhl_circuit,
        scheduled_circuit=scheduled_circuit,
        backend=backend,
    )


if __name__ == "__main__":
    main()
