"""Stage 7: unitary coupled-cluster with singles and doubles.

Builds |psi(theta)> = exp(T(theta) - T(theta)+) |HF> and compiles it into the engine's
gate IR.

The construction, and why each step is where it is:

1. **Excitations.** One variational amplitude per spin-conserving single and double
   excitation from the Hartree-Fock reference.
2. **Anti-Hermitian generator.** ``G = T - T+`` for each excitation, so ``exp(G)`` is
   unitary.
3. **Jordan-Wigner.** ``JW(G) = sum_k i*c_k*P_k`` with real ``c_k`` -- an anti-Hermitian
   operator maps to imaginary coefficients on Hermitian Pauli strings.
4. **First-order Trotter.** ``exp(sum_k i*theta*c_k*P_k) ~= prod_k exp(i*theta*c_k*P_k)``.
   The Pauli strings within one excitation generally do not commute, so this is an
   approximation, and it is the dominant *method* error in the budget -- it is reported
   in ``ErrorBudget`` rather than ignored. (For H2 in a minimal basis there is a single
   double excitation and the approximation is exact, which is why the H2 gate reaches FCI
   rather than merely approaching it.)
5. **Compilation.** Each ``exp(i*alpha*P)`` becomes a basis change, a CNOT ladder that
   accumulates parity onto one qubit, an ``Rz``, and the inverse of both.

All excitations belonging to one amplitude share a single ``param_index`` and carry their
own ``param_scale``; that is what ties them to one variational parameter.
"""

import math

import numpy as np
from numpy.typing import NDArray
from openfermion import FermionOperator, QubitOperator, hermitian_conjugated, jordan_wigner

from qdr_compute_engine.types import Circuit, Gate, GateName
from qdr_core.errors import ValidationError

ANSATZ_NAME = "uccsd"

#: JW coefficients below this are dropped: they are numerical noise from the transform,
#: and each one retained would cost a full CNOT ladder for no physical effect.
_COEFFICIENT_CUTOFF = 1e-12


def single_excitations(n_electrons: int, n_spin_orbitals: int) -> list[tuple[int, int]]:
    """Spin-conserving single excitations ``(occupied, virtual)``.

    Spin is conserved when both indices have the same parity, since spin orbital ``2p``
    is alpha and ``2p+1`` is beta.
    """
    occupied = range(n_electrons)
    virtual = range(n_electrons, n_spin_orbitals)
    return [(i, a) for i in occupied for a in virtual if i % 2 == a % 2]


def double_excitations(n_electrons: int, n_spin_orbitals: int) -> list[tuple[int, int, int, int]]:
    """Spin-conserving double excitations ``(i, j, a, b)`` with ``i<j`` and ``a<b``."""
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_spin_orbitals))
    excitations = []
    for index, i in enumerate(occupied):
        for j in occupied[index + 1 :]:
            for a_index, a in enumerate(virtual):
                for b in virtual[a_index + 1 :]:
                    if (i % 2) + (j % 2) == (a % 2) + (b % 2):
                        excitations.append((i, j, a, b))
    return excitations


def _single_generator(i: int, a: int) -> FermionOperator:
    operator = FermionOperator(f"{a}^ {i}", 1.0)
    return operator - hermitian_conjugated(operator)


def _double_generator(i: int, j: int, a: int, b: int) -> FermionOperator:
    operator = FermionOperator(f"{a}^ {b}^ {j} {i}", 1.0)
    return operator - hermitian_conjugated(operator)


def _pauli_rotation_gates(
    pauli_term: tuple[tuple[int, str], ...],
    param_index: int,
    param_scale: float,
) -> list[Gate]:
    """Compile ``exp(i * alpha * P)`` where ``alpha = param_scale/(-2) * theta``.

    ``Rz(phi) = exp(-i*phi*Z/2)``, so realising ``exp(i*alpha*P)`` needs
    ``phi = -2*alpha``; the caller folds that factor into ``param_scale``.

    An empty Pauli term is the identity, which contributes only a global phase and is
    dropped -- it is unobservable and would otherwise emit a stray rotation.
    """
    if not pauli_term:
        return []

    qubits = [qubit for qubit, _ in pauli_term]
    basis_in: list[Gate] = []
    basis_out: list[Gate] = []

    for qubit, pauli in pauli_term:
        if pauli == "X":
            # H Z H = X
            basis_in.append(Gate(GateName.H, (qubit,)))
            basis_out.append(Gate(GateName.H, (qubit,)))
        elif pauli == "Y":
            # Rx(pi/2) Y Rx(-pi/2) = Z
            basis_in.append(Gate(GateName.RX, (qubit,), angle=math.pi / 2))
            basis_out.append(Gate(GateName.RX, (qubit,), angle=-math.pi / 2))
        elif pauli != "Z":
            raise ValidationError(f"unexpected Pauli operator {pauli!r}")

    # Accumulate the parity of all involved qubits onto the last one.
    ladder = [Gate(GateName.CX, (qubits[k], qubits[k + 1])) for k in range(len(qubits) - 1)]

    rotation = Gate(
        GateName.RZ,
        (qubits[-1],),
        param_index=param_index,
        param_scale=param_scale,
    )

    return [
        *basis_in,
        *ladder,
        rotation,
        *reversed(ladder),
        *reversed(basis_out),
    ]


def _generator_to_gates(generator: FermionOperator, param_index: int) -> list[Gate]:
    """Trotterise one excitation generator into IR gates."""
    qubit_operator: QubitOperator = jordan_wigner(generator)
    qubit_operator.compress()

    gates: list[Gate] = []
    for pauli_term, coefficient in sorted(qubit_operator.terms.items()):
        # G is anti-Hermitian, so its JW image carries imaginary coefficients.
        if abs(coefficient.real) > 1e-9:
            raise ValidationError(
                f"excitation generator produced a real coefficient {coefficient!r}; "
                "the generator is not anti-Hermitian"
            )
        c_k = coefficient.imag
        if abs(c_k) < _COEFFICIENT_CUTOFF:
            continue
        # exp(i*theta*c_k*P) == Rz(phi) sandwich with phi = -2*c_k*theta.
        gates.extend(_pauli_rotation_gates(pauli_term, param_index, -2.0 * c_k))
    return gates


def reference_state_gates(occupied_orbitals: tuple[int, ...]) -> list[Gate]:
    """Prepare the Hartree-Fock determinant.

    Under Jordan-Wigner the occupation number *is* the computational basis state, so the
    reference is a product state and a row of X gates suffices.
    """
    return [Gate(GateName.X, (orbital,)) for orbital in sorted(occupied_orbitals)]


def build_uccsd_ansatz(
    n_electrons: int,
    n_spin_orbitals: int,
    *,
    include_singles: bool = True,
    include_doubles: bool = True,
) -> Circuit:
    """Build the UCCSD ansatz circuit for ``n_electrons`` in ``n_spin_orbitals``."""
    if n_electrons <= 0:
        raise ValidationError(f"n_electrons must be positive, got {n_electrons}")
    if n_electrons > n_spin_orbitals:
        raise ValidationError(
            f"cannot place {n_electrons} electrons in {n_spin_orbitals} spin orbitals",
            n_electrons=n_electrons,
            n_spin_orbitals=n_spin_orbitals,
        )

    gates: list[Gate] = reference_state_gates(tuple(range(n_electrons)))

    generators: list[FermionOperator] = []
    if include_singles:
        generators.extend(
            _single_generator(i, a) for i, a in single_excitations(n_electrons, n_spin_orbitals)
        )
    if include_doubles:
        generators.extend(
            _double_generator(i, j, a, b)
            for i, j, a, b in double_excitations(n_electrons, n_spin_orbitals)
        )

    if not generators:
        raise ValidationError(
            "UCCSD has no excitations for this system; the active space is fully "
            "occupied or fully virtual",
            n_electrons=n_electrons,
            n_spin_orbitals=n_spin_orbitals,
        )

    for param_index, generator in enumerate(generators):
        gates.extend(_generator_to_gates(generator, param_index))

    return Circuit(
        n_qubits=n_spin_orbitals,
        gates=tuple(gates),
        n_params=len(generators),
        metadata={
            "ansatz": ANSATZ_NAME,
            "n_singles": len(single_excitations(n_electrons, n_spin_orbitals))
            if include_singles
            else 0,
            "n_doubles": len(double_excitations(n_electrons, n_spin_orbitals))
            if include_doubles
            else 0,
            "trotter_order": 1,
        },
    )


def initial_parameters(circuit: Circuit, seed: int = 0, spread: float = 0.0) -> NDArray[np.float64]:
    """Starting amplitudes.

    Zeros by default, which makes the ansatz the identity and the starting energy exactly
    Hartree-Fock -- a physically meaningful, deterministic starting point that guarantees
    the optimiser can only improve on the mean-field answer. A non-zero ``spread`` adds
    seeded jitter, useful for escaping a symmetric stationary point.
    """
    if spread == 0.0:
        return np.zeros(circuit.n_params, dtype=float)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, spread, size=circuit.n_params)
