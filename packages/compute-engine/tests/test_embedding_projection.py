"""Projection-based embedding core (backend-free) -- T2 scaffolding.

These tests need no PySCF: the embedding math is plain array arithmetic on synthetic
inputs. The load-bearing test is the recovery property -- with an empty environment the
embedded operator must collapse to the bare core Hamiltonian exactly, which is the
unit-level backbone of the separated-fragment correctness gate.
"""

import numpy as np
import pytest

from qdr_compute_engine.embedding import (
    embed_one_body_operator,
    levelshift_projector,
    partition_orbitals_by_population,
)
from qdr_core.errors import ValidationError

# --------------------------------------------------------------------------------------
# Level-shift projector
# --------------------------------------------------------------------------------------


def test_projector_is_s_gamma_s() -> None:
    overlap = np.array([[1.0, 0.2], [0.2, 1.0]])
    gamma_env = np.array([[0.5, 0.0], [0.0, 0.0]])
    expected = overlap @ gamma_env @ overlap
    np.testing.assert_allclose(levelshift_projector(overlap, gamma_env), expected)


def test_empty_environment_gives_zero_projector() -> None:
    overlap = np.eye(3)
    zero_env = np.zeros((3, 3))
    np.testing.assert_array_equal(
        levelshift_projector(overlap, zero_env), np.zeros((3, 3))
    )


def test_projector_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValidationError):
        levelshift_projector(np.eye(2), np.zeros((3, 3)))


# --------------------------------------------------------------------------------------
# Embedded one-body operator
# --------------------------------------------------------------------------------------


def test_empty_environment_recovers_core_hamiltonian() -> None:
    """Recovery gate (unit level): no environment => h_emb == h_core exactly."""
    rng = np.random.default_rng(0)
    core = rng.standard_normal((4, 4))
    core = core + core.T  # a plausible Hermitian one-electron operator
    potential = rng.standard_normal((4, 4))
    potential = potential + potential.T

    # Empty environment: g[gamma_tot] == g[gamma_A] and P_env == 0.
    h_emb = embed_one_body_operator(
        core_hamiltonian=core,
        potential_total=potential,
        potential_active=potential,
        projector=np.zeros((4, 4)),
        level_shift=1e6,
    )
    np.testing.assert_allclose(h_emb, core, atol=1e-12)


def test_embedding_adds_environment_field_and_level_shift() -> None:
    core = np.eye(2)
    g_tot = np.array([[2.0, 0.0], [0.0, 2.0]])
    g_act = np.array([[1.0, 0.0], [0.0, 1.0]])
    projector = np.array([[0.0, 0.0], [0.0, 1.0]])
    mu = 10.0

    h_emb = embed_one_body_operator(core, g_tot, g_act, projector, mu)
    # core + (g_tot - g_act) + mu * projector
    expected = np.array([[2.0, 0.0], [0.0, 12.0]])
    np.testing.assert_allclose(h_emb, expected)


def test_embedding_rejects_basis_shape_mismatch() -> None:
    with pytest.raises(ValidationError, match="same basis"):
        embed_one_body_operator(
            core_hamiltonian=np.eye(2),
            potential_total=np.eye(3),
            potential_active=np.eye(2),
            projector=np.eye(2),
            level_shift=1.0,
        )


def test_embedding_rejects_negative_level_shift() -> None:
    with pytest.raises(ValidationError, match="level_shift"):
        embed_one_body_operator(
            np.eye(2), np.eye(2), np.eye(2), np.eye(2), level_shift=-1.0
        )


# --------------------------------------------------------------------------------------
# Orbital partition
# --------------------------------------------------------------------------------------


def test_partition_assigns_by_threshold() -> None:
    # Orbitals 0,3 sit on the active atoms; 1,2 on the environment.
    result = partition_orbitals_by_population([0.9, 0.1, 0.05, 0.8])
    assert result.active_indices == (0, 3)
    assert result.environment_indices == (1, 2)


def test_partition_logs_boundary_orbitals() -> None:
    # 0.52 is within 10% of the 0.5 cut (margin 0.05) -> boundary warning.
    result = partition_orbitals_by_population([0.52, 0.95])
    assert any("boundary" in w for w in result.warnings)
    assert result.active_indices == (0, 1)  # 0.52 > 0.5, still active


def test_partition_refuses_empty_active_region() -> None:
    with pytest.raises(ValidationError, match="empty active region"):
        partition_orbitals_by_population([0.1, 0.2, 0.3])


def test_partition_requires_one_dimensional_populations() -> None:
    with pytest.raises(ValidationError, match="1-D"):
        partition_orbitals_by_population(np.zeros((2, 2)))
