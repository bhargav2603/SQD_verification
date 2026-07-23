"""PySCF embedding adapter -- T2 (adapter slice).

Two layers of tests:

* **Input validation** runs everywhere: ``build_embedding_artifacts`` checks its atom
  selection before importing PySCF, so those guards are verified on Windows too.
* **SCF-dependent behaviour** is ``requires_pyscf`` (skipped where PySCF has no wheel). The
  load-bearing one is the operator-level recovery gate: selecting every atom as active
  leaves an empty environment, so the embedded one-electron operator must equal the bare
  core Hamiltonian to machine precision.
"""

import numpy as np
import pytest

from qdr_compute_engine.embedding.pyscf_embedding import build_embedding_artifacts
from qdr_core.errors import ValidationError

# --------------------------------------------------------------------------------------
# Input validation (no PySCF needed -- these raise before the import)
# --------------------------------------------------------------------------------------


def test_empty_active_selection_is_rejected(h2_geometry) -> None:
    with pytest.raises(ValidationError, match="no active atoms"):
        build_embedding_artifacts(h2_geometry, active_atom_indices=())


def test_out_of_range_active_atom_is_rejected(h2_geometry) -> None:
    with pytest.raises(ValidationError, match="outside the molecule"):
        build_embedding_artifacts(h2_geometry, active_atom_indices=(0, 5))


# --------------------------------------------------------------------------------------
# SCF-dependent behaviour (Colab / Linux VM)
# --------------------------------------------------------------------------------------


@pytest.mark.requires_pyscf
def test_whole_molecule_active_recovers_core_hamiltonian(h2_geometry) -> None:
    """Recovery gate: active = all atoms => empty environment => h_emb == h_core."""
    art = build_embedding_artifacts(h2_geometry, active_atom_indices=(0, 1))

    np.testing.assert_allclose(art.h_emb, art.core_hamiltonian, atol=1e-8)
    np.testing.assert_allclose(
        art.environment_potential, np.zeros_like(art.environment_potential), atol=1e-8
    )
    assert art.partition.environment_indices == ()
    assert art.result.scheme == "projection-wf-in-dft"


@pytest.mark.requires_pyscf
def test_partial_active_region_leaves_an_environment(h2_geometry) -> None:
    """Selecting only one H leaves the other's orbital in the environment."""
    art = build_embedding_artifacts(h2_geometry, active_atom_indices=(0,))

    assert len(art.partition.active_indices) >= 1
    assert len(art.partition.environment_indices) >= 1
    # A non-empty environment must actually shift the operator away from the bare core.
    assert not np.allclose(art.h_emb, art.core_hamiltonian, atol=1e-6)
