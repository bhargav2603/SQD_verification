"""The variational optimisation loop.

Deliberately backend-agnostic: it takes an energy callable and knows nothing about how
that energy was produced. The statevector and sampled drivers both use it, so the
optimiser's behaviour is identical across them and a difference in results is
attributable to the estimator rather than the search.

Optimiser choice depends on whether an *analytic* gradient is available, because the two
regimes have opposite requirements:

``no gradient`` (sampled backends)
    Energy evaluations carry shot noise. Gradient-based methods driven by finite
    differences respond to that noise by taking nonsense steps, so a derivative-free
    method -- COBYLA -- degrades far more gracefully. This is the default.

``analytic gradient`` (exact statevector backend)
    The energy is smooth and deterministic, and ``drivers/statevector/adjoint.py`` returns
    every partial derivative for the cost of roughly three circuit simulations. Passing
    those to L-BFGS-B converges in tens of iterations where COBYLA needs thousands of
    evaluations, and the gap widens with parameter count: at 117 parameters COBYLA has not
    converged after the point L-BFGS-B has finished.

Supplying ``value_and_gradient`` selects the second regime. Nothing else changes, so a
difference between the two is attributable to the search rather than to the physics.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from qdr_core.errors import ValidationError

EnergyFunction = Callable[[NDArray[np.float64]], float]
ValueAndGradientFunction = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]

#: Optimisers we have actually exercised. Others may work, but they are not claimed to.
SUPPORTED_OPTIMIZERS = frozenset({"COBYLA", "L-BFGS-B", "Powell", "Nelder-Mead", "SLSQP"})

#: Optimisers that can consume an analytic Jacobian. Handing a gradient to any other
#: method silently wastes it, so the choice is made explicit rather than assumed.
GRADIENT_CAPABLE_OPTIMIZERS = frozenset({"L-BFGS-B", "SLSQP"})

#: Chosen automatically when an analytic gradient is supplied and no method is named.
DEFAULT_GRADIENT_OPTIMIZER = "L-BFGS-B"

#: Chosen automatically when no gradient is available.
DEFAULT_GRADIENT_FREE_OPTIMIZER = "COBYLA"


@dataclass
class VQEOutcome:
    energy: float
    optimal_params: NDArray[np.float64]
    converged: bool
    n_iterations: int
    n_energy_evaluations: int
    method: str
    initial_energy: float
    message: str = ""
    history: list[float] = field(default_factory=list)
    #: Whether the search consumed an analytic Jacobian. Recorded because the same
    #: optimiser reaching the same energy by two different routes is evidence the
    #: gradient is right, and a run that silently fell back to finite differences would
    #: otherwise be indistinguishable from one that did not.
    used_analytic_gradient: bool = False


def run_vqe(
    energy_function: EnergyFunction,
    initial_params: NDArray[np.float64],
    *,
    method: str | None = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    track_history: bool = True,
    value_and_gradient: ValueAndGradientFunction | None = None,
) -> VQEOutcome:
    """Minimise ``energy_function`` over the ansatz parameters.

    ``value_and_gradient``, when supplied, must return ``(energy, gradient)`` at a point
    and is used in place of ``energy_function`` during the search. ``method`` then
    defaults to L-BFGS-B; naming a derivative-free method explicitly is still honoured,
    and the gradient is simply not used.
    """
    if method is None:
        method = (
            DEFAULT_GRADIENT_OPTIMIZER
            if value_and_gradient is not None
            else DEFAULT_GRADIENT_FREE_OPTIMIZER
        )

    if method not in SUPPORTED_OPTIMIZERS:
        raise ValidationError(
            f"unsupported optimizer {method!r}",
            requested=method,
            supported=sorted(SUPPORTED_OPTIMIZERS),
        )

    # A gradient handed to a method that cannot use it is a silent performance cliff,
    # not an error -- but the caller should not be left believing it took effect.
    use_gradient = value_and_gradient is not None and method in GRADIENT_CAPABLE_OPTIMIZERS

    params = np.asarray(initial_params, dtype=float)

    # A zero-parameter ansatz is legitimate (a fixed reference state); there is simply
    # nothing to optimise, and the energy is whatever the state gives.
    if params.size == 0:
        value = float(energy_function(params))
        return VQEOutcome(
            energy=value,
            optimal_params=params,
            converged=True,
            n_iterations=0,
            n_energy_evaluations=1,
            method=method,
            initial_energy=value,
            message="no free parameters",
            history=[value],
            used_analytic_gradient=False,
        )

    history: list[float] = []
    evaluations = 0

    def objective(x: NDArray[np.float64]) -> float:
        nonlocal evaluations
        evaluations += 1
        value = float(energy_function(x))
        if track_history:
            history.append(value)
        return value

    def objective_with_gradient(
        x: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        nonlocal evaluations
        evaluations += 1
        assert value_and_gradient is not None  # guarded by use_gradient
        value, gradient = value_and_gradient(x)
        if track_history:
            history.append(float(value))
        return float(value), np.asarray(gradient, dtype=float)

    initial_energy = float(energy_function(params))

    options: dict[str, object] = {"maxiter": max_iterations}
    if method == "COBYLA":
        # COBYLA reads its convergence threshold from `tol`, not from options, and
        # counts function evaluations rather than iterations.
        options = {"maxiter": max_iterations}

    if use_gradient:
        result = minimize(
            objective_with_gradient,
            params,
            method=method,
            jac=True,
            tol=tolerance,
            options=options,
        )
    else:
        result = minimize(objective, params, method=method, tol=tolerance, options=options)

    # The variational principle guarantees the optimiser cannot legitimately end above
    # its own starting point; if it has, keep the better of the two rather than
    # reporting a regression.
    best_energy = float(result.fun)
    best_params = np.asarray(result.x, dtype=float)
    if initial_energy < best_energy:
        best_energy, best_params = initial_energy, params

    return VQEOutcome(
        energy=best_energy,
        optimal_params=best_params,
        converged=bool(result.success),
        n_iterations=int(getattr(result, "nit", 0) or 0),
        n_energy_evaluations=evaluations,
        method=method,
        initial_energy=initial_energy,
        message=str(result.message),
        history=history,
        used_analytic_gradient=use_gradient,
    )
