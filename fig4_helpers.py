"""
fig4_helpers.py
===============

Region-classification helpers for the Figure 4 bifurcation diagram in the
(delta, sigma) parameter plane.

The five dynamical regions are distinguished by:

  * ``classify_local``: local information only (number of interior steady
    states + stability of the boundary equilibrium A). Returns 1, 2, 4, or
    3, where the returned 3 actually encodes the *unresolved* "region 3 or
    region 5" case, since those two regions are locally identical.

  * ``in_region_5``: the global (heteroclinic) test that separates region
    3 from region 5 by shooting forward along A's unstable manifold and
    checking whether the trajectory escapes to the tumor-escape state D.

Unlike the notebook's original closures, these functions take the shared
parameters (rho, eta, mu, alpha, beta) explicitly. They default to the
estimated nondimensional values from ``kuznetsov_model``, so the simple
call sites ``classify_local(s, d)`` and ``in_region_5(s, d)`` still work
unchanged, but the same functions can now be reused for any parameter set
(e.g. sweeping mu) without touching global state.
"""

import numpy as np
from scipy.integrate import solve_ivp

from kuznetsov_model import nd_params as _nd

# Default shared parameters (the Fig 4 scan holds these fixed while it
# sweeps sigma and delta).
_RHO = _nd['rho']
_ETA = _nd['eta']
_MU = _nd['mu']
_ALPHA = _nd['alpha']
_BETA = _nd['beta']


def jacobian_at(x_s, y_s, sigma, delta, rho=_RHO, eta=_ETA, mu=_MU,
                alpha=_ALPHA, beta=_BETA):
    """2x2 Jacobian at (x_s, y_s) for given (sigma, delta) and shared params."""
    return np.array([
        [rho * y_s / (eta + y_s) - mu * y_s - delta,
         rho * x_s * eta / (eta + y_s) ** 2 - mu * x_s],
        [-y_s,
         alpha * (1 - 2 * beta * y_s) - x_s],
    ])


def is_stable(x_s, y_s, sigma, delta, **shared):
    eigs = np.linalg.eigvals(jacobian_at(x_s, y_s, sigma, delta, **shared))
    return all(np.real(e) < 0 for e in eigs)


def interior_steady_states(sigma, delta, rho=_RHO, eta=_ETA, mu=_MU,
                           alpha=_ALPHA, beta=_BETA):
    """Positive interior fixed points as a list of (y_s, x_s, stable)."""
    C0 = eta * (sigma / alpha - delta)
    C1 = sigma / alpha + rho - mu * eta - delta + delta * eta * beta
    C2 = -mu + (mu * eta + delta - rho) * beta
    C3 = mu * beta
    roots = np.roots([C3, C2, C1, C0])
    shared = dict(rho=rho, eta=eta, mu=mu, alpha=alpha, beta=beta)
    states = []
    for r in roots:
        if np.isreal(r) and np.real(r) > 0:
            y_s = np.real(r)
            x_s = alpha * (1 - beta * y_s)
            if x_s > 0:
                states.append((y_s, x_s, is_stable(x_s, y_s, sigma, delta, **shared)))
    return states


def classify_local(sigma, delta, rho=_RHO, eta=_ETA, mu=_MU,
                   alpha=_ALPHA, beta=_BETA):
    """Local classification from steady-state count and boundary stability.

    Returns 1, 2, 4, or 3 (where 3 encodes the unresolved 3-or-5 case),
    and 0 for boundary / transitional points.
    """
    shared = dict(rho=rho, eta=eta, mu=mu, alpha=alpha, beta=beta)
    states = interior_steady_states(sigma, delta, **shared)
    n = len(states)
    A_stab = (sigma / delta) > alpha    # A = (sigma/delta, 0) stable iff sigma/delta > alpha
    if n == 0 and A_stab:
        return 1   # total regression only
    if n == 1 and not A_stab:
        return 2   # single interior stable steady state
    if n == 2 and A_stab:
        return 4   # D + total regression (tiny region)
    if n == 3 and not A_stab:
        return 3   # bistability, needs heteroclinic test
    return 0       # boundary / transitional


def in_region_5(sigma, delta, t_end=3000, rho=_RHO, eta=_ETA, mu=_MU,
                alpha=_ALPHA, beta=_BETA):
    """Heteroclinic test for points locally classified as 3-or-5.

    Shoots forward from A = (sigma/delta, 0) along its 1-D unstable manifold
    (the eigenvector for the positive eigenvalue of J_A, perturbed into
    y > 0).

      * Region 5: A's unstable manifold goes to D -> trajectory escapes
        (y -> large).
      * Region 3: A's unstable manifold goes to B -> trajectory settles
        (y stays low).

    Near the boundary, trajectories linger near the saddle C for a long time
    before deciding; t_end=3000 suffices for the grid resolution used here.
    Increase to 5000 if artifacts appear along the boundary line.
    """
    shared = dict(rho=rho, eta=eta, mu=mu, alpha=alpha, beta=beta)
    x_A = sigma / delta
    J_A = jacobian_at(x_A, 0, sigma, delta, **shared)
    eigvals, eigvecs = np.linalg.eig(J_A)
    unstable_idx = np.argmax(np.real(eigvals))
    v = np.real(eigvecs[:, unstable_idx])
    v = v / np.linalg.norm(v)
    if v[1] < 0:                # ensure perturbation enters y > 0 half-plane
        v = -v
    ic = [x_A + 1e-4 * v[0], max(1e-4 * v[1], 1e-6)]
    try:
        sol = solve_ivp(
            lambda t, z: [
                sigma + rho * z[0] * z[1] / (eta + z[1]) - mu * z[0] * z[1] - delta * z[0],
                alpha * z[1] * (1 - beta * z[1]) - z[0] * z[1],
            ],
            (0, t_end), ic,
            method='RK45', rtol=1e-8, atol=1e-10, max_step=2.0,
        )
        return sol.y[1, -1] > 100   # escaped to D (tumor escape)
    except Exception:
        return False
