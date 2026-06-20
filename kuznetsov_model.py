"""
kuznetsov_model.py
==================

Model definitions, parameters, experimental data, and nondimensionalization
for the recreation of:

    Kuznetsov, Makalkin, Taylor & Perelson (1994),
    "Nonlinear Dynamics of Immunogenic Tumors: Parameter Estimation and
     Global Bifurcation Analysis", Bull. Math. Biol. 56(2), 295-321.

Importing this module exposes:

  * Dimensional system:        model(), params
  * Experimental data:         times, log_values, values
  * Nondimensional system:     model_nd(), nd_params,
                               and the unpacked scalars
                               sigma, rho, eta, mu, delta, alpha, beta
  * Integration safety net:    SAFETY_BOUNDS, safety_event

The two reference scales E0 = T0 = 1e6 cells are exposed as E0_nd, T0_nd.
"""

import numpy as np


# =============================================================================
# Dimensional model (equations 4a / 4b after quasi-steady-state reduction)
# =============================================================================

def model(t, y, s, p, g, m, d, a, b, n):
    """Dimensional 2-ODE tumor-immune system.

    State y = (E, T) where E = effector cells, T = tumor cells.

        dE/dt = s + p*E*T/(g+T) - m*E*T - d*E
        dT/dt = a*T*(1 - b*T) - n*E*T
    """
    E, T = y
    dE_dt = s + p * E * T / (g + T) - m * E * T - d * E
    dT_dt = a * T * (1 - b * T) - n * E * T
    return [dE_dt, dT_dt]


# Dimensional parameters (Section 3)
#  name  value           units             meaning
params = {
    'a': 0.18,          # day^-1,          tumor growth rate
    'b': 2.0e-9,        # cells^-1,        inverse carrying capacity
    's': 1.3e4,         # cells/day,       effector source rate
    'p': 0.1245,        # day^-1,          stimulated recruitment rate
    'g': 2.019e7,       # cells,           half-saturation constant
    'm': 3.422e-10,     # day^-1 cells^-1, effector inactivation rate
    'n': 1.101e-7,      # day^-1 cells^-1, tumor kill rate
    'd': 0.0412,        # day^-1,          effector death rate
}


# =============================================================================
# Experimental data
# =============================================================================
# Digitized from Figure 1 of the paper using WebPlotDigitizer.
# BCL1 tumor growth in chimeric mice, curve 2 (5e5 initial tumor cells),
# originally from Siu et al. 1986.

times = [0, 20, 30, 50, 60, 70, 90]                              # days
log_values = [5.692, 7.298, 8.195, 8.535, 8.652, 8.704, 8.836]   # log10(cells)
values = [round(10 ** i) for i in log_values]                    # actual counts


# =============================================================================
# Nondimensionalization (Section 4)
# =============================================================================
# Scale: x = E/E0, y = T/T0 with E0 = T0 = 1e6 cells; time tau = n*T0*t.

E0_nd = 1e6
T0_nd = 1e6


def _compute_nd_params(p):
    """Derive the seven nondimensional parameters from a dimensional dict."""
    n = p['n']
    return {
        'sigma': p['s'] / (n * E0_nd * T0_nd),
        'rho':   p['p'] / (n * T0_nd),
        'eta':   p['g'] / T0_nd,
        'mu':    p['m'] / p['n'],          # equals k3/k2
        'delta': p['d'] / (n * T0_nd),
        'alpha': p['a'] / (n * T0_nd),
        'beta':  p['b'] * T0_nd,
    }


nd_params = _compute_nd_params(params)

# Unpacked scalars for convenience (match the paper's symbols).
# Expected: sigma=0.1181, rho=1.131, eta=20.19, mu=0.00311,
#           delta=0.3743, alpha=1.636, beta=2e-3
sigma = nd_params['sigma']
rho   = nd_params['rho']
eta   = nd_params['eta']
mu    = nd_params['mu']
delta = nd_params['delta']
alpha = nd_params['alpha']
beta  = nd_params['beta']


def model_nd(tau, z, sigma, rho, eta, mu, delta, alpha, beta):
    """Nondimensional 2-ODE system (equations 6a / 6b).

        dx/dtau = sigma + rho*x*y/(eta+y) - mu*x*y - delta*x
        dy/dtau = alpha*y*(1 - beta*y) - x*y
    """
    x, y = z
    dx = sigma + rho * x * y / (eta + y) - mu * x * y - delta * x
    dy = alpha * y * (1 - beta * y) - x * y
    return [dx, dy]


# =============================================================================
# Integration safety net
# =============================================================================
# A terminal event that stops integration once a trajectory diverges well
# outside any plotted range, instead of burning steps on a blowup.

SAFETY_BOUNDS = (-2, 50, -2, 2000)   # (lo_x, hi_x, lo_y, hi_y)


def safety_event(t, z):
    x, y = z
    lo_x, hi_x, lo_y, hi_y = SAFETY_BOUNDS
    return min(x - lo_x, hi_x - x, y - lo_y, hi_y - y)


safety_event.terminal = True


if __name__ == "__main__":
    print("Dimensional parameters:", params)
    print("Nondimensional parameters:")
    for k, v in nd_params.items():
        print(f"  {k:6s} = {v:.5g}")
