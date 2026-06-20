# Kuznetsov et al. 1994: Recreation

Computational recreation of:

> Kuznetsov, Makalkin, Taylor & Perelson (1994),
> *Nonlinear Dynamics of Immunogenic Tumors: Parameter Estimation and
> Global Bifurcation Analysis*, Bulletin of Mathematical Biology 56(2), 295-321.

## Layout

| File | Contents |
|------|----------|
| `Kuznetsov_Figure_Plots.ipynb` | The notebook. Run top to bottom to reproduce every figure. Its first code cell imports everything from the three modules below. |
| `kuznetsov_model.py` | Dimensional model, parameters, experimental data, nondimensionalization, and the integration safety event. All parameter values are defined here, nowhere else. |
| `phase_portrait.py` | `PhasePortraitPlotter`, shared phase-portrait machinery (fixed points, manifolds, streamplots, basin boundaries, trajectories, panel assembly) used by Figures 3, 5, 6, 8 and 9. |
| `fig4_helpers.py` | Region-classification helpers for the Figure 4 bifurcation diagram (`classify_local`, `in_region_5`, and supporting functions). |

The four files must sit in the same directory so the notebook's imports
resolve.

## How the notebook uses the modules

A single import cell near the top of the notebook pulls in everything the
figures need:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq
from IPython.display import display, Image

from kuznetsov_model import (params, nd_params, sigma, rho, eta, mu,
                             delta, alpha, beta, times, log_values, values,
                             model, model_nd, E0_nd, T0_nd)
from phase_portrait import PhasePortraitPlotter
from fig4_helpers import classify_local, in_region_5
```

Every figure cell after that uses these names directly (`sigma`,
`nd_params`, `PhasePortraitPlotter`, `classify_local`, ...).

## Note on `fig4_helpers`

The helpers take the shared parameters (`rho, eta, mu, alpha, beta`) as
explicit keyword arguments, defaulting to the estimated values from
`kuznetsov_model`. The simple call sites `classify_local(s, d)` and
`in_region_5(s, d)` work without passing anything extra, but the same
functions can be reused for alternate parameter sets (e.g. sweeping mu)
without relying on global state.

## Requirements

- numpy
- scipy
- matplotlib

Figure 4 (`in_region_5` over the grid) is the slow one, roughly 5-15 minutes
depending on grid resolution. All other figures render in seconds to a couple
of minutes.
