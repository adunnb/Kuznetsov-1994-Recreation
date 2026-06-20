"""
phase_portrait.py
=================

Shared phase-portrait machinery for the nondimensional Kuznetsov model,
used by Figures 3, 5, 6, 8 and 9.

A single ``PhasePortraitPlotter`` instance is built per (sigma, delta, mu)
combination (the other parameters rho, eta, alpha, beta are shared) and
knows how to find and classify fixed points, draw streamplots, trace
saddle manifolds and basin boundaries, integrate trajectories, and
assemble a complete panel.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from kuznetsov_model import safety_event


class PhasePortraitPlotter:
    """Phase-portrait machinery for the nondimensional model.

    Build directly with the seven parameters, or via
    :meth:`from_nd_params` to pull them from an ``nd_params`` dict
    (with optional per-figure overrides of sigma / delta / mu).
    """

    def __init__(self, sigma, delta, mu, rho, eta, alpha, beta):
        self.sigma = sigma
        self.delta = delta
        self.mu = mu
        self.rho = rho
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self._fixed_point_cache = {}
        self._curve_cache = {}

    @classmethod
    def from_nd_params(cls, nd_params, **overrides):
        kwargs = {k: nd_params[k] for k in
                  ('sigma', 'delta', 'mu', 'rho', 'eta', 'alpha', 'beta')}
        kwargs.update(overrides)
        return cls(**kwargs)

    # ---- dynamics ---------------------------------------------------------

    def ode(self, t, z):
        x, y = z
        dx = self.sigma + self.rho * x * y / (self.eta + y) - self.mu * x * y - self.delta * x
        dy = self.alpha * y * (1 - self.beta * y) - x * y
        return [dx, dy]

    def jacobian(self, x, y):
        rho, eta, mu, delta, alpha, beta = (
            self.rho, self.eta, self.mu, self.delta, self.alpha, self.beta
        )
        return np.array([
            [rho * y / (eta + y) - mu * y - delta,  rho * x * eta / (eta + y) ** 2 - mu * x],
            [-y,                                     alpha * (1 - 2 * beta * y) - x],
        ])

    # ---- fixed points -----------------------------------------------------

    def classify_fixed_point(self, x, y):
        eigs = np.linalg.eigvals(self.jacobian(x, y))
        if np.all(np.real(eigs) < 0):
            return "stable"
        if np.all(np.real(eigs) > 0):
            return "unstable"
        return "saddle"

    def _interior_residual(self, y):
        denom = self.delta + self.mu * y - self.rho * y / (self.eta + y)
        if abs(denom) < 1e-12:
            return np.nan
        return self.sigma / denom - self.alpha * (1 - self.beta * y)

    def find_fixed_points(self, y_max=500, n_grid=1500):
        cache_key = (round(y_max, 6), n_grid)
        cached = self._fixed_point_cache.get(cache_key)
        if cached is not None:
            return cached

        x_A = self.sigma / self.delta
        fixed_points = [(x_A, 0.0, self.classify_fixed_point(x_A, 0.0))]

        ys = np.linspace(1e-6, y_max, n_grid)
        denom = self.delta + self.mu * ys - self.rho * ys / (self.eta + ys)
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = np.where(np.abs(denom) > 1e-12,
                            self.sigma / denom - self.alpha * (1 - self.beta * ys), np.nan)

        finite_pair = np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
        sign_change = np.nonzero((vals[:-1] * vals[1:] < 0) & finite_pair)[0]

        for idx in sign_change:
            y1, y2 = ys[idx], ys[idx + 1]
            try:
                y_root = brentq(self._interior_residual, y1, y2, maxiter=100)
            except ValueError:
                continue
            x_root = self.alpha * (1 - self.beta * y_root)
            if x_root <= 0 or y_root <= 0:
                continue
            if any(np.hypot(x_root - x0, y_root - y0) < 1e-3 for x0, y0, _ in fixed_points):
                continue
            fixed_points.append((x_root, y_root, self.classify_fixed_point(x_root, y_root)))

        fixed_points.sort(key=lambda p: p[1])
        self._fixed_point_cache[cache_key] = fixed_points
        return fixed_points

    def get_interior_saddles(self):
        return [(x, y) for x, y, t in self.find_fixed_points() if t == "saddle" and y > 0]

    # ---- plotting primitives ---------------------------------------------

    def draw_streamplot(self, ax, x_max=5, y_max=500, log_y=False):
        x_vals = np.linspace(0.01, x_max, 30)
        y_vals = np.linspace(0.3 if log_y else 0.01, y_max, 30)
        X, Y = np.meshgrid(x_vals, y_vals)
        dX = self.sigma + self.rho * X * Y / (self.eta + Y) - self.mu * X * Y - self.delta * X
        dY = self.alpha * Y * (1 - self.beta * Y) - X * Y
        speed = np.sqrt(dX ** 2 + dY ** 2)
        speed[speed == 0] = 1
        ax.streamplot(x_vals, y_vals, dX / speed, dY / speed,
                      color='lightgrey', density=1.0, linewidth=0.6, arrowsize=0.7)

    def plot_nullclines(self, ax, y_max=800, n=1000, denom_threshold=0.01):
        y = np.linspace(0, y_max, n)
        denom = self.delta + self.mu * y - self.rho * y / (self.eta + y)
        mask = np.abs(denom) > denom_threshold
        f_vals = np.where(mask, self.sigma / denom, np.nan)
        g_vals = self.alpha * (1 - self.beta * y)
        ax.plot(f_vals, y, 'C0', lw=1.5, label=r'x-nullcline $\dot{x}=0$')
        ax.plot(g_vals, y, 'C1', lw=1.5, label=r'y-nullcline $\dot{y}=0$')

    def plot_fixed_points(self, ax, log_y=False):
        for x_fp, y_fp, fp_type in self.find_fixed_points(y_max=max(500, ax.get_ylim()[1])):
            if log_y and y_fp <= 0:
                continue
            if not (ax.get_xlim()[0] <= x_fp <= ax.get_xlim()[1] and
                    ax.get_ylim()[0] <= y_fp <= ax.get_ylim()[1]):
                continue
            if fp_type == "stable":
                ax.plot(x_fp, y_fp, 'ko', ms=7, zorder=10)
            elif fp_type == "saddle":
                ax.plot(x_fp, y_fp, 'o', ms=7, markerfacecolor='white',
                        markeredgecolor='black', markeredgewidth=1.2, zorder=10)
            else:
                ax.plot(x_fp, y_fp, 'o', ms=7, markerfacecolor='lightgrey',
                        markeredgecolor='black', markeredgewidth=1.2, zorder=10)

    def label_fixed_points(self, ax, log_y=False):
        fixed_points = self.find_fixed_points(y_max=max(500, ax.get_ylim()[1]))
        boundary = [p for p in fixed_points if abs(p[1]) < 1e-8]
        interior = sorted((p for p in fixed_points if p[1] > 1e-8), key=lambda p: p[1])
        labeled = ([("A", boundary[0])] if boundary else []) + list(zip(["B", "C", "D"], interior))

        for label, (x, y, fp_type) in labeled:
            if log_y and y <= 0:
                continue
            if not (ax.get_xlim()[0] <= x <= ax.get_xlim()[1] and
                    ax.get_ylim()[0] <= y <= ax.get_ylim()[1]):
                continue
            ax.text(x + 0.06, y + (12 if not log_y else y * 0.12), label,
                    fontsize=10, color='darkred', zorder=12)

    def plot_trajectory(self, ax, z0, T=250, dt_mark=0.1, rtol=1e-6, atol=1e-8):
        sol = solve_ivp(self.ode, (0, T), z0, max_step=0.2, rtol=rtol, atol=atol,
                        dense_output=True, events=safety_event)
        t_end = sol.t[-1]
        ts = np.arange(0, t_end, dt_mark)
        if len(ts) == 0:
            return
        x, y = sol.sol(ts)
        mask = (x >= ax.get_xlim()[0]) & (x <= ax.get_xlim()[1]) & (y > 0) & (y <= ax.get_ylim()[1])
        ax.plot(x[mask], y[mask], 'k.', ms=0.8)

    def _saddle_manifold_raw(self, eps, T_stable, T_unstable):
        key = (eps, T_stable, T_unstable)
        cached = self._curve_cache.get(('saddle_manifolds', key))
        if cached is not None:
            return cached

        curves = []
        for x_c, y_c in self.get_interior_saddles():
            eigvals, eigvecs = np.linalg.eig(self.jacobian(x_c, y_c))
            for eigval, eigvec in zip(eigvals, eigvecs.T):
                eigvec = np.real(eigvec)
                eigvec = eigvec / np.linalg.norm(eigvec)
                stable = np.real(eigval) < 0
                t_span = (0, -T_stable) if stable else (0, T_unstable)
                linestyle = '-' if stable else '--'
                linewidth = 1.5 if stable else 1.2
                for sign in (-1, 1):
                    z0 = np.array([x_c, y_c]) + sign * eps * eigvec
                    if z0[0] <= 0 or z0[1] <= 0:
                        continue
                    sol = solve_ivp(self.ode, t_span, z0, max_step=0.75, rtol=1e-6, atol=1e-8,
                                    events=safety_event)
                    curves.append((linestyle, linewidth, sol.y[0], sol.y[1]))

        self._curve_cache[('saddle_manifolds', key)] = curves
        return curves

    def plot_saddle_manifolds(self, ax, eps=1e-4, T_stable=120, T_unstable=180):
        for linestyle, linewidth, x, y in self._saddle_manifold_raw(eps, T_stable, T_unstable):
            mask = ((x >= ax.get_xlim()[0]) & (x <= ax.get_xlim()[1]) &
                    (y > 0) & (y >= ax.get_ylim()[0]) & (y <= ax.get_ylim()[1]))
            ax.plot(x[mask], y[mask], color='k', linestyle=linestyle,
                    linewidth=linewidth, zorder=8)

    def plot_basin_boundary(self, ax, x_scan, y_bracket=(5.0, 499.0), n_bisect=14,
                            y_threshold=100, T=300, **solve_kwargs):
        """Trace a basin boundary by bisecting on y for each x in x_scan.

        Used where the separatrix isn't fully captured by eigenvector
        shooting alone. Cached per (x_scan, y_bracket, n_bisect, ...) since
        callers may re-plot the same boundary on multiple panels.
        """
        x_scan = tuple(np.round(np.asarray(x_scan, dtype=float), 6))
        key = ('basin_boundary', x_scan, y_bracket, n_bisect, y_threshold, T,
               tuple(sorted(solve_kwargs.items())))
        cached = self._curve_cache.get(key)
        if cached is None:
            kwargs = {'max_step': 5.0, 'rtol': 1e-5, 'atol': 1e-7, 'events': safety_event,
                      **solve_kwargs}
            sep_x, sep_y = [], []
            for x0 in x_scan:
                lo, hi = y_bracket
                for _ in range(n_bisect):
                    mid = (lo + hi) / 2
                    sol = solve_ivp(self.ode, (0, T), [x0, mid], **kwargs)
                    if sol.y[1, -1] < y_threshold:
                        lo = mid
                    else:
                        hi = mid
                sep_x.append(x0)
                sep_y.append((lo + hi) / 2)
            cached = (sep_x, sep_y)
            self._curve_cache[key] = cached
        ax.plot(*cached, 'k-', linewidth=1.5)

    def plot_unstable_manifold_of_A(self, ax, eps=1e-4, T=300):
        x_A, y_A = self.sigma / self.delta, 0.0
        eigvals, eigvecs = np.linalg.eig(self.jacobian(x_A, y_A))
        for eigval, eigvec in zip(eigvals, eigvecs.T):
            if np.real(eigval) <= 0:
                continue
            eigvec = np.real(eigvec)
            eigvec = eigvec / np.linalg.norm(eigvec)
            for sign in (-1, 1):
                z0 = np.array([x_A, y_A]) + sign * eps * eigvec
                if z0[0] <= 0 or z0[1] <= 0:
                    continue
                sol = solve_ivp(self.ode, (0, T), z0, max_step=1.0, rtol=1e-6, atol=1e-8,
                                events=safety_event)
                x, y = sol.y
                mask = ((x >= ax.get_xlim()[0]) & (x <= ax.get_xlim()[1]) &
                        (y > 0) & (y >= ax.get_ylim()[0]) & (y <= ax.get_ylim()[1]))
                ax.plot(x[mask], y[mask], 'k--', lw=1.2, zorder=8)

    def stable_manifold_branches(self, ax, eps=1e-4, T=250, max_step=1.0):
        """Stable-manifold branches of the first interior saddle, clipped to ax's limits."""
        saddles = self.get_interior_saddles()
        if not saddles:
            return []
        x_c, y_c = saddles[0]
        eigvals, eigvecs = np.linalg.eig(self.jacobian(x_c, y_c))
        stable_idx = np.where(np.real(eigvals) < 0)[0]
        if len(stable_idx) == 0:
            return []
        v = np.real(eigvecs[:, stable_idx[0]])
        v = v / np.linalg.norm(v)

        branches = []
        for sign in (-1, 1):
            z0 = np.array([x_c, y_c]) + sign * eps * v
            if z0[0] <= 0 or z0[1] <= 0:
                continue
            sol = solve_ivp(self.ode, (0, -T), z0, max_step=max_step, rtol=1e-6, atol=1e-8,
                            events=safety_event)
            x, y = sol.y
            mask = ((x >= ax.get_xlim()[0]) & (x <= ax.get_xlim()[1]) &
                    (y >= ax.get_ylim()[0]) & (y <= ax.get_ylim()[1]))
            branches.append((x[mask], y[mask]))
        return branches

    # ---- panel orchestration ---------------------------------------------

    def make_panel(self, ax, initials, title, xlim, ylim, log_y=False,
                   show_streamplot=True, show_nullclines=False,
                   show_saddle_manifolds=True, manifold_kwargs=None,
                   show_unstable_A=False):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if log_y:
            ax.set_yscale('log')

        if show_nullclines:
            self.plot_nullclines(ax, y_max=ylim[1])

        if show_streamplot and not log_y:
            self.draw_streamplot(ax, x_max=xlim[1], y_max=ylim[1], log_y=log_y)

        if show_saddle_manifolds:
            kwargs = {'T_stable': 45, 'T_unstable': 60, **(manifold_kwargs or {})}
            self.plot_saddle_manifolds(ax, **kwargs)

        if show_unstable_A:
            self.plot_unstable_manifold_of_A(ax)

        for z0 in initials:
            ax.plot(z0[0], z0[1], '+', color='k', ms=8, zorder=9)
            self.plot_trajectory(ax, z0)

        self.plot_fixed_points(ax, log_y=log_y)
        self.label_fixed_points(ax, log_y=log_y)

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
