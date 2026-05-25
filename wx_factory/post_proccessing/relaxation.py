from . import post_proccessor
from scipy.optimize import newton, brentq
from numpy.typing import NDArray
from common import Configuration
from geometry import Cartesian2D, DFROperators
from solvers import nonlin
from init import entropy_vars as ev
import math
import numpy as np


class RelaxationPostProcessor(post_proccessor.PostProcessor):
    def __init__(self, config, geom, operators, Q_initial, rhs):
        if type(geom) != Cartesian2D or geom is None:
            raise TypeError("The relaxation post processor works only with a 2D cartesian geometry")
        self.geom = geom
        self.xp = geom.device.xp
        self.Q_prev = None
        self.Q_current = None
        self.operators = operators
        self.rhs = rhs
        self.dt = None
        self.dt_effective = None
        self.entropy_rate_estimate = 0.0

    def residual(self, Q_prev: NDArray, Q_current: NDArray, rconst: float) -> float:
        Q_relaxed = Q_prev + rconst * (Q_current - Q_prev)
        eta_relaxed = ev.global_entropy(Q_relaxed, self.geom, self.operators)
        eta_prev    = ev.global_entropy(Q_prev,    self.geom, self.operators)
        return eta_relaxed - eta_prev - rconst * self.dt * self.entropy_rate_estimate

    def residual_prime(self, Q_prev: NDArray, Q_current: NDArray, rconst: float) -> float:
        Q_relaxed = Q_prev + rconst * (Q_current - Q_prev)
        dQ        = Q_current - Q_prev
        d_eta     = ev.entropy_prime_function(Q_relaxed, dQ, self.geom, self.operators)
        return d_eta - self.dt * self.entropy_rate_estimate

    def update(self, Q_before: NDArray, Q: NDArray, dt: float):
        self.Q_prev = Q_before.copy()
        self.Q_current = Q
        self.dt = dt
        self.dt_effective = dt

        F_check = self.rhs(Q_before)
        dQ_over_dt = (Q - Q_before) / dt

        # Compare F to dQ/dt component-wise
        diff_norm = float(self.xp.linalg.norm(F_check - dQ_over_dt))
        F_norm    = float(self.xp.linalg.norm(F_check))
        dQ_norm   = float(self.xp.linalg.norm(dQ_over_dt))
        print(f"||F||         = {F_norm:.6e}")
        print(f"||dQ/dt||     = {dQ_norm:.6e}")
        print(f"||F - dQ/dt|| = {diff_norm:.6e}")
        print(f"ratio ||F||/||dQ/dt|| = {F_norm/dQ_norm:.4f}")

        # Compute the CN entropy-rate estimate ONCE per step,
        # using the same Q^n and Q^{n+1} the integrator just produced.
        F_prev = self.rhs(Q_before)
        F_curr = self.rhs(Q)
        V_prev = ev.entropy_variables_rhotheta(Q_before, self.geom)
        V_curr = ev.entropy_variables_rhotheta(Q, self.geom)

        ip_prev = ev.entropy_inner_product(V_prev, F_prev, self.geom, self.operators)
        ip_curr = ev.entropy_inner_product(V_curr, F_curr, self.geom, self.operators)
        self.entropy_rate_estimate = 0.5 * (ip_prev + ip_curr)
        print(f"<v_n, F_n>     = {ev.entropy_inner_product(V_prev, F_prev, self.geom, self.operators):.6e}")
        print(f"<v_n, dQ>/dt  = {ev.entropy_inner_product(V_prev, (Q - Q_before)/dt, self.geom, self.operators):.6e}")

    def process(self):
        if self.Q_current is None or self.Q_prev is None:
            return
        self._apply_relaxation(self.Q_prev, self.Q_current)

    def _apply_relaxation(self, Q_prev: NDArray, Q_current: NDArray):
        '''
        r_values = np.concatenate([
            np.linspace(0.0, 0.1, 6),     # near the spurious root
            np.linspace(0.2, 0.9, 8),     # middle
            np.linspace(0.95, 1.05, 11),  # near the expected root
            np.linspace(1.1, 2.0, 5),     # above 1
        ])
        print(f"[sweep]   r            R(r)              R'(r)")
        for r in r_values:
            R  = self.residual(Q_prev, Q_current, float(r))
            Rp = self.residual_prime(Q_prev, Q_current, float(r))
            print(f"[sweep]  {r:8.5f}    {R: .6e}    {Rp: .6e}")
        print("[sweep] ---- end sweep ----")
        '''
        def r_func(gamma):
            return self.residual(Q_prev, Q_current, float(gamma))

        def r_func_prime(gamma):
            return self.residual_prime(Q_prev, Q_current, float(gamma))

        result = newton(r_func, fprime=r_func_prime, x0=1.05, tol=1e-8, maxiter=200)
        gamma = float(result)

        if not (0.5 < gamma < 2.0):
            print(f"WARNING: relaxation gamma = {gamma}, falling back to 1.0")
            gamma = 1.0

        print(f"-------gamma currently at {gamma} --------")

        Q_current[:] = Q_prev + gamma * (Q_current - Q_prev)
        self.dt_effective