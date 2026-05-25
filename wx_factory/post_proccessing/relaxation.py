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
    def __init__(self, config: Configuration, geom: Cartesian2D, operators: DFROperators, Q_initial: NDArray):
        if type(geom) != Cartesian2D or geom is None:
            raise TypeError("The relaxation post processor works only with a 2D cartesian geometry")
        self.geom = geom
        self.xp = geom.device.xp
        self.Q_prev = None
        self.Q_current = None
        self.operators = operators

    def residual(self, Q_prev: NDArray, Q_current: NDArray, rconst: float) -> float:
        # Compute the residual between the two entropies,
        Q_relaxed = Q_prev + rconst * (Q_current - Q_prev)
        first = ev.global_entropy(Q_relaxed, self.geom, self.operators)
        second = ev.global_entropy(Q_prev,   self.geom, self.operators)
        current = first - second
        #print(f"first = {first} ---------- Second = {second}")
        return current
    
    def update(self, Q_before: NDArray, Q: NDArray):
        print(f"[update] sum(Q_prev) = {float(self.xp.sum(Q_before)):.15e}")
        self.Q_prev = Q_before.copy()
        self.Q_current = Q
    
    def process(self):
        if self.Q_current is None or self.Q_prev is None:
            return
        self._apply_relaxation(self.Q_prev, self.Q_current)
    
    def residual_prime(self, Q_prev: NDArray, Q_current: NDArray, rconst: float) -> float:
        """Derivative of `residual` with respect to rconst.

        Since residual(r) = S(Q_prev + r * dQ) - S(Q_prev), and the second term
        is constant in r, this reduces to the directional derivative of S at
        Q_relaxed in the direction dQ = Q_current - Q_prev.
        """
        Q_relaxed = Q_prev + rconst * (Q_current - Q_prev)
        dQ        = Q_current - Q_prev
        return ev.entropy_prime_function(Q_relaxed, dQ, self.geom, self.operators)

    def _apply_relaxation(self, Q_prev: NDArray, Q_current: NDArray):
        '''
        r_values = np.concatenate([
        np.linspace(0.0, 0.1, 6),       # near the spurious root
        np.linspace(0.2, 0.9, 8),       # middle
        np.linspace(0.95, 1.05, 11),    # near the expected root
        np.linspace(1.1, 2.0, 5),       # above 1
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

        
        result = newton(r_func, fprime=r_func_prime, x0=0.95, tol=1e-8, maxiter=200)
        gamma = float(result)

        print(f"-------gamma currently at {gamma} --------")

        Q_current[:] = Q_prev + gamma*(Q_current - Q_prev)