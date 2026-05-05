from . import post_proccessor
from scipy.optimize import brentq
from numpy.typing import NDArray
from common import Configuration
from geometry import Cartesian2D, DFROperators
from solvers import nonlin
from init import entropy_vars as ev
import math
import numpy

class RelaxationPostProcessor(post_proccessor.PostProcessor):
    def __init__(self, config: Configuration, geom: Cartesian2D, operators: DFROperators, Q_initial: NDArray):
        if type(geom) != Cartesian2D or geom is None:
            raise TypeError("The relaxation post processor works only with a 2D cartesian geometry")
        self.geom = geom
        self.xp = geom.device.xp
        self.Q_prev = Q_initial.copy()
        self.Q_current = None
        self.operators = operators

    def residual(self, Q_prev: NDArray, Q_current: NDArray, rconst: float) -> float:
        # Compute the residual between the two entropies,
        Q_relaxed = Q_prev + rconst * (Q_current - Q_prev)
        return ev.global_entropy(Q_relaxed, self.geom, self.operators) - ev.global_entropy(Q_prev, self.geom, self.operators)
    
    def update(self, Q: NDArray):
        self.Q_prev = Q.copy()
        self.Q_current = Q
    
    def process(self):
        if self.Q_current is None:
            return
        self._apply_relaxation(self.Q_prev, self.Q_current)

    def _apply_relaxation(self, Q_prev: NDArray, Q_current: NDArray):

        def r_func(gamma):
            return self.residual(Q_prev, Q_current, float(gamma))

        #gamma_arr, *_ = nonlin.newton_krylov(r_func, x0=numpy.array([1.0]), f_tol=1e-6, maxiter=100, verbose=False, line_search=None,  fgmres_restart=1)
        #gamma = float(gamma_arr[0])
        result = brentq(r_func, a=0.5, b=1.2, xtol=1e-9, maxiter=100)
        gamma = float(result)
        print(f"-------gamma currently at {gamma} --------")

        Q_current[:] = Q_prev + gamma * (Q_current - Q_prev)

    
    