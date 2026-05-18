from . import post_proccessor
from scipy.optimize import newton
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
        self.Q_prev = None
        self.Q_current = None
        self.operators = operators

    def residual(self, Q_prev: NDArray, Q_current: NDArray, rconst: float) -> float:
        # Compute the residual between the two entropies,
        Q_relaxed = Q_prev + rconst * (Q_current - Q_prev)
        first = ev.global_entropy_rhotheta(Q_relaxed, self.geom, self.operators)
        second = ev.global_entropy_rhotheta(Q_prev,    self.geom, self.operators)
        current = first - second
        print(f"first = {first} ---------- Second = {second}")
        return current
    
    def update(self, Q_before: NDArray, Q: NDArray):
        self.Q_prev = Q_before.copy()
        self.Q_current = Q
    
    def process(self):
        if self.Q_current is None or self.Q_prev is None:
            return
        self._apply_relaxation(self.Q_prev, self.Q_current)

    def _apply_relaxation(self, Q_prev: NDArray, Q_current: NDArray):
        if self.residual(Q_prev, Q_current, 0.9) == 0:
            return
        

        def r_func(gamma):
            return self.residual(Q_prev, Q_current, float(gamma))

        #gamma_arr, *_ = nonlin.newton_krylov(r_func, x0=numpy.array([1.0]), f_tol=1e-6, maxiter=100, verbose=False, line_search=None,  fgmres_restart=1)
        #gamma = float(gamma_arr[0])
        
        result = newton(r_func, x0=0.9, x1=1.1, tol=1e-6, maxiter=100)
        gamma = float(result)

        print(f"-------gamma currently at {gamma} --------")

        Q_current[:] = Q_prev + gamma*(Q_current - Q_prev)