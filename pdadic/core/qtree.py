"""
Implements the QTree class.
"""

import numpy as np

from .types import Number
from .types import DiscreteFunction
from .types import DiscreteFunctionFunnel
from .types import RealFunction
from .types import Integrator
from .types import Interval

from .types import Coefficients
from .types import Contribution

na = None
inv = lambda A: np.conj(A.T)


class Layer:
    """
    Encapsulates layer based operations for the QTree class.
    """

    def __init__(self, n: int, q: int, real: bool = False):
        """
        Specifies a layer by the total number of nodes and
        the parent degree.
        
        Optionally only deals with real components.
        """
        assert n % q == 0, f"{n=} must be divisible by {q=}"

        self.n = n
        self.q = q
        
        W = self._get_fft_mat(self.q)
        self.W = np.real(W) if real else W

    @staticmethod
    def _get_fft_mat(q: int) -> np.ndarray:
        """
        Generates an appropriatly sized fft matrix for
        q-wavelets component analysis.
        """
        
        A = np.arange(q)
        B = (-2j * np.pi / q) * (A[:, na] @ A[na, :])
        W = 1/np.sqrt(q) * np.exp(B)
    
        # W = [phi_0, ..., phi_{q-1}]^T
        return W

    def get_coefficients(self, func: DiscreteFunction) -> Coefficients:
        """
        Computes the coefficients of q-wavelet components
        analysis, with wavelets supported in
            [0..q-1], [q..2q-1], ..., [kq..n-1].
        """
        assert len(func) == self.n, f"{len(func)=} is not {self.n}"
        
        f = np.array(func)
        f = f.reshape((self.n//self.q, self.q)).T
        
        coeff = inv(self.W) @ f
        return coeff

    def get_contribution(self, coeff: Coefficients) -> DiscreteFunction:
        """
        Combines the coefficients of q-wavelet component analysis.
        """

        g = (self.W[:,1:] @ coeff[1:,:]).T
        return g.flatten()


class QTree:
    """
    A QTree is a finite regulary branching tree. It is
    specified by providing a tuple of so called branching
    numbers Q=(q_1,...,q_N).
    
    By convention the QTree for the empty tuple is taken to
    be a single root node without any edges.
    """

    def __init__(self, Q: list[int]):
        """
        Instanciates the QTree and precomputes some attributes.
        """
        
        self.Q = Q
        self.Qc = np.cumprod(Q)
        self.N = np.prod(Q)
        
        self.height = len(Q)

        self.layers = [Layer(qc,q) for q, qc in zip(self.Q, self.Qc)]

    @staticmethod
    def _trapezoid_integrator(func: RealFunction, x: float, y: float):
        """
        Cheap method for integrating real functions with
        second derivitive.
        """
        
        mid = x + (y-x) / 2
        _sum = func(x) + 4*func(mid) + func(y)
        return _sum / 6

    def funnel(self, func: DiscreteFunction) -> DiscreteFunctionFunnel:
        """
        Successively averages a function on N elements,
        as specified by the qtree.
        """

        f = np.array(func)
        
        arr = [func]
        for q in self.Q[::-1]:
            f_mat = f.reshape((f.size//q, q)).T
            f = np.sum(f_mat, axis=0) / q
            
            arr.append(tuple(f))
        return arr[::-1]

    def discretize(self,
        func: RealFunction,
        integrator: Integrator | None = None,
        interval: Interval = Interval(0,1)
    ) -> DiscreteFunction:
        """
        Successively approximates the input function by
        functions, locally constant on successive refinements. 
        """

        if integrator is None:
            integrator = self._trapezoid_integrator

        a,b = interval

        _ps = a + (b-a) * np.arange(self.N)/self.N
        points = np.append(_ps, [b])

        return [
            integrator(func, x, y)
            for x,y in zip(points, points[1:])
        ]

    def __repr__(self):
        return f"{self.Q}-Tree(height={self.height}, leaves={self.N:_})"


class NAryTree(QTree):
    """
    A simple N-ary tree.
    """
    
    def __init__(self, regularity, height):
        self.regularity = regularity

        Q = height * [regularity,]

        super().__init__(Q)

    def __repr__(self):
        return f"{self.regularity}AryTree(height={self.height}, leaves={self.N:_})"
