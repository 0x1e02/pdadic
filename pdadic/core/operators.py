"""
Defines certain kinds of linear operators on Q-adic trees (QTree) via kernel functions.

Includes QTreeOperator for kernel-based operators with explicit eigenvalues,
and functions for common kernels (power-law, exponential).

Supports time evolution of QTreeMap functions through exponential decay transforms.
"""


import numpy as np

from typing import Generator

from .qtree import QTree
from .maps import QTreeMap
from .types import RealFunction


class QTreeOperator:
    """
    Models an operator given by a kernel function.
    
    Represents a linear operator on a QTree structure,
    defined via a kernel function.

    The operator acts on functions represented in the QTree
    basis by applying a kernel to pairwise distances in the
    Q-adic hierarchy, yielding a diagonalizable operator
    with an explicit spectral representation.
    """

    def __init__(self, qtree: QTree, kernel: RealFunction) -> None:
        self.domain = qtree
        self._kernel = kernel

    @property
    def eigenvalues(self) -> np.ndarray:
        """
        Compute the eigenvalues of the operator using a
        distance-based discretization over the QTree domain.

        The result is a diagonal spectrum derived from
        evaluating the kernel at scaled Q-adic intervals
        and applying weighted cumulative sums.

        Returns:
            np.ndarray: Array of eigenvalues aligned with
                the QTree basis.
        """

        cohort = self.domain.Qc
        c = cohort - np.append([1], cohort[:-1])
        d = 1 / cohort
    
        kd = np.array([self._kernel(x) for x in d])
    
        arr = [
            d[i] * ( c[:i] @ kd[:i] + kd[i])
            for i in range(len(self.domain.Q))
        ]
        return np.array(arr)

    @staticmethod
    def get_geometric_ts(n:int, r: float = 0) -> np.ndarray:
        """
        Generate a sequence of `n` monotonically increasing
        time steps in [0, 1), where the step sizes grow
        geometrically with ratio `1 + r`.

        Args:
            n (int): Number of time steps.
            r (float, optional): Growth coefficient. Must
                be greater than -1. Default is 0
                (i.e., uniform steps).

        Returns:
            np.ndarray: A 1D array of shape (n,) containing
                the time steps in [0, 1).
        """
        assert r > -1, f"The groth coefficient {r=} must be > -1."
        
        # Compute a geometric progression
        geom = (1+r) ** np.arange(n)

        # Scale it
        sgeom = np.cumsum(geom)
        sgeom = sgeom / sgeom[-1]

        return np.concatenate([[0], sgeom[:-1]])
        

    def get_slice(
        self,
        T: float,
        initial_value: QTreeMap
    ) -> QTreeMap:
        """
        Solve the associated differential equiationfor at
        a single time `T`.

        Args:
            T (float): Time parameter.
            initial_value (QTreeMap): The initial value to
                be transformed.

        Returns:
            QTreeMap: The transformed map at time `T`.
        """

        decomposition = initial_value.to_decomposition()
        factors = np.exp(-T * self.eigenvalues)

        transform = decomposition.transform(factors)
        return transform.to_map()

    def get_slices(
        self,
        initial_value: QTreeMap,
        ts: np.ndarray
    ) -> list[QTreeMap]:
        """
        Solve the associated differential equiationfor at
        multiple time points.

        Args:
            initial_value (QTreeMap): The initial value to
                be transformed.
            ts (np.ndarray): 1D array of time points at
                which to evaluate the transformation.

        Returns:
            list[QTreeMap]: A list of transformed QTreeMaps
                corresponding to each time in `ts`.
        """

        factors_mat = np.exp(-ts[:,None] * self.eigenvalues)
        
        decomposition = initial_value.to_decomposition()

        return [
            decomposition.transform(factors).to_map()
            for factors in factors_mat
        ]

    def get_linear_slices(
        self,
        initial_value: QTreeMap,
        T_start: float,
        T_end: float,
        count: int = 100
    ) -> tuple[np.ndarray, list[QTreeMap]]:
        """
        Apply exponential decay transformations to a
        QTreeMap at multiple time points.

        Args:
            initial_value (QTreeMap): The initial value to
                be transformed.
            ts (np.ndarray): 1D array of time points at
                which to evaluate the transformation.

        Returns:
            list[QTreeMap]: A list of transformed QTreeMaps
                corresponding to each time in `ts`.
        """
        
        ts = T_start + \
            (T_end - T_start) * np.arange(0, count)/count
        
        return ts, self.get_slices(initial_value, ts)

    def get_time_series(
        self,
        initial_value: QTreeMap,
        chunk: int = 100,
        offset: int = 0,
        delta_t: float | None = None,
        total_t: float | None = None,
        ) -> Generator[QTreeMap]:
        """
        Lazily generate an infinite time series of QTreeMap
        instances by repeatedly applying exponential decay
        transformations to the given initial value.

        Time steps are processed in chunks of size `chunk`.
        Each call to `next()` yields the next transformed
        QTreeMap in the sequence. The generator guarantees
        that `next()` always produces a result.

        Args:
            initial_value (QTreeMap): The initial value to
                be transformed over time.
            chunk (int, optional): Number of time steps to
                compute per iteration. Default is 100.
            offset (int, optional): Initial offset applied
                to the time steps. Useful for resuming from
                a previous state. Default is 0.
            delta_t (float, optional): Fixed time step size.
                Must be specified if `total_t` is not
                provided.
            total_t (float, optional): Total time span for
                one chunk. Used to compute `delta_t` as
                `total_t / chunk` if `delta_t` is not given.

        Returns:
            Generator[QTreeMap]: A generator yielding
                QTreeMap instances transformed at
                increasing time steps.
        
        Raises:
            ValueError: If both `delta_t` and `total_t` are None.
        """
        
        if delta_t is not None:
            delta_t = delta_t

        elif total_t is not None:
            delta_t = total_t/chunk

        else:
            raise ValueError("delta_t and total_t cannot be both None")

        # default: offset = 0
        steps = np.arange(0,chunk)

        decomposition = initial_value.to_decomposition()

        while True:
            ts = delta_t * (offset + steps)
            factors_mat = np.exp(-ts[:,None] * self.eigenvalues)

            for c in factors_mat:
                yield decomposition.transform(c).to_map()
            
            offset += chunk

def pow_kernel_operator(qtree: QTree, alpha: float) -> QTreeOperator:
    """
    Construct a QTreeOperator with a power-law kernel of
    the form |x|^{-alpha}.

    This kernel emphasizes long-range interactions and is
    singular at zero.

    Parameters:
        qtree (QTree): The QTree structure on which the
            operator acts.
        alpha (float): Positive exponent controlling the
            decay rate. Must be > 0.

    Returns:
        QTreeOperator: An operator on the given QTree using
            the power-law kernel.

    Raises:
        AssertionError: If `alpha` is not strictly positive.
    """
    assert alpha > 0 

    def kernel(x):
        return np.abs(x) ** (-alpha)
    
    return QTreeOperator(qtree, kernel)


def exp_kernel_operator(qtree: QTree, sigma: float) -> QTreeOperator:
    """
    Construct a QTreeOperator with a Gaussian-like
    exponential kernel of the form exp(-x² / σ²).

    This kernel is smooth, symmetric, and decays rapidly
    with distance, favoring local interactions.

    Parameters:
        qtree (QTree): The QTree structure on which the
            operator acts.
        sigma (float): Positive scale parameter controlling
            the width of the kernel. Must be > 0.

    Returns:
        QTreeOperator: An operator on the given QTree using
            the exponential kernel.

    Raises:
        AssertionError: If `sigma` is not strictly positive.
    """
    assert sigma > 0

    def kernel(x):
        return 1/np.sqrt(np.pi * sigma**2) * np.exp(- x**2 / sigma**2)

    return QTreeOperator(qtree, kernel)
