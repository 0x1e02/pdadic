
import numpy as np

from . import QTree
from . import QTreeKernelOperator

from .core.types import Number

def gaussian(k):
    def func(x):
        return

    return func

def sigmoid(k):
    def func(x):
        return
    
    return func


class ConstantKernelOperator(QTreeKernelOperator):
    def __init__(self, qtree: QTree, const: float) -> None:
        self.const = const
        super().__init__(qtree)

    def _kernel(self, x: float) -> Number:
        return self.const


class PowerKernelOperator(QTreeKernelOperator):
    def __init__(self, qtree: QTree, alpha: float) -> None:
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

        self.alpha = alpha
        super().__init__(qtree)

    def _kernel(self, x: float) -> Number:
        return np.abs(x)**(-self.alpha)


class GaussianKernelOperator(QTreeKernelOperator):
    def __init__(self, qtree: QTree, sigma: float) -> None:
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

        self.sigma = sigma
        self._normalization_factor = 1 / np.sqrt(np.pi * self.sigma**2)
        super().__init__(qtree)

    def _kernel(self, x: float) -> Number:
        return self._normalization_factor * np.exp(np.abs(x)**2 / self.sigma**2)
