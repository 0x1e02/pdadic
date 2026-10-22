import math
import numpy as np

from . import QTree
from . import QTreeKernelOperator
from . import NormalizedQTreeOperator

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
                decay rate. Must be < 1.

        Returns:
            QTreeOperator: An operator on the given QTree using
                the power-law kernel.

        Raises:
            AssertionError: If `alpha` is not strictly positive.
        """
        assert alpha < 1

        self.alpha = alpha
        super().__init__(qtree)

    def _kernel(self, x: float) -> Number:
        return np.abs(x)**(-self.alpha)

    def __str__(self):
        return f"PowerKernel(alpha: {self.alpha})"


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
        self._normalization_factor = 1 / np.sqrt(2* np.pi * self.sigma**2)
        super().__init__(qtree)

    def _kernel(self, x: float) -> Number:
        return self._normalization_factor * np.exp(-(np.abs(x) / (np.sqrt(2) * self.sigma))**2)
        # return np.exp(-(np.abs(x) / self.sigma)**2)

    def __str__(self):
        return f"GaussianKernel(sigma: {self.sigma})"

class TruncatedGaussianKernelOperator(QTreeKernelOperator):
    def __init__(self,
            qtree: QTree,
            sigma: float,
            mu: float = 0,
            a:float = 0,
            b:float = 1
        ) -> None:
        
        assert sigma > 0
        assert a < b

        self.sigma=sigma
        self.mu=mu
        self.a = a
        self.b = b

        self._normalization_factor = \
            1 / (sigma * (self._cdf((b - mu)/sigma) - self._cdf((a - mu)/sigma) ))
        
        super().__init__(qtree)

    def _gauss(self, x:float) -> float:
        return 1 / np.sqrt(2 * np.pi) * np.exp(-1/2 * x**2)

    def _cdf(self, x: float) -> float:
        return 1/2 * (1 + math.erf(x/np.sqrt(2)))
    
    def _kernel(self, x: float) -> Number:
        return self._normalization_factor * self._gauss((x-self.mu)/self.sigma)