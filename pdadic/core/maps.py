"""
Implements the two types QTreeMap and QTreeDecomposition.

The former represents a map on the leaves of a short q-adic
tree before component analysis. The latter represents a
similar map after compenent analysis.
"""

from __future__ import annotations

import numpy as np

from typing import Callable
from typing import Sequence

from .qtree import QTree

from .types import Number
from .types import DiscreteFunction
from .types import DiscreteFunctionFunnel
from .types import RealFunction
from .types import Integrator

class QTreeMap:
    """
    Represents a map in the reals or complex with domain
    as the set of leaves of a short Q-adic tree.

    The internal data is a function on a successive refinement.
    """

    def __init__(self, qtree: QTree, values: DiscreteFunctionFunnel) -> None:
        """
        Factors common functionality for instanciation.
        """
        assert qtree.height == len(values) - 1
        assert all( zip(qtree.Q, ))
        
        self.domain = qtree
        self.values = values
    
    def __iter__(self):
        return iter(self.as_arrays()[-1])

    def __len__(self):
        return self.domain.N

    @classmethod
    def from_tuple(cls, qtree: QTree, func: DiscreteFunction) -> QTreeMap:
        """
        Instanciates a QTreeMap by passing a function on a
        set of leaves and combine these by the data of the
        short Q-adic tree.
        """

        return cls(qtree, qtree.funnel(func))

    @classmethod
    def from_function(
        cls,
        qtree: QTree,
        func: RealFunction,
        **kwargs
    ) -> QTreeMap:
        """
        Instanciates a QTreeMap by passing a real python
        function and discretizing it with the logic provided
        by the short Q-adic tree.
        """

        return cls.from_tuple(qtree, qtree.discretize(func, **kwargs))
    
    def as_arrays(self) -> list[np.ndarray]:
        """
        Exports the internal data as a list of numpy arrays.
        """

        return [
            np.array(arr) for arr in self.values
        ]

    def to_decomposition(self) -> QTreeDecomposition:
        """
        Extracts the wavelets contributions for each layer
        of the Q-adic tree and yields a QTreeDecomposition.
        """

        mean, *tail = self.values
        contributions = []

        for fun, layer in zip(tail, self.domain.layers):
            contributions.append(layer.get_component(fun))

        return QTreeDecomposition(self.domain, mean[0], contributions)
    

class QTreeDecomposition:
    """
    Represents a map in the reals or complex with domain
    as the set of leaves of a short Q-adic tree.

    The internal data is a mean value and a list of
    functions each defined on a layer of a Q-adic tree.
    """
    def __init__(self,
        domain: QTree,
        mean: Number,
        contributions: list[DiscreteFunction]
    ) -> None:
        """
        Instanciates a QTreeDecomposition by passing the
        mean and contributions from each layer.
        """

        self.domain = domain
        self.mean = mean
        self.contributions = contributions


    def as_arrays(self) -> list[np.ndarray]:
        """
        Exports the internal data as a list of numpy arrays.
        """

        layers = list(self.contributions)
        layers.insert(0, [self.mean])

        return [
            np.array(arr) for arr in layers
        ]

    def transform(self, factors) -> QTreeDecomposition:
        """
        Scale the decomposition's contributions layer-wise
        by given factors.

        Args:
            factors (array-like): Multiplicative factors
                applied to each component of the decomposition
                (excluding the mean).

        Returns:
            QTreeDecomposition: A new decomposition with
                scaled contributions.
        """
        return self.__class__(
            domain = self.domain,
            mean = self.mean,
            contributions = [
                c * g
                for c,g in zip(factors, self.as_arrays()[1:])
            ]
        )

    def to_map(self) -> QTreeMap:
        """
        Combines the contributions for each layer
        and the mean to yields a QTreeMap.
        """
        _f = np.array([self.mean], dtype=np.complex128)

        buff = [list(_f)]
        
        for q, g in zip(self.domain.Q, self.contributions):
            _f = _f.repeat(q)
            _f += g
            buff.append(list(_f))

        return QTreeMap(self.domain, buff)
