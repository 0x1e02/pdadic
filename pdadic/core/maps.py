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
    def __init__(self, qtree: QTree, values: DiscreteFunctionFunnel):
        assert qtree.height == len(values) - 1, ""
        assert all( zip(qtree.Q, ))
        
        self.domain = qtree
        self.values = values

    def as_arrays(self):
        return [
            np.array(arr) for arr in self.values
        ]

    @classmethod
    def from_tuple(cls, qtree: QTree, func: DiscreteFunction) -> QTreeMap:
        return cls(qtree, qtree.funnel(func))

    @classmethod
    def from_function(
        cls,
        qtree: QTree,
        func: RealFunction,
        **kwargs
    ) -> QTreeMap:

        return cls.from_tuple(qtree, qtree.discretize(func, **kwargs))
    

    def to_decomposition(self) -> QTreeDecomposition:
        mean, *tail = self.values
        contributions = []

        for fun, layer in zip(tail, self.domain.layers):
            coeff = layer.get_coefficients(fun)
            g = layer.get_contribution(coeff)
            
            contributions.append(g)

        return QTreeDecomposition(self.domain, mean[0], contributions)
    

class QTreeDecomposition:
    def __init__(self,
        domain: QTree,
        mean: Number,
        contributions: list[DiscreteFunction]
    ):
        self.domain = domain
        self.mean = mean
        self.contributions = contributions


    def to_map(self) -> QTreeMap:
        _f = np.array([self.mean], dtype=np.complex128)

        buff = [list(_f)]
        
        for q, g in zip(self.domain.Q, self.contributions):
            _f = _f.repeat(q)
            _f += g
            buff.append(list(_f))

        return QTreeMap(self.domain, buff)
