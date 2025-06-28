import numpy.typing as npt
from typing import Callable
from typing import Sequence

type Number = float | complex
type RealFunction = Callable[[float], Number]
type DiscreteFunction = Sequence[Number]
type DiscreteFunctionFunnel = Sequence[DiscreteFunction]
type Integrator = Callable[[RealFunction, float, float], Number]


class Interval:
    """
    Closed interval usefull for integration.
    """

    def __init__(self, a: float, b: float):
        if not a < b:
            raise ValueError(
                f"The left value {a=} must be smaller than the right value {b=}."
            )

        self.a = a
        self.b = b

    def __repr__(self):
        return f"({self.a}, {self.b})"

    def __contains__(self, x: float) -> bool:
        return self.a <= x <= self.b

    def __iter__(self):
        yield self.a
        yield self.b
