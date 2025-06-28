"""
Implements basic sanity checks.
"""

import numpy as np

from dataclasses import dataclass

from . import QTree
from . import QTreeMap
from . import QTreeDecomposition

@dataclass
class Result:
    success: bool
    text: str
    auxillary: dict | None = None

    def __bool__(self):
        return self.success

    def __repr__(self):
        out = ("[Success]" if self.success else "[Failure]")
        out += " "
        out += self.text
        return out


def means_zero(decomposition: QTreeDecomposition, atol=1e-08) -> Result:
    means = [np.mean(layer) for layer in decomposition.contributions]
    closeness = np.isclose(means, 0, atol=atol)

    if np.all(closeness):
        return Result(True,
            f"The means are within {atol:.0e}."
        )

    return Result(False,
        f"The means are of {means=}."
    )
        

