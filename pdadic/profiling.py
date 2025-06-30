"""
Implements basic profiling.
"""

import time
import tracemalloc

from contextlib import contextmanager
from dataclasses import dataclass

from . import QTree
from . import QTreeMap
from . import QTreeOperator

@dataclass
class TimeResult:
    eta: float = float('nan')


@contextmanager
def timer():
    """
    Syntax sugar for runtime profiling.
    """
    result = TimeResult()

    start = time.perf_counter()
    yield result
    eta = time.perf_counter() - start
    
    result.eta = eta


class Timer:
    class Eta:
        def __init__(self, name):
            self.name = name
            self.value = float('nan')
        
        def __matmul__(self, other):
            if len(other) == 3:
                return self.defer(other[0], *other[1], **other[2])

            elif len(other) == 2:
                return self.defer(other[0], *other[1])

            elif len(other) == 1:
                return self.defer(other[0])

        def defer(self, func, *args, **kwargs):
            start = time.perf_counter()
            
            result = func(*args, **kwargs)

            end = time.perf_counter()
            self.value = end - start

            return result

        def __repr__(self):
            return f"{self.name}: {self.value:.2f} sec"

    def __init__(self) -> None:
        self.buffer: list[Timer.Eta] = []

    def __call__(self, name: str):
        eta = self.Eta(name)
        self.buffer.append(eta)

        return eta

    def __iter__(self):
        return iter(self.buffer)


def profile_qtree(qtree: QTree, func):
    """
    Profiles the execution of applying a function to a
    QTree and processing it through decomposition and
    recomposition.

    This function performs the following steps:
      1. Applies `func` to each node of the input `qtree`
        to produce a `QTreeMap`.
      2. Decomposes the resulting map into a structured
        representation.
      3. Recomposes the decomposition back into a map.
      4. Measures and prints memory usage using `tracemalloc`.
      5. Times each step using a `Timer` instance.
      6. Prints a summary of the input, runtime, and memory usage.

    Parameters:
        qtree (QTree): The input QTree to process.
        func (Callable): A function to apply to each node
            of the QTree.

    Returns:
        tuple: A tuple `(qmap, qdecomp, qrecomp)` containing:
            - `qmap` (QTreeMap): The map produced from
                applying `func` to `qtree`.
            - `qdecomp`: The decomposition of the map.
            - `qrecomp` (QTreeMap): The recomposed map from
                the decomposition.
    """

    print(f"[Input]")
    print(qtree)
    print()

    _timer = Timer()

    tracemalloc.start()

    qmap    = _timer("Preprocessing").defer(QTreeMap.from_function, qtree, func)
    qdecomp = _timer("Decomposition").defer(qmap.to_decomposition)
    qrecomp = _timer("Recomposition").defer(qdecomp.to_map,)
    
    current, peak = tracemalloc.get_traced_memory()

    print("[Memory]")
    print(f"Memory (current): {current/1024**2:.2f} MB")
    print(f"Memory (peak): {peak/1024**2:.2f} MB")
    print()

    tracemalloc.stop()

    print("[Runtime]")
    for eta in _timer:
        print(eta)
    print()

    return qmap, qdecomp, qrecomp

def profile_equilibrium(
    qoperator: QTreeOperator,
    initial_value: QTreeMap,
    eps:float = 1e-3,
    T0:float = 1,
    max_iter_coarse = 100,
    max_iter_tuning = 100
    ) -> float:
    """
    Estimate the smallest time when the operator transforms
    the initial value into approximate equilibrium within eps.

    Equilibrium means the image is confined to a range of size
    at most eps (max - min <= eps).

    The search is two-stage:

    1. Coarse search:
       Starting at T0, checks times T = T0 * 2^k until a slice
       reaches equilibrium.

    2. Tuning:
       Generates finer slices between 0 and found time, recursively
       searching for the earliest equilibrium time.

    Args:
        qoperator (QTreeOperator): Operator on the QTreeMap.
        initial_value (QTreeMap): Initial map to transform.
        eps (float): Tolerance for equilibrium (default 1e-3).
        T0 (float): Start time for coarse search (default 1).
        max_iter_coarse (int): Max iterations for coarse search.
        max_iter_tuning (int): Max iterations for tuning phase.

    Returns:
        float: Earliest time at which equilibrium is reached.

    Raises:
        RuntimeError: If equilibrium not found within iteration limits.
    """

    def is_equilibrium(f):
        return max(f) - min(f) <= eps

    def tuning(T, iteration=0):
        if iteration >= max_iter_tuning:
            raise RuntimeError(
                f"Reached {iteration=} during tuning."
            )

        time_slices = qoperator.get_linear_slices(
            initial_value,
            T_start = 0,
            T_end = T,
            count=1000
        )

        for t, f in zip(*time_slices):
            if is_equilibrium(f):
                return tuning(t, iteration+1)
        
        else:
            return T

    
    # Check slice at T=1
    _slice = qoperator.get_slice(1, initial_value)
    if is_equilibrium(_slice):
        return tuning(T=T0)

    # Search for equilibrium T=2^k
    for k in range(1, max_iter_coarse+1):
        _T = T0 * 2**k
        _slice = qoperator.get_slice(_T, initial_value)
        
        if is_equilibrium(_slice):
            return tuning(T=_T)

    else:
        raise RuntimeError(f"Could not find equilibrium until {_T=}")
    