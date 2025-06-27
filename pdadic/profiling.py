from contextlib import contextmanager
import time
from .maps import QTreeMap

import tracemalloc
from dataclasses import dataclass

@dataclass
class TimeResult:
    eta: float = float('nan')





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


def profile_qtree(qtree, func):
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

def discuss_error(qmap, qrecomp):
    pass

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

with timer() as res:
    ...

