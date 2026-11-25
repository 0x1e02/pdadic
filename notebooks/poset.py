import numpy as np
from scipy.linalg import expm

from collections import defaultdict, deque
from typing import Any


class HasseDiagram:
    """
    Weighted poset stored as its Hasse diagram: only cover relations x ⋖ y.
    """

    def __init__(self):
        self.up = defaultdict(set)    # x -> elements that directly cover x
        self.down = defaultdict(set)  # y -> elements directly covered by y
        self.mu = dict()
        self.elements = set()

    @staticmethod
    def from_elements(
        elements: list[Any],
        weights: list[Any] | None = None
    ) -> HasseDiagram:
        hd = HasseDiagram()

        if weights is None:
            for e in elements:
                hd.add(e)
        else:
            assert len(elements) == len(weights)
            for e, w in zip(elements, weights):
                hd.add(e, w)

        return hd

    @staticmethod
    def from_relations(relations = ()) -> HasseDiagram:
        hd = HasseDiagram()
        for x,y in relations:
            hd.add_relation(x,y)

        return hd

    def add(self, x, weight: float = 1.0):
        self.elements.add(x)
        self.up[x]
        self.down[x]

        self.mu[x] = weight



    def _reach(self, start, graph):
        """
        All elements reachable from start (excluding start) via graph.
        """
        seen, stack = set(), [start]
        while stack:
            for z in graph.get(stack.pop(), ()):
                if z not in seen:
                    seen.add(z)
                    stack.append(z)
        return seen

    def upset(self, x):
        return self._reach(x, self.up) | {x}

    def downset(self, x):
        return self._reach(x, self.down) | {x}

    def leq(self, x, y):
        """x <= y  (reachability, O(n + m))."""
        return x == y or y in self._reach(x, self.up)

    def add_relation(self, x, y):
        """Declare x < y and keep the diagram a transitive reduction."""
        self.add(x)
        self.add(y)
        if x == y:
            return
        if self.leq(y, x):
            raise ValueError(f"{x!r} < {y!r} would create a cycle")
        if self.leq(x, y):
            return  # already implied
        # Edges a⋖b with a <= x and y <= b become redundant.
        above = self.upset(y)
        for a in self.downset(x):
            for b in list(self.up[a] & above):
                self.up[a].remove(b)
                self.down[b].remove(a)
        self.up[x].add(y)
        self.down[y].add(x)

    def minimal(self):
        return {e for e, d in self.down.items() if not d}

    def maximal(self):
        return {e for e, u in self.up.items() if not u}

    def covers(self):
        return [(a, b) for a, bs in self.up.items() for b in bs]

    def linear_extension(self):
        """A topological order (Kahn's algorithm)."""
        indeg = {e: len(d) for e, d in self.down.items()}
        queue = deque(e for e, n in indeg.items() if n == 0)
        order = []
        while queue:
            e = queue.popleft()
            order.append(e)
            for b in self.up[e]:
                indeg[b] -= 1
                if indeg[b] == 0:
                    queue.append(b)
        return order

    def upmu(self, x):
        return sum(self.mu[t] for t in self.upset(x))
    
    def downmu(self, x):
        return sum(self.mu[t] for t in self.downset(x))

    def upkernel(self, x, y):
        if y in self.upset(x):
            return  1/self.upmu(x)
        return 0

    def downkernel(self, x, y):
        if y in self.downset(x):
            return  1/self.downmu(x)
        return 0

    def advection_matrix(self, kernel: str = "up"):
        """
        Return the matrix M of the advection operator K^bullet
        in the basis of point evaluations, together with the
        ordered list of elements used for the matrix indices.

        Parameters
        ----------
        kernel : {"up", "down"}
            "up"   -> K^U with kernel k^U(x,y) = 1_{U(x)}(y)/mu(U(x))
            "down" -> K^V with kernel k^V(x,y) = 1_{V(x)}(y)/mu(V(x))

        Returns
        -------
        M : np.ndarray of shape (n, n)
        elements : list
            The elements in the order used for rows/columns of M.
        """
        if kernel == "up":
            k = self.upkernel
        elif kernel == "down":
            k = self.downkernel
        else:
            raise ValueError("kernel must be 'up' or 'down'")

        elements = self.linear_extension()
        n = len(elements)
        idx = {e: i for i, e in enumerate(elements)}

        # gamma(x) = sum_z k(z,x) * mu(z)
        gamma = {x: sum(k(z, x) * self.mu[z] for z in elements)
                 for x in elements}

        M = np.zeros((n, n), dtype=float)
        for i, x in enumerate(elements):
            for j, y in enumerate(elements):
                M[i, j] = k(x, y) * self.mu[y]
            M[i, i] -= gamma[x]

        return M, elements

    def eigen_advection(self, kernel: str = "up"):
        """
        Compute eigenvalues and eigenvectors of the advection operator
        K^bullet on the finite T0-space.

        Parameters
        ----------
        kernel : {"up", "down"}
            "up"   -> K^U
            "down" -> K^V

        Returns
        -------
        eigenvalues : np.ndarray of shape (n,)
        eigenvectors : np.ndarray of shape (n, n)
            The columns are the eigenvectors.
        elements : list
            The elements in the order corresponding to the rows/columns.
        """
        M, elements = self.advection_matrix(kernel)
        eigenvalues, eigenvectors = np.linalg.eig(M)
        return eigenvalues, eigenvectors, elements

    def eigen_up(self):
        """Convenience wrapper for K^U."""
        return self.eigen_advection("up")

    def eigen_down(self):
        """Convenience wrapper for K^V."""
        return self.eigen_advection("down")


    def cauchy_solution(self, t, f0, kernel: str = "up", sign: int = 1):
        """
        Solve the Cauchy problem
            ∂_t f = sign * K^bullet f,   f(0) = f0.

        Parameters
        ----------
        t : float
            Time.
        f0 : dict or array-like
            Initial value. If dict, maps elements to values.
            If array-like, must be in the order of linear_extension().
        kernel : {"up", "down"}
            "up" -> K^U, "down" -> K^V.
        sign : {1, -1}
            1 for ∂_t f = K^bullet f,
           -1 for ∂_t f = -K^bullet f (as in Theorem 3 with L^bullet).

        Returns
        -------
        dict
            Maps each element to its value at time t.
        """
        M, elements = self.advection_matrix(kernel)
        n = len(elements)

        if isinstance(f0, dict):
            f0_vec = np.array([f0[e] for e in elements], dtype=float)
        else:
            f0_vec = np.asarray(f0, dtype=float)
            assert len(f0_vec) == n

        P = expm(sign * t * M)
        f_t = P @ f0_vec
        return {e: f_t[i] for i, e in enumerate(elements)}

    def heat_kernel_matrix(self, t, kernel: str = "up", sign: int = 1):
        """
        Return the transition matrix P(t) = exp(sign * t * M)
        and the ordered list of elements.
        The heat kernel is p(t, x_i, x_j) = P(t)_{ij} / mu(x_j).
        """
        M, elements = self.advection_matrix(kernel)
        P = expm(sign * t * M)
        return P, elements

    def heat_kernel(self, t, x, y, kernel: str = "up", sign: int = 1):
        """
        Heat kernel p(t, x, y) = (exp(sign * t * M))_{x,y} / mu(y).
        """
        P, elements = self.heat_kernel_matrix(t, kernel, sign)
        i = elements.index(x)
        j = elements.index(y)
        return P[i, j] / self.mu[y]
