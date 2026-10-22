"""
Defines a common interfaces for discussing computations
implemented in core.
"""

import math
import numpy as np
import matplotlib as mpl

from dataclasses import dataclass

from matplotlib.figure import Figure
from matplotlib.backends.backend_nbagg import FigureCanvasNbAgg as FigureCanvas

from .. import QTreeMap
from .. import QTreeKernelOperator
from ..profiling import profile_equilibrium


@dataclass
class PlotParameters:
    columns: int            = 2
    subplot_width: float    = 3
    subplot_height: float   = 2
    
    margin_x: float         = .1
    margin_y: float         = .1

    margin_left: float | None   = None
    margin_right: float | None  = None
    margin_top: float | None    = None
    margin_bottom: float | None = None

    wspace: float           = .3
    hspace: float           = .4

    title: str | None       = None
    title_size: int         = 16

    def __post_init__(self):
        if self.margin_left is None:
            self.margin_left = self.margin_x

        if self.margin_right is None:
            self.margin_right = 1 - self.margin_x

        if self.margin_top is None:
            self.margin_top = 1 - self.margin_y

        if self.margin_bottom is None:
            self.margin_bottom = self.margin_y


class AbstractDiscussion:

    DEFAULT_PARAMS = PlotParameters()

    def __init__(self,
        data: list[np.ndarray],
        params: None | PlotParameters = None
    ):
        self.data = data
        self.params = params

        if params is None:
            self.params = self.DEFAULT_PARAMS

    def single(self, title=None):
        K = len(self.data)
        
        cols = min(self.params.columns, K)
        rows = math.ceil(K / self.params.columns)

        fig = Figure(figsize=(
            self.params.subplot_width * cols,
            self.params.subplot_height * rows
        ))

        fig.subplots_adjust(
            left    = self.params.margin_left,
            right   = self.params.margin_right,
            bottom  = self.params.margin_bottom,
            top     = self.params.margin_top,
            wspace  = self.params.wspace,
            hspace  = self.params.hspace,
        )

        canvas = FigureCanvas(fig)

        for i, frame in enumerate(self.data):
            ax = fig.add_subplot(rows, cols, i+1)
            self._plot(ax, frame)

        if title is not None:
            fig.suptitle(title)
        
        return fig

    def _plot(self, ax, frame):
        return NotImplemented


class Discussion(AbstractDiscussion):
    def _plot(self, ax, frame):
        N = len(frame)
        xs = np.arange(N)

        ax.set_xticks([0, N-1])
        ax.set_xticklabels(['0', '1'])

        ax.bar(np.arange(N), frame)


class ErrorDiscussion(AbstractDiscussion):
    def __init__(self,
        estimate: list[np.ndarray],
        reference: list[np.ndarray],
        *args,
        **kwargs
    ) -> None:

        differences = [ e-r for e, r in zip(estimate, reference) ]
        super().__init__(differences, *args, **kwargs)
        
    def _plot(self, ax, frame):
        N = len(frame)
        xs = np.arange(N)
        
        ax.set_xticks([0, N-1])
        ax.set_xticklabels(['0', '1'])

        ax.bar(xs, np.real(frame), rasterized=True)
        ax.bar(xs, np.imag(frame), alpha=.5, rasterized=True)

class EvolutionDiscussion(Discussion):
    
    DEFAULT_PARAMS = PlotParameters(
        subplot_height  = 4,
        subplot_width   = 6
    )

    def __init__(self,
        qoperator: QTreeKernelOperator,
        initial_value: QTreeMap,
        dilation: float | None = None,
        **kwargs
    ):
        self.T = profile_equilibrium(qoperator, initial_value)
        print(f"{self.T=}")

        if dilation is None:
            _, slices = qoperator.get_linear_slices(
                initial_value,
                T_start=0,
                T_end=self.T,
                count=len(initial_value)
            )
        else:
            slices = qoperator.get_slices(
                initial_value,
                ts = self.T * QTreeKernelOperator.get_geometric_ts(len(initial_value), dilation)
            )
            

        frame = np.array([list(s) for s in slices])
        
            
        super().__init__([frame], **kwargs)

    def _plot(self, ax, frame):
        N = len(frame)
        ax.matshow(np.real(frame))

        ax.set_xlabel("Space")
        ax.set_xticks([0, N-1])
        ax.set_xticklabels(["0", "1"])

        ax.set_ylabel("Time")
        ax.yaxis.set_ticks_position("right")
        ax.set_yticks([0, N-1])
        if np.log10(self.T) < 2:
            ax.set_yticklabels(["0", f"{self.T:.2f}"])
        else:
            ax.set_yticklabels(["0", f"{self.T:.1E}"])
        