import math
import numpy as np
import matplotlib as mpl

from dataclasses import dataclass

from matplotlib.figure import Figure
from matplotlib.backends.backend_nbagg import FigureCanvasNbAgg as FigureCanvas

@dataclass
class PlotParameters:
    columns         = 2
    subplot_width   = 3
    subplot_height  = 2
    
    margin_x        = .1
    margin_y        = .1

    margin_left     = None
    margin_right    = None
    margin_top      = None
    margin_bottom   = None

    wspace          = .3
    hspace          = .4

    title           = None
    title_size      = 16

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
    def __init__(self,
        data: list[np.ndarray],
        params: None | PlotParameters = None
    ):
        self.data = data

        if params is None:
            self.params = PlotParameters()

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
