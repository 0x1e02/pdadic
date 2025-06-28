import io

from IPython.display import Image, SVG
from IPython.display import DisplayObject
from IPython.display import display

from matplotlib.figure import Figure


class Exporter:
    def __init__(self,
        fig: Figure,
        _format: str,
        _display: DisplayObject
    ) -> None:

        self.fig = fig
        self.format = _format
        self.display_strategy = _display

    def show(self):
        buffer = io.BytesIO()
        self.save(buffer)
        buffer.seek(0)
        return display(
            self.display_strategy(data=buffer.getvalue())
        )

    def save(self, buffer):
        self.fig.savefig(buffer, format=self.format)

        return buffer


class PNGExporter(Exporter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
            _format = "png",
            _display = Image 
        )


class SVGExporter(Exporter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
            _format = "svg",
            _display = SVG 
        )