import pyqtgraph as pg

class ScatterPlotWidget(pg.PlotWidget):
    """A self-contained widget that shows two red points at (1,2) and (3,4)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        scatter = pg.ScatterPlotItem(
            x=[1, 3],
            y=[2, 4],
            pen=None,               # no outline
            brush=pg.mkBrush('r'),  # red fill
            size=10
        )
        self.addItem(scatter)