import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scatter Plot Example")
        plot_widget = pg.PlotWidget()
        self.setCentralWidget(plot_widget)

        # two red dots at (1, 2) and (3, 4)
        scatter = pg.ScatterPlotItem(
            x=[1, 3],
            y=[2, 4],
            pen=pg.mkPen(None),        # no outline
            brush=pg.mkBrush('r'),     # red fill
            size=10                    # marker size
        )
        plot_widget.addItem(scatter)
        print('self.scatter',scatter.getData())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
