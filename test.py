
import pyqtgraph as pg, platform
print(pg.__version__, pg.Qt.QT_LIB, pg.QtCore.qVersion())

import os, sys
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"   # eliminates silent fallback

from PySide6.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg

class Win(QMainWindow):
    def __init__(self):
        super().__init__()
        w = pg.PlotWidget()
        self.setCentralWidget(w)
        w.addItem(pg.ScatterPlotItem(x=[1,3], y=[2,4],
                                     pen=None, brush='r', size=10))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Win().show()
    sys.exit(app.exec())

# import sys, uuid, numpy as np
# import pandas as pd
# import numpy as np
# from utils.similarity import SIM_METRIC
# from pathlib import Path
# from PyQt5.QtWidgets import QApplication
# from PySide6.QtWidgets import QMainWindow
# # from PySide6.QtWidgets import (
# #     QTreeView, QMainWindow, QFileDialog, QVBoxLayout,
# #     QWidget, QLabel, QSlider, QMessageBox, QPushButton, QDockWidget,
# #     QStackedWidget
# # )
# # from PySide6.QtGui import QStandardItem, QStandardItemModel, QAction
# # from PySide6.QtCore import Qt, QSignalBlocker, QObject, Signal


# # from utils.data_loader import (
# #     DATA_DIRECTORY, get_h5_files_in_directory, load_session_data
# # )
# # from utils.selection_bus import SelectionBus 
# # from utils.session_model import SessionModel
# # from utils.image_grid import ImageGridDock
# # from utils.hyperedge_matrix import HyperedgeMatrixDock










# # import sys
# #, QMainWindow
# import pyqtgraph as pg

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Scatter Plot Example")
#         plot_widget = pg.PlotWidget()
#         self.setCentralWidget(plot_widget)

#         # two red dots at (1, 2) and (3, 4)
#         scatter = pg.ScatterPlotItem(
#             x=[1, 3],
#             y=[2, 4],
#             pen=pg.mkPen(None),        # no outline
#             brush=pg.mkBrush('r'),     # red fill
#             size=10                    # marker size
#         )
#         plot_widget.addItem(scatter)
#         print('self.scatter',scatter.getData())

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = MainWindow()
#     win.show()
#     sys.exit(app.exec_())






# # import sys
# # from PyQt5.QtWidgets import QApplication, QMainWindow
# # import pyqtgraph as pg

# # class MainWindow(QMainWindow):
# #     def __init__(self):
# #         super().__init__()
# #         self.setWindowTitle("Scatter Plot Example")
# #         plot_widget = pg.PlotWidget()
# #         self.setCentralWidget(plot_widget)

# #         # two red dots at (1, 2) and (3, 4)
# #         scatter = pg.ScatterPlotItem(
# #             x=[1, 3],
# #             y=[2, 4],
# #             pen=pg.mkPen(None),        # no outline
# #             brush=pg.mkBrush('r'),     # red fill
# #             size=10                    # marker size
# #         )
# #         plot_widget.addItem(scatter)
# #         print('self.scatter',scatter.getData())

# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)
# #     win = MainWindow()
# #     win.show()
# #     sys.exit(app.exec_())
