# from PySide6.QtCore import QObject, Signal
from PyQt5.QtCore import QObject, pyqtSignal as Signal

class SelectionBus(QObject):
    """Broadcasts the current logical selection across widgets."""

    edgesChanged = Signal(list)   # list[str]  – selected hyper-edge names
    imagesChanged = Signal(list)  # list[int]  – selected image indices

    def set_edges(self, names: list[str]):
        self.edgesChanged.emit(names)

    def set_images(self, idxs: list[int]):
        self.imagesChanged.emit(idxs)