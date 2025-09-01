from PyQt5.QtCore import QObject, pyqtSignal as Signal

class SelectionBus(QObject):
    """Broadcasts the current selection of hyperedges and images across views."""

    edgesChanged = Signal(list)   # selected hyperedge names
    imagesChanged = Signal(list)  # selected image indices

    def set_edges(self, names: list[str]):
        self.edgesChanged.emit(names)

    def set_images(self, idxs: list[int]):
        self.imagesChanged.emit(idxs)