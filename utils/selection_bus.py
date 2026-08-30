from PyQt5.QtCore import QObject, pyqtSignal as Signal

class SelectionBus(QObject):
    """Send selected edge IDs and image indices to all views."""

    edgesChanged = Signal(list)   # selected edge IDs
    imagesChanged = Signal(list)  # selected image indices

    def set_edges(self, edge_ids: list[str]):
        self.edgesChanged.emit(edge_ids)

    def set_images(self, idxs: list[int]):
        self.imagesChanged.emit(idxs)
