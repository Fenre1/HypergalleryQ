from PySide6.QtCore import QObject, Signal

class SelectionBus(QObject):
    """
    Central hub that broadcasts the *current* logical selection and nothing else.
    The payload can be whatever IDs you use (edge names, image indices, …).
    """
    edgesChanged  = Signal(list)   # list[str]    – selected hyper-edge names
    imagesChanged = Signal(list)   # list[int]    – selected image indices

    def set_edges(self, names: list[str]):
        self.edgesChanged.emit(names)

    def set_images(self, idxs: list[int]):
        self.imagesChanged.emit(idxs)