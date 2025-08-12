from PyQt5.QtGui import QImageReader, QPixmap, QImage
from PyQt5.QtCore import QBuffer, QIODevice


def qimage_from_file(path: str) -> QImage:
    """Read an image from *path* applying EXIF orientation."""
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    return reader.read()


def qimage_from_data(data: bytes) -> QImage:
    """Read an image from raw *data* applying EXIF orientation."""
    buf = QBuffer()
    buf.setData(data)
    buf.open(QIODevice.ReadOnly)
    reader = QImageReader(buf)
    reader.setAutoTransform(True)
    img = reader.read()
    buf.close()
    return img


def pixmap_from_file(path: str) -> QPixmap:
    """Load a QPixmap from *path* respecting EXIF orientation."""
    img = qimage_from_file(path)
    return QPixmap.fromImage(img)