from PyQt5.QtGui import (
    QImageReader,
    QPixmap,
    QImage,
    QPixmapCache,
    QColor,
)
from PyQt5.QtCore import QBuffer, QIODevice, QSize


def qimage_from_file(path: str) -> QImage:
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    return reader.read()


def qimage_from_data(data: bytes) -> QImage:
    buf = QBuffer()
    buf.setData(data)
    buf.open(QIODevice.ReadOnly)
    reader = QImageReader(buf)
    reader.setAutoTransform(True)
    img = reader.read()
    buf.close()
    return img


def pixmap_from_file(path: str) -> QPixmap:
    img = qimage_from_file(path)
    return QPixmap.fromImage(img)



QPixmapCache.setCacheLimit(512 * 1024)  # 512 MB

def load_thumbnail(path: str, w: int, h: int) -> QPixmap:
    key = f"{path}|{w}x{h}"
    pix = QPixmapCache.find(key)
    if pix is not None and not pix.isNull():
        return pix

    reader = QImageReader(path)
    reader.setAutoTransform(True)
    reader.setScaledSize(QSize(w, h))
    img = reader.read()
    if img.isNull():
        pix = QPixmap(w, h)
        pix.fill(QColor(220, 220, 220))
    else:
        pix = QPixmap.fromImage(img)

    QPixmapCache.insert(key, pix)
    return pix