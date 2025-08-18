from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
)
from PyQt5.QtCore import Qt


def _gather_stats(session):
    total_images = len(session.im_list)
    total_edges = len(session.hyperedges)

    origin_data = {}
    for edge, origin in session.edge_origins.items():
        data = origin_data.setdefault(origin, {"edges": set(), "imgs": set()})
        data["edges"].add(edge)
        data["imgs"].update(session.hyperedges.get(edge, set()))

    origin_stats = {
        o: (len(d["edges"]), len(d["imgs"])) for o, d in origin_data.items()
    }

    modified_imgs = set()
    for edge, meta in session.status_map.items():
        if "modified" in str(meta.get("status", "")).lower():
            modified_imgs.update(session.hyperedges.get(edge, set()))

    return {
        "total_images": total_images,
        "total_edges": total_edges,
        "origin_stats": origin_stats,
        "modified_images": len(modified_imgs),
    }


class SessionStatsDialog(QDialog):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session Statistics")
        layout = QVBoxLayout(self)

        stats = _gather_stats(session)
        info = QLabel(
            f"Total images: {stats['total_images']}\n"
            f"Total hyperedges: {stats['total_edges']}\n"
            f"Images in 'modified' hyperedges: {stats['modified_images']}"
        )
        info.setAlignment(Qt.AlignLeft)
        layout.addWidget(info)
        
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels([
            "Origin",
            "Hyperedges",
            "Unique Images",
        ])
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)

        for row, (orig, vals) in enumerate(stats["origin_stats"].items()):
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(str(orig)))
            table.setItem(row, 1, QTableWidgetItem(str(vals[0])))
            table.setItem(row, 2, QTableWidgetItem(str(vals[1])))

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


def show_session_stats(session, parent=None):
    dlg = SessionStatsDialog(session, parent=parent)
    dlg.exec_()