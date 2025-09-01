import pandas as pd
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QPushButton,
)
from PyQt5.QtCore import Qt


def _summarize_column(series: pd.Series) -> tuple[int, int, str]:

    valid = [v for v in series.dropna() if str(v).strip() and str(v).lower() not in {"none", "nan"}]
    valid_count = len(valid)

    unique_count = len({str(v) for v in valid})

    numeric_values: list[float] | None = []
    for v in valid:
        try:
            numeric_values.append(float(v))
        except Exception:
            numeric_values = None
            break

    if numeric_values:
        range_str = f"{min(numeric_values)} - {max(numeric_values)}"
    elif numeric_values == []:
        # no valid numeric entries
        range_str = "n/a"
    else:
        range_str = "categorical"

    return valid_count, unique_count, range_str


class MetadataOverviewDialog(QDialog):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("Metadata Overview")
        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Item",
            "Valid Values",
            "Unique Values",
            "Range/Type",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.btn_add = QPushButton("Add Hyperedges")
        self.btn_add.clicked.connect(self._on_add)
        layout.addWidget(self.btn_add)
        self._populate(session.metadata)

    def _on_add(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 0)
        if not item:
            return
        col = item.text()
        parent = self.parent()
        if parent and hasattr(parent, "add_metadata_hyperedges"):
            parent.add_metadata_hyperedges(col)

    def _populate(self, metadata: pd.DataFrame) -> None:
        row = 0
        for col in metadata.columns:
            if col == "image_path":
                continue
            valid_count, unique_count, range_str = _summarize_column(metadata[col])
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(col)))
            self.table.setItem(row, 1, QTableWidgetItem(str(valid_count)))
            self.table.setItem(row, 2, QTableWidgetItem(str(unique_count)))
            self.table.setItem(row, 3, QTableWidgetItem(range_str))
            row += 1


def show_metadata_overview(session, parent=None):
    dlg = MetadataOverviewDialog(session, parent=parent)
    dlg.exec_()