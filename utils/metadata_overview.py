import pandas as pd
from dataclasses import replace
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QPushButton,
    QSpinBox,
)
from PyQt5.QtCore import Qt

from .metadata_hyperedges import MetadataBinSettings, profile_metadata_column


def _summarize_column(series: pd.Series) -> tuple[int, int, str]:
    profile = profile_metadata_column(series)
    detail = profile.detail if profile.kind == "empty" else f"{profile.kind}: {profile.detail}"
    return profile.valid_count, profile.unique_count, detail


class MetadataOverviewDialog(QDialog):
    def __init__(
        self,
        session,
        parent=None,
        metadata_settings: MetadataBinSettings | None = None,
    ):
        super().__init__(parent)
        self._session = session
        self._metadata_settings = metadata_settings or MetadataBinSettings()
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

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Split bins above:"))
        self.bin_threshold_spin = QSpinBox()
        self.bin_threshold_spin.setRange(2, 1_000_000)
        self.bin_threshold_spin.setSingleStep(100)
        self.bin_threshold_spin.setValue(self._metadata_settings.large_bin_image_threshold)
        threshold_row.addWidget(self.bin_threshold_spin)
        threshold_row.addWidget(QLabel("images"))
        layout.addLayout(threshold_row)

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
            if hasattr(parent, "set_metadata_large_bin_threshold"):
                parent.set_metadata_large_bin_threshold(self.bin_threshold_spin.value())
            parent.add_metadata_hyperedges(col, metadata_settings=self.metadata_settings())

    def metadata_settings(self) -> MetadataBinSettings:
        return replace(
            self._metadata_settings,
            large_bin_image_threshold=self.bin_threshold_spin.value(),
        )

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


def show_metadata_overview(
    session,
    parent=None,
    metadata_settings: MetadataBinSettings | None = None,
):
    dlg = MetadataOverviewDialog(session, parent=parent, metadata_settings=metadata_settings)
    dlg.exec_()
