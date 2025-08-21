from __future__ import annotations

from dataclasses import dataclass
from PyQt5.QtGui import QColor


@dataclass
class ViewState:
    indices: list[int]
    highlight: dict[int, QColor] | list[int] | set[int] | None
    labels: list[str] | None
    separators: set[int] | None
    selected_edges: list[str]