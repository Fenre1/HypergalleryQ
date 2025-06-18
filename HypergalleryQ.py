# prototype.py --------------------------------------------------------------
import sys, uuid, numpy as np
import pandas as pd
import numpy as np
from utils.similarity import SIM_METRIC
from pathlib import Path
import io
import torch
from PIL import Image, ImageGrab

from PyQt5.QtWidgets import (
    QApplication,
    QTreeView,
    QMainWindow,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QLabel,
    QSlider,
    QMessageBox,
    QPushButton,
    QDockWidget,
    QStackedWidget,
    QAction,
    QInputDialog,
    QDialog,
    QListWidget,
    QDialogButtonBox,
    QAbstractItemView,
    QLineEdit,
)
from PyQt5.QtGui import (
    QStandardItem,
    QStandardItemModel,
    QPalette,
    QColor,
    QIcon,
    QPixmap,
)
from PyQt5.QtCore import Qt, QSignalBlocker, QObject, pyqtSignal as Signal

from utils.data_loader import (
    DATA_DIRECTORY, get_h5_files_in_directory, load_session_data
)
from utils.selection_bus import SelectionBus
from utils.session_model import SessionModel
from utils.image_grid import ImageGridDock
from utils.hyperedge_matrix import HyperedgeMatrixDock
from utils.spatial_viewQv3 import SpatialViewQDock
from utils.feature_extraction import Swinv2LargeFeatureExtractor

from utils.temi_clustering_v1 import TEMIClusterer as TEMIClustererV1
from utils.temi_clustering_v2 import TEMIClusterer as TEMIClustererV2
from utils.temi_clustering_v3 import TEMIHypergraphClusterer as TEMIClustererV3
from utils.temi_clustering_v4 import TemiClustering as TEMIClustererV4

try:
    import darkdetect
    SYSTEM_DARK_MODE = darkdetect.isDark()
except Exception:
    SYSTEM_DARK_MODE = False

def apply_dark_palette(app: QApplication) -> None:
    """Apply a dark color palette to the given QApplication."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

THRESHOLD_DEFAULT = 0.8
SIM_COL = 3
INTER_COL = 4
DECIMALS = 3
UNGROUPED = "Ungrouped"

class _MultiSelectDialog(QDialog):
    """Simple dialog presenting a list for multi-selection."""

    def __init__(self, items: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Hyperedges")
        self.list = QListWidget()
        self.list.addItems(items)
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(self.list)
        lay.addWidget(buttons)

    def chosen(self) -> list[str]:
        return [it.text() for it in self.list.selectedItems()]


class HyperedgeSelectDialog(QDialog):
    """Dialog allowing the user to pick a hyperedge from a list with filtering."""

    def __init__(self, names: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Hyperedge")
        layout = QVBoxLayout(self)

        self.filter_edit = QLineEdit(self)
        self.filter_edit.setPlaceholderText("Filter...")
        layout.addWidget(self.filter_edit)

        self.list_widget = QListWidget(self)
        self.list_widget.addItems(names)
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.list_widget.itemDoubleClicked.connect(lambda *_: self.accept())

    def _apply_filter(self, text: str) -> None:
        text = text.lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(text not in item.text().lower())

    def selected_name(self) -> str | None:
        items = self.list_widget.selectedItems()
        return items[0].text() if items else None

class HyperEdgeTree(QTreeView):
    """Navigator tree that lists meta-groups and individual hyper-edges."""
    def __init__(self, bus: SelectionBus, parent=None):
        super().__init__(parent)
        self.bus = bus
        # everything else (uniformRowHeights, sortingEnabled, etc.) stays
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTreeView.SelectRows)

    def _send_bus_update(self, *_):
        # column-0 indexes (name column)
        names = [idx.data(Qt.DisplayRole)
                 for idx in self.selectionModel().selectedRows(0)]
        self.bus.set_edges(names)

# ---------- Qt helpers -----------------------------------------------------
def _make_item(text: str = "", value=None, editable: bool = False):
    it = QStandardItem(text)
    it.setData(value, Qt.UserRole)

    if editable:
        it.setFlags(it.flags() |  Qt.ItemIsEditable)
    else:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)

    return it


def build_qmodel(rows, headers):
    model   = QStandardItemModel()
    model.setHorizontalHeaderLabels(headers)
    parents = {}

    for r in rows:
        g = r["group_name"]

        # ——— A. special-case “Ungrouped”: insert leaf at root
        if g == UNGROUPED:
            _append_leaf(model, r)
            continue

        # ——— B. ordinary meta-group
        if g not in parents:
            group_items = [_make_item(g)] + [_make_item() for _ in headers[1:]]
            parents[g] = group_items[0]
            model.appendRow(group_items)

        _append_leaf(parents[g], r)

    return model


def _append_leaf(parent_or_model, rowdict):
    """Add one leaf row under either a QStandardItem (group) or the model root."""
    container = parent_or_model
    name_item = _make_item(rowdict["name"], rowdict["name"], editable=True)
    color = rowdict.get("color")
    if color:
        pix = QPixmap(12, 12)
        pix.fill(QColor(color))
        name_item.setIcon(QIcon(pix))
    leaf = [
        name_item,
        _make_item(str(rowdict["image_count"]), rowdict["image_count"]),
        _make_item(rowdict["status"]),
                _make_item(
            "" if rowdict["similarity"] is None
            else f"{rowdict['similarity']:.3f}",
            None if rowdict["similarity"] is None else float(rowdict["similarity"]),
        ),
        _make_item(
            "" if rowdict.get("intersection") is None else str(rowdict["intersection"]),
            rowdict.get("intersection"),
        ),
    ]
    container.appendRow(leaf)

def calculate_similarity_matrix(vecs):
    names = list(vecs)
    if not names:
        return pd.DataFrame()
    m = np.array(list(vecs.values()))
    s = SIM_METRIC(m, m)
    np.fill_diagonal(s, -np.inf)
    return pd.DataFrame(s, index=names, columns=names)


def perform_hierarchical_grouping(model, thresh=0.8):
    vecs   = model.hyperedge_avg_features.copy()
    comp   = {k: [k] for k in vecs}
    counts = {k: 1     for k in vecs}

    while len(vecs) > 1:
        sim = calculate_similarity_matrix(vecs)
        if sim.empty or sim.values.max() < thresh:
            break
        col = sim.max().idxmax()
        row = sim[col].idxmax()

        new = f"temp_{uuid.uuid4()}"
        c1, c2 = counts[row], counts[col]
        vecs[new] = (vecs[row] * c1 + vecs[col] * c2) / (c1 + c2)
        comp[new] = comp.pop(row) + comp.pop(col)
        counts[new] = c1 + c2
        vecs.pop(row), vecs.pop(col)
    return comp


def rename_groups_sequentially(raw):
    res, cnt, singles = {}, 1, []
    for k, ch in raw.items():
        if len(ch) > 1:
            res[f"Meta-Group {cnt}"] = ch
            cnt += 1
        else:
            singles.extend(ch)
    if singles:
        res["Ungrouped"] = singles
    return res


def build_row_data(groups, model):
    status = model.status_map
    rows = []
    for g, children in groups.items():
        for child in children:
            meta = status[child]
            rows.append(
                dict(
                    uuid=meta["uuid"],
                    name=child,
                    image_count=len(model.hyperedges[child]),
                    status=meta["status"],
                    similarity=None,
                    intersection=None,
                    group_name=g,
                    color=model.edge_colors.get(child, "#808080"),
                )
            )
    return rows



class MainWin(QMainWindow):
    def _vector_for(self, name: str) -> np.ndarray | None:
        avg = self.model.hyperedge_avg_features
        if name in avg:
            return avg[name][None, :]

        if name in self.groups:
            child_vecs = [avg[c] for c in self.groups[name] if c in avg]
            if child_vecs:
                return np.mean(child_vecs, axis=0, keepdims=True)
        return None

    def compute_similarity(self):
        if not self.model: 
            return
        sel = self.tree.selectionModel().selectedRows(0)
        if not sel: 
            return
        ref_name = sel[0].data(Qt.DisplayRole)

        ref_vec = self._vector_for(ref_name)
        if ref_vec is None:
            QMessageBox.warning(self, "No features", f"No feature vector for “{ref_name}”.")
            return

        avg = self.model.hyperedge_avg_features
        names, vectors = list(avg), np.stack(list(avg.values()))
        sims = SIM_METRIC(ref_vec, vectors)[0]
        sim_map = dict(zip(names, sims))

        ref_imgs = self.model.hyperedges.get(ref_name, set())
        inter_map = {
            name: len(ref_imgs & self.model.hyperedges.get(name, set()))
            for name in names
        }

        model = self.tree.model()
        self._update_similarity_items(model.invisibleRootItem(), sim_map)
        self._update_intersection_items(model.invisibleRootItem(), inter_map)
        model.sort(SIM_COL, Qt.DescendingOrder)

    def _update_similarity_items(self, parent: QStandardItem, sim_map):
        for r in range(parent.rowCount()):
            name_item, sim_item = parent.child(r, 0), parent.child(r, SIM_COL)

            if name_item.hasChildren():
                self._update_similarity_items(name_item, sim_map)
                child_vals = [name_item.child(c, SIM_COL).data(Qt.UserRole) for c in range(name_item.rowCount())]
                val = np.nanmean([v for v in child_vals if v is not None]) if child_vals else None
            else:
                val = sim_map.get(name_item.text())

            if val is None:
                sim_item.setData(None, Qt.UserRole); sim_item.setData("", Qt.DisplayRole)
            else:
                sim_item.setData(float(val), Qt.UserRole); sim_item.setData(f"{val:.{DECIMALS}f}", Qt.DisplayRole)

    def _compute_overview_triplets(self) -> dict[str, tuple[int | None, ...]]:
        """Return up to six image indexes for an overview of each hyperedge."""
        res: dict[str, tuple[int | None, ...]] = {}
        for name, idxs in self.model.hyperedges.items():
            if not idxs:
                continue
            idxs = list(idxs)
            feats = self.model.features[idxs]
            avg = self.model.hyperedge_avg_features[name].reshape(1, -1)

            sims = SIM_METRIC(avg, feats)[0]
            top_order = np.argsort(sims)[::-1]
            top = [idxs[i] for i in top_order[:3]]

            extremes: list[int] = []
            if len(idxs) >= 2:
                sim_mat = SIM_METRIC(feats, feats)
                np.fill_diagonal(sim_mat, 1.0)
                i, j = divmod(np.argmin(sim_mat), sim_mat.shape[1])
                extremes = [idxs[i], idxs[j]]

            far_order = np.argsort(sims)  # ascending
            farthest: int | None = None
            for i in far_order:
                cand = idxs[i]
                if cand not in top and cand not in extremes:
                    farthest = cand
                    break

            final: list[int | None] = []
            for idx in top + extremes:
                if idx not in final:
                    final.append(idx)
            if farthest is not None and farthest not in final:
                final.append(farthest)

            while len(final) < 6:
                final.append(None)

            res[name] = tuple(final[:6])

        return res


    def _update_intersection_items(self, parent: QStandardItem, inter_map):
        for r in range(parent.rowCount()):
            name_item = parent.child(r, 0)
            inter_item = parent.child(r, INTER_COL)

            if name_item.hasChildren():
                self._update_intersection_items(name_item, inter_map)
                child_vals = [name_item.child(c, INTER_COL).data(Qt.UserRole) for c in range(name_item.rowCount())]
                val = sum(v for v in child_vals if v is not None) if child_vals else None
            else:
                val = inter_map.get(name_item.text())

            if val is None:
                inter_item.setData(None, Qt.UserRole)
                inter_item.setData("", Qt.DisplayRole)
            else:
                inter_item.setData(int(val), Qt.UserRole)
                inter_item.setData(str(int(val)), Qt.DisplayRole)

    def __init__(self):
        super().__init__()
        self.setDockNestingEnabled(True)
        self.setWindowTitle("Hypergraph Desktop Prototype")
        self.resize(1200, 800)

        self.model = None
        self._clip_extractor = None
        self._overview_triplets = None
        self.temi_results = {}
        self.bus = SelectionBus()
        self.bus.edgesChanged.connect(self._update_bus_images)
        self.bus.edgesChanged.connect(print)

        # ----------------- WIDGET AND DOCK CREATION ------------------------------------
        # Create all widgets and docks first, then arrange them.

        # --- List Tree ---
        self.tree = HyperEdgeTree(self.bus)
        self.tree_dock = QDockWidget("List tree", self)
        self.tree_dock.setWidget(self.tree)

        # --- Buttons / Tools Dock ---
        self.toolbar_dock = QDockWidget("Buttons", self)
        toolbar_container = QWidget()
        toolbar_layout = QVBoxLayout(toolbar_container)
        toolbar_layout.setContentsMargins(10, 10, 10, 10)
        toolbar_layout.setSpacing(10)

        self.btn_sim = QPushButton("Show similarity and intersection")
        self.btn_sim.clicked.connect(self.compute_similarity)
        
        self.btn_add_hyperedge = QPushButton("Add Hyperedge")
        self.btn_add_hyperedge.clicked.connect(self.on_add_hyperedge)
        
        self.btn_add_img = QPushButton("Add images to hyperedge")
        self.btn_add_img.clicked.connect(self.add_selection_to_hyperedge)
        
        self.btn_del_img = QPushButton("Remove images from hyperedge")
        self.btn_del_img.clicked.connect(self.on_remove_images)

        self.btn_rank = QPushButton("Rank images by selection")
        self.btn_rank.clicked.connect(self.rank_selected_images)

        self.btn_rank_edge = QPushButton("Rank images by hyperedge")
        self.btn_rank_edge.clicked.connect(self.rank_selected_hyperedge)

        self.btn_rank_file = QPushButton("Rank external image")
        self.btn_rank_file.clicked.connect(self.rank_image_file)

        self.btn_rank_clip = QPushButton("Rank clipboard image")
        self.btn_rank_clip.clicked.connect(self.rank_clipboard_image)

        self.btn_overview = QPushButton("Overview")
        self.btn_overview.clicked.connect(self.show_overview)


        self.btn_cluster_v1 = QPushButton("cluster v1")
        self.btn_cluster_v1.clicked.connect(self.cluster_v1)
        self.btn_cluster_v2 = QPushButton("cluster v2")
        self.btn_cluster_v2.clicked.connect(self.cluster_v2)
        self.btn_cluster_v3 = QPushButton("cluster v3")
        self.btn_cluster_v3.clicked.connect(self.cluster_v3)
        self.btn_cluster_v4 = QPushButton("cluster v4")
        self.btn_cluster_v4.clicked.connect(self.cluster_v4)

        self.btn_show_v1 = QPushButton("show v1")
        self.btn_show_v1.clicked.connect(lambda: self.show_cluster_result("v1"))
        self.btn_show_v2 = QPushButton("show v2")
        self.btn_show_v2.clicked.connect(lambda: self.show_cluster_result("v2"))
        self.btn_show_v3 = QPushButton("show v3")
        self.btn_show_v3.clicked.connect(lambda: self.show_cluster_result("v3"))
        self.btn_show_v4 = QPushButton("show v4")
        self.btn_show_v4.clicked.connect(lambda: self.show_cluster_result("v4"))




        toolbar_layout.addWidget(self.btn_sim)
        toolbar_layout.addWidget(self.btn_add_hyperedge)
        toolbar_layout.addWidget(self.btn_add_img)
        toolbar_layout.addWidget(self.btn_del_img)
        toolbar_layout.addWidget(self.btn_rank)        
        toolbar_layout.addWidget(self.btn_rank_edge)
        toolbar_layout.addWidget(self.btn_rank_file)
        toolbar_layout.addWidget(self.btn_rank_clip)
        toolbar_layout.addWidget(self.btn_overview)

        toolbar_layout.addWidget(self.btn_cluster_v1)
        toolbar_layout.addWidget(self.btn_cluster_v2)
        toolbar_layout.addWidget(self.btn_cluster_v3)
        toolbar_layout.addWidget(self.btn_cluster_v4)
        toolbar_layout.addWidget(self.btn_show_v1)
        toolbar_layout.addWidget(self.btn_show_v2)
        toolbar_layout.addWidget(self.btn_show_v3)
        toolbar_layout.addWidget(self.btn_show_v4)



        toolbar_layout.addStretch()
        self.toolbar_dock.setWidget(toolbar_container)

        # --- Image Grid ---
        self.image_grid = ImageGridDock(self.bus, self)
        self.image_grid.setObjectName("ImageGridDock") # Use object name for clarity
        self.image_grid.labelDoubleClicked.connect(lambda name: self.bus.set_edges([name]))
        # --- Spatial View ---
        self.spatial_dock = SpatialViewQDock(self.bus, self)
        self.spatial_dock.setObjectName("SpatialViewDock")

        # --- Hyperedge Matrix ---
        self.matrix_dock = HyperedgeMatrixDock(self.bus, self)
        self.matrix_dock.setObjectName("HyperedgeMatrixDock")

        # --- Grouping Slider (no longer in central widget) ---
        # We can place these controls in one of the docks, e.g., the 'Buttons' dock.
        self.slider = QSlider(Qt.Horizontal, minimum=50, maximum=100, singleStep=5, value=int(THRESHOLD_DEFAULT * 100))
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        toolbar_layout.insertWidget(0, self.label)
        toolbar_layout.insertWidget(1, self.slider)


        # ----------------- DOCK LAYOUT ARRANGEMENT ------------------------------------
        # Arrange the created docks to match the wireframe.
        # No central widget is set, allowing docks to fill the entire window.

        # 1. Add the "List tree" to the left area.
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tree_dock)

        # 2. Add the "Buttons" dock under the "List tree" dock.
        self.addDockWidget(Qt.LeftDockWidgetArea, self.toolbar_dock)

        # 3. Add the "Image grid" to the right area. It will take up the remaining space.
        self.addDockWidget(Qt.RightDockWidgetArea, self.image_grid)

        # 4. Add the "Spatial view" below the "Image grid".
        self.addDockWidget(Qt.RightDockWidgetArea, self.spatial_dock)

        # 5. Split the area occupied by the "Spatial view" to place the "Hyperedge matrix" to its right.
        self.splitDockWidget(self.spatial_dock, self.matrix_dock, Qt.Horizontal)

        # Optional: Set initial relative sizes of the docks
        self.resizeDocks([self.tree_dock, self.image_grid], [300, 900], Qt.Horizontal)
        self.resizeDocks([self.tree_dock, self.toolbar_dock], [550, 250], Qt.Vertical)
        self.resizeDocks([self.spatial_dock, self.matrix_dock], [450, 450], Qt.Horizontal)


        # ----------------- MENU AND STATE ------------------------------------
        open_act = QAction("&Open Session…", self, triggered=self.open_session)
        self.menuBar().addMenu("&File").addAction(open_act)

        self.model = None

        self.temi_results = {}


        self.slider.valueChanged.connect(self.regroup)

        files = get_h5_files_in_directory()
        if files:
            self.load_session(Path(DATA_DIRECTORY) / files[0])


    def on_add_hyperedge(self):
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        # Use QInputDialog to get text from the user
        new_name, ok = QInputDialog.getText(self, "Add New Hyperedge", "Enter name for the new hyperedge:")

        if ok and new_name:
            # User clicked OK and entered text
            clean_name = new_name.strip()

            # --- Validation ---
            if not clean_name:
                QMessageBox.warning(self, "Invalid Name", "Hyperedge name cannot be empty.")
                return

            if clean_name in self.model.hyperedges:
                QMessageBox.warning(self, "Duplicate Name",
                                    f"A hyperedge named '{clean_name}' already exists.")
                return

            # --- Call the model method to perform the addition ---
            self.model.add_empty_hyperedge(clean_name)
            # The model will emit layoutChanged, which is connected to self.regroup in load_session
        else:
            # User clicked Cancel or entered nothing
            print("Add hyperedge cancelled.")

    def on_remove_images(self):
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        if not self.image_grid.view.model():
            QMessageBox.warning(self, "No Images", "No images are currently displayed.")
            return

        sel_indexes = self.image_grid.view.selectionModel().selectedIndexes()
        if not sel_indexes:
            QMessageBox.warning(self, "No Selection", "Select images to remove first.")
            return

        model = self.image_grid.view.model()
        img_idxs = [model._indexes[i.row()] for i in sel_indexes]

        possible_edges = sorted(
            {e for idx in img_idxs for e in self.model.image_mapping.get(idx, set())}
        )
        if not possible_edges:
            QMessageBox.information(
                self,
                "Not in Hyperedge",
                "Selected images are not assigned to any hyperedge.",
            )
            return

        dialog = _MultiSelectDialog(possible_edges, self)
        if dialog.exec() != QDialog.Accepted:
            return
        edges = dialog.chosen()
        if not edges:
            return

        self.model.remove_images_from_edges(img_idxs, edges)
        # refresh displayed images of current selection
        self._update_bus_images(self.image_grid._selected_edges)



    def rank_selected_images(self):
        """Rank all images by similarity to the currently selected images."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        model = self.image_grid.view.model()
        sel = self.image_grid.view.selectionModel().selectedRows() if model else []
        if not sel:
            QMessageBox.information(self, "No Selection", "Select images in the grid first.")
            return

        sel_idxs = [model._indexes[i.row()] for i in sel]
        feats = self.model.features
        ref = feats[sel_idxs].mean(axis=0, keepdims=True)
        sims = SIM_METRIC(ref, feats)[0]
        ranked = np.argsort(sims)[::-1]
        ranked = [i for i in ranked if i not in sel_idxs][:500]
        final_idxs = sel_idxs + ranked
        self.image_grid.update_images(final_idxs, highlight=sel_idxs, sort=False)

    def rank_selected_hyperedge(self):
        """Rank all images by similarity to the selected hyperedge."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        model = self.tree.model()
        sel = self.tree.selectionModel().selectedRows(0) if model else []
        if not sel:
            QMessageBox.information(self, "No Selection", "Select a hyperedge in the tree first.")
            return

        item = model.itemFromIndex(sel[0])
        if item.hasChildren():
            QMessageBox.information(self, "Invalid Selection", "Please select a single hyperedge, not a group.")
            return

        name = item.text()
        if name not in self.model.hyperedges:
            QMessageBox.information(self, "Unknown Hyperedge", "Selected item is not a hyperedge.")
            return

        ref = self.model.hyperedge_avg_features.get(name)
        feats = self.model.features
        sims = SIM_METRIC(ref.reshape(1, -1), feats)[0]
        ranked = np.argsort(sims)[::-1]
        exclude = self.model.hyperedges[name]
        ranked = [i for i in ranked if i not in exclude][:500]
        self.image_grid.update_images(ranked, sort=False)


    def rank_image_file(self):
        """Rank all session images against a user chosen image file."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select image file",
            str(DATA_DIRECTORY),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)",
        )
        if not file:
            return

        try:
            if not hasattr(self, "_ext_feature_extractor"):
                self._ext_feature_extractor = Swinv2LargeFeatureExtractor(batch_size=1)
            vec = self._ext_feature_extractor.extract_features([file])[0]
        except Exception as e:
            QMessageBox.critical(self, "Feature Error", str(e))
            return

        feats = self.model.features
        sims = SIM_METRIC(vec.reshape(1, -1), feats)[0]
        ranked = np.argsort(sims)[::-1][:500]
        self.image_grid.update_images(list(ranked), sort=False)

    def rank_clipboard_image(self):
        """Rank images by similarity to the image currently in the clipboard."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return
        try:
            clip_img = ImageGrab.grabclipboard()
        except Exception as e:
            QMessageBox.critical(self, "Clipboard Error", str(e))
            return

        if clip_img is None:
            QMessageBox.information(self, "No Image", "No image in clipboard.")
            return

        if isinstance(clip_img, list):
            try:
                clip_img = Image.open(clip_img[0])
            except Exception:
                QMessageBox.information(self, "No Image", "Clipboard does not contain an image.")
                return

        if clip_img.mode == "RGBA":
            clip_img = clip_img.convert("RGB")

        if self._clip_extractor is None:
            self._clip_extractor = Swinv2LargeFeatureExtractor(batch_size=1)

        tensor = self._clip_extractor.transform(clip_img).unsqueeze(0).to(self._clip_extractor.device)
        with torch.no_grad():
            feat = self._clip_extractor.model(tensor).cpu().numpy()[0]

        feats = self.model.features
        sims = SIM_METRIC(feat.reshape(1, -1), feats)[0]
        ranked = np.argsort(sims)[::-1][:500]
        self.image_grid.update_images(ranked, sort=False)

    def show_overview(self):
        """Display triplet overview on the image grid."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return
        if self._overview_triplets is None:
            self._overview_triplets = self._compute_overview_triplets()
        self.image_grid.show_overview(self._overview_triplets, self.model)


    # ------------------------------------------------------------------
    def _run_clusterer(self, clusterer) -> np.ndarray | None:
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return None
        feats = self.model.features
        try:
            if hasattr(clusterer, "fit_predict"):
                result = clusterer.fit_predict(feats)
            else:
                clusterer.fit(feats)
                result = clusterer.hyperedges(feats)
            if hasattr(result, "cpu"):
                result = result.cpu().numpy()
        except Exception as e:
            QMessageBox.critical(self, "Clustering Error", str(e))
            return None
        return result

    def cluster_v1(self):
        res = self._run_clusterer(TEMIClustererV1(k=3, epochs=10))
        if res is not None:
            self.temi_results["v1"] = np.array(res)

    def cluster_v2(self):
        res = self._run_clusterer(TEMIClustererV2(3, epochs=10))
        if res is not None:
            self.temi_results["v2"] = np.array(res)

    def cluster_v3(self):
        res = self._run_clusterer(TEMIClustererV3(3, epochs=10))
        if res is not None:
            self.temi_results["v3"] = np.array(res)

    def cluster_v4(self):
        res = self._run_clusterer(TEMIClustererV4(3, epochs=10))
        if res is not None:
            self.temi_results["v4"] = np.array(res)

    def show_cluster_result(self, key: str):
        matrix = self.temi_results.get(key)
        if matrix is None:
            QMessageBox.information(self, "No Result", f"No clustering result for {key}.")
            return
        self.model.apply_clustering_matrix(matrix, prefix=f"v{key}")



    def add_selection_to_hyperedge(self):
        """Add currently selected images in the grid to a chosen hyperedge."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        model = self.image_grid.view.model()
        sel = self.image_grid.view.selectionModel().selectedRows() if model else []
        if not sel:
            QMessageBox.information(self, "No Selection", "Select images in the grid first.")
            return

        img_idxs = [model._indexes[i.row()] for i in sel]

        dialog = HyperedgeSelectDialog(list(self.model.hyperedges), self)
        if dialog.exec_() != QDialog.Accepted:
            return
        name = dialog.selected_name()
        if not name:
            QMessageBox.information(self, "No Hyperedge", "No hyperedge selected.")
            return

        self.model.add_images_to_hyperedge(name, img_idxs)

    def open_session(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select .h5 session", DATA_DIRECTORY, "H5 files (*.h5)")
        if file: 
            self.load_session(Path(file))

    def load_session(self, path: Path):
        try:
            # If a model already exists, disconnect its signal first
            if self.model:
                try:
                    self.model.layoutChanged.disconnect(self.regroup)
                except TypeError:
                    # This can happen if the signal was never connected
                    pass

            self.model = SessionModel.load_h5(path)


            self.model.layoutChanged.connect(self.regroup)

            self._overview_triplets = None
            self.model.layoutChanged.connect(self._on_layout_changed)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        self.image_grid.set_model(self.model)
        self.matrix_dock.set_model(self.model)
        self.spatial_dock.set_model(self.model)
        self.regroup()

    def _on_layout_changed(self):
        self._overview_triplets = None
        self.regroup()

    def _on_item_changed(self, item: QStandardItem):
        if item.column() != 0 or item.hasChildren(): return
        parent = item.parent()
        old_name, new_name = item.data(Qt.UserRole), item.text().strip()
        self.model.rename_edge(old_name, new_name)
        if parent is not None: self._update_group_similarity(parent)

    def _invalidate_similarity_column(self, name_item: QStandardItem):
        row = name_item.row()
        sim_item = name_item.parent().child(row, SIM_COL)
        sim_item.setData(None, Qt.UserRole); sim_item.setData("", Qt.DisplayRole)

    def _update_bus_images(self, names: list[str]):
        if not self.model: self.bus.set_images([]); return
        idxs = set().union(*(self.model.hyperedges.get(n, set()) for n in names))
        self.bus.set_images(sorted(idxs))

    def regroup(self):
        if not self.model: return
        thr = self.slider.value() / 100
        self.label.setText(f"Grouping threshold: {thr:.2f}")

        self.groups = rename_groups_sequentially(perform_hierarchical_grouping(self.model, thresh=thr))
        rows = build_row_data(self.groups, self.model)

        headers = ["Name", "Images", "Status", "Similarity", "Intersection"]
        model = build_qmodel(rows, headers)
        model.itemChanged.connect(self._on_item_changed)
        self.tree.setModel(model)
        self.tree.selectionModel().selectionChanged.connect(self.tree._send_bus_update)
        self.tree.expandAll()

        if hasattr(self, 'matrix_dock'): self.matrix_dock.update_matrix()

    def _update_group_similarity(self, group_item: QStandardItem):
        vals = [v for v in (group_item.child(r, SIM_COL).data(Qt.UserRole) for r in range(group_item.rowCount())) if v is not None]
        parent = group_item.parent() or group_item.model().invisibleRootItem()
        sim_item = parent.child(group_item.row(), SIM_COL)

        if vals:
            mean_val = float(np.mean(vals))
            sim_item.setData(mean_val, Qt.UserRole); sim_item.setData(f"{mean_val:.{DECIMALS}f}", Qt.DisplayRole)
        else:
            sim_item.setData(None, Qt.UserRole); sim_item.setData("", Qt.DisplayRole)

# ---------- main -----------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    if SYSTEM_DARK_MODE:
        apply_dark_palette(app)
    win = MainWin()
    win.show()
    sys.exit(app.exec())