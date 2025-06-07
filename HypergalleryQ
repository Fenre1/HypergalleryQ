# prototype.py --------------------------------------------------------------
import sys, uuid, numpy as np
import pandas as pd
import numpy as np
from utils.similarity import SIM_METRIC
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QTreeView, QMainWindow, QFileDialog, QVBoxLayout,
    QWidget, QLabel, QSlider, QMessageBox, QPushButton
)
from PySide6.QtGui import QStandardItem, QStandardItemModel, QAction
from PySide6.QtCore import Qt, QSignalBlocker, QObject, Signal


from utils.data_loader import (
    DATA_DIRECTORY, get_h5_files_in_directory, load_session_data
)
from utils.selection_bus import SelectionBus 
from utils.session_model import SessionModel
# from utils.hyperedge_list_utils import (
#     calculate_similarity_matrix, perform_hierarchical_grouping, 
#     rename_groups_sequentially, build_row_data
# )

THRESHOLD_DEFAULT = 0.8
SIM_COL = 3
DECIMALS = 3
UNGROUPED = "Ungrouped"


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

        # publish every time the user changes selection
        # self.selectionModel().selectionChanged.connect(self._send_bus_update)

    # ------------------------------------------------------------------
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
            _append_leaf(model, r)           # helper defined just below
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
    leaf = [
        _make_item(rowdict["name"],           rowdict["name"], editable=True),
        _make_item(str(rowdict["image_count"]), rowdict["image_count"]),
        _make_item(rowdict["status"]),
        _make_item("" if rowdict["similarity"] is None
                   else f"{rowdict['similarity']:.3f}",
                   None if rowdict["similarity"] is None
                   else float(rowdict["similarity"])),
    ]
    container.appendRow(leaf) if isinstance(container, QStandardItem) \
        else container.appendRow(leaf)


def calculate_similarity_matrix(vecs):
    names = list(vecs)
    if not names:
        return pd.DataFrame()
    m = np.array(list(vecs.values()))
    s = SIM_METRIC(m, m)                         # cosine for both axes
    np.fill_diagonal(s, -np.inf)                # exclude self-matches
    return pd.DataFrame(s, index=names, columns=names)


def perform_hierarchical_grouping(model, thresh=0.8):
    vecs   = model.hyperedge_avg_features.copy()
    comp   = {k: [k] for k in vecs}
    counts = {k: 1     for k in vecs}

    while len(vecs) > 1:
        sim = calculate_similarity_matrix(vecs)  # uses SIM_METRIC
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
                    group_name=g,
                )
            )
    return rows


class MainWin(QMainWindow):
        # ---------- similarity helpers ---------------------------------------
    def _vector_for(self, name: str) -> np.ndarray | None:
        """Return (1,d) vector for edge or meta-group (mean of children)."""
        avg = self.model.hyperedge_avg_features
        if name in avg:
            return avg[name][None, :]                 # (1,d)

        # meta-group → mean of its children
        if name in self.groups:
            child_vecs = [avg[c] for c in self.groups[name] if c in avg]
            if child_vecs:
                return np.mean(child_vecs, axis=0, keepdims=True)
        return None

    # ---------- slot ------------------------------------------------------
    def compute_similarity(self):
        if not self.model:
            return

        sel = self.tree.selectionModel().selectedRows(0)
        if not sel:
            return
        ref_name = sel[0].data(Qt.DisplayRole)

        ref_vec = self._vector_for(ref_name)
        if ref_vec is None:
            QMessageBox.warning(self, "No features",
                                 f"No feature vector for “{ref_name}”.")
            return

        avg = self.model.hyperedge_avg_features
        names   = list(avg)
        vectors = np.stack([avg[n] for n in names])
        sims    = SIM_METRIC(ref_vec, vectors)[0]
        sim_map = dict(zip(names, sims))

        model = self.tree.model()
        self._update_similarity_items(model.invisibleRootItem(), sim_map)
        model.sort(SIM_COL, Qt.DescendingOrder)

    # ---------- recursive update -----------------------------------------
    def _update_similarity_items(self, parent: QStandardItem, sim_map):
        for r in range(parent.rowCount()):
            name_item = parent.child(r, 0)
            sim_item  = parent.child(r, SIM_COL)

            if name_item.hasChildren():                # meta-group
                self._update_similarity_items(name_item, sim_map)
                child_vals = [name_item.child(c, SIM_COL)
                              .data(Qt.UserRole)
                              for c in range(name_item.rowCount())]
                val = np.nanmean([v for v in child_vals if v is not None]) \
                      if child_vals else None          # mean, not max
            else:                                      # leaf
                val = sim_map.get(name_item.text())

            if val is None:
                sim_item.setData(None, Qt.UserRole)
                sim_item.setData("",   Qt.DisplayRole)
            else:
                sim_item.setData(float(val), Qt.UserRole)
                sim_item.setData(f"{val:.{DECIMALS}f}", Qt.DisplayRole)
                
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hypergraph Desktop Prototype")
        self.resize(1000, 700)

        self.model = None           # your SessionModel; unchanged
        self.bus   = SelectionBus() # ← single shared bus
        self.bus.edgesChanged.connect(print)
        # ───────────────── existing widgets … ─────────────────────────
        self.tree = HyperEdgeTree(self.bus)  


        # buttons
        self.btn_sim = QPushButton("Compute similarity → sort")
        self.btn_sim.clicked.connect(self.compute_similarity)
        
        # self.tree = QTreeView(uniformRowHeights=True, alternatingRowColors=True,
        #                       sortingEnabled=True, selectionBehavior=QTreeView.SelectRows)

        self.slider = QSlider(Qt.Horizontal, minimum=50, maximum=100, singleStep=5,
                              value=int(THRESHOLD_DEFAULT * 100))
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        lay = QVBoxLayout()
        lay.addWidget(self.label)
        lay.addWidget(self.slider)
        lay.addWidget(self.btn_sim)
        lay.addWidget(self.tree)

        w = QWidget(); w.setLayout(lay)
        self.setCentralWidget(w)

        # menu ----------------------------------------------------------------
        open_act = QAction("&Open Session…", self, triggered=self.open_session)
        
        self.menuBar().addMenu("&File").addAction(open_act)

        # state ----------------------------------------------------------------
        self.model = None
        self.slider.valueChanged.connect(self.regroup)

        # auto-load first file if present
        files = get_h5_files_in_directory()
        if files:
            self.load_session(Path(DATA_DIRECTORY) / files[0])
        
    # ------------ slots -----------------------------------------------------
    def open_session(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select .h5 session", DATA_DIRECTORY, "H5 files (*.h5)"
        )
        if file:
            self.load_session(Path(file))

    def load_session(self, path: Path):
        try:
            self.model = SessionModel.load_h5(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return
        self.regroup()


    def _on_item_changed(self, item: QStandardItem):
        if item.column() != 0 or item.hasChildren():
            return                  # ignore group names & other columns

        parent = item.parent()      # may be None (root-level leaf)        
        old_name = item.data(Qt.UserRole)      # stored key
        new_name = item.text().strip()
        self.model.rename_edge(old_name, new_name)

        if parent is not None:          # skip for root-level leaves
            self._update_group_similarity(parent)

    
    def _invalidate_similarity_column(self, name_item: QStandardItem):
        row = name_item.row()
        sim_item = name_item.parent().child(row, SIM_COL)
        sim_item.setData(None, Qt.UserRole)
        sim_item.setData("",   Qt.DisplayRole)
    
    


    def regroup(self):
        if not self.model:
            return
        thr = self.slider.value() / 100
        self.label.setText(f"Grouping threshold: {thr:.2f}")

        self.groups = rename_groups_sequentially(
            perform_hierarchical_grouping(self.model, thresh=thr)
        )
        rows = build_row_data(self.groups, self.model)

        headers = ["Name", "Images", "Status", "Similarity"]
        model = build_qmodel(rows, headers)
        model.itemChanged.connect(self._on_item_changed)
        self.tree.setModel(model)
        # connect *after* the model is set so selectionModel exists
        self.tree.selectionModel().selectionChanged.connect(
            self.tree._send_bus_update)
        self.tree.expandAll()

    def _update_group_similarity(self, group_item: QStandardItem):
        """Recompute mean(similarity of children) and update group's cell."""
        vals = [group_item.child(r, SIM_COL).data(Qt.UserRole)
                for r in range(group_item.rowCount())]
        vals = [v for v in vals if v is not None]
        sim_item = group_item.child(0, SIM_COL).parent().child(group_item.row(), SIM_COL)
        if vals:
            mean_val = float(np.mean(vals))
            sim_item.setData(mean_val, Qt.UserRole)
            sim_item.setData(f"{mean_val:.{DECIMALS}f}", Qt.DisplayRole)
        else:
            sim_item.setData(None, Qt.UserRole)
            sim_item.setData("",   Qt.DisplayRole)

# ---------- main -----------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWin()
    win.show()
    sys.exit(app.exec())
