# session_model.py
from __future__ import annotations  # for -> SessionModel typing
import uuid
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Set, Iterable, Optional
from PIL import Image, ExifTags
import io
from PyQt5.QtCore import QObject, pyqtSignal as Signal
from .similarity import SIM_METRIC
import pyqtgraph as pg


class SessionModel(QObject):
    # ─── signals any view can subscribe to ──────────────────────────────
    edgeRenamed      = Signal(str, str)      # old, new
    layoutChanged    = Signal()              # big regroup or reload
    similarityDirty  = Signal()              # vectors changed; views may flush

    # ─── construction helpers ───────────────────────────────────────────
    @classmethod
    def load_h5(cls, path: Path) -> "SessionModel":
        with h5py.File(path, "r") as hdf:
            im_list = [x.decode() if isinstance(x, bytes) else x
                       for x in hdf["file_list"][()]]
            matrix = hdf["clustering_results"][()]
            cat_raw = (
                hdf["catList"][()] if "catList" in hdf else [f"edge_{i}" for i in range(matrix.shape[1])]
            )
            cat_list = [x.decode() if isinstance(x, bytes) else x for x in cat_raw]
            df_edges = pd.DataFrame(matrix, columns=cat_list)
            features = hdf["features"][()]
            umap_emb = hdf["umap_embedding"][()] if "umap_embedding" in hdf else None
            openclip_feats = hdf["openclip_features"][()] if "openclip_features" in hdf else None

            if "edge_origins" in hdf:
                origin_raw = hdf["edge_origins"][()]
                edge_orig = [o.decode() if isinstance(o, bytes) else str(o) for o in origin_raw]
            else:
                edge_orig = ["Loaded"] * len(cat_list)

            thumbnails_embedded = hdf.attrs.get("thumbnails_are_embedded", True)
            thumbnail_data: Optional[List[bytes] | List[str]] = None
            if thumbnails_embedded and "thumbnail_data_embedded" in hdf:
                thumbnail_data = [arr.tobytes() for arr in hdf["thumbnail_data_embedded"][:]]
            elif not thumbnails_embedded and "thumbnail_relative_paths" in hdf:
                thumbnail_data = [p.decode("utf-8") if isinstance(p, bytes) else str(p)
                                  for p in hdf["thumbnail_relative_paths"][:]]

        return cls(
            im_list,
            df_edges,
            features,
            path,
            openclip_features=openclip_feats,
            umap_embedding=umap_emb,
            thumbnail_data=thumbnail_data,
            thumbnails_are_embedded=thumbnails_embedded,
            edge_origins=edge_orig,
        )

    def save_h5(self, path: Path | None = None) -> None:
        """Write current session state to an HDF5 file."""
        target = Path(path) if path else self.h5_path
        if not target.suffix:
            target = target.with_suffix(".h5")

        if self.thumbnail_data is None:
            self.generate_thumbnails()

        with h5py.File(target, "w") as hdf:
            dt = h5py.string_dtype(encoding="utf-8")
            hdf.create_dataset(
                "file_list", data=np.array(self.im_list, dtype=object), dtype=dt
            )
            hdf.create_dataset(
                "clustering_results",
                data=self.df_edges.values.astype("i8"),
                dtype="i8",
            )
            hdf.create_dataset(
                "catList", data=np.array(self.cat_list, dtype=object), dtype=dt
            )
            hdf.create_dataset(
                "edge_origins",
                data=np.array([self.edge_origins.get(n, "swin") for n in self.cat_list], dtype=object),
                dtype=dt,
            )
            hdf.create_dataset("features", data=self.features, dtype="f4")
            if self.openclip_features is not None:
                hdf.create_dataset("openclip_features", data=self.openclip_features, dtype="f4")
            if self.umap_embedding is not None:
                hdf.create_dataset("umap_embedding", data=self.umap_embedding, dtype="f4")

            hdf.attrs["thumbnails_are_embedded"] = self.thumbnails_are_embedded

            if self.thumbnail_data:
                if self.thumbnails_are_embedded:
                    dt_vlen = h5py.vlen_dtype(np.uint8)
                    arrs = [np.frombuffer(b, dtype=np.uint8) for b in self.thumbnail_data]
                    hdf.create_dataset("thumbnail_data_embedded", data=arrs, dtype=dt_vlen)
                else:
                    hdf.create_dataset(
                        "thumbnail_relative_paths",
                        data=np.array(self.thumbnail_data, dtype=object),
                        dtype=dt,
                    )

        self.h5_path = target


    def __init__(self,
                 im_list: List[str],
                 df_edges: pd.DataFrame,
                 features: np.ndarray,
                 h5_path: Path,
                 *,
                 openclip_features: np.ndarray | None = None,
                 umap_embedding: np.ndarray | None = None,
                 thumbnail_data: Optional[List[bytes] | List[str]] = None,
                 thumbnails_are_embedded: bool = True,
                 edge_origins: Optional[List[str]] | None = None):
        super().__init__()
        self.im_list  = im_list                              # list[str]
        self.cat_list = list(df_edges.columns)               # list[str]
        self.df_edges = df_edges                             # DataFrame (images×edges)
        self.hyperedges, self.image_mapping = self._prepare_hypergraph_structures(df_edges)

        self.features = features                             # np.ndarray (N×D)
        self.openclip_features = openclip_features
        self.hyperedge_avg_features = self._calculate_hyperedge_avg_features(features)

        self.edge_origins = edge_origins or {name: "swin" for name in self.cat_list}

                # Collect EXIF metadata for all images
        self.metadata = self._extract_image_metadata(im_list)

        self.status_map = {n: {"uuid": str(uuid.uuid4()), "status": "Original"}
                           for n in self.cat_list}
        cmap_hues = max(len(self.cat_list), 16)
        self.edge_colors = {
            name: pg.mkColor(pg.intColor(i, hues=cmap_hues)).name()
            for i, name in enumerate(self.cat_list)
        }
        self.edge_origins = {
            name: (edge_origins[i] if edge_origins and i < len(edge_origins) else "Loaded")
            for i, name in enumerate(self.cat_list)
        }
        self.umap_embedding = umap_embedding
        self.h5_path = h5_path
        
        self.thumbnail_data: Optional[List[bytes] | List[str]] = thumbnail_data
        self.thumbnails_are_embedded: bool = thumbnails_are_embedded

        # overview triplets cache -------------------------------------------------
        self.overview_triplets: Dict[str, tuple[int | None, ...]] | None = None
        self.compute_overview_triplets()


    # ─── internal helpers (static) ──────────────────────────────────────
    @staticmethod
    def _prepare_hypergraph_structures(df):
        hyperedges = {col: set(np.where(df[col] == 1)[0]) for col in df.columns}
        image_mapping: Dict[int, Set[str]] = {}
        rows, cols = np.where(df.values == 1)
        for r, c in zip(rows, cols):
            image_mapping.setdefault(r, set()).add(df.columns[c])
        return hyperedges, image_mapping

    def _calculate_hyperedge_avg_features(self, features):
        n_feat = features.shape[1]
        return {name: features[list(idx)].mean(axis=0) if idx else np.zeros(n_feat)
                for name, idx in self.hyperedges.items()}


    @staticmethod
    def _extract_image_metadata(im_list: List[str]) -> pd.DataFrame:
        """Return a DataFrame with EXIF metadata for each image."""
        meta_rows = []
        for path in im_list:
            entry = {"image_path": path}
            try:
                with Image.open(path) as img:
                    exif = img._getexif()
                    if exif:
                        for k, v in exif.items():
                            tag = ExifTags.TAGS.get(k, k)
                            entry[tag] = v
            except Exception:
                pass
            meta_rows.append(entry)

        return pd.DataFrame(meta_rows)


    def generate_thumbnails(self, size: tuple[int, int] = (100, 100)) -> None:
        """Generate thumbnail JPEG bytes for all images."""
        thumbs: List[bytes] = []
        for p in self.im_list:
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail(size, Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", size, "black")
                off_x = (size[0] - img.width) // 2
                off_y = (size[1] - img.height) // 2
                canvas.paste(img, (off_x, off_y))
                buf = io.BytesIO()
                canvas.save(buf, format="JPEG", quality=90)
                thumbs.append(buf.getvalue())
            except Exception as e:
                print(f"Thumbnail generation failed for {p}: {e}")
                thumbs.append(b"")

        self.thumbnail_data = thumbs
        self.thumbnails_are_embedded = True

    # ─── public API used by the GUI today ───────────────────────────────
    def rename_edge(self, old: str, new: str) -> bool:
        """Return True on success, False if duplicate/invalid."""
        new = new.strip()
        if (not new) or (new in self.hyperedges):
            return False

        # raw structures -------------------------------------------------
        self.hyperedges[new] = self.hyperedges.pop(old)
        self.df_edges.rename(columns={old: new}, inplace=True)
        self.cat_list[self.cat_list.index(old)] = new
        self.hyperedge_avg_features[new] = self.hyperedge_avg_features.pop(old)
        self.status_map[new] = self.status_map.pop(old)
        if old in self.edge_origins:
            self.edge_origins[new] = self.edge_origins.pop(old)
        if old in self.edge_colors:
            self.edge_colors[new] = self.edge_colors.pop(old)
        for imgs in self.image_mapping.values():
            if old in imgs:
                imgs.remove(old)
                imgs.add(new)

        # tell views -----------------------------------------------------
        self.overview_triplets = None
        self.edgeRenamed.emit(old, new)
        self.similarityDirty.emit()
        
        return True

    # ------------------ NEW METHOD --------------------------------------
    def add_empty_hyperedge(self, name: str) -> None:
        """Adds a new, empty hyperedge to the model."""
        # This assumes 'name' has already been validated for uniqueness and is not empty.
        # 1. Update hyperedges dictionary
        self.hyperedges[name] = set()

        # 2. Add a new column of zeros to the DataFrame
        self.df_edges[name] = 0

        # 3. Add to the category list
        self.cat_list.append(name)

        # 4. Create a zero-vector for the new edge's average features
        n_features = self.features.shape[1]
        self.hyperedge_avg_features[name] = np.zeros(n_features)

        # 5. Add a status entry for the new edge
        self.status_map[name] = {"uuid": str(uuid.uuid4()), "status": "New"}
        self.edge_origins[name] = "New"

        idx = len(self.edge_colors)
        cmap_hues = max(idx + 1, 16)
        self.edge_colors[name] = pg.mkColor(pg.intColor(idx, hues=cmap_hues)).name()

        # 6. Signal to the UI that the overall layout has changed
        self.overview_triplets = None
        self.layoutChanged.emit()
    # --------------------------------------------------------------------

    def add_images_to_hyperedge(self, name: str, idxs: Iterable[int]) -> None:
        """Add given image indices to an existing hyperedge."""
        if name not in self.hyperedges:
            return

        changed = False
        for idx in idxs:
            if idx not in self.hyperedges[name]:
                self.hyperedges[name].add(idx)
                self.df_edges.at[idx, name] = 1
                self.image_mapping.setdefault(idx, set()).add(name)
                changed = True

        if changed:
            indices = list(self.hyperedges[name])
            if indices:
                self.hyperedge_avg_features[name] = self.features[indices].mean(axis=0)
            else:
                self.hyperedge_avg_features[name] = np.zeros(self.features.shape[1])
            self.overview_triplets = None
            self.layoutChanged.emit()
            self.similarityDirty.emit()


    def remove_images_from_edges(self, img_idxs: List[int], edges: List[str]) -> None:
        """Remove given image indices from the specified hyperedges."""
        changed = False
        for edge in edges:
            if edge not in self.hyperedges:
                continue
            members = self.hyperedges[edge]
            removed = [i for i in img_idxs if i in members]
            if not removed:
                continue
            changed = True
            for idx in removed:
                members.remove(idx)
                if idx in self.image_mapping:
                    self.image_mapping[idx].discard(edge)
                    if not self.image_mapping[idx]:
                        del self.image_mapping[idx]
                if idx < len(self.df_edges.index):
                    self.df_edges.at[idx, edge] = 0

            # update average features for this edge
            if members:
                self.hyperedge_avg_features[edge] = self.features[list(members)].mean(axis=0)
            else:
                self.hyperedge_avg_features[edge] = np.zeros(self.features.shape[1])

        if changed:
            self.overview_triplets = None
            self.layoutChanged.emit()
            self.similarityDirty.emit()




    # convenience read-only properties -----------------------------------
    def vector_for(self, name: str) -> np.ndarray | None:
        return self.hyperedge_avg_features.get(name)

    def similarity_map(self, ref_name: str) -> Dict[str, float]:
        ref = self.vector_for(ref_name)
        if ref is None:
            return {}
        names = list(self.hyperedge_avg_features)
        mat   = np.stack([self.hyperedge_avg_features[n] for n in names])
        sims  = SIM_METRIC(ref.reshape(1, -1), mat)[0]
        return dict(zip(names, sims))
    
    def similarity_std(self, name: str) -> float | None:
        """Return the standard deviation of image-to-average similarities."""
        idxs = list(self.hyperedges.get(name, []))
        if not idxs:
            return None
        feats = self.features[idxs]
        avg = self.hyperedge_avg_features[name][None, :]
        sims = SIM_METRIC(avg, feats)[0]
        return float(np.std(sims))

    # ------------------------------------------------------------------
    def compute_overview_triplets(self) -> Dict[str, tuple[int | None, ...]]:
        """Return and cache up to six representative image indices per edge."""
        if self.overview_triplets is not None:
            return self.overview_triplets

        res: Dict[str, tuple[int | None, ...]] = {}
        for name, idxs in self.hyperedges.items():
            if not idxs:
                continue
            idx_list = list(idxs)
            feats = self.features[idx_list]
            avg = self.hyperedge_avg_features[name].reshape(1, -1)
            sims = SIM_METRIC(avg, feats)[0]
            top_order = np.argsort(sims)[::-1]
            top = [idx_list[i] for i in top_order[:3]]

            extremes: list[int] = []
            if len(idx_list) >= 2:
                sim_mat = SIM_METRIC(feats, feats)
                np.fill_diagonal(sim_mat, 1.0)
                i, j = divmod(np.argmin(sim_mat), sim_mat.shape[1])
                extremes = [idx_list[i], idx_list[j]]

            far_order = np.argsort(sims)
            farthest: int | None = None
            for i in far_order:
                cand = idx_list[i]
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

        self.overview_triplets = res
        return res

    def apply_clustering_matrix(self, matrix: np.ndarray, *, prefix: str = "edge", origin: str = "swin") -> None:
        """Replace current hyperedges with clustering results."""
        if matrix.ndim != 2:
            raise ValueError("clustering matrix must be 2D")

        df_edges = pd.DataFrame(matrix.astype(int),
                                columns=[f"{prefix}_{i}" for i in range(matrix.shape[1])])
        self.df_edges = df_edges
        self.cat_list = list(df_edges.columns)
        self.edge_origins = {name: origin for name in self.cat_list}

        self.hyperedges, self.image_mapping = self._prepare_hypergraph_structures(df_edges)
        self.hyperedge_avg_features = self._calculate_hyperedge_avg_features(self.features)
        self.status_map = {
            name: {"uuid": str(uuid.uuid4()), "status": "Cluster"}
            for name in self.cat_list
        }
        cmap_hues = max(len(self.cat_list), 16)
        self.edge_colors = {
            name: pg.mkColor(pg.intColor(i, hues=cmap_hues)).name()
            for i, name in enumerate(self.cat_list)
        }
        self.overview_triplets = None
        self.layoutChanged.emit()
        self.similarityDirty.emit()

    def append_clustering_matrix(self, matrix: np.ndarray, *, prefix: str = "edge", origin: str = "swin") -> None:
        """Append new hyperedges from a clustering matrix."""
        if matrix.ndim != 2:
            raise ValueError("clustering matrix must be 2D")
        if matrix.shape[0] != len(self.im_list):
            raise ValueError("matrix row count must match number of images")

        start_idx = len(self.cat_list)
        for i in range(matrix.shape[1]):
            name = f"{prefix}_{start_idx + i}"
            col = matrix[:, i].astype(int)
            self.df_edges[name] = col
            self.cat_list.append(name)

            idxs = set(np.where(col == 1)[0])
            self.hyperedges[name] = idxs
            for idx in idxs:
                self.image_mapping.setdefault(idx, set()).add(name)
            self.hyperedge_avg_features[name] = (
                self.features[list(idxs)].mean(axis=0) if idxs else np.zeros(self.features.shape[1])
            )
            self.status_map[name] = {"uuid": str(uuid.uuid4()), "status": "Cluster"}
            cmap_hues = max(len(self.cat_list), 16)
            self.edge_colors[name] = pg.mkColor(pg.intColor(len(self.edge_colors), hues=cmap_hues)).name()
            self.edge_origins[name] = origin

        self.overview_triplets = None
        self.layoutChanged.emit()
        self.similarityDirty.emit()