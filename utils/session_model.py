from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import h5py
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PIL import ExifTags, Image, ImageOps
from PyQt5.QtCore import QObject, pyqtSignal as Signal
from PyQt5.QtGui import QColor

from .similarity import SIM_METRIC

SCHEMA_VERSION = 2


def generate_n_colors(n: int, saturation: int = 150, value: int = 230) -> list[str]:
    colors: list[str] = []
    for i in range(max(n, 1)):
        hue = int(360 * i / n) if n else 0
        color = QColor()
        color.setHsv(hue, saturation, value)
        colors.append(color.name())
    return colors


def jaccard_similarity(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _decode_strings(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


class SessionModel(QObject):
    edgeRenamed = Signal(str, str)       # edge_id, new label
    layoutChanged = Signal()
    similarityDirty = Signal()
    hyperedgeModified = Signal(str)      # edge_id

    @classmethod
    def load_h5(cls, path: Path) -> "SessionModel":
        with h5py.File(path, "r") as hdf:
            version = int(hdf.attrs["schema_version"])
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported session schema {version}; expected {SCHEMA_VERSION}."
                )

            im_list = _decode_strings(hdf["file_list"][()])
            edge_ids = _decode_strings(hdf["edge_ids"][()])
            edge_names_list = _decode_strings(hdf["edge_names"][()])
            edge_names = dict(zip(edge_ids, edge_names_list))
            matrix = hdf["clustering_results"][()]
            df_edges = pd.DataFrame(matrix, columns=edge_ids)
            features = hdf["features"][()]
            openclip_feats = (
                hdf["openclip_features"][()] if "openclip_features" in hdf else None
            )
            umap_emb = hdf["umap_embedding"][()] if "umap_embedding" in hdf else None

            image_umap: Optional[Dict[str, Dict[int, np.ndarray]]] = None
            if "image_umap" in hdf:
                image_umap = {}
                grp = hdf["image_umap"]
                for edge_id in grp:
                    data = grp[edge_id][()]
                    image_umap[edge_id] = (
                        {
                            int(i): data[row, 1:]
                            for row, i in enumerate(data[:, 0].astype(int))
                        }
                        if data.size > 0
                        else {}
                    )

            edge_origins = _decode_strings(hdf["edge_origins"][()])
            edge_seen = (
                hdf["edge_seen_times"][()]
                if "edge_seen_times" in hdf
                else np.zeros(len(edge_ids), dtype="f8")
            )
            edge_statuses = (
                _decode_strings(hdf["edge_statuses"][()])
                if "edge_statuses" in hdf
                else ["Original"] * len(edge_ids)
            )
            edge_colors = (
                _decode_strings(hdf["edge_colors"][()])
                if "edge_colors" in hdf
                else generate_n_colors(len(edge_ids))
            )

            thumbnails_embedded = bool(hdf.attrs.get("thumbnails_are_embedded", True))
            thumbnail_data: Optional[List[bytes] | List[str]] = None
            if thumbnails_embedded and "thumbnail_data_embedded" in hdf:
                thumbnail_data = [
                    arr.tobytes() for arr in hdf["thumbnail_data_embedded"][:]
                ]
            elif not thumbnails_embedded and "thumbnail_relative_paths" in hdf:
                thumbnail_data = _decode_strings(hdf["thumbnail_relative_paths"][:])

            metadata_df: pd.DataFrame | None = None
            if "metadata" in hdf:
                meta_json = hdf["metadata"][()]
                if isinstance(meta_json, bytes):
                    meta_json = meta_json.decode("utf-8")
                metadata_df = pd.read_json(meta_json, orient="table")

        return cls(
            im_list,
            df_edges,
            features,
            path,
            openclip_features=openclip_feats,
            umap_embedding=umap_emb,
            image_umap=image_umap,
            thumbnail_data=thumbnail_data,
            thumbnails_are_embedded=thumbnails_embedded,
            edge_ids=edge_ids,
            edge_names=edge_names,
            edge_origins=edge_origins,
            edge_last_seen=edge_seen,
            edge_statuses=edge_statuses,
            edge_colors=edge_colors,
            metadata=metadata_df,
        )

    def save_h5(self, path: Path | None = None) -> None:
        target = Path(path) if path else self.h5_path
        if not target.suffix:
            target = target.with_suffix(".h5")

        with h5py.File(target, "w") as hdf:
            hdf.attrs["schema_version"] = SCHEMA_VERSION
            hdf.attrs["thumbnails_are_embedded"] = self.thumbnails_are_embedded
            dt = h5py.string_dtype(encoding="utf-8")

            hdf.create_dataset(
                "file_list", data=np.array(self.im_list, dtype=object), dtype=dt
            )
            hdf.create_dataset(
                "clustering_results",
                data=self.df_edges[self.cat_list].values.astype("i8"),
                dtype="i8",
            )
            hdf.create_dataset(
                "edge_ids", data=np.array(self.cat_list, dtype=object), dtype=dt
            )
            hdf.create_dataset(
                "edge_names",
                data=np.array([self.edge_label(eid) for eid in self.cat_list], dtype=object),
                dtype=dt,
            )
            hdf.create_dataset(
                "edge_origins",
                data=np.array(
                    [self.edge_origins.get(eid, "swinv2") for eid in self.cat_list],
                    dtype=object,
                ),
                dtype=dt,
            )
            hdf.create_dataset(
                "edge_statuses",
                data=np.array(
                    [
                        self.status_map.get(eid, {}).get("status", "Original")
                        for eid in self.cat_list
                    ],
                    dtype=object,
                ),
                dtype=dt,
            )
            hdf.create_dataset(
                "edge_colors",
                data=np.array(
                    [self.edge_colors.get(eid, "#808080") for eid in self.cat_list],
                    dtype=object,
                ),
                dtype=dt,
            )
            hdf.create_dataset(
                "edge_seen_times",
                data=np.array(
                    [self.edge_seen_times.get(eid, 0.0) for eid in self.cat_list],
                    dtype="f8",
                ),
                dtype="f8",
            )
            hdf.create_dataset("features", data=self.features, dtype="f4")
            if self.openclip_features is not None:
                hdf.create_dataset(
                    "openclip_features", data=self.openclip_features, dtype="f4"
                )
            if self.umap_embedding is not None:
                hdf.create_dataset("umap_embedding", data=self.umap_embedding, dtype="f4")

            if self.image_umap:
                grp = hdf.create_group("image_umap")
                for edge_id, mapping in self.image_umap.items():
                    arr = np.array(
                        [[idx, vec[0], vec[1]] for idx, vec in mapping.items()],
                        dtype="f4",
                    )
                    grp.create_dataset(edge_id, data=arr, dtype="f4")

            try:
                meta_json = self._sanitize_metadata(self.metadata).to_json(orient="table")
            except Exception:
                meta_json = self.metadata.astype(str).to_json(orient="table")
            hdf.create_dataset("metadata", data=meta_json, dtype=dt)

            if self.thumbnail_data:
                if self.thumbnails_are_embedded:
                    dt_vlen = h5py.vlen_dtype(np.uint8)
                    arrs = [
                        np.frombuffer(b, dtype=np.uint8) for b in self.thumbnail_data
                    ]
                    hdf.create_dataset(
                        "thumbnail_data_embedded", data=arrs, dtype=dt_vlen
                    )
                else:
                    hdf.create_dataset(
                        "thumbnail_relative_paths",
                        data=np.array(self.thumbnail_data, dtype=object),
                        dtype=dt,
                    )

        self.h5_path = target

    def __init__(
        self,
        im_list: List[str],
        df_edges: pd.DataFrame,
        features: np.ndarray,
        h5_path: Path,
        *,
        openclip_features: np.ndarray | None = None,
        umap_embedding: np.ndarray | None = None,
        image_umap: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        thumbnail_data: Optional[List[bytes] | List[str]] = None,
        thumbnails_are_embedded: bool = True,
        edge_ids: Optional[List[str]] | None = None,
        edge_names: Optional[Dict[str, str]] | None = None,
        edge_origins: Optional[List[str] | Dict[str, str]] | None = None,
        edge_last_seen: Optional[List[float] | Dict[str, float]] | None = None,
        edge_statuses: Optional[List[str] | Dict[str, str]] | None = None,
        edge_colors: Optional[List[str] | Dict[str, str]] | None = None,
        metadata: pd.DataFrame | None = None,
    ):
        super().__init__()

        incoming_cols = list(df_edges.columns)
        if edge_ids is None:
            edge_ids = [str(uuid.uuid4()) for _ in incoming_cols]
            if edge_names is None:
                edge_names = dict(zip(edge_ids, [str(c) for c in incoming_cols]))
        elif edge_names is None:
            edge_names = {edge_id: edge_id for edge_id in edge_ids}

        if len(edge_ids) != len(incoming_cols):
            raise ValueError("edge_ids length must match clustering matrix columns")

        self.im_list = im_list
        self.cat_list = list(edge_ids)
        self.edge_names = {
            edge_id: str(edge_names.get(edge_id, edge_id)) for edge_id in self.cat_list
        }
        self.df_edges = df_edges.copy()
        self.df_edges.columns = self.cat_list

        self.hyperedges, self.image_mapping = self._prepare_hypergraph_structures(
            self.df_edges
        )

        self.features = features
        self.openclip_features = openclip_features
        self.hyperedge_avg_features = self._calculate_hyperedge_avg_features(features)

        feats32 = self.features.astype(np.float32, copy=False)
        norms = np.linalg.norm(feats32, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._features_unit = feats32 / norms

        self.metadata = metadata if metadata is not None else self._extract_image_metadata(im_list)
        self.status_map = self._init_status_map(edge_statuses)
        self.edge_seen_times = self._init_float_map(edge_last_seen, 0.0)
        colors = generate_n_colors(len(self.cat_list))
        self.edge_colors = self._init_str_map(edge_colors, colors, "#808080")
        self.edge_origins = self._init_str_map(edge_origins, None, "swinv2")

        self.umap_embedding = umap_embedding
        self.image_umap = image_umap
        self.h5_path = h5_path
        self.thumbnail_data: Optional[List[bytes] | List[str]] = thumbnail_data
        self.thumbnails_are_embedded = thumbnails_are_embedded
        self.overview_triplets: Dict[str, tuple[int | None, ...]] | None = None
        self.compute_overview_triplets()

    def _init_str_map(
        self,
        values: Optional[List[str] | Dict[str, str]],
        defaults: Optional[list[str]],
        fallback: str,
    ) -> dict[str, str]:
        if isinstance(values, dict):
            return {edge_id: values.get(edge_id, fallback) for edge_id in self.cat_list}
        if isinstance(values, list):
            return {
                edge_id: values[i] if i < len(values) else fallback
                for i, edge_id in enumerate(self.cat_list)
            }
        if defaults is not None:
            return {
                edge_id: defaults[i % len(defaults)]
                for i, edge_id in enumerate(self.cat_list)
            }
        return {edge_id: fallback for edge_id in self.cat_list}

    def _init_float_map(
        self,
        values: Optional[List[float] | Dict[str, float]],
        fallback: float,
    ) -> dict[str, float]:
        if isinstance(values, dict):
            return {edge_id: float(values.get(edge_id, fallback)) for edge_id in self.cat_list}
        if values is not None:
            return {
                edge_id: float(values[i]) if i < len(values) else fallback
                for i, edge_id in enumerate(self.cat_list)
            }
        return {edge_id: fallback for edge_id in self.cat_list}

    def _init_status_map(
        self,
        statuses: Optional[List[str] | Dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        if isinstance(statuses, dict):
            return {
                edge_id: {
                    "uuid": str(uuid.uuid4()),
                    "status": statuses.get(edge_id, "Original"),
                }
                for edge_id in self.cat_list
            }
        if isinstance(statuses, list):
            return {
                edge_id: {
                    "uuid": str(uuid.uuid4()),
                    "status": statuses[i] if i < len(statuses) else "Original",
                }
                for i, edge_id in enumerate(self.cat_list)
            }
        return {
            edge_id: {"uuid": str(uuid.uuid4()), "status": "Original"}
            for edge_id in self.cat_list
        }

    @property
    def features_unit(self) -> np.ndarray:
        return self._features_unit

    def edge_label(self, edge_id: str) -> str:
        return self.edge_names.get(edge_id, edge_id)

    def edge_id_for_name(self, name: str) -> str | None:
        for edge_id, label in self.edge_names.items():
            if label == name:
                return edge_id
        return None

    def edge_choices(self) -> list[tuple[str, str]]:
        return [(edge_id, self.edge_label(edge_id)) for edge_id in self.cat_list]

    @staticmethod
    def _prepare_hypergraph_structures(df: pd.DataFrame):
        hyperedges = {col: set(np.where(df[col] == 1)[0]) for col in df.columns}
        image_mapping: Dict[int, Set[str]] = {}
        rows, cols = np.where(df.values == 1)
        for r, c in zip(rows, cols):
            image_mapping.setdefault(r, set()).add(df.columns[c])
        return hyperedges, image_mapping

    def _calculate_hyperedge_avg_features(self, features):
        n_feat = features.shape[1]
        return {
            edge_id: features[list(idxs)].mean(axis=0) if idxs else np.zeros(n_feat)
            for edge_id, idxs in self.hyperedges.items()
        }

    @staticmethod
    def _sanitize_metadata(df: pd.DataFrame) -> pd.DataFrame:
        return df.applymap(
            lambda x: x if isinstance(x, (int, float, str, bool)) else str(x)
        )

    @staticmethod
    def _extract_image_metadata(im_list: List[str]) -> pd.DataFrame:
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
        thumbs: List[bytes] = []
        for p in self.im_list:
            try:
                img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
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

    def _update_edit_status(
        self, edge_id: str, *, renamed: bool = False, modified: bool = False
    ) -> None:
        entry = self.status_map.get(edge_id)
        if not entry:
            return
        status = entry.get("status", "Original")
        if status in {"New", "Orphaned", "Cluster"}:
            return
        has_renamed = "Renamed" in status
        has_modified = "modified" in status.lower()
        if renamed:
            has_renamed = True
        if modified:
            has_modified = True
        if has_renamed and has_modified:
            entry["status"] = "Renamed and modified"
        elif has_renamed:
            entry["status"] = "Renamed"
        elif has_modified:
            entry["status"] = "Modified"
        else:
            entry["status"] = "Original"

    def rename_edge(self, edge_id: str, new: str) -> bool:
        new = new.strip()
        if not new or edge_id not in self.hyperedges:
            return False
        if any(label == new and eid != edge_id for eid, label in self.edge_names.items()):
            return False
        self.edge_names[edge_id] = new
        self._update_edit_status(edge_id, renamed=True)
        self.edgeRenamed.emit(edge_id, new)
        return True

    def add_empty_hyperedge(self, name: str) -> str:
        edge_id = str(uuid.uuid4())
        self.hyperedges[edge_id] = set()
        self.df_edges[edge_id] = 0
        self.cat_list.append(edge_id)
        self.edge_names[edge_id] = name
        self.hyperedge_avg_features[edge_id] = np.zeros(self.features.shape[1])
        self.status_map[edge_id] = {"uuid": str(uuid.uuid4()), "status": "New"}
        self.edge_origins[edge_id] = "New"
        idx = len(self.edge_colors)
        cmap_hues = max(idx + 1, 16)
        self.edge_colors[edge_id] = pg.mkColor(pg.intColor(idx, hues=cmap_hues)).name()
        self.edge_seen_times[edge_id] = 0.0
        self.overview_triplets = None
        self.layoutChanged.emit()
        self.hyperedgeModified.emit(edge_id)
        return edge_id

    def add_images_to_hyperedge(self, edge_id: str, idxs: Iterable[int]) -> None:
        if edge_id not in self.hyperedges:
            return

        changed = False
        for idx in idxs:
            if idx not in self.hyperedges[edge_id]:
                self.hyperedges[edge_id].add(idx)
                self.df_edges.at[idx, edge_id] = 1
                self.image_mapping.setdefault(idx, set()).add(edge_id)
                changed = True

        if changed:
            indices = list(self.hyperedges[edge_id])
            self.hyperedge_avg_features[edge_id] = (
                self.features[indices].mean(axis=0)
                if indices
                else np.zeros(self.features.shape[1])
            )
            self._update_edit_status(edge_id, modified=True)
            if self.overview_triplets is not None:
                self.overview_triplets.pop(edge_id, None)
            self.layoutChanged.emit()
            self.similarityDirty.emit()
            self.hyperedgeModified.emit(edge_id)

    def remove_images_from_edges(self, img_idxs: List[int], edges: List[str]) -> None:
        changed_edges: set[str] = set()
        for edge_id in edges:
            if edge_id not in self.hyperedges:
                continue
            members = self.hyperedges[edge_id]
            removed = [i for i in img_idxs if i in members]
            if not removed:
                continue
            changed_edges.add(edge_id)
            for idx in removed:
                members.remove(idx)
                if idx in self.image_mapping:
                    self.image_mapping[idx].discard(edge_id)
                    if not self.image_mapping[idx]:
                        del self.image_mapping[idx]
                if idx < len(self.df_edges.index):
                    self.df_edges.at[idx, edge_id] = 0

            self.hyperedge_avg_features[edge_id] = (
                self.features[list(members)].mean(axis=0)
                if members
                else np.zeros(self.features.shape[1])
            )
            self._update_edit_status(edge_id, modified=True)

        if changed_edges:
            if self.overview_triplets is not None:
                for edge_id in changed_edges:
                    self.overview_triplets.pop(edge_id, None)
            self.layoutChanged.emit()
            self.similarityDirty.emit()
            for edge_id in changed_edges:
                self.hyperedgeModified.emit(edge_id)

    def delete_hyperedge(self, edge_id: str, orphan_name: str = "orphaned images") -> None:
        if edge_id not in self.hyperedges:
            return

        members = self.hyperedges.pop(edge_id)
        if edge_id in self.df_edges.columns:
            self.df_edges.drop(columns=[edge_id], inplace=True)
        if edge_id in self.cat_list:
            self.cat_list.remove(edge_id)
        self.hyperedge_avg_features.pop(edge_id, None)
        self.status_map.pop(edge_id, None)
        self.edge_origins.pop(edge_id, None)
        self.edge_colors.pop(edge_id, None)
        self.edge_seen_times.pop(edge_id, None)
        self.edge_names.pop(edge_id, None)
        if self.image_umap:
            self.image_umap.pop(edge_id, None)

        orphan_id = self.edge_id_for_name(orphan_name)
        if orphan_id is None:
            orphan_id = str(uuid.uuid4())
            self.hyperedges[orphan_id] = set()
            self.df_edges[orphan_id] = 0
            self.cat_list.append(orphan_id)
            self.edge_names[orphan_id] = orphan_name
            self.hyperedge_avg_features[orphan_id] = np.zeros(self.features.shape[1])
            self.status_map[orphan_id] = {
                "uuid": str(uuid.uuid4()),
                "status": "Orphaned",
            }
            self.edge_colors[orphan_id] = generate_n_colors(len(self.edge_colors) + 1)[-1]
            self.edge_origins[orphan_id] = "system"
            self.edge_seen_times[orphan_id] = 0.0

        for idx in members:
            if idx in self.image_mapping:
                self.image_mapping[idx].discard(edge_id)
                if not self.image_mapping[idx]:
                    self.image_mapping[idx].add(orphan_id)
                    self.hyperedges[orphan_id].add(idx)
                    self.df_edges.at[idx, orphan_id] = 1

        orphans = self.hyperedges[orphan_id]
        self.hyperedge_avg_features[orphan_id] = (
            self.features[list(orphans)].mean(axis=0)
            if orphans
            else np.zeros(self.features.shape[1])
        )

        if self.overview_triplets is not None:
            self.overview_triplets.pop(edge_id, None)
            self.overview_triplets.pop(orphan_id, None)
        self.layoutChanged.emit()
        self.similarityDirty.emit()
        self.hyperedgeModified.emit(orphan_id)

    def prune_similar_edges(self, threshold: float) -> None:
        precedence = ["swinv2", "openclip"]
        ordered_ids = self.cat_list[:]
        kept: list[str] = []
        remove: list[str] = []
        for origin in precedence:
            for edge_id in ordered_ids:
                if self.edge_origins.get(edge_id) != origin or edge_id in remove:
                    continue
                if origin == "swinv2":
                    kept.append(edge_id)
                    continue
                is_dup = False
                for kept_id in kept:
                    if jaccard_similarity(
                        self.hyperedges[edge_id], self.hyperedges[kept_id]
                    ) > threshold:
                        is_dup = True
                        break
                if is_dup:
                    remove.append(edge_id)
                else:
                    kept.append(edge_id)

        for edge_id in remove:
            self.delete_hyperedge(edge_id)

    def vector_for(self, edge_id: str) -> np.ndarray | None:
        return self.hyperedge_avg_features.get(edge_id)

    def similarity_map(self, ref_edge_id: str) -> Dict[str, float]:
        ref = self.vector_for(ref_edge_id)
        if ref is None:
            return {}
        edge_ids = list(self.hyperedge_avg_features)
        mat = np.stack([self.hyperedge_avg_features[eid] for eid in edge_ids])
        sims = SIM_METRIC(ref.reshape(1, -1), mat)[0]
        return dict(zip(edge_ids, sims))

    def similarity_std(self, edge_id: str) -> float | None:
        idxs = list(self.hyperedges.get(edge_id, []))
        if not idxs:
            return None
        feats = self.features[idxs]
        avg = self.hyperedge_avg_features[edge_id][None, :]
        sims = SIM_METRIC(avg, feats)[0]
        return float(np.std(sims))

    def overview_triplet_for(self, edge_id: str) -> tuple[int | None, ...]:
        if self.overview_triplets is None:
            self.overview_triplets = {}
        trip = self.overview_triplets.get(edge_id)
        if trip is not None:
            return trip

        idxs = self.hyperedges.get(edge_id)
        if not idxs:
            trip = ()
        else:
            idx_list = list(idxs)
            feats = self.features[idx_list]
            avg = self.hyperedge_avg_features[edge_id].reshape(1, -1)
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
            trip = tuple(final[:6])

        self.overview_triplets[edge_id] = trip
        return trip

    def compute_overview_triplets(self) -> Dict[str, tuple[int | None, ...]]:
        if self.overview_triplets is None:
            self.overview_triplets = {}
        for edge_id in self.hyperedges:
            if edge_id not in self.overview_triplets:
                self.overview_triplets[edge_id] = self.overview_triplet_for(edge_id)
        return self.overview_triplets

    def apply_clustering_matrix(
        self, matrix: np.ndarray, *, prefix: str = "edge", origin: str = "swinv2"
    ) -> None:
        if matrix.ndim != 2:
            raise ValueError("clustering matrix must be 2D")
        edge_ids = [str(uuid.uuid4()) for _ in range(matrix.shape[1])]
        self.df_edges = pd.DataFrame(matrix.astype(int), columns=edge_ids)
        self.cat_list = edge_ids
        self.edge_names = {eid: f"{prefix}_{i}" for i, eid in enumerate(edge_ids)}
        self.edge_origins = {eid: origin for eid in self.cat_list}
        self.hyperedges, self.image_mapping = self._prepare_hypergraph_structures(
            self.df_edges
        )
        self.hyperedge_avg_features = self._calculate_hyperedge_avg_features(self.features)
        status = "Original" if origin == "swinv2" else "Origin"
        self.status_map = {
            eid: {"uuid": str(uuid.uuid4()), "status": status}
            for eid in self.cat_list
        }
        colors = generate_n_colors(len(self.cat_list))
        self.edge_colors = {
            eid: colors[i % len(colors)] for i, eid in enumerate(self.cat_list)
        }
        self.edge_seen_times = {eid: 0.0 for eid in self.cat_list}
        self.layoutChanged.emit()
        self.similarityDirty.emit()

    def append_clustering_matrix(
        self, matrix: np.ndarray, *, prefix: str = "edge", origin: str = "swinv2"
    ) -> None:
        if matrix.ndim != 2:
            raise ValueError("clustering matrix must be 2D")
        if matrix.shape[0] != len(self.im_list):
            raise ValueError("matrix row count must match number of images")

        start_idx = len(self.cat_list)
        n_new = matrix.shape[1]
        color_list = generate_n_colors(start_idx + n_new)
        for i in range(n_new):
            edge_id = str(uuid.uuid4())
            label = f"{prefix}_{start_idx + i}"
            col = matrix[:, i].astype(int)
            self.df_edges[edge_id] = col
            self.cat_list.append(edge_id)
            self.edge_names[edge_id] = label

            idxs = set(np.where(col == 1)[0])
            self.hyperedges[edge_id] = idxs
            for idx in idxs:
                self.image_mapping.setdefault(idx, set()).add(edge_id)
            self.hyperedge_avg_features[edge_id] = (
                self.features[list(idxs)].mean(axis=0)
                if idxs
                else np.zeros(self.features.shape[1])
            )
            status = "Original" if origin == "swinv2" else "Origin"
            self.status_map[edge_id] = {"uuid": str(uuid.uuid4()), "status": status}
            self.edge_colors[edge_id] = color_list[start_idx + i]
            self.edge_origins[edge_id] = origin
            self.edge_seen_times[edge_id] = 0.0

        self.layoutChanged.emit()
        self.similarityDirty.emit()
