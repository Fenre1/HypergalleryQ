from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MISSING_STRINGS = {"", "none", "nan", "nat", "null", "<na>"}
MAX_CATEGORY_EDGES = 40
MAX_EXACT_NUMERIC_VALUES = 10
MAX_NUMERIC_BINS = 5
MAX_DATE_GROUPS = 36
MAX_DATE_DAY_GROUPS = 14
LARGE_BIN_IMAGE_THRESHOLD = 1000
MAX_NUMERIC_REFINEMENT_DEPTH = 3


@dataclass(frozen=True)
class MetadataBinSettings:
    max_category_edges: int = MAX_CATEGORY_EDGES
    max_exact_numeric_values: int = MAX_EXACT_NUMERIC_VALUES
    max_numeric_bins: int = MAX_NUMERIC_BINS
    max_date_groups: int = MAX_DATE_GROUPS
    max_date_day_groups: int = MAX_DATE_DAY_GROUPS
    large_bin_image_threshold: int = LARGE_BIN_IMAGE_THRESHOLD
    max_numeric_refinement_depth: int = MAX_NUMERIC_REFINEMENT_DEPTH


@dataclass(frozen=True)
class MetadataColumnProfile:
    kind: str
    valid_count: int
    unique_count: int
    detail: str


@dataclass(frozen=True)
class MetadataEdgePlan:
    name: str
    image_indices: list[int]


@dataclass(frozen=True)
class MetadataHyperedgePlan:
    column: str
    kind: str
    valid_count: int
    mapped_count: int
    unique_count: int
    detail: str
    edges: list[MetadataEdgePlan]
    warnings: list[str]


def display_value(value) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def is_valid_metadata_value(value) -> bool:
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return False
    except Exception:
        pass

    return display_value(value).casefold() not in MISSING_STRINGS


def valid_metadata_mask(series: pd.Series) -> pd.Series:
    return series.map(is_valid_metadata_value).astype(bool)


def profile_metadata_column(series: pd.Series) -> MetadataColumnProfile:
    mask = valid_metadata_mask(series)
    valid = series[mask]
    valid_count = int(mask.sum())
    if valid_count == 0:
        return MetadataColumnProfile("empty", 0, 0, "n/a")

    strings = valid.map(display_value)
    unique_count = int(strings.nunique(dropna=True))

    dates = _coerce_dates(strings)
    date_count = int(dates.notna().sum())
    if _looks_date_like(series, strings) and date_count / valid_count >= 0.8:
        parsed = dates.dropna()
        return MetadataColumnProfile(
            "date",
            valid_count,
            unique_count,
            f"{_format_date(parsed.min())} - {_format_date(parsed.max())}",
        )

    numeric = pd.to_numeric(strings, errors="coerce")
    numeric_count = int(numeric.notna().sum())
    if numeric_count / valid_count >= 0.9:
        nums = numeric.dropna()
        return MetadataColumnProfile(
            "numeric",
            valid_count,
            unique_count,
            f"{_format_number(float(nums.min()))} - {_format_number(float(nums.max()))}",
        )

    return MetadataColumnProfile("categorical", valid_count, unique_count, "categorical")


def plan_metadata_hyperedges(
    metadata: pd.DataFrame,
    im_list: Iterable[str],
    column: str,
    existing_names: Iterable[str],
    settings: MetadataBinSettings | None = None,
) -> MetadataHyperedgePlan:
    settings = settings or MetadataBinSettings()
    if column not in metadata.columns:
        return MetadataHyperedgePlan(
            column=column,
            kind="missing",
            valid_count=0,
            mapped_count=0,
            unique_count=0,
            detail="n/a",
            edges=[],
            warnings=[f"No metadata column '{column}'."],
        )

    series = metadata[column]
    profile = profile_metadata_column(series)
    warnings: list[str] = []
    if profile.valid_count == 0:
        return MetadataHyperedgePlan(
            column=column,
            kind=profile.kind,
            valid_count=0,
            mapped_count=0,
            unique_count=0,
            detail=profile.detail,
            edges=[],
            warnings=[f"Metadata column '{column}' has no usable values."],
        )

    row_to_image, mapping_warnings = _map_metadata_rows_to_images(metadata, list(im_list))
    warnings.extend(mapping_warnings)

    valid_mask = valid_metadata_mask(series)
    items = []
    for row_pos, value in enumerate(series):
        if not bool(valid_mask.iloc[row_pos]):
            continue
        image_idx = row_to_image.get(row_pos)
        if image_idx is None:
            continue
        items.append(
            {
                "row_pos": row_pos,
                "image_idx": int(image_idx),
                "value": value,
                "text": display_value(value),
            }
        )

    mapped_images = sorted({item["image_idx"] for item in items})
    if not mapped_images:
        return MetadataHyperedgePlan(
            column=column,
            kind=profile.kind,
            valid_count=profile.valid_count,
            mapped_count=0,
            unique_count=profile.unique_count,
            detail=profile.detail,
            edges=[],
            warnings=warnings + [f"No valid '{column}' metadata values could be mapped to images."],
        )

    unmapped_count = profile.valid_count - len({item["row_pos"] for item in items})
    if unmapped_count > 0:
        warnings.append(f"{unmapped_count} valid metadata rows could not be mapped to images.")

    used_names = set(existing_names)
    edges = [
        MetadataEdgePlan(
            name=_unique_edge_name(column, used_names),
            image_indices=mapped_images,
        )
    ]

    if profile.kind == "date":
        _append_date_edges(edges, items, column, used_names, mapped_images, warnings, settings)
    elif profile.kind == "numeric":
        _append_numeric_edges(edges, items, column, used_names, mapped_images, warnings, settings)
    elif profile.kind == "categorical":
        _append_categorical_edges(edges, items, column, used_names, mapped_images, warnings, settings)

    return MetadataHyperedgePlan(
        column=column,
        kind=profile.kind,
        valid_count=profile.valid_count,
        mapped_count=len(mapped_images),
        unique_count=profile.unique_count,
        detail=profile.detail,
        edges=edges,
        warnings=warnings,
    )


def _append_categorical_edges(
    edges: list[MetadataEdgePlan],
    items: list[dict],
    column: str,
    used_names: set[str],
    aggregate_images: list[int],
    warnings: list[str],
    settings: MetadataBinSettings,
) -> None:
    groups: OrderedDict[str, set[int]] = OrderedDict()
    for item in items:
        groups.setdefault(item["text"], set()).add(item["image_idx"])

    mostly_unique = len(groups) > 12 and len(groups) / max(len(aggregate_images), 1) > 0.8

    if len(groups) <= settings.max_category_edges and not mostly_unique:
        selected = sorted(groups.items(), key=lambda pair: pair[0].casefold())
    else:
        repeated = [(value, idxs) for value, idxs in groups.items() if len(idxs) > 1]
        selected = sorted(repeated, key=lambda pair: (-len(pair[1]), pair[0].casefold()))[
            :settings.max_category_edges
        ]
        if not selected:
            warnings.append(
                f"'{column}' has {len(groups)} mostly unique values, so only the aggregate hyperedge was created."
            )
            return
        if mostly_unique:
            warnings.append(
                f"'{column}' is mostly unique; created hyperedges only for repeated values."
            )
        else:
            warnings.append(
                f"'{column}' has {len(groups)} unique values; created hyperedges for the most common repeated values."
            )

    selected_images: set[int] = set()
    for value, idxs in selected:
        selected_images.update(idxs)
        _append_edge(edges, f"{_short_label(value)} {column}", idxs, used_names, aggregate_images)

    if len(groups) > settings.max_category_edges:
        other = set(aggregate_images) - selected_images
        _append_edge(edges, f"Other {column}", other, used_names, aggregate_images)


def _append_numeric_edges(
    edges: list[MetadataEdgePlan],
    items: list[dict],
    column: str,
    used_names: set[str],
    aggregate_images: list[int],
    warnings: list[str],
    settings: MetadataBinSettings,
) -> None:
    numeric_values = pd.to_numeric(pd.Series([item["text"] for item in items]), errors="coerce")
    numeric_items = [
        (items[i]["image_idx"], float(value))
        for i, value in enumerate(numeric_values)
        if pd.notna(value)
    ]
    if not numeric_items:
        return

    unique_values = sorted({value for _, value in numeric_items})
    if len(unique_values) <= settings.max_exact_numeric_values:
        for value in unique_values:
            idxs = {idx for idx, num in numeric_items if num == value}
            _append_edge(edges, f"{_format_number(value)} {column}", idxs, used_names, aggregate_images)
        return

    refined = _append_numeric_bins(
        edges,
        numeric_items,
        column,
        used_names,
        aggregate_images,
        depth=0,
        settings=settings,
    )
    if refined:
        warnings.append(
            f"Refined numeric bins with {settings.large_bin_image_threshold} or more images into smaller ranges."
        )


def _append_numeric_bins(
    edges: list[MetadataEdgePlan],
    numeric_items: list[tuple[int, float]],
    column: str,
    used_names: set[str],
    aggregate_images: list[int],
    *,
    depth: int,
    settings: MetadataBinSettings,
) -> bool:
    groups = _numeric_bin_groups(numeric_items, settings)
    refined_any = False

    for bin_items in groups:
        idxs = {idx for idx, _ in bin_items}
        unique_values = {value for _, value in bin_items}
        can_refine = (
            len(idxs) >= settings.large_bin_image_threshold
            and depth < settings.max_numeric_refinement_depth
            and len(unique_values) > settings.max_exact_numeric_values
        )
        if can_refine:
            refined_any = (
                _append_numeric_bins(
                    edges,
                    bin_items,
                    column,
                    used_names,
                    aggregate_images,
                    depth=depth + 1,
                    settings=settings,
                )
                or True
            )
            continue

        values = [value for _, value in bin_items]
        label = f"{_format_number(min(values))} - {_format_number(max(values))} {column}"
        _append_edge(edges, label, idxs, used_names, aggregate_images)

    return refined_any


def _numeric_bin_groups(
    numeric_items: list[tuple[int, float]],
    settings: MetadataBinSettings,
) -> list[list[tuple[int, float]]]:
    unique_values = sorted({value for _, value in numeric_items})
    if len(unique_values) <= settings.max_exact_numeric_values:
        return [
            [(idx, num) for idx, num in numeric_items if num == value]
            for value in unique_values
        ]

    nums = pd.Series([value for _, value in numeric_items], dtype="float64")
    q = min(settings.max_numeric_bins, len(unique_values), len(nums))
    try:
        codes = pd.qcut(nums, q=q, labels=False, duplicates="drop")
    except ValueError:
        codes = pd.cut(nums, bins=q, labels=False, duplicates="drop")

    grouped: OrderedDict[int, list[tuple[int, float]]] = OrderedDict()
    for row, code in enumerate(codes):
        if pd.isna(code):
            continue
        grouped.setdefault(int(code), []).append(numeric_items[row])

    return list(grouped.values())


def _append_date_edges(
    edges: list[MetadataEdgePlan],
    items: list[dict],
    column: str,
    used_names: set[str],
    aggregate_images: list[int],
    warnings: list[str],
    settings: MetadataBinSettings,
) -> None:
    parsed = _coerce_dates(pd.Series([item["text"] for item in items]))
    date_items = [
        (items[i]["image_idx"], pd.Timestamp(value))
        for i, value in enumerate(parsed)
        if pd.notna(value)
    ]
    if not date_items:
        return

    missing_dates = len(items) - len(date_items)
    if missing_dates > 0:
        warnings.append(f"{missing_dates} mapped values could not be parsed into date bins.")

    refined = _append_date_bins(
        edges,
        date_items,
        column,
        used_names,
        aggregate_images,
        current_granularity=None,
        settings=settings,
    )
    if refined:
        warnings.append(
            f"Refined date bins with {settings.large_bin_image_threshold} or more images into smaller time ranges."
        )


def _append_date_bins(
    edges: list[MetadataEdgePlan],
    date_items: list[tuple[int, pd.Timestamp]],
    column: str,
    used_names: set[str],
    aggregate_images: list[int],
    *,
    current_granularity: str | None,
    settings: MetadataBinSettings,
) -> bool:
    granularity = _date_next_granularity(date_items, current_granularity, settings)
    if granularity is None:
        return False

    groups: OrderedDict[object, list[tuple[int, pd.Timestamp]]] = OrderedDict()
    for image_idx, ts in date_items:
        groups.setdefault(_date_group_key(ts, granularity), []).append((image_idx, ts))

    refined_any = False
    for key, group_items in sorted(
        groups.items(),
        key=lambda pair: _date_sort_key(pair[0], granularity),
    ):
        idxs = {idx for idx, _ in group_items}
        can_refine = (
            len(idxs) >= settings.large_bin_image_threshold
            and _date_next_granularity(group_items, granularity, settings) is not None
        )
        if can_refine:
            refined_any = (
                _append_date_bins(
                    edges,
                    group_items,
                    column,
                    used_names,
                    aggregate_images,
                    current_granularity=granularity,
                    settings=settings,
                )
                or True
            )
            continue

        _append_edge(
            edges,
            _date_group_label(key, granularity, column, group_items),
            idxs,
            used_names,
            aggregate_images,
        )

    return refined_any


def _date_next_granularity(
    date_items: list[tuple[int, pd.Timestamp]],
    current_granularity: str | None,
    settings: MetadataBinSettings,
) -> str | None:
    timestamps = [ts for _, ts in date_items]
    years = sorted({ts.year for ts in timestamps})
    months = sorted({ts.to_period("M") for ts in timestamps})
    days = sorted({ts.date() for ts in timestamps})
    hours = sorted({ts.floor("h") for ts in timestamps})

    if current_granularity is None:
        if len(years) > settings.max_date_groups:
            return "decade"
        if len(years) > 1:
            return "year"

    if current_granularity in {None, "decade"} and len(years) > 1:
        return "year"

    if current_granularity in {None, "decade", "year"} and len(months) > 1:
        return "month"

    if current_granularity in {None, "decade", "year", "month"}:
        if len(days) > settings.max_date_day_groups:
            return "week"
        if len(days) > 1:
            return "day"

    if current_granularity == "week" and len(days) > 1:
        return "day"

    if len(hours) > 1:
        return "hour"

    return None


def _date_group_key(ts: pd.Timestamp, granularity: str):
    if granularity == "decade":
        return (ts.year // 10) * 10
    if granularity == "year":
        return ts.year
    if granularity == "month":
        return ts.to_period("M")
    if granularity == "week":
        return ts.to_period("W")
    if granularity == "day":
        return ts.date()
    if granularity == "hour":
        return ts.floor("h")
    return ts


def _date_group_label(
    key,
    granularity: str,
    column: str,
    group_items: list[tuple[int, pd.Timestamp]],
) -> str:
    if granularity == "decade":
        return f"{key}s {column}"
    if granularity == "year":
        return f"{key} {column}"
    if granularity == "month":
        return f"{key} {column}"
    if granularity == "week":
        dates = [ts.date() for _, ts in group_items]
        start = min(dates)
        end = max(dates)
        if start == end:
            return f"{start.isoformat()} {column}"
        return (
            f"{start.isoformat()} - "
            f"{end.isoformat()} {column}"
        )
    if granularity == "day":
        return f"{key.isoformat()} {column}"
    if granularity == "hour":
        return f"{key.strftime('%Y-%m-%d %H:00')} {column}"
    return f"{key} {column}"


def _date_sort_key(key, granularity: str):
    if granularity == "week":
        return key.start_time
    return key


def _append_edge(
    edges: list[MetadataEdgePlan],
    label: str,
    image_indices: Iterable[int],
    used_names: set[str],
    aggregate_images: list[int],
) -> None:
    idxs = sorted(set(image_indices))
    if not idxs:
        return
    if set(idxs) == set(aggregate_images):
        return
    edges.append(
        MetadataEdgePlan(
            name=_unique_edge_name(label, used_names),
            image_indices=idxs,
        )
    )


def _map_metadata_rows_to_images(
    metadata: pd.DataFrame,
    im_list: list[str],
) -> tuple[dict[int, int], list[str]]:
    warnings: list[str] = []
    row_to_image: dict[int, int] = {}
    path_column = "image_path" if "image_path" in metadata.columns else None

    path_map = _build_image_path_map(im_list)
    path_matches = 0
    index_matches = 0
    position_matches = 0

    for pos, row_index in enumerate(metadata.index):
        image_idx = None
        if path_column is not None:
            image_idx = _lookup_path_index(metadata.iloc[pos][path_column], path_map)
            if image_idx is not None:
                path_matches += 1

        if image_idx is None:
            image_idx = _index_to_image_idx(row_index, len(im_list))
            if image_idx is not None:
                index_matches += 1

        if image_idx is None and len(metadata) == len(im_list) and pos < len(im_list):
            image_idx = pos
            position_matches += 1

        if image_idx is not None:
            row_to_image[pos] = image_idx

    if path_column is not None and path_matches == 0 and (index_matches or position_matches):
        warnings.append("No metadata image_path values matched image files; used index/row order fallback.")
    elif path_column is not None and position_matches:
        warnings.append("Some metadata rows used row order fallback because their image_path did not match.")
    elif path_column is None and position_matches:
        warnings.append("Metadata has no image_path column; rows were mapped by table order.")
    elif path_column is None and index_matches:
        warnings.append("Metadata has no image_path column; rows were mapped by DataFrame index.")

    return row_to_image, warnings


def _build_image_path_map(im_list: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    duplicates: set[str] = set()
    basename_counts: dict[str, int] = {}

    for path in im_list:
        basename = os.path.basename(os.path.normpath(str(path))).casefold()
        basename_counts[basename] = basename_counts.get(basename, 0) + 1

    for idx, path in enumerate(im_list):
        for key in _path_keys(path):
            if key in mapping and mapping[key] != idx:
                duplicates.add(key)
            else:
                mapping[key] = idx

        basename = os.path.basename(os.path.normpath(str(path))).casefold()
        if basename_counts.get(basename) == 1:
            mapping.setdefault(f"basename:{basename}", idx)

    for key in duplicates:
        mapping.pop(key, None)
    return mapping


def _lookup_path_index(value, path_map: dict[str, int]) -> int | None:
    if not is_valid_metadata_value(value):
        return None
    for key in _path_keys(value):
        if key in path_map:
            return path_map[key]
    basename = os.path.basename(os.path.normpath(display_value(value))).casefold()
    return path_map.get(f"basename:{basename}")


def _path_keys(value) -> list[str]:
    text = display_value(value)
    if not text:
        return []

    keys = {
        os.path.normcase(os.path.normpath(text)),
    }
    try:
        resolved = Path(text).expanduser().resolve(strict=False)
        keys.add(os.path.normcase(os.path.normpath(str(resolved))))
    except Exception:
        pass
    return [key.casefold() for key in keys if key]


def _index_to_image_idx(index_value, image_count: int) -> int | None:
    if isinstance(index_value, (int, np.integer)) and 0 <= int(index_value) < image_count:
        return int(index_value)
    return None


def _unique_edge_name(base: str, used_names: set[str]) -> str:
    root = _clean_edge_name(base) or "metadata"
    name = root
    suffix = 2
    while name in used_names:
        name = f"{root} ({suffix})"
        suffix += 1
    used_names.add(name)
    return name


def _clean_edge_name(name: str) -> str:
    text = display_value(name)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120]


def _short_label(value: str, limit: int = 80) -> str:
    value = _clean_edge_name(value)
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _looks_date_like(original: pd.Series, strings: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(original):
        return True
    pattern = re.compile(
        r"(\d{4}[-/:]\d{1,2}[-/:]\d{1,2})|"
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|"
        r"([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
    )
    return strings.map(lambda value: bool(pattern.search(value))).mean() >= 0.6


def _coerce_dates(strings: pd.Series) -> pd.Series:
    normalized = strings.astype(str).str.strip()
    normalized = normalized.str.replace(
        r"^(\d{4}):(\d{1,2}):(\d{1,2})(.*)$",
        r"\1-\2-\3\4",
        regex=True,
    )
    return normalized.map(_parse_datetime_value)


def _parse_datetime_value(value):
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT

    if pd.isna(ts):
        return pd.NaT

    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def _format_number(value: float) -> str:
    if np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _format_date(value) -> str:
    ts = pd.Timestamp(value)
    if ts.hour or ts.minute or ts.second:
        return ts.strftime("%Y-%m-%d %H:%M")
    return ts.strftime("%Y-%m-%d")
