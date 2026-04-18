from __future__ import annotations

from dataclasses import dataclass

from .metadata_hyperedges import MetadataBinSettings


@dataclass
class AppSettings:
    max_grid_results: int = 1000
    default_grouping_threshold: float = 1.0
    default_duplicate_jaccard_threshold: float = 0.5

    spatial_limit_images_enabled: bool = False
    spatial_limit_images_value: int = 10
    spatial_limit_edges_enabled: bool = False
    spatial_limit_edges_value: int = 10

    metadata_large_bin_threshold: int = 1000
    metadata_max_category_edges: int = 40
    metadata_max_exact_numeric_values: int = 10
    metadata_max_numeric_bins: int = 5
    metadata_max_date_groups: int = 36
    metadata_max_date_day_groups: int = 14
    metadata_max_numeric_refinement_depth: int = 3

    def metadata_bin_settings(self) -> MetadataBinSettings:
        return MetadataBinSettings(
            large_bin_image_threshold=self.metadata_large_bin_threshold,
            max_category_edges=self.metadata_max_category_edges,
            max_exact_numeric_values=self.metadata_max_exact_numeric_values,
            max_numeric_bins=self.metadata_max_numeric_bins,
            max_date_groups=self.metadata_max_date_groups,
            max_date_day_groups=self.metadata_max_date_day_groups,
            max_numeric_refinement_depth=self.metadata_max_numeric_refinement_depth,
        )
