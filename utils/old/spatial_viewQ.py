from __future__ import annotations

import logging
from typing import List, Dict, Optional, Protocol
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from matplotlib.path import Path as MplPath
from PySide6.QtCore import Qt, QPointF, Signal, QObject, QTimer
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout

try:
    import imageio.v3 as iio
    from skimage.transform import resize
    import umap
except ImportError as e:
    logging.error(f"Missing required dependency: {e}")
    raise

from .selection_bus import SelectionBus

__all__ = ["SpatialViewDock"]

# Constants
DARK_MODE_THRESHOLD = 128
DEFAULT_SCATTER_SIZE = 8
DEFAULT_THUMBNAIL_SIZE = 64
MIN_THUMBNAIL_SIZE = 16
LASSO_MIN_POINTS = 3
THUMBNAIL_BORDER_WIDTH = 2
ZOOM_PADDING = 0.1

class SessionModelProtocol(Protocol):
    """Protocol defining the expected interface for SessionModel."""
    
    @property
    def features(self) -> np.ndarray:
        """Feature vectors for each image."""
        ...
    
    @property
    def image_paths(self) -> List[str]:
        """List of image file paths."""
        ...
        
    @property
    def hyperedges(self) -> Dict[str, set]:
        """Mapping from edge names to sets of image indices."""
        ...
        
    @property
    def image_to_edges(self) -> Dict[int, set]:
        """Mapping from image indices to sets of edge names."""
        ...


class UMAPWorker(QObject):
    """Worker for computing UMAP embeddings in a separate thread."""
    
    finished = Signal(np.ndarray)
    error = Signal(str)
    
    def __init__(self, features: np.ndarray, **umap_params):
        super().__init__()
        self.features = features
        self.umap_params = umap_params or {
            'n_components': 2,
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean'
        }
    
    def compute(self):
        """Compute UMAP embedding."""
        try:
            reducer = umap.UMAP(**self.umap_params)
            embedding = reducer.fit_transform(self.features)
            self.finished.emit(embedding)
        except Exception as e:
            self.error.emit(str(e))


class SpatialViewDock(QDockWidget):
    """Interactive 2D embedding view implemented with PyQtGraph."""

    def __init__(self, bus: SelectionBus, parent=None):
        super().__init__("Spatial View", parent)
        self.bus = bus
        self.session: Optional[SessionModelProtocol] = None
        self.embedding: Optional[np.ndarray] = None
        self.color_map: Dict[str, str] = {}
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._umap_worker: Optional[UMAPWorker] = None
        
        # UI Components
        self._setup_ui()
        self._setup_plot()
        
        # State
        self._lasso_path: List[QPointF] = []
        self._lasso_item: Optional[pg.PlotCurveItem] = None
        self._dragging_lasso = False
        self._thumbnail_items: List[pg.ImageItem] = []
        self._thumbnail_rois: List[pg.ROI] = []
        self._thumbnail_size = DEFAULT_THUMBNAIL_SIZE
        
        # Connections
        self._connect_signals()
        self._apply_theme()

    def _setup_ui(self):
        """Initialize UI components."""
        self._plot = pg.PlotWidget()
        self._plot.setRenderHint(pg.QtGui.QPainter.RenderHint.Antialiasing)
        self._plot.setLabel('left', 'UMAP 2')
        self._plot.setLabel('bottom', 'UMAP 1')
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)
        self.setWidget(container)

    def _setup_plot(self):
        """Configure plot settings."""
        self._viewbox = self._plot.getViewBox()
        self._viewbox.setAspectLocked(False)
        self._scatter: Optional[pg.ScatterPlotItem] = None

    def _connect_signals(self):
        """Connect all signal handlers."""
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_click)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        self.bus.edgesChanged.connect(self._on_edges_changed)

    def _is_dark_mode(self) -> bool:
        """Detect if the application is in dark mode."""
        color = self.palette().color(QPalette.ColorRole.Window)
        luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        return luminance < DARK_MODE_THRESHOLD

    def _apply_theme(self):
        """Apply appropriate theme colors."""
        if self._is_dark_mode():
            bg, fg = "#303030", "#DDDDDD"
        else:
            bg, fg = "#ffffff", "#000000"
            
        pg.setConfigOption("background", bg)
        pg.setConfigOption("foreground", fg)
        self._plot.setBackground(bg)
        
        for axis_name in ['left', 'bottom']:
            axis = self._plot.getAxis(axis_name)
            axis.setPen(fg)

    # Lasso Selection Methods
    def _on_mouse_click(self, event):
        """Handle mouse click events for lasso selection."""
        if event.button() != Qt.MouseButton.LeftButton:
            if event.button() == Qt.MouseButton.RightButton and self._dragging_lasso:
                self._cancel_lasso()
            return
            
        if not self._dragging_lasso:
            self._start_lasso(event.scenePos())
        else:
            self._finish_lasso()

    def _on_mouse_move(self, pos):
        """Handle mouse move events during lasso drawing."""
        if not self._dragging_lasso:
            return
            
        self._lasso_path.append(pos)
        self._update_lasso_display()

    def _start_lasso(self, pos: QPointF):
        """Start lasso selection."""
        self._lasso_path = [pos]
        self._clear_lasso_display()
        
        self._lasso_item = pg.PlotCurveItem(pen=pg.mkPen("yellow", width=2))
        self._plot.addItem(self._lasso_item)
        self._dragging_lasso = True

    def _update_lasso_display(self):
        """Update the visual representation of the lasso."""
        if not self._lasso_item or len(self._lasso_path) < 2:
            return
            
        # Convert scene coordinates to view coordinates
        view_points = [self._viewbox.mapSceneToView(p) for p in self._lasso_path]
        xs, ys = zip(*[(p.x(), p.y()) for p in view_points])
        
        # Close the lasso loop for display
        closed_xs = list(xs) + [xs[0]]
        closed_ys = list(ys) + [ys[0]]
        
        self._lasso_item.setData(closed_xs, closed_ys)

    def _finish_lasso(self):
        """Complete lasso selection and identify selected points."""
        if not self._validate_lasso():
            self._cancel_lasso()
            return
            
        selected_indices = self._get_lasso_selection()
        if selected_indices:
            self.bus.set_images(selected_indices)
            
        self._clear_lasso()

    def _cancel_lasso(self):
        """Cancel current lasso selection."""
        self._clear_lasso()

    def _validate_lasso(self) -> bool:
        """Check if lasso has enough points and embedding exists."""
        return (self.embedding is not None and 
                len(self._lasso_path) >= LASSO_MIN_POINTS)

    def _get_lasso_selection(self) -> List[int]:
        """Get indices of points within the lasso."""
        if self.embedding is None:
            return []
            
        # Convert lasso path to view coordinates
        view_points = [self._viewbox.mapSceneToView(p) for p in self._lasso_path]
        vertices = np.array([(p.x(), p.y()) for p in view_points])
        
        # Find points inside the lasso
        path = MplPath(vertices)
        inside_mask = path.contains_points(self.embedding)
        return np.where(inside_mask)[0].tolist()

    def _clear_lasso(self):
        """Clear lasso state completely."""
        self._clear_lasso_display()
        self._lasso_path.clear()
        self._dragging_lasso = False

    def _clear_lasso_display(self):
        """Remove lasso visual elements."""
        if self._lasso_item:
            self._plot.removeItem(self._lasso_item)
            self._lasso_item = None

    # Model and Data Methods
    def set_model(self, session: Optional[SessionModelProtocol]):
        """Set the session model and trigger embedding computation."""
        self.session = session
        self.embedding = None
        self._reset_plot()
        
        if session is None:
            return
            
        self._compute_embedding_async(session.features)
        self._generate_color_map()

    def _reset_plot(self):
        """Reset plot to initial state."""
        self._viewbox.enableAutoRange()
        self._clear_scatter()
        self._clear_thumbnails()
        self._clear_lasso()

    def _clear_scatter(self):
        """Remove existing scatter plot."""
        if self._scatter:
            self._plot.removeItem(self._scatter)
            self._scatter = None

    def _compute_embedding_async(self, features: np.ndarray):
        """Compute UMAP embedding asynchronously."""
        if features.size == 0:
            logging.warning("Empty feature array provided")
            return
            
        self._umap_worker = UMAPWorker(features)
        self._umap_worker.finished.connect(self._on_embedding_ready)
        self._umap_worker.error.connect(self._on_embedding_error)
        
        # Run in thread pool
        self._executor.submit(self._umap_worker.compute)

    def _on_embedding_ready(self, embedding: np.ndarray):
        """Handle completed UMAP computation."""
        self.embedding = embedding
        self._create_scatter_plot()
        self._viewbox.autoRange()
        self._viewbox.enableAutoRange(enable=False)

    def _on_embedding_error(self, error_msg: str):
        """Handle UMAP computation error."""
        logging.error(f"UMAP computation failed: {error_msg}")
        # Could emit a signal here to show error in UI

    def _create_scatter_plot(self):
        """Create the scatter plot from embedding."""
        if self.embedding is None:
            return
            
        self._scatter = pg.ScatterPlotItem(
            pxMode=True, 
            size=DEFAULT_SCATTER_SIZE, 
            pen=None, 
            symbol="o"
        )
        
        # Set initial colors (all gray)
        gray_brush = pg.mkBrush(128, 128, 128)
        brushes = [gray_brush] * len(self.embedding)
        
        self._scatter.setData(
            x=self.embedding[:, 0], 
            y=self.embedding[:, 1], 
            brush=brushes
        )
        
        self._plot.addItem(self._scatter)

    def _generate_color_map(self):
        """Generate colors for each edge."""
        if not self.session:
            return
            
        edge_names = list(self.session.hyperedges.keys())
        num_edges = max(len(edge_names), 1)
        
        self.color_map = {
            name: pg.intColor(i, hues=num_edges).name() 
            for i, name in enumerate(edge_names)
        }

    # Edge Handling Methods
    def _on_edges_changed(self, edge_names: List[str]):
        """Handle changes in selected edges."""
        if not self._is_ready_for_updates():
            return
            
        self._clear_thumbnails()
        
        if not edge_names:
            self._reset_colors()
            return
            
        primary_edge = edge_names[0]
        self._update_colors(primary_edge)
        self._update_view_range(primary_edge)
        self._add_thumbnails_for_edges(edge_names)

    def _is_ready_for_updates(self) -> bool:
        """Check if view is ready for edge updates."""
        return (self.session is not None and 
                self.embedding is not None and 
                self._scatter is not None)

    def _reset_colors(self):
        """Reset all points to default gray color."""
        gray_brush = pg.mkBrush(128, 128, 128)
        colors = [gray_brush] * len(self.embedding)
        self._scatter.setBrush(colors)

    def _update_colors(self, primary_edge: str):
        """Update point colors based on edge membership."""
        colors = ["#808080"] * len(self.embedding)  # Default gray
        
        primary_indices = self.session.hyperedges.get(primary_edge, set())
        primary_color = self.color_map.get(primary_edge, "#ff0000")
        
        # Color points based on edge membership
        for idx in range(len(self.embedding)):
            edges = self.session.image_to_edges.get(idx, set())
            
            if primary_edge in edges:
                colors[idx] = primary_color
            else:
                # Check for other edge memberships
                for edge_name in edges:
                    if edge_name in self.color_map:
                        colors[idx] = self.color_map[edge_name]
                        break
        
        # Apply colors
        brushes = [pg.mkBrush(color) for color in colors]
        self._scatter.setBrush(brushes)

    def _update_view_range(self, edge_name: str):
        """Update view to focus on selected edge."""
        edge_indices = self.session.hyperedges.get(edge_name, set())
        if not edge_indices:
            return
            
        # Get bounding box of selected points
        indices_list = list(edge_indices)
        points = self.embedding[indices_list]
        
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        # Add padding
        x_range = max(x_max - x_min, 0.1)  # Minimum range
        y_range = max(y_max - y_min, 0.1)
        
        x_padding = x_range * ZOOM_PADDING
        y_padding = y_range * ZOOM_PADDING
        
        self._viewbox.setXRange(x_min - x_padding, x_max + x_padding, padding=0)
        self._viewbox.setYRange(y_min - y_padding, y_max + y_padding, padding=0)

    # Thumbnail Methods
    def set_thumbnail_size(self, size: int):
        """Set the size of thumbnail images."""
        self._thumbnail_size = max(MIN_THUMBNAIL_SIZE, size)
        self._clear_thumbnails()

    def _add_thumbnails_for_edges(self, edge_names: List[str]):
        """Add thumbnail images for the specified edges."""
        for edge_name in edge_names:
            edge_indices = self.session.hyperedges.get(edge_name, set())
            if edge_indices:
                # Limit to first few images to avoid clutter
                sample_indices = list(edge_indices)[:3]
                edge_color = self.color_map.get(edge_name, "yellow")
                self._add_thumbnails(sample_indices, edge_color)

    def _add_thumbnails(self, indices: List[int], border_color: str):
        """Add thumbnail images at specified indices."""
        if self.embedding is None or not self.session:
            return
            
        # Calculate scaling factor
        x_range = self._viewbox.viewRange()[0]
        view_width = x_range[1] - x_range[0]
        plot_width = max(self._plot.width(), 1)
        data_per_pixel = view_width / plot_width
        
        for idx in indices:
            try:
                self._add_single_thumbnail(idx, border_color, data_per_pixel)
            except Exception as e:
                logging.warning(f"Failed to load thumbnail for index {idx}: {e}")

    def _add_single_thumbnail(self, idx: int, border_color: str, data_per_pixel: float):
        """Add a single thumbnail image."""
        # Load and process image
        image_path = self.session.image_paths[idx]
        if not Path(image_path).exists():
            logging.warning(f"Image not found: {image_path}")
            return
            
        image_array = iio.imread(image_path)
        processed_array = self._process_thumbnail_image(image_array)
        
        # Create image item
        image_item = pg.ImageItem(processed_array)
        
        # Scale and position
        image_item.resetTransform()
        image_item.scale(data_per_pixel, data_per_pixel)
        
        width_data = processed_array.shape[1] * data_per_pixel
        height_data = processed_array.shape[0] * data_per_pixel
        
        x, y = self.embedding[idx]
        image_item.setPos(x - width_data/2, y - height_data/2)
        image_item.setZValue(10)
        
        # Add border
        roi = pg.ROI(
            [x - width_data/2, y - height_data/2], 
            [width_data, height_data],
            pen=pg.mkPen(border_color, width=THUMBNAIL_BORDER_WIDTH),
            movable=False
        )
        roi.setZValue(11)
        
        # Add to plot and track
        self._plot.addItem(image_item)
        self._plot.addItem(roi)
        self._thumbnail_items.append(image_item)
        self._thumbnail_rois.append(roi)

    def _process_thumbnail_image(self, image_array: np.ndarray) -> np.ndarray:
        """Process image for thumbnail display."""
        # Convert grayscale to RGB if needed
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        
        # Resize if too large
        max_dimension = max(image_array.shape[:2])
        if max_dimension > self._thumbnail_size:
            scale_factor = self._thumbnail_size / max_dimension
            new_shape = (
                int(image_array.shape[0] * scale_factor),
                int(image_array.shape[1] * scale_factor)
            )
            image_array = resize(
                image_array, 
                new_shape, 
                anti_aliasing=True, 
                preserve_range=True
            ).astype(image_array.dtype)
        
        return image_array

    def _clear_thumbnails(self):
        """Remove all thumbnail images and borders."""
        all_items = self._thumbnail_items + self._thumbnail_rois
        for item in all_items:
            self._plot.removeItem(item)
        
        self._thumbnail_items.clear()
        self._thumbnail_rois.clear()

    def closeEvent(self, event):
        """Clean up when dock is closed."""
        self._executor.shutdown(wait=True)
        super().closeEvent(event)