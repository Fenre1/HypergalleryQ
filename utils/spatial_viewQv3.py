from __future__ import annotations

import os
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QApplication,
    QPushButton,
)
from PyQt5.QtGui import QPalette, QPixmap, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPointF, pyqtSignal as Signal, QTimer 

from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem

from matplotlib.path import Path as MplPath
from .selection_bus import SelectionBus
from .session_model import SessionModel
import umap

      
from .fast_sim_engine   import SimulationEngine 

# class SimulationEngine:
#     def __init__(self, initial_positions, hyperedges):
#         self.num_nodes = len(initial_positions)
#         self.positions = np.copy(initial_positions).astype(np.float32)
#         self.velocities = np.zeros((self.num_nodes, 2), dtype=np.float32)

#         # Build efficient lookups for hyperedge membership
#         self.hyperedges = [list(he) for he in hyperedges] # Ensure lists for indexing
#         self.num_hyperedges = len(self.hyperedges)
        
#         self.node_to_hyperedges = [[] for _ in range(self.num_nodes)]
#         for he_idx, he in enumerate(self.hyperedges):
#             for node_idx in he:
#                 self.node_to_hyperedges[node_idx].append(he_idx)
                
#         self.centroids = np.zeros((self.num_hyperedges, 2), dtype=np.float32)

#     def simulation_step(self, dt=0.02, k_attraction=0.05, k_repulsion=50.0, damping=0.95):
#         """Performs one step of the physics simulation."""
#         if self.num_nodes == 0:
#             return

#         forces = np.zeros((self.num_nodes, 2), dtype=np.float32)

#         # 1. Calculate Geometric Centroids (Dynamic)
#         for i, he in enumerate(self.hyperedges):
#             if he:
#                 member_positions = self.positions[he]
#                 self.centroids[i] = np.mean(member_positions, axis=0)

#         # 2. Calculate Node-Centroid Attraction Forces
#         for i in range(self.num_nodes):
#             for he_idx in self.node_to_hyperedges[i]:
#                 force_vec = self.centroids[he_idx] - self.positions[i]
#                 forces[i] += force_vec # k_attraction is applied later

#         forces *= k_attraction

#         # 3. Calculate Node-Node Repulsion (Approximated for performance)
#         # A full N^2 is too slow. We use a random sample.
#         # For a more robust solution, a Quadtree is needed.
#         num_samples = 100 # Tune this for balance of performance/quality
#         for i in range(self.num_nodes):
#             # Select random samples, excluding self
#             samples_idx = np.random.choice(self.num_nodes, num_samples, replace=False)
            
#             delta = self.positions[samples_idx] - self.positions[i]
#             distance_sq = np.sum(delta**2, axis=1)
#             distance_sq[distance_sq == 0] = 1e-6 # Avoid division by zero
            
#             # Simplified force calculation
#             force_magnitude = k_repulsion / distance_sq
#             repulsion_force = np.sum(delta * force_magnitude[:, np.newaxis], axis=0)
#             forces[i] -= repulsion_force / num_samples # Average the force

#         # 4. Update Physics (Euler Integration)
#         self.velocities += forces * dt
#         self.velocities *= damping
#         self.positions += self.velocities * dt

    

class LassoViewBox(pg.ViewBox):
    """ViewBox that emits a polygon drawn with Shift + left mouse."""
  
    sigLassoFinished = Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drawing = False
        self._path = QPainterPath()
        self._item = None

    def mousePressEvent(self, ev):
        if (
            ev.button() == Qt.LeftButton
            and ev.modifiers() & Qt.ShiftModifier
        ):
            self._drawing = True
            self._path = QPainterPath(self.mapToView(ev.pos()))
            pen = QPen(pg.mkColor("y"))
            pen.setWidth(2)
            pen.setCosmetic(True)
            self._item = pg.QtWidgets.QGraphicsPathItem()
            self._item.setPen(pen)
            self.addItem(self._item)
            ev.accept()
            return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drawing:
            self._path.lineTo(self.mapToView(ev.pos()))
            self._item.setPath(self._path)
            ev.accept()
            return

        super().mouseMoveEvent(ev)


    def mouseReleaseEvent(self, ev):
        if self._drawing and ev.button() == Qt.LeftButton:
            self._drawing = False
            self.removeItem(self._item)
            pts = [self.mapToView(ev.pos())]
            path = []
            for i in range(self._path.elementCount()):
                el = self._path.elementAt(i)
                path.append(QPointF(el.x, el.y))
            if len(path) > 2:
                self.sigLassoFinished.emit(path)
            ev.accept()
            return

        super().mouseReleaseEvent(ev)


class SpatialViewQDock(QDockWidget):
    """PyQtGraph-based spatial view with force-directed layout."""
    
    def __init__(self, bus: SelectionBus, parent=None):
        super().__init__("Spatial View", parent)
        self.bus = bus
        self.session: SessionModel | None = None
        self.embedding: np.ndarray | None = None # Will now be the *initial* embedding
        self.color_map: dict[str, str] = {}
        
        # --- NEW: Engine and Timer for dynamic layout ---
        self.engine: SimulationEngine | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(16)  # Target ~60 FPS
        self.timer.timeout.connect(self._update_frame)

        self.auto_stop_ms = 5000  # Stop after 5 seconds of running
        self.run_button = QPushButton("Pause Layout")
        self.run_button.clicked.connect(self.on_run_button_clicked)
        self.run_button.setEnabled(False) # Disabled until model is loaded


        # --- NEW: Track sample images for dynamic updates ---
        self.sample_image_items: dict[int, QGraphicsPixmapItem] = {} # Maps node_idx to item

        # --- Your existing setup (mostly unchanged) ---
        self.view = LassoViewBox()
        # self.view = GLViewWidget()   # OpenGL 3D view, but we’ll just use x/y
        # self.plot = self.view        # keep attribute names for rest of class
        self.view.sigLassoFinished.connect(self._on_lasso_select)
        self.plot = pg.PlotWidget(viewBox=self.view)
        self.plot.setBackground('#444444')



        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.run_button) # Add button at the top
        layout.addWidget(self.plot)
        self.setWidget(widget)

        self.scatter = None

        self.scatter_colors: np.ndarray | None = None
        self.scatter_symbols: np.ndarray | None = None

        self.bus.edgesChanged.connect(self._on_edges_changed)

    def start_simulation(self):
        """Starts or resumes the layout simulation."""
        if not self.timer.isActive():
            self.timer.start()
            self.run_button.setText("Pause Layout")
            print("Layout simulation started.")
            # Set a timer to automatically stop it
            QTimer.singleShot(self.auto_stop_ms, self.stop_simulation)

    def stop_simulation(self):
        """Pauses the layout simulation."""
        if self.timer.isActive():
            self.timer.stop()
            self.run_button.setText("Resume Layout")
            print("Layout simulation paused.")

    def on_run_button_clicked(self):
        self.auto_stop_ms = 60000*5
        """Toggles the simulation state when the user clicks the button."""
        if self.timer.isActive():
            self.stop_simulation()
        else:
            self.start_simulation()
            


    def set_model(self, session: SessionModel | None):
        # --- MODIFIED: This now initializes the simulation ---
        self.timer.stop()
        self.plot.clear()
        self._clear_image_items() # Your existing clear method
        
        if not session:
            self.session = None
            self.embedding = None
            self.scatter = None
            self.engine = None
            self.scatter_colors = None # Clear state
            self.scatter_symbols = None
            return

        self.session = session
        
        # 1. Calculate the INITIAL embedding (great starting point!)
        print("Calculating initial UMAP embedding...")
        self.embedding = umap.UMAP(n_components=2).fit_transform(session.features)
        
        # 2. Initialize the simulation engine with this layout
        print("Initializing simulation engine...")
        # We need the hyperedges as a list of lists of node indices
        hyperedge_list = [list(he) for he in session.hyperedges.values()]
        self.engine = SimulationEngine(self.embedding, hyperedge_list)
        
        self.scatter_colors = np.array(['#808080'] * self.engine.num_nodes)
        self.scatter_symbols = np.array(['o'] * self.engine.num_nodes, dtype=object)

        # 3. Create the scatter plot item

        # self.scatter = GLScatterPlotItem(
        #     pos=self.engine.positions, size=7.0,
        #     color=pg.glColor('#808080'), pxMode=False)


        self.scatter = pg.ScatterPlotItem(
            pos=self.engine.positions,
            size=9,
            brush=[pg.mkBrush(c) for c in self.scatter_colors], # Use initial colors
            symbol=self.scatter_symbols, # Use initial symbols
            pen=None,
            pxMode=True,
            useOpenGL=True
        )
        self.plot.addItem(self.scatter)

        # # 4. Setup colors (your existing logic)
        # edges = list(session.hyperedges)
        # print(edges)
        # self.color_map = {
        #     n: pg.mkColor(pg.intColor(i, hues=len(edges))).name()
        #     for i, n in enumerate(edges)
        # }
        self.color_map = session.edge_colors.copy() 

        self.run_button.setEnabled(True)
        self.start_simulation() # Use the new method to start

        # 5. Start the simulation!
        # print("Starting simulation.")
        # self.timer.start()

          
    def _update_frame(self):
        """The core animation loop."""
        if not self.engine or not self.scatter:
            return
                
        # 1. Advance the simulation
        self.engine.simulation_step()
        
        # 2. Redraw the scatter plot with updated positions and current styles
        self._refresh_scatter_plot()
        
        # 3. Update the positions of any sample images
        for idx, item in self.sample_image_items.items():
            pos = self.engine.positions[idx]
            item.setPos(pos[0], pos[1])

    def _refresh_scatter_plot(self):
        """
        Updates the scatter plot with the current state of positions, 
        colors, and symbols. This is the single point of truth for drawing.
        """
        if not all([self.engine, self.scatter, self.scatter_colors is not None, self.scatter_symbols is not None]):
            return
            
        # Convert color strings to QBrush objects for pyqtgraph
        brushes = [pg.mkBrush(c) for c in self.scatter_colors]
        
        # The single, correct setData call
        self.scatter.setData(
            pos=self.engine.positions, 
            brush=brushes, 
            symbol=self.scatter_symbols
        )
        # self.scatter.setData(
        #     pos=self.engine.positions,
        #     color=[pg.glColor(c) for c in self.scatter_colors]
        # )
            
    def _on_lasso_select(self, pts: list[QPointF]):
        # --- MODIFIED: Use engine positions for selection ---
        if self.engine is None:
            return
        poly = [(p.x(), p.y()) for p in pts]
        path = MplPath(poly)
        # Use the CURRENT positions from the engine, not the static embedding
        idxs = np.nonzero(path.contains_points(self.engine.positions))[0]
        self.bus.set_images(list(map(int, idxs)))

    def _on_edges_changed(self, names: list[str]):
        """
        Updates the desired appearance of the scatter plot by modifying the
        class's state variables (self.scatter_colors and self.scatter_symbols).
        It does NOT call setData directly; it calls _refresh_scatter_plot at the end.
        """
        # 1. --- GUARD CLAUSES ---
        # Ensure everything is initialized before proceeding.
        if not all([self.session, self.engine, self.scatter_colors is not None, self.scatter_symbols is not None]):
            return

        # 2. --- CLEAR PREVIOUS STATE ---
        self._clear_image_items()
        # Reset colors and symbols to their default state.
        # We use [:] to modify the array in-place.
        self.scatter_colors[:] = '#808080'
        self.scatter_symbols[:] = 'o'

        # 3. --- HANDLE "NO SELECTION" ---
        # If the selection is cleared, we're done. Just refresh the view.
        if not names:
            self._refresh_scatter_plot()
            return

        # 4. --- CALCULATE NEW STATE BASED ON SELECTION ---
        main_edge_name = names[0]
        selected_nodes = self.session.hyperedges.get(main_edge_name, set())

        # Find all overlapping hyperedges
        overlapping_edges = []
        if selected_nodes:
            for edge in self.session.hyperedges:
                if edge == main_edge_name:
                    continue
                if selected_nodes & self.session.hyperedges.get(edge, set()):
                    overlapping_edges.append(edge)

        # Rule 3: Color "Neighboring" Nodes
        for ov_name in overlapping_edges:
            ov_nodes = list(self.session.hyperedges.get(ov_name, set()))
            ov_color = self.color_map.get(ov_name, 'blue')
            # Modify the class-level state array
            self.scatter_colors[ov_nodes] = ov_color

        # Rules 1 & 2: Process nodes within the main selection
        nodes_to_show_pure = []
        nodes_to_show_intersect = {}
        if selected_nodes:
            main_color = self.color_map.get(main_edge_name, 'red')
            for node_idx in selected_nodes:
                member_of_edges = self.session.image_mapping.get(node_idx, set())
                
                if len(member_of_edges) == 1:
                    # Rule 1: Node is "Pure"
                    self.scatter_colors[node_idx] = main_color
                    self.scatter_symbols[node_idx] = 'o' # Circle
                    if len(nodes_to_show_pure) < 3:
                        nodes_to_show_pure.append(node_idx)
                else:
                    # Rule 2: Node is "Intersecting"
                    self.scatter_symbols[node_idx] = '+'  # Star shape

                    other_edges = member_of_edges - {main_edge_name}
                    if other_edges:
                        other_edge_name = other_edges.pop()
                        intersect_color = self.color_map.get(other_edge_name, 'magenta')
                        self.scatter_colors[node_idx] = intersect_color
                        if len(nodes_to_show_intersect) < 3:
                            nodes_to_show_intersect[node_idx] = intersect_color
                    else:
                        self.scatter_colors[node_idx] = 'white' # Fallback
        
        # 5. --- UPDATE SAMPLE IMAGES ---
        self._add_sample_images(nodes_to_show_pure, self.color_map.get(main_edge_name, 'yellow'))
        for idx, color in nodes_to_show_intersect.items():
            self._add_sample_images([idx], color)

        # 6. --- TRIGGER A REDRAW ---
        # Now that the state arrays are updated, tell the renderer to draw them.
        # This ensures the view updates even if the simulation is paused.
        self._refresh_scatter_plot()

    def _add_sample_images(self, idxs: list[int], color: str):
        # --- MODIFIED: To store items in our new dictionary ---
        for idx in idxs:
            if idx in self.sample_image_items: # Don't add duplicates
                continue
            # ... (your existing pixmap loading code) ...
            path = self.session.im_list[idx]
            pix = QPixmap(path)
            if pix.isNull(): continue
            pix = pix.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            item = QGraphicsPixmapItem(pix)
            item.setOffset(-pix.width()/2, -pix.height()/2)
            # Set initial position from the engine
            item.setPos(self.engine.positions[idx, 0], self.engine.positions[idx, 1])
            item.setFlag(QGraphicsPixmapItem.ItemIgnoresTransformations)
            
            rect = QGraphicsRectItem(-pix.width()/2, -pix.height()/2, pix.width(), pix.height(), item) 
            # rect.setOffset(-pix.width()/2, -pix.height()/2)
            pen = QPen(QColor(color))
            pen.setWidth(2) # Make it thicker
            pen.setCosmetic(True)
            rect.setPen(pen)
            
            self.plot.addItem(item)
            self.sample_image_items[idx] = item # Store by index

    def _clear_image_items(self):
        # --- MODIFIED: To work with the dictionary ---
        for item in self.sample_image_items.values():
            if item.scene():
                self.plot.removeItem(item)
        self.sample_image_items.clear()
        
    # def closeEvent(self, event):
    #     # --- NEW: Ensure the timer is stopped on close ---
    #     self.timer.stop()
    #     super().closeEvent(event)

    def closeEvent(self, event):
        self.stop_simulation() # Use the new method
        super().closeEvent(event)