from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QPolygonF, QBrush, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from shapely.geometry import Polygon

class ImageCanvas(QGraphicsView):
    # Signal emitted when a polygon is selected
    polygon_selected = pyqtSignal(object)  # Shapely Polygon
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.sequence = []
        self.pixmap_item = None
        self.current_index = 0
        self.setRenderHint(QPainter.Antialiasing)
        
        # Size policy to maximize space usage
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Allow zoom with mouse wheel
        self.setDragMode(QGraphicsView.RubberBandDrag)
        
        # Variables for polygon selection
        self.polygon_points = []
        self.polygon_item = None
        self.is_drawing_polygon = False
        self.current_polygon = None
        
        # Variables for displaying selected nodes
        self.ps_data = None
        self.selected_nodes = []
        self.node_overlays = []  # Overlays for selected nodes
        self.cached_masks = None  # Cache des masques calculés
        self.last_nodes_hash = None  # Hash des derniers nœuds pour détecter les changements
        self.use_unique_colors = False  # Flag to enable unique colors for nodes
        self.cached_colored_masks = None  # Cache des masques colorés
        
        self.load_fixed_sequence()

    def load_fixed_sequence(self):
        base_path = "data/sits_example"
        files = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".png")])
        for f in files:
            img = imread(f)
            if img.ndim == 3:
                img = (rgb2gray(img) * 255).astype(np.uint8)
            self.sequence.append(img)

        self.display_index(0)

    def display_index(self, idx):
        if not self.sequence:
            return
        self.current_index = idx
        img = self.sequence[idx]
        h, w = img.shape
        
        # Créer l'image Qt avec la bonne stride
        bytes_per_line = w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)

        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(pix)
        else:
            self.pixmap_item.setPixmap(pix)

        self.scene.setSceneRect(0, 0, w, h)
        # Ajuster automatiquement la vue pour optimiser l'affichage
        self.fit_image_to_view()
        
        # Mettre à jour les overlays pour le nouveau timestep
        self.update_node_overlays()
    
    def fit_image_to_view(self):
        """Ajuster l'image pour qu'elle prenne le maximum d'espace disponible"""
        if self.pixmap_item is not None:
            # Utiliser fitInView pour ajuster automatiquement
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
    
    def resizeEvent(self, event):
        """Réajuster l'image quand la fenêtre est redimensionnée"""
        super().resizeEvent(event)
        if self.pixmap_item is not None:
            self.fit_image_to_view()
    
    def fit_to_window(self):
        """Ajuster l'image pour qu'elle tienne dans la fenêtre"""
        if self.pixmap_item is not None:
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
    
    def reset_zoom(self):
        """Réinitialiser le zoom à 100%"""
        if self.pixmap_item is not None:
            self.resetTransform()
    
    def zoom_to_fit(self):
        """Zoomer pour que l'image occupe le maximum d'espace sans déborder"""
        if self.pixmap_item is not None:
            # Calculer le facteur de zoom optimal
            view_rect = self.viewport().rect()
            scene_rect = self.scene.itemsBoundingRect()
            
            if scene_rect.width() > 0 and scene_rect.height() > 0:
                scale_x = view_rect.width() / scene_rect.width()
                scale_y = view_rect.height() / scene_rect.height()
                scale = min(scale_x, scale_y)
                
                self.resetTransform()
                self.scale(scale, scale)
                self.centerOn(scene_rect.center())
    
    def mousePressEvent(self, event):
        """Mouse click handling for polygon selection."""
        if event.button() == Qt.LeftButton:
            # Convert coordinates from view to scene
            scene_pos = self.mapToScene(event.pos())
            
            if not self.is_drawing_polygon:
                # Start a new polygon
                self.start_polygon(scene_pos)
            else:
                # Add a point to the current polygon
                self.add_polygon_point(scene_pos)
                
        elif event.button() == Qt.RightButton:
            # Finish the current polygon
            if self.is_drawing_polygon:
                self.finish_polygon()
            
        super().mousePressEvent(event)
    
    def start_polygon(self, scene_pos):
        """Starts selection of a new polygon."""
        self.is_drawing_polygon = True
        self.polygon_points = [scene_pos]
        
        # Supprimer l'ancien polygone s'il existe
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
            self.polygon_item = None
    
    def add_polygon_point(self, scene_pos):
        """Ajoute un point au polygone en cours."""
        if self.is_drawing_polygon:
            self.polygon_points.append(scene_pos)
            self.update_polygon_display()
    
    def finish_polygon(self):
        """Termine la sélection du polygone."""
        if self.is_drawing_polygon and len(self.polygon_points) >= 3:
            self.is_drawing_polygon = False
            
            # Créer le polygone Shapely
            coords = [(p.x(), p.y()) for p in self.polygon_points]
            self.current_polygon = Polygon(coords)
            
            # Émettre le signal
            self.polygon_selected.emit(self.current_polygon)
            
            # Mettre à jour l'affichage
            self.update_polygon_display()
        else:
            # Annuler si pas assez de points
            self.cancel_polygon()
    
    def cancel_polygon(self):
        """Annule la sélection du polygone en cours."""
        self.is_drawing_polygon = False
        self.polygon_points = []
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
            self.polygon_item = None
    
    def update_polygon_display(self):
        """Met à jour l'affichage du polygone."""
        if len(self.polygon_points) < 2:
            return
            
        # Supprimer l'ancien polygone
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
        
        # Créer le nouveau polygone Qt
        qt_polygon = QPolygonF(self.polygon_points)
        
        # Ajouter le polygone à la scène
        pen = QPen(Qt.red, 2)
        if self.is_drawing_polygon:
            pen.setStyle(Qt.DashLine)
        
        self.polygon_item = self.scene.addPolygon(qt_polygon, pen)
    
    def clear_polygon(self):
        """Efface le polygone actuel."""
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
            self.polygon_item = None
        
        self.polygon_points = []
        self.is_drawing_polygon = False
        self.current_polygon = None
    
    def keyPressEvent(self, event):
        """Gestion des touches du clavier."""
        if event.key() == Qt.Key_Escape:
            # Annuler la sélection en cours
            self.cancel_polygon()
        elif event.key() == Qt.Key_Delete:
            # Supprimer le polygone actuel
            self.clear_polygon()
        
        super().keyPressEvent(event)
    
    def set_ps_data(self, ps_data):
        """Sets the pattern spectra data for node display."""
        self.ps_data = ps_data
        
    def set_color_nodes_mode(self, use_unique_colors):
        """Sets whether to use unique colors for each node."""
        if self.use_unique_colors != use_unique_colors:
            self.use_unique_colors = use_unique_colors
            # Invalidate colored masks cache when mode changes
            self.cached_colored_masks = None
            # Update display if there are selected nodes
            if self.selected_nodes:
                self.update_node_overlays()
        
    def set_selected_nodes(self, selected_nodes):
        """Sets the selected nodes to display in highlight."""
        self.selected_nodes = selected_nodes
        
        # Invalidate cache if nodes have changed
        nodes_hash = hash(tuple(sorted(selected_nodes))) if selected_nodes else None
        if nodes_hash != self.last_nodes_hash:
            self.cached_masks = None
            self.cached_colored_masks = None  # Also invalidate colored cache
            self.last_nodes_hash = nodes_hash
            
        self.update_node_overlays()
        
    def update_node_overlays(self):
        """Updates the overlay display for selected nodes (optimized version)."""
        # Remove old overlays
        for overlay in self.node_overlays:
            self.scene.removeItem(overlay)
        self.node_overlays.clear()
        
        if not self.selected_nodes or not self.ps_data or not self.sequence:
            return
        
        if self.use_unique_colors:
            # Use unique colors for each node
            if self.cached_colored_masks is None:
                from core.pattern_spectra import compute_node_masks_per_timestep_with_colors
                cube_shape = (len(self.sequence), self.sequence[0].shape[0], self.sequence[0].shape[1])
                # Call with use_unique_colors=True to get colored overlays
                _, self.cached_colored_masks = compute_node_masks_per_timestep_with_colors(
                    self.ps_data, self.selected_nodes, cube_shape, use_unique_colors=True
                )
            
            # Display colored overlay for current index
            if self.current_index < len(self.cached_colored_masks):
                colored_overlay = self.cached_colored_masks[self.current_index]
                self.add_colored_overlay(colored_overlay)
        else:
            # Use single color (red) for all nodes
            if self.cached_masks is None:
                from core.pattern_spectra import compute_node_masks_per_timestep_optimized
                cube_shape = (len(self.sequence), self.sequence[0].shape[0], self.sequence[0].shape[1])
                self.cached_masks = compute_node_masks_per_timestep_optimized(
                    self.ps_data, self.selected_nodes, cube_shape
                )
            
            # Display mask for current index
            if self.current_index < len(self.cached_masks):
                mask = self.cached_masks[self.current_index]
                self.add_mask_overlay_optimized(mask)
            
    def add_colored_overlay(self, colored_overlay):
        """Add a colored overlay with unique colors for each node."""
        if colored_overlay is None:
            return
            
        h, w = colored_overlay.shape[:2]
        
        # colored_overlay should be an RGBA image (h, w, 4)
        # Convert to QImage format
        qimg = QImage(colored_overlay.data, w, h, w * 4, QImage.Format_RGBA8888)
        overlay_pixmap = QPixmap.fromImage(qimg)
        
        # Add to scene
        overlay_item = self.scene.addPixmap(overlay_pixmap)
        overlay_item.setZValue(1)  # Above the main image
        self.node_overlays.append(overlay_item)
            
    def add_mask_overlay_optimized(self, mask):
        """Optimized version to add a colored overlay."""
        if not mask.any():
            return
            
        h, w = mask.shape
        
        # Optimization: create image directly with correct dimensions
        # and avoid unnecessary copies
        overlay_img = np.zeros((h, w, 4), dtype=np.uint8)
        overlay_img[mask, :] = [255, 0, 0, 120]  # Red semi-transparent, more efficient
        
        # Optimized conversion
        qimg = QImage(overlay_img.data, w, h, w * 4, QImage.Format_RGBA8888)
        overlay_pixmap = QPixmap.fromImage(qimg)
        
        # Ajouter à la scène
        overlay_item = self.scene.addPixmap(overlay_pixmap)
        overlay_item.setZValue(1)  # Au-dessus de l'image principale
        self.node_overlays.append(overlay_item)
