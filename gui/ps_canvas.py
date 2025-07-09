"""
Canvas interactif pour l'affichage et la sélection dans le Pattern Spectra.
"""

from PyQt5.QtCore import pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib import cm
import numpy as np
from core.pattern_spectra import (plot_ps_with_highlights, get_bin_coordinates_from_point, 
                                get_bins_in_polygon_selection, compute_nodes_from_bins)


class PSCanvas(FigureCanvas):
    """Canvas matplotlib interactif pour le Pattern Spectra."""
    
    # Signal émis quand des bins sont sélectionnés
    bins_selected = pyqtSignal(object)  # List de nœuds sélectionnés
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(6, 6))
        super().__init__(self.figure)
        self.setParent(parent)
        
        # Données du PS
        self.ps_data = None
        self.current_highlight_bins = None
        self.current_highlight_nodes = None
        
        # Variables pour la sélection
        self.selection_mode = 'click'  # 'click' ou 'polygon'
        self.selected_bins = set()
        self.selected_nodes = []
        
        # Variables pour la sélection par clic maintenu
        self.is_mouse_pressed = False
        self.last_selected_bin = None
        
        # Variables pour la sélection par polygone
        self.polygon_points = []
        self.is_drawing_polygon = False
        self.selection_polygon = None
        
        # Cache pour optimiser le rendu
        self.base_plot_cache = None
        self.last_ps_data_hash = None
        
        # Connecter les événements
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('key_press_event', self.on_key_press)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        self.ax = None
        
    def set_ps_data(self, ps_data):
        """Définit les données du pattern spectra."""
        self.ps_data = ps_data
        self.clear_selection()
        
    def set_selection_mode(self, mode):
        """Change le mode de sélection ('click' ou 'polygon')."""
        self.selection_mode = mode
        self.clear_selection()
        
    def update_display(self, highlight_bins=None, highlight_nodes=None):
        """Met à jour l'affichage du pattern spectra."""
        if self.ps_data is None:
            return
            
        # Sauvegarder la surbrillance externe (depuis sélection image)
        self.current_highlight_bins = highlight_bins
        self.current_highlight_nodes = highlight_nodes
        
        # Effacer la figure
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        # Afficher le PS avec surbrillance externe
        pcm = plot_ps_with_highlights(self.ax, self.ps_data, highlight_bins, highlight_nodes)
        
        # Afficher la sélection interne (bins sélectionnés dans le PS)
        if self.selected_bins:
            self.draw_selected_bins()
        
        # Afficher le polygone de sélection en cours
        if self.selection_polygon:
            self.ax.add_patch(self.selection_polygon)
        
        # Ajouter la colorbar
        self.figure.colorbar(pcm, ax=self.ax, label='Volume')
        
        # Configuration
        selection_info = ""
        if self.selected_bins:
            selection_info = f" - {len(self.selected_bins)} bins sélectionnés"
        elif self.is_drawing_polygon and len(self.polygon_points) > 0:
            selection_info = f" - Polygone en cours ({len(self.polygon_points)} points)"
            
        self.ax.set_title(f'Pattern Spectra{selection_info}')
        
        self.draw()
        
    def draw_selected_bins(self):
        """Dessine les bins sélectionnés en surbrillance."""
        if not self.selected_bins or self.ps_data is None:
            return
            
        bins1 = self.ps_data['bins1']
        bins2 = self.ps_data['bins2']
        hist2d = self.ps_data['hist2d']
        
        # Créer un masque pour les bins sélectionnés
        selected_mask = np.zeros_like(hist2d)
        for (i, j) in self.selected_bins:
            if 0 <= i < hist2d.shape[0] and 0 <= j < hist2d.shape[1]:
                selected_mask[i, j] = hist2d[i, j]
        
        # Afficher en surbrillance avec une couleur distincte
        if selected_mask.sum() > 0:
            pcm_selected = self.ax.pcolormesh(
                bins1, bins2, selected_mask.T,
                cmap=cm.get_cmap('plasma'), 
                shading='auto', 
                alpha=0.8
            )
            
        # Dessiner les contours pour mieux voir la sélection
        contour_mask = np.zeros_like(hist2d)
        for (i, j) in self.selected_bins:
            if 0 <= i < hist2d.shape[0] and 0 <= j < hist2d.shape[1]:
                contour_mask[i, j] = 1
                
        if contour_mask.sum() > 0:
            self.ax.contour(bins1[:-1], bins2[:-1], contour_mask.T, levels=[0.5], colors='yellow', linewidths=2)
        
    def on_mouse_press(self, event):
        """Gestionnaire de clic de souris."""
        if event.inaxes != self.ax or self.ps_data is None:
            return
            
        if self.selection_mode == 'click':
            self.handle_click_selection(event)
        elif self.selection_mode == 'polygon':
            self.handle_polygon_selection(event)
            
    def on_mouse_release(self, event):
        """Gestionnaire de relâchement de souris."""
        if event.button == 1:  # Relâchement du clic gauche
            self.is_mouse_pressed = False
            self.last_selected_bin = None
        
    def on_key_press(self, event):
        """Gestionnaire de touches du clavier."""
        if event.key == 'escape':
            self.cancel_polygon_selection()
        elif event.key == 'delete':
            self.clear_selection()
        elif event.key == 'c':
            self.set_selection_mode('click')
        elif event.key == 'p':
            self.set_selection_mode('polygon')
            
    def on_mouse_move(self, event):
        """Gestionnaire de mouvement de souris pour afficher des informations et sélection par glissement."""
        if event.inaxes == self.ax and self.ps_data is not None:
            # Trouver le bin sous le curseur
            bin_coord = get_bin_coordinates_from_point(self.ps_data, event.xdata, event.ydata)
            
            # Sélection par glissement en mode clic
            if (self.is_mouse_pressed and self.selection_mode == 'click' and 
                bin_coord and bin_coord != self.last_selected_bin):
                self.last_selected_bin = bin_coord
                if bin_coord not in self.selected_bins:
                    self.selected_bins.add(bin_coord)
                    self.update_selection()
            
            # Afficher des informations sur le bin
            if bin_coord and bin_coord in self.ps_data['bin_contributions']:
                num_nodes = len(self.ps_data['bin_contributions'][bin_coord])
                volume = sum(vol for _, vol in self.ps_data['bin_contributions'][bin_coord])
                self.setToolTip(f"Bin ({bin_coord[0]}, {bin_coord[1]}): {num_nodes} nœuds, volume: {volume:.4f}")
            else:
                self.setToolTip("")

    def handle_click_selection(self, event):
        """Gestion de la sélection par clic."""
        if event.button == 1:  # Clic gauche
            self.is_mouse_pressed = True
            
            # Trouver le bin correspondant
            bin_coord = get_bin_coordinates_from_point(self.ps_data, event.xdata, event.ydata)
            
            if bin_coord:
                self.last_selected_bin = bin_coord
                if bin_coord in self.selected_bins:
                    # Désélectionner
                    self.selected_bins.remove(bin_coord)
                else:
                    # Sélectionner
                    self.selected_bins.add(bin_coord)
                
                self.update_selection()
                
        elif event.button == 3:  # Clic droit
            self.clear_selection()
            
    def handle_polygon_selection(self, event):
        """Gestion de la sélection par polygone (comme sur l'image)."""
        if event.button == 1:  # Clic gauche
            if not self.is_drawing_polygon:
                # Commencer un nouveau polygone
                self.start_polygon_selection(event.xdata, event.ydata)
            else:
                # Ajouter un point
                self.add_polygon_point(event.xdata, event.ydata)
                
        elif event.button == 3:  # Clic droit - terminer le polygone
            if self.is_drawing_polygon:
                self.finish_polygon_selection()
            else:
                self.clear_selection()
                
    def start_polygon_selection(self, x, y):
        """Commence la sélection par polygone."""
        self.is_drawing_polygon = True
        self.polygon_points = [(x, y)]
        
    def add_polygon_point(self, x, y):
        """Ajoute un point au polygone."""
        if self.is_drawing_polygon:
            self.polygon_points.append((x, y))
            self.update_polygon_display()
            
    def finish_polygon_selection(self):
        """Termine la sélection par polygone."""
        if self.is_drawing_polygon and len(self.polygon_points) >= 3:
            self.is_drawing_polygon = False
            
            # Trouver les bins dans le polygone (fermer le polygone pour la sélection)
            closed_polygon = self.polygon_points + [self.polygon_points[0]]
            selected_bins = get_bins_in_polygon_selection(self.ps_data, closed_polygon)
            
            # Ajouter à la sélection (ou remplacer selon le mode)
            self.selected_bins.update(selected_bins)
            
            self.update_selection()
            
        self.cancel_polygon_selection()
        
    def cancel_polygon_selection(self):
        """Annule la sélection par polygone."""
        self.is_drawing_polygon = False
        self.polygon_points = []
        if self.selection_polygon:
            self.selection_polygon.remove()
            self.selection_polygon = None
        self.draw()
        
    def update_polygon_display(self):
        """Met à jour l'affichage du polygone en cours."""
        if len(self.polygon_points) < 2:
            return
            
        # Supprimer l'ancien polygone
        if self.selection_polygon:
            self.selection_polygon.remove()
            
        # Créer le nouveau polygone 
        polygon_points = self.polygon_points.copy()
        
        # Pour l'affichage, fermer le polygone si on a plus de 2 points
        display_points = polygon_points.copy()
        if len(display_points) >= 3:
            display_points.append(display_points[0])  # Fermer visuellement
            
        self.selection_polygon = MplPolygon(
            display_points, 
            closed=(len(display_points) >= 4), 
            fill=False, 
            edgecolor='orange', 
            linewidth=2,
            linestyle='--',
            alpha=0.7
        )
        self.ax.add_patch(self.selection_polygon)
        self.draw()
        
    def auto_finish_polygon_selection(self):
        """Termine automatiquement la sélection du polygone après chaque ajout de point."""
        if len(self.polygon_points) >= 3:
            # Sélectionner les bins dans le polygone fermé
            closed_polygon = self.polygon_points + [self.polygon_points[0]]
            selected_bins = get_bins_in_polygon_selection(self.ps_data, closed_polygon)
            
            # Remplacer la sélection précédente
            self.selected_bins = selected_bins
            self.update_selection()
        
    def update_selection(self):
        """Met à jour la sélection et émet le signal."""
        # Calculer les nœuds correspondants
        self.selected_nodes = compute_nodes_from_bins(self.ps_data, self.selected_bins)
        
        # Émettre le signal
        self.bins_selected.emit(self.selected_nodes)
        
        # Mettre à jour l'affichage
        self.update_display(self.current_highlight_bins, self.current_highlight_nodes)
        
    def clear_selection(self):
        """Efface toute la sélection."""
        self.selected_bins.clear()
        self.selected_nodes = []
        self.cancel_polygon_selection()
        
        if self.ps_data:
            self.update_display(self.current_highlight_bins, self.current_highlight_nodes)
        
        # Émettre un signal vide
        self.bins_selected.emit([])
        
    def get_selected_nodes(self):
        """Retourne les nœuds actuellement sélectionnés."""
        return self.selected_nodes.copy()
    
    def get_selection_info(self):
        """Retourne des informations sur la sélection actuelle."""
        if self.selected_bins:
            num_bins = len(self.selected_bins)
            num_nodes = len(self.selected_nodes)
            return f"{num_bins} bins sélectionnés ({num_nodes} nœuds)"
        elif self.is_drawing_polygon:
            num_points = len(self.polygon_points)
            return f"Polygone en cours ({num_points} points - clic droit pour terminer)"
        else:
            return "Aucune sélection"
