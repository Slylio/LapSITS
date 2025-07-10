from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QSlider, 
                             QHBoxLayout, QPushButton, QLabel, QSplitter, QButtonGroup, QRadioButton, QCheckBox,
                             QMenuBar, QAction, QFileDialog, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt
from gui.image_canvas import ImageCanvas
from gui.ps_canvas import PSCanvas
from core.pattern_spectra import compute_global_ps, compute_local_ps_highlight
from core.attributes import ATTR_FUNCS, BINS
from core.pattern_spectra import compute_2d_ps_with_tracking
import numpy as np
import os
from skimage.io import imread
from skimage.color import rgb2gray

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LapSITS")
        self.setGeometry(100, 100, 1400, 800)  # Larger window for PS

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Pattern spectra data
        self.ps_data = None
        self.current_polygon = None
        self.current_selected_nodes = []  # Nodes selected from PS
        
        self.setup_menu()
        self.setup_ui()
        self.setup_connections()
        self.compute_initial_ps()

    def setup_menu(self):
        """Configure the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Open temporal sequence action
        open_sequence_action = QAction('Open Temporal Sequence...', self)
        open_sequence_action.setShortcut('Ctrl+O')
        open_sequence_action.setStatusTip('Open a temporal sequence of PNG files')
        open_sequence_action.triggered.connect(self.open_temporal_sequence)
        file_menu.addAction(open_sequence_action)
        
        file_menu.addSeparator()
        
        # Reset to default sequence
        reset_sequence_action = QAction('Reset to Default Sequence', self)
        reset_sequence_action.setStatusTip('Reset to the default example sequence')
        reset_sequence_action.triggered.connect(self.reset_to_default_sequence)
        file_menu.addAction(reset_sequence_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        # Reset PS zoom action
        reset_zoom_action = QAction('Reset PS Zoom', self)
        reset_zoom_action.setShortcut('Ctrl+R')
        reset_zoom_action.setStatusTip('Reset Pattern Spectra zoom to full view')
        reset_zoom_action.triggered.connect(self.reset_ps_zoom)
        view_menu.addAction(reset_zoom_action)

    def setup_ui(self):
        """User interface configuration."""
        # Horizontal splitter to divide image and PS
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left part: Image and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        self.canvas_view = ImageCanvas()
        self.canvas_view.setMinimumWidth(400)
        self.canvas_view.setMinimumHeight(400)
        
        # Connect polygon selection
        self.canvas_view.polygon_selected.connect(self.on_polygon_selected)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.canvas_view.sequence) - 1)
        self.slider.valueChanged.connect(self.on_slider_changed)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        self.btn_compute_ps = QPushButton("Recalculate PS")
        self.btn_compute_ps.clicked.connect(self.compute_initial_ps)
        self.btn_clear_selection = QPushButton("Clear selection")
        self.btn_clear_selection.clicked.connect(self.clear_all_selections)
        
        # Radio buttons for PS selection mode
        self.ps_mode_group = QButtonGroup()
        self.radio_click = QRadioButton("Click")
        self.radio_polygon = QRadioButton("Polygon")
        self.radio_click.setChecked(True)
        self.ps_mode_group.addButton(self.radio_click, 0)
        self.ps_mode_group.addButton(self.radio_polygon, 1)
        
        # Checkbox to show nodes
        from PyQt5.QtWidgets import QCheckBox
        self.checkbox_show_nodes = QCheckBox("Show nodes")
        self.checkbox_show_nodes.setChecked(False)
        self.checkbox_show_nodes.stateChanged.connect(self.on_show_nodes_changed)
        
        # Checkbox to color nodes with unique colors
        self.checkbox_color_nodes = QCheckBox("Color nodes")
        self.checkbox_color_nodes.setChecked(False)
        self.checkbox_color_nodes.setEnabled(False)  # Disabled by default
        self.checkbox_color_nodes.stateChanged.connect(self.on_color_nodes_changed)
        
        controls_layout.addWidget(QLabel("PS Mode:"))
        controls_layout.addWidget(self.radio_click)
        controls_layout.addWidget(self.radio_polygon)
        controls_layout.addWidget(self.checkbox_show_nodes)
        controls_layout.addWidget(self.checkbox_color_nodes)
        controls_layout.addWidget(self.btn_compute_ps)
        controls_layout.addWidget(self.btn_clear_selection)
        controls_layout.addStretch()
        
        left_layout.addWidget(self.canvas_view)
        left_layout.addWidget(self.slider)
        left_layout.addLayout(controls_layout)
        left_widget.setLayout(left_layout)
        
        # Partie droite : Pattern Spectra
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        self.ps_label = QLabel("Pattern Spectra Global")
        self.ps_label.setAlignment(Qt.AlignCenter)
        
        # Dropdowns pour sélectionner les attributs du PS 2D
        self.attr1_combo = QComboBox()
        self.attr2_combo = QComboBox()
        self.attr1_combo.addItems(list(ATTR_FUNCS.keys()))
        self.attr2_combo.addItems(list(ATTR_FUNCS.keys()))
        self.attr1_combo.setCurrentIndex(0)
        self.attr2_combo.setCurrentIndex(1 if len(ATTR_FUNCS)>1 else 0)
        self.attr1_combo.currentTextChanged.connect(self.compute_initial_ps)
        self.attr2_combo.currentTextChanged.connect(self.compute_initial_ps)
        attr_layout = QHBoxLayout()
        attr_layout.addWidget(QLabel("Axe X:"))
        attr_layout.addWidget(self.attr1_combo)
        attr_layout.addWidget(QLabel("Axe Y:"))
        attr_layout.addWidget(self.attr2_combo)
        right_layout.addLayout(attr_layout)
        
        # Canvas PS interactif
        self.ps_canvas = PSCanvas()
        self.ps_canvas.setMinimumWidth(400)
        self.ps_canvas.setMinimumHeight(400)
        
        right_layout.addWidget(self.ps_label)
        right_layout.addWidget(self.ps_canvas)
        right_widget.setLayout(right_layout)
        
        # Ajouter les widgets au splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        
        # Proportions : 60% pour l'image, 40% pour le PS
        main_splitter.setStretchFactor(0, 6)
        main_splitter.setStretchFactor(1, 4)
        
        # Layout principal
        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
        self.central_widget.setLayout(main_layout)

    def setup_connections(self):
        """Configure les connexions entre les widgets."""
        # Connexion pour le changement de mode PS
        self.ps_mode_group.buttonClicked.connect(self.on_ps_mode_changed)
        
        # Connexion pour la sélection de bins dans le PS
        self.ps_canvas.bins_selected.connect(self.on_bins_selected)

    def compute_initial_ps(self):
        """Calcule le pattern spectra initial."""
        try:
            # Récupérer la séquence d'images
            if not self.canvas_view.sequence:
                return
            
            # Vérifier que toutes les images ont la même taille
            if len(self.canvas_view.sequence) > 1:
                first_shape = self.canvas_view.sequence[0].shape
                for i, img in enumerate(self.canvas_view.sequence[1:], 1):
                    if img.shape != first_shape:
                        QMessageBox.critical(
                            self, 
                            "Error", 
                            f"All images must have the same dimensions!\n"
                            f"Image 1: {first_shape}\n"
                            f"Image {i+1}: {img.shape}\n"
                            f"Please load images with consistent dimensions."
                        )
                        return
                
            cube = np.array(self.canvas_view.sequence)
            print(f"Calcul du PS pour cube de forme: {cube.shape}")
            
            # Calcule le PS global 2D avec attributs dynamiques
            # Construire l'arbre et altitudes via compute_global_ps
            base_data = compute_global_ps(cube)
            tree = base_data['tree']
            altitudes = base_data['altitudes']
            # Calculer les attributs selon sélection
            attr1 = ATTR_FUNCS[self.attr1_combo.currentText()](tree, cube)
            attr2 = ATTR_FUNCS[self.attr2_combo.currentText()](tree, cube)
            bins1 = BINS[self.attr1_combo.currentText()]
            bins2 = BINS[self.attr2_combo.currentText()]
            # Calculer le pattern spectra 2D avec tracking
            tree, hist2d, bins1, bins2, node_to_bin2d, bin_contributions = \
                compute_2d_ps_with_tracking(tree, altitudes, attr1, attr2, bins1, bins2)
            self.ps_data = {
                'tree': tree,
                'altitudes': altitudes,
                'hist2d': hist2d,
                'bins1': bins1,
                'bins2': bins2,
                'node_to_bin2d': node_to_bin2d,
                'bin_contributions': bin_contributions
            }
            
            # Configurer le canvas PS
            self.ps_canvas.set_ps_data(self.ps_data)
            
            # Transmettre les données PS à l'image canvas
            self.canvas_view.set_ps_data(self.ps_data)
            
            # Afficher le PS
            self.ps_canvas.update_display()
            
        except Exception as e:
            print(f"Erreur lors du calcul du PS: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, 
                "Pattern Spectra Error", 
                f"Failed to compute Pattern Spectra:\n{str(e)}\n\nPlease check that all images have the same dimensions."
            )

    def on_slider_changed(self, idx):
        """Slider change handler."""
        self.canvas_view.display_index(idx)
        
    def on_ps_mode_changed(self, button):
        """Changes the PS selection mode."""
        if button == self.radio_click:
            self.ps_canvas.set_selection_mode('click')
        else:
            self.ps_canvas.set_selection_mode('polygon')
            
    def on_bins_selected(self, selected_nodes):
        """Handler for bin selection in PS."""
        self.current_selected_nodes = selected_nodes
        
        # Automatically respect the "Show nodes" option
        if self.checkbox_show_nodes.isChecked():
            self.canvas_view.set_selected_nodes(selected_nodes)
        else:
            self.canvas_view.set_selected_nodes([])
        
        # Update label with detailed information
        if selected_nodes:
            selection_info = self.ps_canvas.get_selection_info()
            self.ps_label.setText(f"Pattern Spectra - {selection_info}")
        else:
            self.ps_label.setText("Pattern Spectra Global")

    def on_polygon_selected(self, polygon):
        """Polygon selection handler."""
        try:
            self.current_polygon = polygon
            
            if self.ps_data is None:
                return
                
            # Calculate bins to highlight
            cube_shape = np.array(self.canvas_view.sequence).shape
            highlight_bins, contained_nodes = compute_local_ps_highlight(
                self.ps_data, polygon, cube_shape
            )
            
            # Update PS display with highlighting
            self.ps_canvas.update_display(highlight_bins, contained_nodes)
            
            # Automatically show nodes if the option is enabled
            if self.checkbox_show_nodes.isChecked():
                self.canvas_view.set_selected_nodes(contained_nodes)
                status_text = f"Pattern Spectra - {len(contained_nodes)} nodes selected (Image + Nodes)"
            else:
                # Clear nodes display if option is disabled
                self.canvas_view.set_selected_nodes([])
                status_text = f"Pattern Spectra - {len(contained_nodes)} nodes selected (Image)"
            
            # Update label
            self.ps_label.setText(status_text)
            
        except Exception as e:
            print(f"Selection error: {e}")
            import traceback
            traceback.print_exc()

    def clear_all_selections(self):
        """Efface toutes les sélections (image et PS)."""
        # Effacer la sélection dans l'image
        self.current_polygon = None
        self.canvas_view.clear_polygon()
        
        # Effacer la sélection dans le PS
        self.ps_canvas.clear_selection()
        
        # Effacer la sélection de nœuds dans l'image
        self.canvas_view.set_selected_nodes([])
        
        # Remettre le PS global
        if self.ps_data:
            self.ps_canvas.update_display()
            
        self.ps_label.setText("Pattern Spectra Global")

    def on_show_nodes_changed(self, state):
        """Handler for Show nodes option change."""
        # Enable/disable the color nodes checkbox based on show nodes state
        if state == Qt.Checked:
            self.checkbox_color_nodes.setEnabled(True)
        else:
            self.checkbox_color_nodes.setEnabled(False)
            self.checkbox_color_nodes.setChecked(False)  # Uncheck when disabled
            
        # If there's an active polygon selection, update immediately
        if self.current_polygon is not None:
            # Re-trigger the polygon selection to respect the new state
            self.on_polygon_selected(self.current_polygon)
        # If there's an active PS bin selection, update immediately
        elif self.current_selected_nodes:
            if state == Qt.Checked:
                self.canvas_view.set_selected_nodes(self.current_selected_nodes)
            else:
                self.canvas_view.set_selected_nodes([])

    def on_color_nodes_changed(self, state):
        """Handler for Color nodes option change."""
        # Update the canvas view with the color mode preference
        use_unique_colors = state == Qt.Checked
        self.canvas_view.set_color_nodes_mode(use_unique_colors)
        
        # If there's an active node selection, refresh the display
        if self.checkbox_show_nodes.isChecked() and (self.current_polygon is not None or self.current_selected_nodes):
            if self.current_polygon is not None:
                # Re-trigger polygon selection to update colors
                self.on_polygon_selected(self.current_polygon)
            elif self.current_selected_nodes:
                # Re-apply node selection with new color mode
                self.canvas_view.set_selected_nodes(self.current_selected_nodes)

    def open_temporal_sequence(self):
        """Open a dialog to select multiple PNG files for temporal sequence."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Temporal Sequence Images",
            "",
            "PNG files (*.png);;JPEG files (*.jpg);;All files (*.*)",
            options=QFileDialog.DontUseNativeDialog
        )
        
        if files:
            try:
                # Sort files by name to ensure temporal order
                files.sort()
                
                # Load the new sequence
                new_sequence = []
                for file_path in files:
                    img = imread(file_path)
                    if img.ndim == 3:
                        img = (rgb2gray(img) * 255).astype(np.uint8)
                    new_sequence.append(img)
                
                if len(new_sequence) == 0:
                    QMessageBox.warning(self, "Warning", "No valid images found!")
                    return
                
                # Validate that all images have the same dimensions
                first_shape = new_sequence[0].shape
                for i, img in enumerate(new_sequence[1:], 1):
                    if img.shape != first_shape:
                        QMessageBox.critical(
                            self, 
                            "Error", 
                            f"All images must have the same dimensions!\n"
                            f"First image: {first_shape}\n"
                            f"Image {i+1}: {img.shape}"
                        )
                        return
                
                # Update the image canvas with new sequence
                self.canvas_view.load_custom_sequence(new_sequence)
                
                # Update slider range
                self.slider.setMaximum(len(new_sequence) - 1)
                self.slider.setValue(0)
                
                # Clear all existing selections
                self.clear_all_selections()
                
                # Recalculate pattern spectra
                self.compute_initial_ps()
                
                # Update window title
                self.setWindowTitle(f"LapSITS - Custom Sequence ({len(new_sequence)} images)")
                
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Successfully loaded {len(new_sequence)} images!\n"
                    f"Image dimensions: {first_shape}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Error", 
                    f"Failed to load images:\n{str(e)}"
                )
    
    def reset_to_default_sequence(self):
        """Reset to the default example sequence."""
        try:
            # Reload default sequence
            self.canvas_view.load_fixed_sequence()
            
            # Update slider range
            self.slider.setMaximum(len(self.canvas_view.sequence) - 1)
            self.slider.setValue(0)
            
            # Clear all existing selections
            self.clear_all_selections()
            
            # Recalculate pattern spectra
            self.compute_initial_ps()
            
            # Reset window title
            self.setWindowTitle("LapSITS")
            
            QMessageBox.information(
                self, 
                "Success", 
                "Reset to default sequence successfully!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to reset to default sequence:\n{str(e)}"
            )
    
    def reset_ps_zoom(self):
        """Reset Pattern Spectra zoom to full view."""
        self.ps_canvas.reset_zoom()
