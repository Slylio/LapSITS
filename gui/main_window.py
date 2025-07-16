from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QSlider, 
                             QHBoxLayout, QPushButton, QLabel, QSplitter, QButtonGroup, QRadioButton, QCheckBox,
                             QMenuBar, QAction, QFileDialog, QMessageBox, QComboBox, QSpinBox, QLineEdit)
from PyQt5.QtCore import Qt
from gui.image_canvas import ImageCanvas
from gui.ps_canvas import PSCanvas
from core.pattern_spectra import compute_global_ps, compute_local_ps_highlight
from core.attributes import ATTR_FUNCS, BINS
from core.pattern_spectra import compute_2d_ps_with_tracking
from core.tree import get_available_tree_types, get_tree_info, TREE_DISPLAY_NAMES, validate_cube_for_tree_type
import numpy as np
import os
from skimage.io import imread
from skimage.color import rgb2gray
import traceback

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
        self.current_tree_type = 'tree_of_shapes'  # Default tree type
        # Supprimer current_detail_level - plus de slider de niveau de détail
        self.cube_rgb = None  # Store original RGB cube for watershed reconstruction
        self.original_rgb_sequence = None  # Store original RGB sequence for watershed
        self.max_watershed_level = 0  # Store max level for watershed reconstruction
        self.current_watershed_level = 0  # Current level for watershed reconstruction
        
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
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # Tree types info action
        tree_info_action = QAction('Types d\'arbres disponibles', self)
        tree_info_action.setStatusTip('Afficher les informations sur les types d\'arbres')
        tree_info_action.triggered.connect(self.show_tree_types_info)
        help_menu.addAction(tree_info_action)

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
        
        # Dropdown pour sélectionner le type d'arbre
        self.tree_type_combo = QComboBox()
        available_trees = get_available_tree_types()
        for tree_type in available_trees:
            display_name = TREE_DISPLAY_NAMES.get(tree_type, tree_type)
            self.tree_type_combo.addItem(display_name, tree_type)
        self.tree_type_combo.setCurrentIndex(0)
        self.tree_type_combo.currentTextChanged.connect(self.on_tree_type_changed)
        
        # Layout pour les contrôles
        controls_layout = QVBoxLayout()
        
        # Ligne 1: Attributs
        attr_layout = QHBoxLayout()
        attr_layout.addWidget(QLabel("Axe X:"))
        attr_layout.addWidget(self.attr1_combo)
        attr_layout.addWidget(QLabel("Axe Y:"))
        attr_layout.addWidget(self.attr2_combo)
        controls_layout.addLayout(attr_layout)
        
        # Ligne 2: Type d'arbre
        tree_layout = QHBoxLayout()
        tree_layout.addWidget(QLabel("Type d'arbre:"))
        tree_layout.addWidget(self.tree_type_combo)
        tree_layout.addStretch()
        controls_layout.addLayout(tree_layout)
        
        # Contrôles watershed (visibles seulement pour le type watershed)
        self.watershed_controls = QWidget()
        watershed_layout = QVBoxLayout()
        
        # Ligne pour le niveau de reconstruction
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Niveau de reconstruction:"))
        
        self.watershed_level_slider = QSlider(Qt.Horizontal)
        self.watershed_level_slider.setMinimum(0)
        self.watershed_level_slider.setMaximum(100)  # Sera ajusté dynamiquement
        self.watershed_level_slider.setValue(50)  # Commence à 50% (équivalent à max_level//2)
        self.watershed_level_slider.valueChanged.connect(self.on_watershed_level_changed)
        
        self.watershed_level_label = QLabel("50%")
        self.watershed_level_label.setMinimumWidth(40)
        
        level_layout.addWidget(self.watershed_level_slider)
        level_layout.addWidget(self.watershed_level_label)
        
        watershed_layout.addLayout(level_layout)
        
        self.watershed_controls.setLayout(watershed_layout)
        self.watershed_controls.setVisible(False)  # Masqué par défaut
        
        controls_layout.addWidget(self.watershed_controls)
        
        # Supprimer les contrôles de niveau de détail - affichage watershed sans slider
        
        right_layout.addLayout(controls_layout)
        
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
            if self.canvas_view.sequence is None or len(self.canvas_view.sequence) == 0:
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
            
            # Pour le watershed, nous avons besoin des données RGB originales
            if self.current_tree_type == 'watershed':
                # Charger les données RGB originales
                if self.original_rgb_sequence is not None:
                    # Utiliser les données RGB chargées par l'utilisateur
                    cube_rgb = np.array(self.original_rgb_sequence)
                else:
                    # Charger les données RGB par défaut
                    base_path = "data/sits_example"
                    files = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".png")])
                    rgb_sequence = []
                    for f in files:
                        img = imread(f)
                        if img.ndim == 3:
                            # Garder les données RGB pour le watershed
                            rgb_sequence.append(img)
                        else:
                            # Convertir en RGB si l'image est en niveaux de gris
                            rgb_sequence.append(np.stack([img, img, img], axis=-1))
                    
                    cube_rgb = np.array(rgb_sequence)
                    # Stocker pour utilisation ultérieure
                    self.original_rgb_sequence = rgb_sequence
                
                print(f"Calcul du PS watershed pour cube RGB de forme: {cube_rgb.shape}")
                
                # Utiliser le cube RGB pour le calcul du PS watershed
                cube_for_ps = cube_rgb
            else:
                cube_for_ps = cube
            
            print(f"Calcul du PS pour cube de forme: {cube_for_ps.shape} avec arbre: {self.current_tree_type}")
            
            # Vérifier la compatibilité avec le type d'arbre sélectionné
            if not validate_cube_for_tree_type(cube_for_ps, self.current_tree_type):
                tree_info = get_tree_info(self.current_tree_type)
                QMessageBox.warning(
                    self,
                    "Incompatibilité de format",
                    f"Le format de données actuel n'est pas optimal pour {tree_info['name']}.\n\n"
                    f"Type d'arbre: {tree_info['name']}\n"
                    f"Format requis: {tree_info['image_type']}\n"
                    f"Format actuel: {'RGB' if cube_for_ps.ndim == 4 else 'Niveaux de gris'}\n\n"
                    f"Les données seront converties automatiquement, mais vous pourriez "
                    f"obtenir de meilleurs résultats en choisissant un type d'arbre plus adapté."
                )
            
            # Calcule le PS global 2D avec attributs dynamiques et type d'arbre
            # Construire l'arbre et altitudes via compute_global_ps
            # Pas de niveau de détail pour le watershed - utiliser l'arbre complet
            base_data = compute_global_ps(cube_for_ps, self.current_tree_type)
            tree = base_data['tree']
            altitudes = base_data['altitudes']
            
            # Stocker les données pour le watershed
            if self.current_tree_type == 'watershed':
                self.cube_rgb = cube_for_ps  # Utiliser directement le cube RGB
            else:
                self.cube_rgb = None
            
            # Calculer les attributs selon sélection
            attr1 = ATTR_FUNCS[self.attr1_combo.currentText()](tree, cube_for_ps)
            attr2 = ATTR_FUNCS[self.attr2_combo.currentText()](tree, cube_for_ps)
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
                'bin_contributions': bin_contributions,
                'tree_type': self.current_tree_type
            }
            
            # Configurer le canvas PS
            self.ps_canvas.set_ps_data(self.ps_data)
            
            # Transmettre les données PS à l'image canvas
            self.canvas_view.set_ps_data(self.ps_data)
            
            # Afficher le PS
            self.ps_canvas.update_display()
            
            # Si c'est un watershed, afficher l'image RGB reconstruite
            if self.current_tree_type == 'watershed':
                self.update_watershed_image_display()
            
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
    
    def on_tree_type_changed(self, display_name):
        """Handler for tree type change."""
        # Get the actual tree type from the combo box data
        current_index = self.tree_type_combo.currentIndex()
        if current_index >= 0:
            tree_type = self.tree_type_combo.itemData(current_index)
            if tree_type != self.current_tree_type:
                self.current_tree_type = tree_type
                
                # Afficher/masquer les contrôles watershed selon le type d'arbre
                is_watershed = (tree_type == 'watershed')
                self.watershed_controls.setVisible(is_watershed)
                
                # Show info about the selected tree type
                tree_info = get_tree_info(tree_type)
                print(f"Changement vers {tree_info['name']}: {tree_info['description']}")
                print(f"Compatible avec: {tree_info['image_type']}")
                
                # Show info dialog to user
                info_message = f"Description: {tree_info['description']}\n\n" \
                              f"Compatible avec: {tree_info['image_type']}\n\n" \
                              f"Le Pattern Spectra sera recalculé avec ce nouveau type d'arbre."
                
                if tree_type == 'watershed':
                    info_message += "\n\nLe watershed affichera automatiquement les couleurs moyennes des régions.\n" \
                                    "Utilisez le slider de niveau pour contrôler le degré de simplification."
                
                QMessageBox.information(
                    self,
                    f"Type d'arbre changé: {tree_info['name']}",
                    info_message
                )
                
                # Recalculate PS with new tree type
                self.compute_initial_ps()
    
    def on_watershed_level_changed(self, value):
        """Handler for watershed level slider change."""
        if self.current_tree_type == 'watershed' and self.ps_data is not None:
            # Convertir le pourcentage en niveau réel
            level = int((value / 100.0) * self.max_watershed_level)
            self.current_watershed_level = level
            
            # Mettre à jour l'affichage du pourcentage
            self.watershed_level_label.setText(f"{value}%")
            
            # Mettre à jour l'affichage de l'image
            self.update_watershed_image_display()
            
            # Mettre à jour l'affichage des nœuds pour le nouveau niveau
            if self.current_selected_nodes and self.checkbox_show_nodes.isChecked():
                self.update_nodes_for_watershed_level()
    
    def update_nodes_for_watershed_level(self):
        """Met à jour l'affichage des nœuds pour le niveau de reconstruction watershed actuel."""
        if (self.current_tree_type != 'watershed' or 
            not self.current_selected_nodes or 
            not self.checkbox_show_nodes.isChecked()):
            return
            
        try:
            # Filtrer les nœuds selon le niveau de reconstruction actuel
            filtered_nodes = self.filter_nodes_by_watershed_level(
                self.current_selected_nodes, 
                self.current_watershed_level
            )
            
            # Mettre à jour l'affichage avec les nœuds filtrés
            self.canvas_view.set_selected_nodes(filtered_nodes)
            
            print(f"Nœuds filtrés pour niveau {self.current_watershed_level}: {len(filtered_nodes)}/{len(self.current_selected_nodes)}")
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour des nœuds: {e}")
            import traceback
            traceback.print_exc()
    
    def filter_nodes_by_watershed_level(self, nodes, level):
        """Filtre les nœuds selon le niveau de reconstruction watershed."""
        if not nodes or self.ps_data is None:
            return nodes
            
        try:
            import higra as hg
            
            tree = self.ps_data['tree']
            altitudes = self.ps_data['altitudes']
            
            # Créer un HorizontalCutExplorer pour le niveau actuel
            hce = hg.HorizontalCutExplorer(tree, altitudes)
            
            # Faire la coupe au niveau spécifié
            try:
                cut = hce.horizontal_cut_from_index(level)
            except:
                # Si le niveau n'est pas accessible, utiliser le niveau maximum accessible
                max_accessible = min(level, tree.num_vertices() - tree.num_leaves() - 1)
                cut = hce.horizontal_cut_from_index(max_accessible)
            
            # Obtenir les nœuds actifs à ce niveau
            active_nodes = set()
            
            # Parcourir tous les nœuds de la coupe
            for node in cut.nodes():
                active_nodes.add(node)
                
                # Ajouter aussi les ancêtres jusqu'à un certain niveau
                current = node
                while current != tree.root() and current in tree.ancestors(node):
                    active_nodes.add(current)
                    current = tree.parent(current)
            
            # Filtrer les nœuds sélectionnés pour ne garder que ceux actifs
            filtered_nodes = [node for node in nodes if node in active_nodes]
            
            return filtered_nodes
            
        except Exception as e:
            print(f"Erreur lors du filtrage des nœuds: {e}")
            # En cas d'erreur, retourner les nœuds originaux
            return nodes
    
    # Fonctions de niveau de détail supprimées - plus de slider
    
    def update_watershed_image_display(self):
        """Met à jour l'affichage de l'image watershed avec les couleurs moyennes."""
        if self.current_tree_type != 'watershed' or self.ps_data is None or self.cube_rgb is None:
            return
        
        try:
            # Importer la fonction de reconstruction depuis watershed.py
            from watershed import reconstruct_rgb_at_level, get_max_hierarchy_level
            
            tree = self.ps_data['tree']
            altitudes = self.ps_data['altitudes']
            print(self.cube_rgb.shape)
            # Obtenir le niveau maximum de la hiérarchie
            max_level = get_max_hierarchy_level(tree, altitudes)
            self.max_watershed_level = max_level
            
            # Utiliser le niveau sélectionné par l'utilisateur, ou max_level//2 par défaut
            if hasattr(self, 'current_watershed_level'):
                level = self.current_watershed_level
            else:
                level = max_level // 2
                self.current_watershed_level = level
            
            # Mettre à jour le slider si nécessaire
            if hasattr(self, 'watershed_level_slider'):
                # Calculer le pourcentage correspondant au niveau actuel
                percentage = int((level / max_level) * 100) if max_level > 0 else 50
                self.watershed_level_slider.blockSignals(True)  # Éviter la récursion
                self.watershed_level_slider.setValue(percentage)
                self.watershed_level_label.setText(f"{percentage}%")
                self.watershed_level_slider.blockSignals(False)
            
            print(f"Reconstruction watershed au niveau {level}/{max_level} ({int((level/max_level)*100) if max_level > 0 else 50}%)")
            
            # Reconstruire l'image RGB au niveau spécifié
            recon_image = reconstruct_rgb_at_level(tree, altitudes, self.cube_rgb, level)
            
            # Mettre à jour la séquence d'images dans le canvas
            self.canvas_view.load_custom_sequence(recon_image)
            
            print(f"Image RGB reconstruite avec couleurs moyennes watershed au niveau {level}")
            
        except Exception as e:
            print(f"Erreur lors de la reconstruction de l'image: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Erreur de reconstruction",
                f"Impossible de reconstruire l'image watershed:\n{str(e)}"
            )
            
    def on_bins_selected(self, selected_nodes):
        """Handler for bin selection in PS."""
        self.current_selected_nodes = selected_nodes
        
        # Automatically respect the "Show nodes" option
        if self.checkbox_show_nodes.isChecked():
            # Pour le watershed, filtrer les nœuds selon le niveau actuel
            if self.current_tree_type == 'watershed' and hasattr(self, 'current_watershed_level'):
                filtered_nodes = self.filter_nodes_by_watershed_level(
                    selected_nodes, 
                    self.current_watershed_level
                )
                self.canvas_view.set_selected_nodes(filtered_nodes)
            else:
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
                # Pour le watershed, filtrer les nœuds selon le niveau actuel
                if self.current_tree_type == 'watershed' and hasattr(self, 'current_watershed_level'):
                    filtered_nodes = self.filter_nodes_by_watershed_level(
                        contained_nodes, 
                        self.current_watershed_level
                    )
                    self.canvas_view.set_selected_nodes(filtered_nodes)
                    status_text = f"Pattern Spectra - {len(contained_nodes)} nodes selected ({len(filtered_nodes)} visible at current level)"
                else:
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
                # Pour le watershed, filtrer les nœuds selon le niveau actuel
                if self.current_tree_type == 'watershed' and hasattr(self, 'current_watershed_level'):
                    filtered_nodes = self.filter_nodes_by_watershed_level(
                        self.current_selected_nodes, 
                        self.current_watershed_level
                    )
                    self.canvas_view.set_selected_nodes(filtered_nodes)
                else:
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
                # Pour le watershed, filtrer les nœuds selon le niveau actuel
                if self.current_tree_type == 'watershed' and hasattr(self, 'current_watershed_level'):
                    filtered_nodes = self.filter_nodes_by_watershed_level(
                        self.current_selected_nodes, 
                        self.current_watershed_level
                    )
                    self.canvas_view.set_selected_nodes(filtered_nodes)
                else:
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
                new_rgb_sequence = []
                for file_path in files:
                    img = imread(file_path)
                    if img.ndim == 3:
                        # Stocker l'image RGB originale
                        new_rgb_sequence.append(img)
                        # Convertir en niveaux de gris pour l'affichage normal
                        img = (rgb2gray(img) * 255).astype(np.uint8)
                    else:
                        # Si l'image est déjà en niveaux de gris, créer une version RGB
                        new_rgb_sequence.append(np.stack([img, img, img], axis=-1))
                    new_sequence.append(img)
                
                if len(new_sequence) == 0:
                    QMessageBox.warning(self, "Warning", "No valid images found!")
                    return
                
                # Stocker les données RGB originales pour le watershed
                self.original_rgb_sequence = new_rgb_sequence
                
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
            
            # Reset RGB sequence for watershed
            self.original_rgb_sequence = None
            
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
    
    def show_tree_types_info(self):
        """Affiche les informations sur les types d'arbres disponibles."""
        info_text = "Types d'arbres disponibles pour le calcul des Pattern Spectra:\n\n"
        
        available_trees = get_available_tree_types()
        for tree_type in available_trees:
            tree_info = get_tree_info(tree_type)
            info_text += f"• {tree_info['name']}\n"
            info_text += f"  Description: {tree_info['description']}\n"
            info_text += f"  Compatible avec: {tree_info['image_type']}\n\n"
        
        info_text += "Recommandations:\n"
        info_text += "• Tree of Shapes: Idéal pour l'analyse de formes et contours\n"
        info_text += "• Min-Tree: Parfait pour analyser les régions sombres\n"
        info_text += "• Max-Tree: Optimal pour analyser les régions claires\n"
        info_text += "• Watershed: Excellent pour la segmentation basée sur la couleur\n"
        
        QMessageBox.information(
            self,
            "Types d'arbres - Informations",
            info_text
        )
