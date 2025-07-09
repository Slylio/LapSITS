from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QSlider, 
                             QHBoxLayout, QPushButton, QLabel, QSplitter, QButtonGroup, QRadioButton, QCheckBox)
from PyQt5.QtCore import Qt
from gui.image_canvas import ImageCanvas
from gui.ps_canvas import PSCanvas
from core.pattern_spectra import compute_global_ps, compute_local_ps_highlight
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SITS + Pattern Spectra")
        self.setGeometry(100, 100, 1400, 800)  # Fenêtre plus large pour le PS

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Données du pattern spectra
        self.ps_data = None
        self.current_polygon = None
        self.current_selected_nodes = []  # Nœuds sélectionnés depuis le PS
        
        self.setup_ui()
        self.setup_connections()
        self.compute_initial_ps()

    def setup_ui(self):
        """Configuration de l'interface utilisateur."""
        # Splitter horizontal pour diviser image et PS
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Partie gauche : Image et contrôles
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        self.canvas_view = ImageCanvas()
        self.canvas_view.setMinimumWidth(400)
        self.canvas_view.setMinimumHeight(400)
        
        # Connecter la sélection de polygone
        self.canvas_view.polygon_selected.connect(self.on_polygon_selected)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.canvas_view.sequence) - 1)
        self.slider.valueChanged.connect(self.on_slider_changed)
        
        # Boutons de contrôle
        controls_layout = QHBoxLayout()
        self.btn_compute_ps = QPushButton("Recalculer PS")
        self.btn_compute_ps.clicked.connect(self.compute_initial_ps)
        self.btn_clear_selection = QPushButton("Effacer sélection")
        self.btn_clear_selection.clicked.connect(self.clear_all_selections)
        
        # Boutons radio pour le mode de sélection PS
        self.ps_mode_group = QButtonGroup()
        self.radio_click = QRadioButton("Clic")
        self.radio_polygon = QRadioButton("Polygone")
        self.radio_click.setChecked(True)
        self.ps_mode_group.addButton(self.radio_click, 0)
        self.ps_mode_group.addButton(self.radio_polygon, 1)
        
        # Checkbox pour afficher les nœuds
        from PyQt5.QtWidgets import QCheckBox
        self.checkbox_show_nodes = QCheckBox("Show nodes")
        self.checkbox_show_nodes.setChecked(False)
        self.checkbox_show_nodes.stateChanged.connect(self.on_show_nodes_changed)
        
        controls_layout.addWidget(QLabel("Mode PS:"))
        controls_layout.addWidget(self.radio_click)
        controls_layout.addWidget(self.radio_polygon)
        controls_layout.addWidget(self.checkbox_show_nodes)
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
                
            cube = np.array(self.canvas_view.sequence)
            print(f"Calcul du PS pour cube de forme: {cube.shape}")
            
            # Calculer le PS global
            self.ps_data = compute_global_ps(cube)
            
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

    def on_slider_changed(self, idx):
        """Gestionnaire de changement du slider."""
        self.canvas_view.display_index(idx)
        
    def on_ps_mode_changed(self, button):
        """Change le mode de sélection du PS."""
        if button == self.radio_click:
            self.ps_canvas.set_selection_mode('click')
        else:
            self.ps_canvas.set_selection_mode('polygon')
            
    def on_bins_selected(self, selected_nodes):
        """Gestionnaire de sélection de bins dans le PS."""
        self.current_selected_nodes = selected_nodes
        
        # Mettre à jour l'affichage dans l'image
        self.canvas_view.set_selected_nodes(selected_nodes)
        
        # Mettre à jour le label avec des informations détaillées
        if selected_nodes:
            selection_info = self.ps_canvas.get_selection_info()
            self.ps_label.setText(f"Pattern Spectra - {selection_info}")
        else:
            self.ps_label.setText("Pattern Spectra Global")

    def on_polygon_selected(self, polygon):
        """Gestionnaire de sélection de polygone."""
        try:
            self.current_polygon = polygon
            
            if self.ps_data is None:
                return
                
            # Calculer les bins à mettre en surbrillance
            cube_shape = np.array(self.canvas_view.sequence).shape
            highlight_bins, contained_nodes = compute_local_ps_highlight(
                self.ps_data, polygon, cube_shape
            )
            
            # Mettre à jour l'affichage du PS avec surbrillance
            self.ps_canvas.update_display(highlight_bins, contained_nodes)
            
            # Mettre à jour le label
            self.ps_label.setText(f"Pattern Spectra - {len(contained_nodes)} nœuds sélectionnés (Image)")
            
        except Exception as e:
            print(f"Erreur lors de la sélection: {e}")
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
        """Gestionnaire de changement de l'option Show nodes."""
        # Si l'option est activée et qu'il y a une sélection de polygone active
        if state == Qt.Checked and self.current_polygon is not None:
            # Réappliquer la sélection de polygone pour afficher les nœuds
            self.apply_polygon_selection_with_nodes()
        elif state == Qt.Unchecked:
            # Masquer les nœuds dans l'image
            self.canvas_view.set_selected_nodes([])
            
    def apply_polygon_selection_with_nodes(self):
        """Applique la sélection de polygone avec affichage des nœuds si l'option est activée."""
        if self.current_polygon is None or self.ps_data is None:
            return
            
        # Calculer les bins à mettre en surbrillance
        cube_shape = np.array(self.canvas_view.sequence).shape
        highlight_bins, contained_nodes = compute_local_ps_highlight(
            self.ps_data, self.current_polygon, cube_shape
        )
        
        # Mettre à jour l'affichage du PS avec surbrillance
        self.ps_canvas.update_display(highlight_bins, contained_nodes)
        
        # Si l'option "Show nodes" est activée, afficher les nœuds dans l'image
        if self.checkbox_show_nodes.isChecked():
            self.canvas_view.set_selected_nodes(contained_nodes)
            status_text = f"Pattern Spectra - {len(contained_nodes)} nœuds sélectionnés (Image + Nodes)"
        else:
            status_text = f"Pattern Spectra - {len(contained_nodes)} nœuds sélectionnés (Image)"
            
        self.ps_label.setText(status_text)
