# LapSITS - Pattern Spectra Viewer

## Description du Projet

LapSITS est une application d'analyse et de visualisation de séquences d'images temporelles (SITS - Satellite Image Time Series) utilisant les Pattern Spectra basés sur les arbres de composantes. Ce mini 

### Fonctionnalités principales

1. **Visualisation des SITS** : Affichage des séquences d'images avec navigation temporelle
2. **Calcul de Pattern Spectra** : Génération automatique du PS global basé sur Area vs Stability
3. **Sélection bidirectionnelle** :
   - **Image → PS** : Sélection de polygones dans l'image pour mettre en surbrillance les bins correspondants dans le PS
   - **PS → Image** : Sélection de bins dans le PS pour visualiser les nœuds correspondants dans la SITS

## Structure du Projet

```
LapSITS/
├── main.py                 # Point d'entrée de l'application
├── core/
│   └── pattern_spectra.py  # Calculs des Pattern Spectra
├── gui/
│   ├── main_window.py      # Fenêtre principale
│   ├── image_canvas.py     # Canvas pour les images SITS
│   └── ps_canvas.py        # Canvas interactif pour les Pattern Spectra
└── data/
    └── sits_example/       # Images de la séquence temporelle



## Guide d'Utilisation

### Interface Principale

L'interface est divisée en deux parties :
- **Gauche** : Visualisation des images SITS avec contrôles
- **Droite** : Pattern Spectra interactif

### Sélection dans l'Image (Image → PS)

1. **Créer un polygone** :
   - Clic gauche pour ajouter des points
   - Clic droit pour fermer le polygone
   - Les bins correspondants s'allument automatiquement dans le PS

2. **Navigation temporelle** :
   - Utilisez le slider pour changer de timestep
   - Le polygone reste actif et les bins correspondants sont mis à jour

### Sélection dans le Pattern Spectra (PS → Image)

1. **Mode Clic** :
   - Activez "Clic" dans les contrôles
   - Cliquez sur les bins du PS pour les sélectionner/désélectionner
   - Les nœuds correspondants s'allument en rouge dans l'image

2. **Mode Polygone** :
   - Activez "Polygone" dans les contrôles
   - Dessinez un polygone dans le PS (comme dans l'image)
   - Tous les bins contenus sont sélectionnés

3. **Navigation temporelle** :
   - Changez de timestep avec le slider
   - Les zones affectées par les nœuds sélectionnés sont visualisées pour chaque temps

### Raccourcis Clavier

**Dans l'image** :
- `Échap` : Annuler la sélection de polygone en cours
- `Suppr` : Effacer le polygone actuel

**Dans le Pattern Spectra** :
- `c` : Mode clic
- `p` : Mode polygone
- `Échap` : Annuler la sélection en cours
- `Suppr` : Effacer toute la sélection

### Boutons de Contrôle

- **Recalculer PS** : Recalcule le Pattern Spectra (utile après changement de données)
- **Effacer sélection** : Efface toutes les sélections (image et PS)

## Algorithmes Utilisés

### Pattern Spectra
- **Arbre de composantes** : Tree of Shapes 3D pour capturer la structure spatio-temporelle
- **Attributs** :
  - Area : Aire des composantes (échelle logarithmique)
  - Stability : Stabilité temporelle des composantes (échelle linéaire)

### Sélection Bidirectionnelle
- **Image → PS** : Utilise les masques 3D et la propagation dans l'arbre
- **PS → Image** : Remonte des bins vers les nœuds puis vers les pixels par timestep

## Framework

- **Interface** : PyQt5
- **Calculs** : Higra (arbres de composantes), NumPy
- **Visualisation** : Matplotlib intégré dans PyQt5
- **Géométrie** : Shapely pour la gestion des polygones
