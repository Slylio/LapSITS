"""
Module pour gérer différents types d'arbres de composantes
Compatible avec les images en niveaux de gris et RGB
"""

import numpy as np
import higra as hg
from watershed import compute_watershed


def compute_tree_of_shapes(cube):
    """
    Calcule un arbre de formes (Tree of Shapes) 3D pour images en niveaux de gris.
    
    Args:
        cube: Image 3D (T, H, W) en niveaux de gris
        
    Returns:
        tuple: (tree, altitudes)
    """
    cube = cube.astype(np.uint8)
    tree, altitudes = hg.component_tree_tree_of_shapes_image3d(cube)
    altitudes = altitudes.astype(np.float32)
    return tree, altitudes


def compute_min_tree(cube):
    """
    Calcule un arbre min (Min Tree) 3D pour images en niveaux de gris.
    
    Args:
        cube: Image 3D (T, H, W) en niveaux de gris
        
    Returns:
        tuple: (tree, altitudes)
    """
    cube = cube.astype(np.uint8)
    
    # Masque de connectivité 3D (6-connexité)
    mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    
    graph = hg.get_nd_regular_graph(cube.shape, hg.mask_2_neighbours(mask))
    tree, altitudes = hg.component_tree_min_tree(graph, cube.ravel())
    altitudes = altitudes.astype(np.float32)
    return tree, altitudes


def compute_max_tree(cube):
    """
    Calcule un arbre max (Max Tree) 3D pour images en niveaux de gris.
    
    Args:
        cube: Image 3D (T, H, W) en niveaux de gris
        
    Returns:
        tuple: (tree, altitudes)
    """
    cube = cube.astype(np.uint8)
    
    # Masque de connectivité 3D (6-connexité)
    mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    
    graph = hg.get_nd_regular_graph(cube.shape, hg.mask_2_neighbours(mask))
    tree, altitudes = hg.component_tree_max_tree(graph, cube.ravel())
    altitudes = altitudes.astype(np.float32)
    return tree, altitudes


def compute_watershed_tree(cube, detail_level=None):
    """
    Calcule un arbre watershed pour images RGB.
    
    Args:
        cube: Image 3D (T, H, W) en niveaux de gris ou 4D (T, H, W, C) RGB
        detail_level: Niveau de détail (0.0 à 1.0). Si None, pas de filtrage.
        
    Returns:
        tuple: (tree, altitudes)
    """
    # Convertir en RGB si nécessaire
    if cube.ndim == 3:
        # Convertir grayscale en RGB
        cube_rgb = np.stack([cube, cube, cube], axis=-1)
    else:
        cube_rgb = cube
    
    # Utiliser la fonction watershed existante avec le niveau de détail
    tree, altitudes = compute_watershed(cube_rgb, detail_level)
    return tree, altitudes


def is_compatible_with_tree_type(cube, tree_type):
    """
    Vérifie si le cube est compatible avec le type d'arbre spécifié.
    
    Args:
        cube: Image 3D ou 4D
        tree_type: Type d'arbre ('tree_of_shapes', 'min_tree', 'max_tree', 'watershed')
        
    Returns:
        bool: True si compatible, False sinon
    """
    if tree_type == 'watershed':
        # Watershed peut gérer RGB et grayscale
        return cube.ndim in [3, 4]
    else:
        # Tree of shapes, min/max trees nécessitent des images en niveaux de gris
        return cube.ndim == 3

"""
def convert_cube_for_tree_type(cube, tree_type):
    if tree_type == 'watershed':
        print("Conversion pour watershed")
        if cube.ndim == 3:
            print("STACKING CHELOU ------")
            # Convertir grayscale en RGB
            return np.stack([cube, cube, cube], axis=-1)
        else:
            return cube
    else:
        # Pour tree of shapes, min/max trees, convertir en grayscale si nécessaire
        if cube.ndim == 4:
            # Convertir RGB en grayscale
            from skimage.color import rgb2gray
            cube_gray = np.zeros((cube.shape[0], cube.shape[1], cube.shape[2]), dtype=np.uint8)
            for t in range(cube.shape[0]):
                cube_gray[t] = (rgb2gray(cube[t]) * 255).astype(np.uint8)
            return cube_gray
        else:
            return cube
"""

# Dictionnaire des fonctions de calcul d'arbre
TREE_FUNCTIONS = {
    'tree_of_shapes': compute_tree_of_shapes,
    'min_tree': compute_min_tree,
    'max_tree': compute_max_tree,
    'watershed': compute_watershed_tree
}

# Dictionnaire des noms d'affichage
TREE_DISPLAY_NAMES = {
    'tree_of_shapes': 'Tree of Shapes',
    'min_tree': 'Min Tree',
    'max_tree': 'Max Tree',
    'watershed': 'Watershed'
}

# Dictionnaire des descriptions
TREE_DESCRIPTIONS = {
    'tree_of_shapes': 'Arbre de formes (ToS) - capture les structures spatiales complexes',
    'min_tree': 'Arbre Min - capture les composantes sombres (minima)',
    'max_tree': 'Arbre Max - capture les composantes claires (maxima)',
    'watershed': 'Watershed - hiérarchie basée sur la segmentation par ligne de partage des eaux'
}

# Dictionnaire des types d'images supportées
TREE_IMAGE_TYPES = {
    'tree_of_shapes': 'Niveaux de gris',
    'min_tree': 'Niveaux de gris',
    'max_tree': 'Niveaux de gris',
    'watershed': 'RGB et niveaux de gris'
}


def compute_tree_with_type(cube, tree_type, detail_level=None):
    """
    Calcule un arbre du type spécifié.
    
    Args:
        cube: Image 3D ou 4D
        tree_type: Type d'arbre ('tree_of_shapes', 'min_tree', 'max_tree', 'watershed')
        detail_level: Niveau de détail pour watershed (0.0 à 1.0). Ignoré pour les autres types.
        
    Returns:
        tuple: (tree, altitudes)
        
    Raises:
        ValueError: Si le type d'arbre n'est pas supporté ou incompatible
    """
    if tree_type not in TREE_FUNCTIONS:
        raise ValueError(f"Type d'arbre non supporté: {tree_type}. Types disponibles: {list(TREE_FUNCTIONS.keys())}")
    
    if not is_compatible_with_tree_type(cube, tree_type):
        print(f"Conversion du cube pour le type d'arbre {tree_type}")
        
    # Convertir le cube si nécessaire
    print("Origine du cube:", cube.shape)
    print("Type d'arbre:", tree_type)
    #converted_cube = convert_cube_for_tree_type(cube, tree_type)

    # Calculer l'arbre
    tree_func = TREE_FUNCTIONS[tree_type]
    
    # Passer le niveau de détail seulement pour le watershed
    if tree_type == 'watershed' and detail_level is not None:
        tree, altitudes = tree_func(cube, detail_level)
    else:
        tree, altitudes = tree_func(cube)
    
    return tree, altitudes


def get_available_tree_types():
    """
    Retourne la liste des types d'arbres disponibles.
    
    Returns:
        list: Liste des types d'arbres disponibles
    """
    return list(TREE_FUNCTIONS.keys())


def validate_cube_for_tree_type(cube, tree_type):
    """
    Valide que le cube est dans le format optimal pour le type d'arbre spécifié.
    
    Args:
        cube: Image 3D ou 4D
        tree_type: Type d'arbre ('tree_of_shapes', 'min_tree', 'max_tree', 'watershed')
        
    Returns:
        bool: True si le format est optimal, False sinon
    """
    if tree_type not in TREE_FUNCTIONS:
        return False
    
    requirements = get_tree_info(tree_type)
    required_format = requirements.get('image_type', '')
    
    if 'RGB' in required_format and cube.ndim == 4:
        return True
    elif 'Niveaux de gris' in required_format and cube.ndim == 3:
        return True
    elif 'RGB et niveaux de gris' in required_format:
        return True
    
    return False


def get_tree_info(tree_type):
    """
    Retourne les informations sur un type d'arbre.
    
    Args:
        tree_type: Type d'arbre
        
    Returns:
        dict: Informations sur l'arbre (nom, description, type d'image)
    """
    if tree_type not in TREE_FUNCTIONS:
        raise ValueError(f"Type d'arbre non supporté: {tree_type}")
    
    return {
        'name': TREE_DISPLAY_NAMES[tree_type],
        'description': TREE_DESCRIPTIONS[tree_type],
        'image_type': TREE_IMAGE_TYPES[tree_type]
    }
