import numpy as np
import higra as hg
import matplotlib.pyplot as plt

def compute_watershed(cube, detail_level=None):
    """
    Calcule un arbre watershed avec filtrage par niveau de détail.
    
    Args:
        cube: Image 4D (T, H, W, C) RGB
        detail_level: Niveau de détail (0.0 à 1.0). Si None, pas de filtrage.
                     0.0 = très simplifié, 1.0 = tous les détails
    
    Returns:
        tuple: (tree, altitudes)
    """
    """
    print(cube[0][0][0])
    print(cube[1][0][0])
    print(cube[2][0][0])
    print(cube[3][0][0])
    print(cube[4][0][0])  
    """  
    print(f"Calcul de l'arbre watershed pour une image de forme {cube.shape} avec niveau de détail {detail_level}")
    cube_rgb = cube.astype(np.float32) / 255.0  # Normalisation des valeurs RGB
    mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]

    graph = hg.get_nd_regular_graph(cube_rgb.shape[:3], hg.mask_2_neighbours(mask))
    edge_weights = hg.weight_graph(graph, cube_rgb, hg.WeightFunction.L2) #norme 
    tree, altitudes = hg.watershed_hierarchy_by_area(graph, edge_weights)
    
    # Appliquer le filtrage par niveau de détail si spécifié
    if detail_level is not None:
        tree, altitudes = apply_detail_level_filter(tree, altitudes, detail_level)
    return tree, altitudes


def reconstruct_rgb_at_level(tree, altitudes, cube_rgb, level):
    """
    Reconstruit l'image RGB à un niveau donné de la hiérarchie.
    
    Args:
        tree: Arbre watershed
        altitudes: Altitudes de l'arbre
        cube_rgb: Image RGB originale (T, H, W, C)
        level: Niveau de la coupe horizontale
    
    Returns:
        np.ndarray: Image reconstruite (T, H, W, C) avec couleurs moyennes (uint8)
    """
    # S'assurer que l'image d'entrée est en uint8
    print(f"Reconstruction à partir du niveau {level} de l'arbre avec {tree.num_vertices()} nœuds")
    if cube_rgb.dtype != np.uint8:
        cube_rgb = (cube_rgb * 255).astype(np.uint8)
    
    # Calculer les couleurs moyennes pour chaque nœud
    cube_vertex_weights = cube_rgb.reshape(-1, 3).astype(np.float32)
    print(f"Cube RGB reshaped: {cube_vertex_weights.shape}")
    mean_colors = hg.attribute_mean_vertex_weights(tree, cube_vertex_weights)
    print(f"Mean colors shape: {mean_colors.shape}")
    
    # Créer le graphe pour la labellisation
    mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    graph = hg.get_nd_regular_graph(cube_rgb.shape[:3], hg.mask_2_neighbours(mask))
    
    # Faire la coupe horizontale
    hce = hg.HorizontalCutExplorer(tree, altitudes)
    cut = hce.horizontal_cut_from_index(level)
    
    # Obtenir le label de région pour chaque pixel
    labels_pixels = cut.labelisation_leaves(tree, graph)
    
    # Remplacer chaque label par la couleur moyenne correspondante
    recon_pixels = mean_colors[labels_pixels]
    recon_image = recon_pixels.reshape(cube_rgb.shape)
    
    # S'assurer que le résultat est en uint8 dans la plage [0, 255]
    recon_image = np.clip(recon_image, 0, 255).astype(np.uint8)
    print(f"Image reconstruite: {recon_image.shape}, dtype: {recon_image.dtype}")
    return recon_image


def reconstruct_rgb_with_detail_level(tree, altitudes, cube_rgb, detail_level):
    """
    Reconstruit l'image RGB avec un niveau de détail donné.
    
    Args:
        tree: Arbre watershed
        altitudes: Altitudes de l'arbre
        cube_rgb: Image RGB originale (T, H, W, C)
        detail_level: Niveau de détail (0.0 à 1.0)
    
    Returns:
        np.ndarray: Image reconstruite (T, H, W, C) avec couleurs moyennes
    """
    # Calculer le niveau de coupe basé sur le niveau de détail
    max_level = get_max_hierarchy_level(tree, altitudes)
    
    # Mapper le niveau de détail (0.0 à 1.0) vers le niveau de coupe (0 à max_level)
    # 0.0 = niveau 0 (très simplifié), 1.0 = niveau max (tous les détails)
    cut_level = int(detail_level * max_level)
    cut_level = min(cut_level, max_level)  # S'assurer qu'on ne dépasse pas le maximum
    
    return reconstruct_rgb_at_level(tree, altitudes, cube_rgb, cut_level)


def get_max_hierarchy_level(tree, altitudes):
    """
    Obtient le niveau maximum de la hiérarchie.
    
    Args:
        tree: Arbre watershed
        altitudes: Altitudes de l'arbre
    
    Returns:
        int: Niveau maximum valide pour HorizontalCutExplorer
    """
    hce = hg.HorizontalCutExplorer(tree, altitudes)
    # Tester différents niveaux pour trouver le maximum valide
    max_level = tree.num_vertices() - tree.num_leaves() - 1
    
    # Tester si ce niveau est accessible
    try:
        cut = hce.horizontal_cut_from_index(max_level)
        return max_level
    except:
        # Si le niveau est trop élevé, essayer des niveaux plus bas
        for level in range(max_level - 1, -1, -1):
            try:
                cut = hce.horizontal_cut_from_index(level)
                return level
            except:
                continue
        return 0  # En dernier recours


def apply_detail_level_filter(tree, altitudes, detail_level):
    """
    Applique un filtrage par niveau de détail sur l'arbre watershed.
    Cette approche utilise une coupe horizontale basée sur le niveau de détail.
    
    Args:
        tree: Arbre watershed original
        altitudes: Altitudes originales
        detail_level: Niveau de détail (0.0 à 1.0)
    
    Returns:
        tuple: (tree, altitudes) - Les altitudes ne sont pas modifiées
    """
    # Calculer les aires des composantes
    area = hg.attribute_area(tree)
    
    # Déterminer le seuil d'aire basé sur le niveau de détail
    max_area = np.max(area)
    min_area = 1
    
    # Calculer le seuil d'aire selon le niveau de détail
    if detail_level <= 0.1:
        # Niveau très bas : garder seulement les grandes composantes
        area_threshold = max_area * 0.05
    elif detail_level >= 0.9:
        # Niveau très haut : garder presque tous les détails
        area_threshold = min_area * 2
    else:
        # Interpolation logarithmique pour un contrôle plus naturel
        log_max = np.log10(max_area)
        log_min = np.log10(min_area)
        # Inverser le detail_level pour que 0.0 = seuil élevé
        inv_detail = 1.0 - detail_level
        log_threshold = log_min + inv_detail * (log_max - log_min) * 0.7
        area_threshold = 10 ** log_threshold
    
    # Compter les nœuds qui seraient filtrés pour information
    small_nodes = area < area_threshold
    
    print(f"Filtrage niveau {detail_level}: seuil aire = {area_threshold:.1f}, "
          f"nœuds filtrés = {np.sum(small_nodes)}/{len(area)}")
    
    # Ne pas modifier les altitudes - retourner l'arbre original
    # Le filtrage se fera au niveau de la coupe horizontale
    return tree, altitudes


"""
def display_watershed_lvl(tree,graph,altitudes, cube_rgb, level):
    cube_vertex_weights = cube_rgb.reshape(-1, 3)
    mean_colors = hg.attribute_mean_vertex_weights(tree, cube_vertex_weights)

    Z, H, W, C = cube_rgb.shape  # Z = nombre de frames (profondeur), H = hauteur, W = largeur, C = canaux (3 pour RGB)

    # Parcourir toutes les coupes (frames) de la hiérarchie
    hce = hg.HorizontalCutExplorer(tree, altitudes)
    cut = hce.horizontal_cut_from_index(level)
    # Obtenir le label de région pour chaque pixel de cette coupe
    labels_pixels = cut.labelisation_leaves(tree, graph)  # tableau de taille Z*H*W
    
    # Remplacer chaque label par la couleur moyenne correspondante
    recon_pixels = mean_colors[labels_pixels]  # shape: (Z*H*W, 3)
    recon_image = recon_pixels.reshape((Z, H, W, 3))  # reconstituer aux dimensions originales
    
    fig, axes = plt.subplots(1, Z, figsize=(4 * Z, 4))
    for t in range(Z):
        axes[t].imshow(recon_image[t])
        axes[t].axis('off')
        axes[t].set_title(f"t={t}")
    plt.suptitle(f"Coupe {level} – {hce.num_regions_cut(level)} régions")
    plt.tight_layout()
    plt.show()"""