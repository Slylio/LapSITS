"""
Module pour les calculs de Pattern Spectra (PS) sur les séquences d'images temporelles (SITS).
Basé sur compute_ps_from_cube.py mais adapté pour l'interface graphique.
"""

import numpy as np
import higra as hg
from matplotlib.colors import LogNorm
from skimage.draw import polygon as sk_polygon
from shapely.geometry import Polygon
from matplotlib import cm


def compute_STstability_attribute_v2(tree, time_indices):
    n_leaves = tree.num_leaves()
    T = len(time_indices) #nombre de timestemps
    frame_size = n_leaves // T #taille des tranches de feuilles par timestep
  
    """
    np.arange(n_leaves) // frame_size == i : true sur les feuilles de la tranche i (ex t1 : True x16, False x32, t2 : False x16, True x16, False x16 etc..)
    hg.accumulate_sequential(tree, (np.arange(n_leaves) // frame_size == i).astype(np.float32), hg.Accumulators.sum) : calcul des aires en ne prenant que les feuilles de la tranche i
    t1 :[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.
        0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
        0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  2.  1.  3.
        4.  6.  0. 16.]
    Et on stack ça dans comp
    """
    comp = np.stack([
        hg.accumulate_sequential(tree, (np.arange(n_leaves) // frame_size == i).astype(np.float32), hg.Accumulators.sum)
        for i in range(T)
    ], axis=1) 
    a, b = comp[:, :-1], comp[:, 1:]
    
    valid = b > 0
    r = np.divide(np.minimum(a, b), np.maximum(a, b), out=np.zeros_like(a), where=valid)
    dt = np.diff(time_indices).astype(np.float32)
    stability = (r * dt).sum(axis=1) / dt.sum()
    stability[tree.root()] = 1.0
    return stability


def shapely_to_mask(poly: Polygon, shape):
    """Convertit un polygone Shapely en masque booléen."""
    if poly.is_empty:
        return np.zeros(shape, dtype=bool)
    x, y = poly.exterior.coords.xy
    rr, cc = sk_polygon(y, x, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def pixelset_to_contained_subtree(tree, mask3d):
    """Trouve les nœuds de l'arbre entièrement contenus dans le masque 3D."""
    return hg.accumulate_and_min_sequential(
        tree,
        np.ones(tree.num_vertices(), dtype=np.uint8),
        mask3d.ravel().astype(np.uint8),
        hg.Accumulators.min
    ).astype(bool)


def extract_related_bins_list(tree, node_to_bin2d, node_list):
    """Extrait tous les bins associés à une liste de nœuds."""
    bins_set = set()
    for node_id in node_list:
        # Obtenir le sous-arbre du nœud
        _, node_map = tree.sub_tree(node_id)
        # Ajouter tous les bins associés aux nœuds du sous-arbre
        for n in node_map:
            if n in node_to_bin2d:
                bins_set.add(node_to_bin2d[n])
    return bins_set


def compute_2d_ps_with_tracking(tree, altitudes, attr1, attr2, bins1, bins2):
    """
    Calcule un pattern spectra 2D avec suivi des contributions des nœuds.
    
    Args:
        tree: Arbre de composantes
        altitudes: Altitudes des nœuds
        attr1, attr2: Attributs pour les deux dimensions
        bins1, bins2: Bins pour l'histogramme 2D
    
    Returns:
        tuple: (tree, hist2d, bins1, bins2, node_to_bin2d, bin_contributions)
    """
    area = hg.attribute_area(tree)
    vol_nodes = area * np.abs(altitudes - altitudes[tree.parents()])

    # Histogramme 2D
    hist2d, _, _ = np.histogram2d(attr1, attr2, bins=[bins1, bins2], weights=vol_nodes)
    if hist2d.sum() > 0:
        hist2d /= hist2d.sum()

    # Calcul des indices de bin pour chaque nœud
    bin_idx_1 = np.digitize(attr1, bins1, right=False) - 1
    bin_idx_2 = np.digitize(attr2, bins2, right=False) - 1

    node_to_bin2d = {}
    for node, (i, j) in enumerate(zip(bin_idx_1, bin_idx_2)):
        if 0 <= i < len(bins1) - 1 and 0 <= j < len(bins2) - 1:
            node_to_bin2d[node] = (i, j)

    # Contributions des nœuds aux bins
    bin_contributions = {}
    for n in range(tree.num_vertices()):
        i, j = node_to_bin2d.get(n, (None, None))
        if i is not None:
            v = area[n] * np.abs(altitudes[n] - altitudes[tree.parents()[n]])
            if (i, j) not in bin_contributions:
                bin_contributions[(i, j)] = []
            bin_contributions[(i, j)].append((n, v))

    return tree, hist2d, bins1, bins2, node_to_bin2d, bin_contributions


def compute_global_ps(cube):
    """
    Calcule le pattern spectra global pour une séquence d'images 3D.
    
    Args:
        cube: Séquence d'images 3D (T, H, W)
    
    Returns:
        dict: Dictionnaire contenant tous les résultats du calcul PS
    """
    cube = cube.astype(np.uint8)
    
    # Construction de l'arbre 3D
    tree, altitudes = hg.component_tree_tree_of_shapes_image3d(cube)
    altitudes = altitudes.astype(np.float32)
    
    # Calcul des attributs
    area = hg.attribute_area(tree)
    stability = compute_STstability_attribute_v2(tree, [1, 2, 3, 4, 5])
    
    # Bins fixes pour la cohérence
    bins1 = np.logspace(0, 7, 100)  # Area (échelle log)
    bins2 = np.linspace(0, 1, 100)  # Stability (échelle linéaire)
    
    # Calcul du PS avec suivi
    tree, hist2d, bins1, bins2, node_to_bin2d, bin_contributions = compute_2d_ps_with_tracking(
        tree, altitudes, area, stability, bins1, bins2
    )
    
    return {
        'tree': tree,
        'altitudes': altitudes,
        'area': area,
        'stability': stability,
        'hist2d': hist2d,
        'bins1': bins1,
        'bins2': bins2,
        'node_to_bin2d': node_to_bin2d,
        'bin_contributions': bin_contributions
    }


def compute_local_ps_highlight(ps_data, polygon, cube_shape):
    """
    Calcule les bins à mettre en surbrillance pour un polygone donné.
    
    Args:
        ps_data: Données du PS global (retour de compute_global_ps)
        polygon: Polygone Shapely
        cube_shape: Forme du cube (T, H, W)
    
    Returns:
        tuple: (highlight_bins, contained_nodes)
    """
    tree = ps_data['tree']
    node_to_bin2d = ps_data['node_to_bin2d']
    
    # Masque 3D du polygone
    mask3d = np.stack([shapely_to_mask(polygon, cube_shape[1:]) for _ in range(cube_shape[0])])
    
    # Nœuds contenus dans le polygone
    node_mask = pixelset_to_contained_subtree(tree, mask3d)
    contained_nodes = np.where(node_mask)[0]
    contained_nodes = [n for n in contained_nodes if n >= tree.num_leaves()]
    
    # Bins associés
    highlight_bins = extract_related_bins_list(tree, node_to_bin2d, contained_nodes)
    
    return highlight_bins, contained_nodes


def plot_ps_with_highlights(ax, ps_data, highlight_bins=None, contained_nodes=None):
    """
    Affiche le pattern spectra avec mise en surbrillance optionnelle.
    
    Args:
        ax: Axes matplotlib
        ps_data: Données du PS global
        highlight_bins: Bins à mettre en surbrillance (optionnel)
        contained_nodes: Nœuds sélectionnés (optionnel)
    
    Returns:
        matplotlib.collections.QuadMesh: L'objet mesh pour la colorbar
    """
    hist2d = ps_data['hist2d']
    bins1 = ps_data['bins1']
    bins2 = ps_data['bins2']
    bin_contributions = ps_data['bin_contributions']
    
    # Affichage du PS global en arrière-plan
    cmap = cm.get_cmap('viridis')
    norm_bg = LogNorm(vmin=hist2d[hist2d > 0].min(), vmax=hist2d.max())
    pcm = ax.pcolormesh(bins1, bins2, hist2d.T, norm=norm_bg, cmap=cmap, shading='auto', alpha=0.3)
    
    # Mise en surbrillance des bins sélectionnés
    if highlight_bins and contained_nodes:
        hist_mask = np.zeros_like(hist2d)
        selected_set = set(contained_nodes)
        
        for (i, j) in highlight_bins:
            if i < hist_mask.shape[0] and j < hist_mask.shape[1]:
                contribs = bin_contributions.get((i, j), [])
                v = sum(vol for n, vol in contribs if n in selected_set)
                hist_mask[i, j] = v
        
        if hist_mask.sum() > 0:
            norm_fg = LogNorm(vmin=hist_mask[hist_mask > 0].min(), vmax=hist_mask.max())
            pcm_highlight = ax.pcolormesh(bins1, bins2, hist_mask.T, norm=norm_fg,
                                        cmap=cm.get_cmap('coolwarm'), shading='auto', alpha=0.9)
    
    # Configuration des axes
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlabel('Area (log scale)')
    ax.set_ylabel('Stability')
    ax.set_title('Pattern Spectra')
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    
    return pcm


def compute_nodes_from_bins(ps_data, selected_bins):
    """
    Calcule les nœuds correspondant aux bins sélectionnés dans le PS.
    
    Args:
        ps_data: Données du PS global
        selected_bins: Set de tuples (i, j) représentant les bins sélectionnés
    
    Returns:
        list: Liste des nœuds correspondant aux bins sélectionnés
    """
    bin_contributions = ps_data['bin_contributions']
    selected_nodes = set()
    
    for bin_coord in selected_bins:
        if bin_coord in bin_contributions:
            # Ajouter tous les nœuds qui contribuent à ce bin
            for node_id, _ in bin_contributions[bin_coord]:
                selected_nodes.add(node_id)
    
    return list(selected_nodes)


def compute_node_masks_per_timestep_optimized(ps_data, selected_nodes, cube_shape):
    """
    Version optimisée du calcul des masques de nœuds pour chaque timestep.
    Utilise un cache et évite la récursion.
    
    Args:
        ps_data: Données du PS global
        selected_nodes: Liste des nœuds sélectionnés
        cube_shape: Forme du cube (T, H, W)
    
    Returns:
        list: Liste de masques booléens pour chaque timestep
    """
    tree = ps_data['tree']
    T, H, W = cube_shape
    n_leaves = tree.num_leaves()
    frame_size = n_leaves // T
    
    # Cache pour les feuilles de chaque nœud (éviter la récursion répétée)
    if not hasattr(tree, '_leaves_cache'):
        tree._leaves_cache = {}
    
    def get_leaves_cached(node):
        """Version optimisée avec cache pour obtenir les feuilles d'un nœud."""
        if node in tree._leaves_cache:
            return tree._leaves_cache[node]
            
        if node < tree.num_leaves():
            leaves = [node]
        else:
            # Utiliser une pile au lieu de récursion
            stack = [node]
            leaves = []
            while stack:
                current = stack.pop()
                if current < tree.num_leaves():
                    leaves.append(current)
                else:
                    children = tree.children(current)
                    stack.extend(children)
        
        tree._leaves_cache[node] = leaves
        return leaves
    
    # Pré-calculer toutes les feuilles pour tous les nœuds sélectionnés
    all_affected_leaves = set()
    for node in selected_nodes:
        leaves = get_leaves_cached(node)
        all_affected_leaves.update(leaves)
    
    # Créer les masques plus efficacement
    masks_per_timestep = []
    for t in range(T):
        start_leaf = t * frame_size
        end_leaf = (t + 1) * frame_size
        
        # Filtrer les feuilles pour ce timestep
        timestep_leaves = [leaf for leaf in all_affected_leaves 
                          if start_leaf <= leaf < end_leaf]
        
        # Créer le masque directement
        mask_2d = np.zeros((H, W), dtype=bool)
        if timestep_leaves:
            # Conversion vectorisée
            local_indices = np.array(timestep_leaves) - start_leaf
            y_coords, x_coords = np.divmod(local_indices, W)
            
            # Filtrer les coordonnées valides
            valid_mask = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)
            y_coords = y_coords[valid_mask]
            x_coords = x_coords[valid_mask]
            
            if len(y_coords) > 0:
                mask_2d[y_coords, x_coords] = True
        
        masks_per_timestep.append(mask_2d)
    
    return masks_per_timestep


def compute_node_masks_per_timestep(ps_data, selected_nodes, cube_shape):
    """Wrapper pour la compatibilité - utilise la version optimisée."""
    return compute_node_masks_per_timestep_optimized(ps_data, selected_nodes, cube_shape)


def get_bin_coordinates_from_point(ps_data, x_coord, y_coord):
    """
    Convertit les coordonnées continues en indices de bins.
    
    Args:
        ps_data: Données du PS global
        x_coord: Coordonnée X (area, échelle log)
        y_coord: Coordonnée Y (stability, échelle linéaire)
    
    Returns:
        tuple: (i, j) indices du bin, ou None si hors limites
    """
    bins1 = ps_data['bins1']  # Area (log scale)
    bins2 = ps_data['bins2']  # Stability (linear scale)
    
    # Trouver l'index du bin pour area (échelle log)
    i = np.digitize(x_coord, bins1, right=False) - 1
    # Trouver l'index du bin pour stability (échelle linéaire)
    j = np.digitize(y_coord, bins2, right=False) - 1
    
    # Vérifier que les indices sont valides
    if 0 <= i < len(bins1) - 1 and 0 <= j < len(bins2) - 1:
        return (i, j)
    return None


def get_bins_in_polygon_selection(ps_data, polygon_coords):
    """
    Trouve tous les bins contenus dans un polygone de sélection sur le PS.
    
    Args:
        ps_data: Données du PS global
        polygon_coords: Liste de tuples (x, y) définissant le polygone
    
    Returns:
        set: Set de tuples (i, j) représentant les bins sélectionnés
    """
    from shapely.geometry import Polygon, Point
    
    # Créer le polygone de sélection
    selection_poly = Polygon(polygon_coords)
    
    bins1 = ps_data['bins1']
    bins2 = ps_data['bins2']
    hist2d = ps_data['hist2d']
    
    selected_bins = set()
    
    # Parcourir tous les bins et vérifier s'ils sont dans le polygone
    for i in range(len(bins1) - 1):
        for j in range(len(bins2) - 1):
            if hist2d[i, j] > 0:  # Seulement les bins non vides
                # Centre du bin
                x_center = (bins1[i] + bins1[i + 1]) / 2
                y_center = (bins2[j] + bins2[j + 1]) / 2
                
                # Vérifier si le centre est dans le polygone
                if selection_poly.contains(Point(x_center, y_center)):
                    selected_bins.add((i, j))
    
    return selected_bins
