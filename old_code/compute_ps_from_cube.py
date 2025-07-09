import numpy as np
import higra as hg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.draw import polygon as sk_polygon
from shapely.geometry import Polygon
from matplotlib import cm
from utils import load_3D_sequence, load_npz_file

#Toolbox to compute pattern spectra from 3D cubes.

# ========== utils ==========
def shapely_to_mask(poly: Polygon, shape):
    if poly.is_empty:
        return np.zeros(shape, dtype=bool)
    x, y = poly.exterior.coords.xy
    rr, cc = sk_polygon(y, x, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask

def pixelset_to_contained_subtree(tree, mask3d):
    return hg.accumulate_and_min_sequential(
        tree,
        np.ones(tree.num_vertices(), dtype=np.uint8),
        mask3d.ravel().astype(np.uint8),
        hg.Accumulators.min
    ).astype(bool)

def extract_related_bins(tree, node_to_bin2d, node_id):
    _, node_map = tree.sub_tree(node_id)
    return {node_to_bin2d[n] for n in node_map if n in node_to_bin2d}

def extract_related_bins_list(tree, node_to_bin2d, node_list):
    bins_set = set()
    for node_id in node_list:
        bins_set.update(extract_related_bins(tree, node_to_bin2d, node_id))
    return bins_set

def associate_sequence_polygon(sequence, polygon_folder="output_images/polygon"):
    """
    Loads a 3D image sequence and its associated polygon from a .npz file.

    Args:
        sequence (list of str): List of image file paths for the sequence.
        polygon_folder (str): Folder where polygon .npz files are stored.

    Returns:
        tuple: (cube, polygon) if successful, else None.
    """
    cube = load_3D_sequence(sequence)
    # Construct the expected npz path
    npz_path = sequence[0].replace(".png", "_poly.npz")
    npz_path = npz_path.replace("output_images/larger", polygon_folder)
    npz_path = npz_path.replace("_50_", "_")
    # Try loading the npz file
    npz_data = load_npz_file(npz_path)
    if npz_data is None:
        # Try alternative path for simplified images
        alt_npz_path = npz_path.replace("simplified_images", "output_images/polygon")
        print(f"Trying alternative path: {alt_npz_path}")
        npz_data = load_npz_file(alt_npz_path)
        
        if npz_data is None:
            print(f"Failed to load polygon data from {npz_path} or {alt_npz_path}")
            return None
        npz_path = alt_npz_path
    if 'coords' not in npz_data:
        print(f"No 'coords' found in the npz file: {npz_path}")
        return None
    coords = npz_data['coords']
    if len(coords) < 3:
        print(f"Polygon coordinates are insufficient in {npz_path}")
        return None
    poly = Polygon(coords)
    cube = cube.astype(np.uint8)
    return cube, poly

#========== PS COMPUTATION ==========
def compute_2d_ps_fixed_bins_with_tracking(tree, altitudes, attr1, attr2, bins1=np.logspace(2, 7, 100), bins2=np.linspace(0, 1, 100)):
    import numpy as np
    area = hg.attribute_area(tree)
    vol_nodes = area * abs(altitudes - altitudes[tree.parents()])

    # Histogramme 2D (inchangé)
    hist2d, _, _ = np.histogram2d(attr1, attr2, bins=[bins1, bins2], weights=vol_nodes)
    hist2d /= hist2d.sum()

    # Calcul des indices de bin pour chaque nœud
    bin_idx_1 = np.digitize(attr1, bins1, right=False) - 1
    bin_idx_2 = np.digitize(attr2, bins2, right=False) - 1

    node_to_bin2d = {}
    for node, (i, j) in enumerate(zip(bin_idx_1, bin_idx_2)):
        if 0 <= i < len(bins1) - 1 and 0 <= j < len(bins2) - 1:
            node_to_bin2d[node] = (i, j)
            
    bin_contributions = dict() #bin_contributions[(i, j)] = [(node, contribution_volumique associée)] -> permet de mesurer quels noeuds contribuentà quel point. 
    for n in range(tree.num_vertices()):
        i, j = node_to_bin2d.get(n, (None, None))
        if i is not None:
            v = area[n] * abs(altitudes[n] - altitudes[tree.parents()[n]])
            if (i, j) not in bin_contributions:
                bin_contributions[(i, j)] = []
            bin_contributions[(i, j)].append((n, v))

    return tree, hist2d, bins1, bins2, node_to_bin2d, bin_contributions


#==========Affichage===========
def plot_2d_ps_on_ax(ax, hist, bins1, bins2, attr1, attr2,
                     highlight_bins=None, bin_contributions=None, node_subset=None):

    X, Y = np.meshgrid(bins1, bins2, indexing='ij')
    assert np.isfinite(hist).all() and np.isfinite(bins1).all() and np.isfinite(bins2).all()

    cmap = cm.get_cmap('viridis')
    norm_bg = LogNorm(vmin=hist[hist > 0].min(), vmax=hist.max())
    pcm = ax.pcolormesh(X, Y, hist, norm=norm_bg, cmap=cmap, shading='auto', alpha=0.2)

    if highlight_bins and bin_contributions is not None and node_subset is not None:
        hist_mask = np.zeros_like(hist)
        selected_set = set(node_subset)
        for (i, j) in highlight_bins:
            contribs = bin_contributions.get((i, j), [])
            v = sum(v for n, v in contribs if n in selected_set)
            hist_mask[i, j] = v

        if hist_mask.sum() > 0:
            norm_fg = LogNorm(vmin=hist_mask[hist_mask > 0].min(), vmax=hist_mask.max())
            pcm = ax.pcolormesh(X, Y, hist_mask, norm=norm_fg,
                                cmap=cm.get_cmap('coolwarm'), shading='auto', alpha=1.0)
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlabel(attr1)
    ax.set_ylabel(attr2)
    ax.grid(True, which='major', linestyle='-', linewidth=0.5)
    return pcm

def show_2d_ps_log(hist, attr1, attr2, bins1, bins2,
                   highlight_bins=None, bin_contributions=None, node_subset=None,
                   return_fig=False):
    fig, ax = plt.subplots()
    pcm = plot_2d_ps_on_ax(
        ax, hist, bins1, bins2, attr1, attr2,
        highlight_bins, bin_contributions, node_subset
    )
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlabel(attr1)
    ax.set_ylabel(attr2)
    ax.grid(True, which='major', linestyle='-', linewidth=0.5)
    plt.title('2D Pattern Spectra')
    fig.colorbar(pcm, ax=ax, label='Volume')
    if return_fig:
        return fig
    else:
        plt.show()

def show_seq_ps(cube, hist, bins1, bins2, attr1, attr2,
                highlight_bins=None, bin_contributions=None, node_subset=None,
                polygon=None, titles=None, return_fig=False):

    T = cube.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(4 * T, 8))

    if polygon is not None:
        mask2d = shapely_to_mask(polygon, cube[0].shape)
    else:
        mask2d = None

    for t in range(T):
        ax = axes[0, t]
        ax.imshow(cube[t], cmap='gray')
        if mask2d is not None:
            ax.contour(mask2d, colors='red', linewidths=1)
        if titles:
            ax.set_title(titles[t], fontsize=10)
        ax.axis('off')

    for i in range(T):
        if i != T // 2:
            axes[1, i].axis('off')

    ax_ps = axes[1, T // 2]
    pcm = plot_2d_ps_on_ax(
        ax_ps, hist, bins1, bins2, attr1, attr2,
        highlight_bins, bin_contributions, node_subset
    )
    ax_ps.set_title('2D Pattern Spectra')
    fig.colorbar(pcm, ax=ax_ps, label='Volume')
    plt.tight_layout()
    if return_fig:
        return fig
    else:
        plt.show()
        
def custom_process_cube(cube, tree, altitudes, attr1, attr2, attr1_str, attr2_str, bins1, bins2, polygon=None, display=False):
    tree, hist2d, bins1, bins2, node_to_bin2d, bin_contributions = compute_2d_ps_fixed_bins_with_tracking(
        tree, altitudes, attr1, attr2, bins1, bins2
    )
    highlight_bins = None
    contained_nodes = None

    if polygon is not None:
        mask3d = np.stack([shapely_to_mask(polygon, cube[0].shape) for _ in range(cube.shape[0])])
        node_mask = pixelset_to_contained_subtree(tree, mask3d)
        contained_nodes = np.where(node_mask)[0]
        contained_nodes = [n for n in contained_nodes if n >= tree.num_leaves()]
        highlight_bins = extract_related_bins_list(tree, node_to_bin2d, contained_nodes)

        if display:
            fig = show_seq_ps(
                cube, hist2d, bins1, bins2, attr1_str, attr2_str,
                highlight_bins=highlight_bins, bin_contributions=bin_contributions,
                node_subset=contained_nodes, polygon=polygon, return_fig=True
            )
        else:
            contained_nodes = None
            fig = None
    else:
        contained_nodes = None
        fig = None

    return hist2d, bins1, bins2, highlight_bins, bin_contributions, fig, contained_nodes


def compute_nd_ps_global(tree, altitudes, attr_list, bins_list):
    internal_nodes = np.arange(tree.num_leaves(), tree.num_vertices())
    attr_vals = [attr[internal_nodes] for attr in attr_list]
    vol = hg.attribute_area(tree)[internal_nodes] * np.abs(
        altitudes[internal_nodes] - altitudes[tree.parents()[internal_nodes]]
    )
    hist, _ = np.histogramdd(
        sample=np.stack(attr_vals, axis=1),
        bins=bins_list,
        weights=vol
    )
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


"""Compute un ps avec n attributs, uniquement en local."""
def compute_nd_ps_local(cube, tree, altitudes, attr_list, bins_list, polygon):
    # Masque spatial 3D (polygone répété sur T)
    mask3d = np.stack([shapely_to_mask(polygon, cube[0].shape) for _ in range(cube.shape[0])])
    node_mask = pixelset_to_contained_subtree(tree, mask3d)
    contained_nodes = np.where(node_mask)[0]
    contained_nodes = contained_nodes[contained_nodes >= tree.num_leaves()]

    if len(contained_nodes) == 0:
        return None

    attr_vals = [attr[contained_nodes] for attr in attr_list]

    # Volume pondéré
    vol_local = hg.attribute_area(tree)[contained_nodes] * np.abs(
        altitudes[contained_nodes] - altitudes[tree.parents()[contained_nodes]]
    )

    # nD histogramme pondéré
    hist, _ = np.histogramdd(
        sample=np.stack(attr_vals, axis=1),
        bins=bins_list,
        weights=vol_local
    )
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def custom_process_cube_local(cube, tree, altitudes, attr1, attr2, attr1_str, attr2_str, bins1, bins2, polygon, display=False):
    tree, hist2d, bins1, bins2, node_to_bin2d, bin_contributions = compute_2d_ps_fixed_bins_with_tracking(
        tree, altitudes, attr1, attr2, bins1, bins2
    )
    highlight_bins = None
    contained_nodes = None

    mask3d = np.stack([shapely_to_mask(polygon, cube[0].shape) for _ in range(cube.shape[0])])
    node_mask = pixelset_to_contained_subtree(tree, mask3d)
    contained_nodes = np.where(node_mask)[0]
    contained_nodes = [n for n in contained_nodes if n >= tree.num_leaves()]
    highlight_bins = extract_related_bins_list(tree, node_to_bin2d, contained_nodes)

    # Histogramme local restreint
    attr1_local = attr1[contained_nodes]
    attr2_local = attr2[contained_nodes]
    vol_local = hg.attribute_area(tree)[contained_nodes] * np.abs(
        altitudes[contained_nodes] - altitudes[tree.parents()[contained_nodes]]
    )
    hist2d_local, _, _ = np.histogram2d(attr1_local, attr2_local, bins=[bins1, bins2], weights=vol_local)
    if hist2d_local.sum() > 0:
        hist2d_local /= hist2d_local.sum()

    if display:
        fig = show_seq_ps(
            cube, hist2d, bins1, bins2, attr1_str, attr2_str,
            highlight_bins=highlight_bins, bin_contributions=bin_contributions,
            node_subset=contained_nodes, polygon=polygon, return_fig=True
        )
    else:
        fig = None

    return hist2d_local, bins1, bins2, highlight_bins, bin_contributions, fig

# ========== PIPELINE PRINCIPAL ==========
def process_cube(cube, polygon=None):
    from st_features import compute_STstability_attribute_v2
    cube = cube.astype(np.uint8)
    # Construction arbre 3D
    """
    mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    """
    #graph = hg.get_nd_regular_graph(cube.shape, hg.mask_2_neighbours(mask))
    #tree, altitudes = hg.component_tree_max_tree(graph, cube.ravel())
    tree, altitudes = hg.component_tree_tree_of_shapes_image3d(cube)
    altitudes = altitudes.astype(np.float32)
    #print(f"Altitudes : {altitudes.shape}, min: {altitudes.min()}, max: {altitudes.max()}")
    area = hg.attribute_area(tree)
    stability = compute_STstability_attribute_v2(tree, [1,2,3,4,5])
    #bins1 = np.logspace(np.log10(max(1, area.min())), np.log10(area.max() + 1), 100)
    #bins2 = np.linspace(stability.min(), stability.max(), 100)
    
    bins1 = np.logspace(0,7,100)
    bins2 = np.linspace(0, 1, 100)

    tree, hist2d, bins1, bins2, node_to_bin2d, bin_contributions = compute_2d_ps_fixed_bins_with_tracking( #from
        tree, altitudes, area, stability, bins1, bins2
    )

    highlight_bins = None
    if polygon is not None:
        mask3d = np.stack([shapely_to_mask(polygon, cube[0].shape) for _ in range(cube.shape[0])])
        node_mask = pixelset_to_contained_subtree(tree, mask3d)
        contained_nodes = np.where(node_mask)[0]
        contained_nodes = [n for n in contained_nodes if n >= tree.num_leaves()]
        highlight_bins = extract_related_bins_list(tree, node_to_bin2d, contained_nodes)

        
        fig = show_seq_ps(cube, hist2d, bins1, bins2, "Area", "Stability",
                    highlight_bins=highlight_bins, bin_contributions=bin_contributions,
                    node_subset=contained_nodes, polygon=polygon, return_fig=True)
    else :
        print("No polygon.")
    return hist2d, bins1, bins2, highlight_bins, bin_contributions, fig, contained_nodes
