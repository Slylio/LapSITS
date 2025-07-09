"""
Module for Pattern Spectra (PS) calculations on temporal image sequences (SITS).
Based on compute_ps_from_cube.py but adapted for the graphical interface.
"""

import numpy as np
import higra as hg
from matplotlib.colors import LogNorm
from skimage.draw import polygon as sk_polygon
from shapely.geometry import Polygon
from matplotlib import cm


def compute_STstability_attribute_v2(tree, time_indices):
    n_leaves = tree.num_leaves()
    T = len(time_indices) # number of timesteps
    frame_size = n_leaves // T # size of leaf slices per timestep
  
    """
    np.arange(n_leaves) // frame_size == i : true on leaves of slice i (ex t1 : True x16, False x32, t2 : False x16, True x16, False x16 etc..)
    hg.accumulate_sequential(tree, (np.arange(n_leaves) // frame_size == i).astype(np.float32), hg.Accumulators.sum) : area calculation taking only leaves from slice i
    t1 :[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.
        0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
        0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  2.  1.  3.
        4.  6.  0. 16.]
    And we stack this in comp
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
    """Converts a Shapely polygon to a boolean mask."""
    if poly.is_empty:
        return np.zeros(shape, dtype=bool)
    x, y = poly.exterior.coords.xy
    rr, cc = sk_polygon(y, x, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def pixelset_to_contained_subtree(tree, mask3d):
    """Finds tree nodes entirely contained in the 3D mask."""
    return hg.accumulate_and_min_sequential(
        tree,
        np.ones(tree.num_vertices(), dtype=np.uint8),
        mask3d.ravel().astype(np.uint8),
        hg.Accumulators.min
    ).astype(bool)


def extract_related_bins_list(tree, node_to_bin2d, node_list):
    """Extracts all bins associated with a list of nodes."""
    bins_set = set()
    for node_id in node_list:
        # Get the subtree of the node
        _, node_map = tree.sub_tree(node_id)
        # Add all bins associated with nodes in the subtree
        for n in node_map:
            if n in node_to_bin2d:
                bins_set.add(node_to_bin2d[n])
    return bins_set


def compute_2d_ps_with_tracking(tree, altitudes, attr1, attr2, bins1, bins2):
    """
    Computes a 2D pattern spectra with node contribution tracking.
    
    Args:
        tree: Component tree
        altitudes: Node altitudes
        attr1, attr2: Attributes for the two dimensions
        bins1, bins2: Bins for the 2D histogram
    
    Returns:
        tuple: (tree, hist2d, bins1, bins2, node_to_bin2d, bin_contributions)
    """
    area = hg.attribute_area(tree)
    vol_nodes = area * np.abs(altitudes - altitudes[tree.parents()])

    # 2D Histogram
    hist2d, _, _ = np.histogram2d(attr1, attr2, bins=[bins1, bins2], weights=vol_nodes)
    if hist2d.sum() > 0:
        hist2d /= hist2d.sum()

    # Calculate bin indices for each node
    bin_idx_1 = np.digitize(attr1, bins1, right=False) - 1
    bin_idx_2 = np.digitize(attr2, bins2, right=False) - 1

    node_to_bin2d = {}
    for node, (i, j) in enumerate(zip(bin_idx_1, bin_idx_2)):
        if 0 <= i < len(bins1) - 1 and 0 <= j < len(bins2) - 1:
            node_to_bin2d[node] = (i, j)

    # Node contributions to bins
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
    Computes the global pattern spectra for a 3D image sequence.
    
    Args:
        cube: 3D image sequence (T, H, W)
    
    Returns:
        dict: Dictionary containing all PS calculation results
    """
    cube = cube.astype(np.uint8)
    
    # Build 3D tree
    tree, altitudes = hg.component_tree_tree_of_shapes_image3d(cube)
    altitudes = altitudes.astype(np.float32)
    
    # Calculate attributes
    area = hg.attribute_area(tree)
    stability = compute_STstability_attribute_v2(tree, [1, 2, 3, 4, 5])
    
    # Fixed bins for consistency
    bins1 = np.logspace(0, 7, 100)  # Area (log scale)
    bins2 = np.linspace(0, 1, 100)  # Stability (linear scale)
    
    # Calculate PS with tracking
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
    Calculates the bins to highlight for a given polygon.
    
    Args:
        ps_data: Global PS data (return from compute_global_ps)
        polygon: Shapely polygon
        cube_shape: Cube shape (T, H, W)
    
    Returns:
        tuple: (highlight_bins, contained_nodes)
    """
    tree = ps_data['tree']
    node_to_bin2d = ps_data['node_to_bin2d']
    
    # 3D polygon mask
    mask3d = np.stack([shapely_to_mask(polygon, cube_shape[1:]) for _ in range(cube_shape[0])])
    
    # Nodes contained in the polygon
    node_mask = pixelset_to_contained_subtree(tree, mask3d)
    contained_nodes = np.where(node_mask)[0]
    contained_nodes = [n for n in contained_nodes if n >= tree.num_leaves()]
    
    # Associated bins
    highlight_bins = extract_related_bins_list(tree, node_to_bin2d, contained_nodes)
    
    return highlight_bins, contained_nodes


def plot_ps_with_highlights(ax, ps_data, highlight_bins=None, contained_nodes=None):
    """
    Displays the pattern spectra with optional highlighting.
    
    Args:
        ax: Matplotlib axes
        ps_data: Global PS data
        highlight_bins: Bins to highlight (optional)
        contained_nodes: Selected nodes (optional)
    
    Returns:
        matplotlib.collections.QuadMesh: The mesh object for the colorbar
    """
    hist2d = ps_data['hist2d']
    bins1 = ps_data['bins1']
    bins2 = ps_data['bins2']
    bin_contributions = ps_data['bin_contributions']
    
    # Display global PS in background
    cmap = cm.get_cmap('viridis')
    norm_bg = LogNorm(vmin=hist2d[hist2d > 0].min(), vmax=hist2d.max())
    pcm = ax.pcolormesh(bins1, bins2, hist2d.T, norm=norm_bg, cmap=cmap, shading='auto', alpha=0.3)
    
    # Highlight selected bins
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
    
    # Axis configuration
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlabel('Area (log scale)')
    ax.set_ylabel('Stability')
    ax.set_title('Pattern Spectra')
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    
    return pcm


def compute_nodes_from_bins(ps_data, selected_bins):
    """
    Calculates the nodes corresponding to selected bins in the PS.
    
    Args:
        ps_data: Global PS data
        selected_bins: Set of tuples (i, j) representing selected bins
    
    Returns:
        list: List of nodes corresponding to selected bins
    """
    bin_contributions = ps_data['bin_contributions']
    selected_nodes = set()
    
    for bin_coord in selected_bins:
        if bin_coord in bin_contributions:
            # Add all nodes that contribute to this bin
            for node_id, _ in bin_contributions[bin_coord]:
                selected_nodes.add(node_id)
    
    return list(selected_nodes)


def compute_node_masks_per_timestep_optimized(ps_data, selected_nodes, cube_shape):
    """
    Optimized version of node mask calculation for each timestep.
    Uses cache and avoids recursion.
    
    Args:
        ps_data: Global PS data
        selected_nodes: List of selected nodes
        cube_shape: Cube shape (T, H, W)
    
    Returns:
        list: List of boolean masks for each timestep
    """
    tree = ps_data['tree']
    T, H, W = cube_shape
    n_leaves = tree.num_leaves()
    frame_size = n_leaves // T
    
    # Cache for leaves of each node (avoid repeated recursion)
    if not hasattr(tree, '_leaves_cache'):
        tree._leaves_cache = {}
    
    def get_leaves_cached(node):
        """Optimized version with cache to get leaves of a node."""
        if node in tree._leaves_cache:
            return tree._leaves_cache[node]
            
        if node < tree.num_leaves():
            leaves = [node]
        else:
            # Use stack instead of recursion
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
    
    # Pre-calculate all leaves for all selected nodes
    all_affected_leaves = set()
    for node in selected_nodes:
        leaves = get_leaves_cached(node)
        all_affected_leaves.update(leaves)
    
    # Create masks more efficiently
    masks_per_timestep = []
    for t in range(T):
        start_leaf = t * frame_size
        end_leaf = (t + 1) * frame_size
        
        # Filter leaves for this timestep
        timestep_leaves = [leaf for leaf in all_affected_leaves 
                          if start_leaf <= leaf < end_leaf]
        
        # Create the mask directly
        mask_2d = np.zeros((H, W), dtype=bool)
        if timestep_leaves:
            # Vectorized conversion
            local_indices = np.array(timestep_leaves) - start_leaf
            y_coords, x_coords = np.divmod(local_indices, W)
            
            # Filter valid coordinates
            valid_mask = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)
            y_coords = y_coords[valid_mask]
            x_coords = x_coords[valid_mask]
            
            if len(y_coords) > 0:
                mask_2d[y_coords, x_coords] = True
        
        masks_per_timestep.append(mask_2d)
    
    return masks_per_timestep


def compute_node_masks_per_timestep(ps_data, selected_nodes, cube_shape):
    """Wrapper for compatibility - uses the optimized version."""
    return compute_node_masks_per_timestep_optimized(ps_data, selected_nodes, cube_shape)


def get_bin_coordinates_from_point(ps_data, x_coord, y_coord):
    """
    Converts continuous coordinates to bin indices.
    
    Args:
        ps_data: Global PS data
        x_coord: X coordinate (area, log scale)
        y_coord: Y coordinate (stability, linear scale)
    
    Returns:
        tuple: (i, j) bin indices, or None if out of bounds
    """
    bins1 = ps_data['bins1']  # Area (log scale)
    bins2 = ps_data['bins2']  # Stability (linear scale)
    
    # Find bin index for area (log scale)
    i = np.digitize(x_coord, bins1, right=False) - 1
    # Find bin index for stability (linear scale)
    j = np.digitize(y_coord, bins2, right=False) - 1
    
    # Check that indices are valid
    if 0 <= i < len(bins1) - 1 and 0 <= j < len(bins2) - 1:
        return (i, j)
    return None


def get_bins_in_polygon_selection(ps_data, polygon_coords):
    """
    Finds all bins contained in a selection polygon on the PS.
    
    Args:
        ps_data: Global PS data
        polygon_coords: List of tuples (x, y) defining the polygon
    
    Returns:
        set: Set of tuples (i, j) representing selected bins
    """
    from shapely.geometry import Polygon, Point
    
    # Create the selection polygon
    selection_poly = Polygon(polygon_coords)
    
    bins1 = ps_data['bins1']
    bins2 = ps_data['bins2']
    hist2d = ps_data['hist2d']
    
    selected_bins = set()
    
    # Go through all bins and check if they are in the polygon
    for i in range(len(bins1) - 1):
        for j in range(len(bins2) - 1):
            if hist2d[i, j] > 0:  # Only non-empty bins
                # Bin center
                x_center = (bins1[i] + bins1[i + 1]) / 2
                y_center = (bins2[j] + bins2[j + 1]) / 2
                
                # Check if the center is in the polygon
                if selection_poly.contains(Point(x_center, y_center)):
                    selected_bins.add((i, j))
    
    return selected_bins


def generate_unique_colors(num_colors):
    """
    Generates unique colors for visualizing different nodes.
    
    Args:
        num_colors: Number of unique colors needed
    
    Returns:
        list: List of RGBA tuples for each node
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    if num_colors == 0:
        return []
    
    # Use a colormap that provides good contrast
    if num_colors <= 10:
        # For small numbers, use qualitative colors
        colors = cm.tab10(np.linspace(0, 1, min(num_colors, 10)))
    elif num_colors <= 20:
        # For medium numbers, use tab20
        colors = cm.tab20(np.linspace(0, 1, min(num_colors, 20)))
    else:
        # For large numbers, use hsv for maximum differentiation
        colors = cm.hsv(np.linspace(0, 1, num_colors))
    
    # Convert to RGBA with good alpha for visibility
    rgba_colors = []
    for color in colors:
        rgba = list(color)
        rgba[3] = 0.7  # Set alpha to 0.7 for good visibility
        rgba_colors.append(tuple(int(c * 255) for c in rgba))
    
    return rgba_colors


def compute_node_masks_per_timestep_with_colors(ps_data, selected_nodes, cube_shape, use_unique_colors=False):
    """
    Optimized version of node mask calculation with optional unique colors per node.
    
    Args:
        ps_data: Global PS data
        selected_nodes: List of selected nodes
        cube_shape: Cube shape (T, H, W)
        use_unique_colors: If True, assign unique colors to each node
    
    Returns:
        tuple: (masks_per_timestep, colors_per_timestep) if use_unique_colors=True
               or just masks_per_timestep if use_unique_colors=False
    """
    tree = ps_data['tree']
    T, H, W = cube_shape
    n_leaves = tree.num_leaves()
    frame_size = n_leaves // T
    
    # Cache for leaves of each node
    if not hasattr(tree, '_leaves_cache'):
        tree._leaves_cache = {}
    
    def get_leaves_cached(node):
        """Optimized version with cache to get leaves of a node."""
        if node in tree._leaves_cache:
            return tree._leaves_cache[node]
            
        if node < tree.num_leaves():
            leaves = [node]
        else:
            # Use stack instead of recursion
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
    
    # Generate unique colors if requested
    unique_colors = None
    if use_unique_colors:
        unique_colors = generate_unique_colors(len(selected_nodes))
        # Create a mapping from node to color
        node_to_color = {node: unique_colors[i] for i, node in enumerate(selected_nodes)}
    
    # Pre-calculate leaves for each node
    node_leaves = {}
    for node in selected_nodes:
        node_leaves[node] = get_leaves_cached(node)
    
    # Create masks and color maps for each timestep
    masks_per_timestep = []
    colors_per_timestep = [] if use_unique_colors else None
    
    for t in range(T):
        start_leaf = t * frame_size
        end_leaf = (t + 1) * frame_size
        
        if use_unique_colors:
            # Create a color map for this timestep
            color_map = np.zeros((H, W, 4), dtype=np.uint8)
            mask_2d = np.zeros((H, W), dtype=bool)
            
            for node in selected_nodes:
                # Get leaves for this node in this timestep
                timestep_leaves = [leaf for leaf in node_leaves[node] 
                                 if start_leaf <= leaf < end_leaf]
                
                if timestep_leaves:
                    # Convert to 2D coordinates
                    local_indices = np.array(timestep_leaves) - start_leaf
                    y_coords, x_coords = np.divmod(local_indices, W)
                    
                    # Filter valid coordinates
                    valid_mask = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)
                    y_coords = y_coords[valid_mask]
                    x_coords = x_coords[valid_mask]
                    
                    if len(y_coords) > 0:
                        # Set the unique color for this node
                        color = node_to_color[node]
                        color_map[y_coords, x_coords] = color
                        mask_2d[y_coords, x_coords] = True
            
            colors_per_timestep.append(color_map)
            masks_per_timestep.append(mask_2d)
            
        else:
            # Original single-color implementation
            all_affected_leaves = set()
            for node in selected_nodes:
                leaves = node_leaves[node]
                all_affected_leaves.update(leaves)
            
            # Filter leaves for this timestep
            timestep_leaves = [leaf for leaf in all_affected_leaves 
                              if start_leaf <= leaf < end_leaf]
            
            # Create the mask directly
            mask_2d = np.zeros((H, W), dtype=bool)
            if timestep_leaves:
                # Vectorized conversion
                local_indices = np.array(timestep_leaves) - start_leaf
                y_coords, x_coords = np.divmod(local_indices, W)
                
                # Filter valid coordinates
                valid_mask = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)
                y_coords = y_coords[valid_mask]
                x_coords = x_coords[valid_mask]
                
                if len(y_coords) > 0:
                    mask_2d[y_coords, x_coords] = True
            
            masks_per_timestep.append(mask_2d)
    
    if use_unique_colors:
        return masks_per_timestep, colors_per_timestep
    else:
        return masks_per_timestep
