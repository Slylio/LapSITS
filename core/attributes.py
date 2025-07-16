"""
Attributs et fonctions pour le calcul des features spatio-temporelles
Refactor des fonctions de old_code/st_attributes.py dans core/attributes.py
"""
import higra as hg
import numpy as np

# --- Fonctions d'attributs vectorisées (sur comp_area) ---
def compute_appear(comp_area):
    """Retourne True pour les nœuds apparaissant (0 -> >0)."""
    return (comp_area[:, 0] == 0) & (comp_area[:, -1] > 0)

def compute_disappear(comp_area):
    """Retourne True pour les nœuds disparaissant (>0 -> 0)."""
    return (comp_area[:, 0] > 0) & (comp_area[:, -1] == 0)

def compute_abruptness_std(comp_area):
    """Écart-type des différences relatives des aires."""
    return np.std(np.diff(comp_area, axis=1), axis=1)

def compute_abruptness_max(comp_area):
    """Max des différences relatives des aires normalisées."""
    diffs = np.diff(comp_area, axis=1)
    total = comp_area[:, -1] - comp_area[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.max(diffs, axis=1) / total
    result[~np.isfinite(result)] = 0
    return result

def compute_change_center(comp_area):
    """Centre de changement pondéré par la position des différences."""
    diffs = np.diff(comp_area, axis=1)
    weights = np.arange(1, diffs.shape[1] + 1).reshape(1, -1)
    total = np.sum(diffs, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        center = np.sum(diffs * weights, axis=1) / total
    center[~np.isfinite(center)] = -1
    return center

def compute_growth_trend(comp_area, epsilon=1e-8):
    """Tendance globale de croissance (>0 ou <0)."""
    diffs = np.diff(comp_area, axis=1)
    return np.sum(diffs, axis=1) / (np.sum(np.abs(diffs), axis=1) + epsilon)

def compute_signed_abruptness(comp_area, epsilon=1e-8):
    """Abruptness signée (tendance * intensité)."""
    diffs = np.diff(comp_area, axis=1)
    abs_sum = np.sum(np.abs(diffs), axis=1) + epsilon
    trend = np.sum(diffs, axis=1) / abs_sum
    abrupt = np.max(np.abs(diffs), axis=1) / abs_sum
    return trend * abrupt

# --- Fonctions d'attributs sur arbre et séquence ---
def compute_comp_area(tree, sits, time_indices=None):
    """Calcule comp_area: matrice (n_nodes, T) des aires temporelles."""
    T = sits.shape[0]
    n_leaves = tree.num_leaves()
    frame_size = n_leaves // T
    comp_area = np.stack([
        hg.accumulate_sequential(
            tree,
            (np.arange(n_leaves) // frame_size == i).astype(np.float32),
            hg.Accumulators.sum
        )
        for i in range(T)
    ], axis=1)
    return comp_area

def compute_growth_oriented_attr(tree, sits):
    """Attribut 'growth_oriented': moyenne des diff. d'aire sur durée active."""
    comp_area = compute_comp_area(tree, sits)
    delta = np.mean(comp_area[:, 1:] - comp_area[:, :-1], axis=1)
    duration = np.sum(comp_area > 0, axis=1)
    return delta / np.maximum(duration, 1)

def compute_STstability_attribute_v2(tree, time_indices):
    """Attribut de stabilité ST optimisé."""
    n_leaves = tree.num_leaves()
    T = len(time_indices)
    frame_size = n_leaves // T
    comp = np.stack([
        hg.accumulate_sequential(
            tree,
            (np.arange(n_leaves) // frame_size == i).astype(np.float32),
            hg.Accumulators.sum
        )
        for i in range(T)
    ], axis=1)
    a, b = comp[:, :-1], comp[:, 1:]
    valid = b > 0
    r = np.divide(np.minimum(a, b), np.maximum(a, b), out=np.zeros_like(a), where=valid)
    dt = np.diff(time_indices).astype(np.float32)
    stability = (r * dt).sum(axis=1) / dt.sum()
    stability[tree.root()] = 1.0
    return stability

def compute_STcentroid_attribute_v2(tree, time_indices):
    """Attribut centroïde ST des nœuds."""
    n_leaves = tree.num_leaves()
    t0 = min(time_indices)
    times = np.repeat(time_indices, n_leaves // len(time_indices)).astype(np.float32) - t0
    sum_t = hg.accumulate_sequential(tree, times, hg.Accumulators.sum)
    area = hg.attribute_area(tree)
    centroid = np.divide(sum_t, area, out=np.zeros_like(sum_t)) + t0
    centroid[tree.root()] = (time_indices[0] + time_indices[-1]) / 2.0
    return centroid

# --- Dictionnaires des fonctions et des bins ---
ATTR_FUNCS = {
    'area': lambda tree, sits: hg.attribute_area(tree),
    'stability': lambda tree, sits: compute_STstability_attribute_v2(tree, list(range(1, sits.shape[0]+1))),
    'growth_oriented': lambda tree, sits: compute_growth_oriented_attr(tree, sits),
    'abruptness_oriented': lambda tree, sits: compute_signed_abruptness(compute_comp_area(tree, sits)),
    'appear': lambda tree, sits: compute_appear(compute_comp_area(tree, sits)),
    'disappear': lambda tree, sits: compute_disappear(compute_comp_area(tree, sits)),
    'height': lambda tree, sits: hg.attribute_height(tree, hg.attribute_area(tree)),
    'depth': lambda tree, sits: hg.attribute_depth(tree),
    'mean_vertex_weights': lambda tree, sits: hg.attribute_mean_vertex_weights(tree, sits.ravel(), hg.attribute_area(tree)),
}

BINS = {
    'area': np.logspace(0, 7, 100),
    'stability': np.linspace(0, 1, 100),
    'growth_oriented': np.linspace(-10000, 10000, 100),
    'abruptness_oriented': np.linspace(-1, 1, 100),
    'appear': np.array([-0.5, 0.5, 1.5]),
    'disappear': np.array([-0.5, 0.5, 1.5]),
    'height': np.linspace(0, 255, 100),
    'depth': np.linspace(0, 100, 100),
    'mean_vertex_weights': np.linspace(0, 255, 100),
}

def compute_disappear(comp_area):
    return (comp_area[:, 0] > 0) & (comp_area[:, -1] == 0)

def compute_abruptness_std(comp_area):
    return np.std(np.diff(comp_area, axis=1), axis=1)

def compute_abruptness_max(comp_area):
    diffs = np.diff(comp_area, axis=1)
    total = comp_area[:, -1] - comp_area[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.max(diffs, axis=1) / total
    result[~np.isfinite(result)] = 0
    return result

def compute_change_center(comp_area):
    diffs = np.diff(comp_area, axis=1)
    weights = np.arange(1, diffs.shape[1] + 1).reshape(1, -1)
    total = np.sum(diffs, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        center = np.sum(diffs * weights, axis=1) / total
    center[~np.isfinite(center)] = -1
    return center

def compute_growth_trend(comp_area, epsilon=1e-8):
    diffs = np.diff(comp_area, axis=1)
    return np.sum(diffs, axis=1) / (np.sum(np.abs(diffs), axis=1) + epsilon)

def compute_signed_abruptness(comp_area, epsilon=1e-8):
    diffs = np.diff(comp_area, axis=1)
    abs_sum = np.sum(np.abs(diffs), axis=1) + epsilon
    trend = np.sum(diffs, axis=1) / abs_sum
    abrupt = np.max(np.abs(diffs), axis=1) / abs_sum
    return trend * abrupt

# --- Fonctions d'attributs sur arbre ---
def compute_STstability_attribute_v2(tree, time_indices):
    n_leaves = tree.num_leaves()
    T = len(time_indices)
    frame_size = n_leaves // T
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

def compute_STcentroid_attribute_v2(tree, time_indices):
    n_leaves = tree.num_leaves()
    t0 = min(time_indices)
    times = np.repeat(time_indices, n_leaves // len(time_indices)).astype(np.float32) - t0
    sum_t = hg.accumulate_sequential(tree, times, hg.Accumulators.sum)
    area = hg.attribute_area(tree)
    centroid = np.divide(sum_t, area, out=np.zeros_like(sum_t)) + t0
    centroid[tree.root()] = (time_indices[0] + time_indices[-1]) / 2.0
    return centroid

def compute_amplitude_attribute(tree, values):
    max_vals = hg.accumulate_sequential(tree, values, hg.Accumulators.max)
    min_vals = hg.accumulate_sequential(tree, values, hg.Accumulators.min)
    return max_vals - min_vals

def compute_comp_area(tree, sits, time_indices):
    T = sits.shape[0]
    n_leaves = tree.num_leaves()
    frame_size = n_leaves // T
    comp_area = np.stack([
        hg.accumulate_sequential(tree, (np.arange(n_leaves) // frame_size == i).astype(np.float32), hg.Accumulators.sum)
        for i in range(T)
    ], axis=1)
    return comp_area

def compute_delta_area_mean(comp_area):
    return np.mean(comp_area[:, 1:] - comp_area[:, :-1], axis=1)

def compute_growth_oriented_attr(tree, sits):
    comp_area = compute_comp_area(tree, sits, list(range(sits.shape[0])))
    delta = compute_delta_area_mean(comp_area)
    active_duration = np.sum(comp_area > 0, axis=1)
    return delta / np.maximum(1, active_duration)

# --- Dictionnaires de fonctions et bins ---
ATTR_FUNCS = {
    'area': lambda tree, sits: hg.attribute_area(tree),
    'stability': lambda tree, sits: compute_STstability_attribute_v2(tree, list(range(1, sits.shape[0]+1))),
    'growth_oriented': lambda tree, sits: compute_growth_oriented_attr(tree, sits),
    'abruptness_oriented': lambda tree, sits: compute_signed_abruptness(compute_comp_area(tree, sits, list(range(1, sits.shape[0]+1)))),
    'appear': lambda tree, sits: compute_appear(compute_comp_area(tree, sits, list(range(1, sits.shape[0]+1)))),
    'disappear': lambda tree, sits: compute_disappear(compute_comp_area(tree, sits, list(range(1, sits.shape[0]+1)))),
    'height': lambda tree, sits: hg.attribute_height(tree, hg.attribute_area(tree)),
    'depth': lambda tree, sits: hg.attribute_depth(tree),
    'mean_vertex_weights': lambda tree, sits: hg.attribute_mean_vertex_weights(tree, sits.ravel(), hg.attribute_area(tree)),
}

BINS = {
    'area': np.logspace(0, 7, 100),
    'stability': np.linspace(0, 1, 100),
    'growth_oriented': np.linspace(-10000, 10000, 100),
    'abruptness_oriented': np.linspace(-1, 1, 100),
    'appear': np.array([-0.5, 0.5, 1.5]),
    'disappear': np.array([-0.5, 0.5, 1.5]),
    'height': np.linspace(0, 255, 100),
    'depth': np.linspace(0, 100, 100),
    'mean_vertex_weights': np.linspace(0, 255, 100),
}

BINS_SCALES = {
    'area' : 'log',
    'stability' : 'linear',
    'growth_oriented' : 'linear',
    'abruptness_oriented' : 'linear',
    'appear' : 'linear',
    'disappear' : 'linear',
    'height' : 'linear',
    'depth' : 'linear',
    'mean_vertex_weights' : 'linear',
}