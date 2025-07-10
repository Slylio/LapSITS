import higra as hg
import numpy as np
import timeit

def benchmark_equal(func_v1, func_v2, args, n_iter=20):
    t1 = timeit.timeit(lambda: func_v1(*args), number=n_iter) / n_iter
    t2 = timeit.timeit(lambda: func_v2(*args), number=n_iter) / n_iter
    out1 = func_v1(*args)
    out2 = func_v2(*args)
    #On vérifie seulement que les noeuds soient bons
    out1 = out1[args[0].num_leaves():]
    out2 = out2[args[0].num_leaves():]
    print(out1)
    print(out2)

    if not np.array_equal(out1, out2):
        raise AssertionError("Changement d'output entre les deux versions")
    print(f"avg_time_v1: {t1:.6f} s")
    print(f"avg_time_v2: {t2:.6f} s")
    return t1, t2

def compute_appear(comp_area):
    return (comp_area[:, 0] == 0) & (comp_area[:, -1] > 0)

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

#bins : 0,1,2,3 ... len of time series
def compute_change_center_base(comp_area):
    diffs = np.diff(comp_area)
    weights = np.arange(1, len(comp_area)) 
    total = np.sum(diffs)
    if total == 0:
        return -1 #si pas de variation, on retourne -1
    return np.sum(diffs * weights) / total

#Vectorized version of compute_change_center_base
def compute_change_center(comp_area):
    diffs = np.diff(comp_area, axis=1)  # (n_nodes, T-1)
    weights = np.arange(1, diffs.shape[1] + 1).reshape(1, -1)  # (1, T-1)
    total = np.sum(diffs, axis=1)  # (n_nodes,)
    with np.errstate(divide='ignore', invalid='ignore'):
        center = np.sum(diffs * weights, axis=1) / total
    center[~np.isfinite(center)] = -1
    return center


#1 if increasing, -1 if decreasing
def compute_growth_trend(comp_area, epsilon=1e-8):
    diffs = np.diff(comp_area, axis=1)
    return np.sum(diffs, axis=1) / (np.sum(np.abs(diffs), axis=1) + epsilon)

def compute_signed_abruptness(comp_area, epsilon=1e-8):
    diffs = np.diff(comp_area, axis=1)
    abs_sum = np.sum(np.abs(diffs), axis=1) + epsilon
    trend = np.sum(diffs, axis=1) / abs_sum
    abrupt = np.max(np.abs(diffs), axis=1) / abs_sum
    return trend * abrupt


def compute_STcentroid_attribute_v2(tree, time_indices):
    n_leaves = tree.num_leaves()
    t0 = min(time_indices)
    times = np.repeat(time_indices, n_leaves // len(time_indices)).astype(np.float32) - t0 #times des feuilles : [0,0,0 ... , 1, 1, 1, ... , 2, 2, 2, ...]
    sum_t = hg.accumulate_sequential(tree, times, hg.Accumulators.sum) #accumulation bottom-up du temps
    area = hg.attribute_area(tree) #area
    centroid = np.divide(sum_t, area, out=np.zeros_like(sum_t)) + t0 #calcul du centroïde
    centroid[tree.root()] = (time_indices[0] + time_indices[-1]) / 2.0 #cas particulier de la racine
    return centroid


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
    
    
def compute_amplitude_attribute(tree, values):
    max_vals = hg.accumulate_sequential(tree, values, hg.Accumulators.max)
    min_vals = hg.accumulate_sequential(tree, values, hg.Accumulators.min)
    amplitude = max_vals - min_vals
    return amplitude


def compute_begin_end_duration(tree, time_indices):
    n_leaves = tree.num_leaves()
    T = len(time_indices)
    time_indices = np.asarray(time_indices)
    frame_size = n_leaves // T
    comp_area = np.stack([
    hg.accumulate_sequential(tree, (np.arange(n_leaves) // frame_size == i).astype(np.float32), hg.Accumulators.sum)
    for i in range(T)
], axis=1)  # shape: (n_nodes, T)
    T = comp_area.shape[1]

    support = comp_area > 0  # (n_nodes, T)

    # Begin : premier t où A_t > 0
    t_begin_idx = np.argmax(support, axis=1)  # indices temporels (0-based)
    begin = time_indices[t_begin_idx]         # valeurs réelles de temps

    # End : dernier t où A_t > 0
    t_end_idx = T - 1 - np.argmax(support[:, ::-1], axis=1)
    end = time_indices[t_end_idx]

    # Duration : nombre de pas entre begin et end (inclus)
    duration = end - begin + 1

    return begin, end, duration


def compute_begin_end_duration_from_comp_area(comp_area, time_indices):
    T = comp_area.shape[1]
    time_indices = np.asarray(time_indices)

    support = comp_area > 0  # (n_nodes, T)

    t_begin_idx = np.argmax(support, axis=1)
    begin = time_indices[t_begin_idx]

    t_end_idx = T - 1 - np.argmax(support[:, ::-1], axis=1)
    end = time_indices[t_end_idx]

    duration = end - begin + 1
    return begin, end, duration


def compute_height(tree, sits, time_indices):
    return hg.attribute_height(tree, hg.attribute_area(tree))

def compute_comp_area(tree, sits, time_indices):
    time_indices = list(range(1, sits.shape[0] + 1))
    T = len(time_indices)
    n_leaves = tree.num_leaves()
    frame_size = n_leaves // T

    # Aires temporelles par noeud
    comp_area = np.stack([
        hg.accumulate_sequential(tree, (np.arange(n_leaves) // frame_size == i).astype(np.float32), hg.Accumulators.sum)
        for i in range(T)
    ], axis=1)
    
    return comp_area

def compute_delta_area_mean(comp_area):
    area_diffs = comp_area[:, 1:] - comp_area[:, :-1]
    delta_area_mean = np.mean(area_diffs, axis=1)
    return delta_area_mean



def compute_growth_oriented(comp_area, delta_area_mean):
    # Croissance orientée
    active_duration = np.sum(comp_area > 0, axis=1)
    growth_oriented = delta_area_mean / np.maximum(1, active_duration)
    return growth_oriented

def compute_growth_oriented_attr(tree, sits):
    comp_area = compute_comp_area(tree, sits, list(range(1, sits.shape[0] + 1)))
    delta_area = compute_delta_area_mean(comp_area)
    growth_oriented = compute_growth_oriented(comp_area, delta_area)
    return growth_oriented
    

if __name__ == "__main__":
    sits = np.array([[[0, 0, 0, 0],
                    [0, 5, 0, 1],
                    [0, 0, 0, 3],
                    [2, 5, 4, 0]],

                    [[0, 1, 0, 0],
                    [1, 5, 0, 1],
                    [2, 5, 0, 8],
                    [0, 3, 4, 0]],
        
                    [[1, 0, 0, 0],
                    [0, 5, 5, 1],
                    [2, 0, 0, 3],
                    [0, 4, 4, 1]]], dtype=np.uint8)
    print(sits.shape)
    #sits = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)

    # Create a tree from the 3D array
    mask = [
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,1,0],[1,0,1],[0,1,0]],
        [[0,0,0],[0,1,0],[0,0,0]]
    ]
    print("compute tree")
    graph = hg.get_nd_regular_graph(sits.shape, hg.mask_2_neighbours(mask))
    tree, altitudes = hg.component_tree_max_tree(graph, sits.ravel())
    """
    amplitudes = compute_amplitude_attribute(tree, sits)
    print("amplitudes", amplitudes[tree.num_leaves():])
    
    maxt = compute_maxt_attribute(tree, sits, [1, 2, 3])
    print("maxt", maxt[tree.num_leaves():])
    
    mint = compute_mint_attribute(tree, sits, [1, 2, 3])
    print("mint", mint[tree.num_leaves():])
    
    st_stability = compute_STstability_attribute(tree, [1, 2, 3])
    print("st_stability", st_stability[tree.num_leaves():])
    
    st_centroid = compute_STcentroid_attribute(tree, [1, 2, 3])
    print("st_centroid", st_centroid[tree.num_leaves():])
    
    begin = compute_begin_attribute(tree, sits, [1, 2, 3])
    print("begin", begin[tree.num_leaves():])
    end = compute_end_attribute(tree, sits, [1, 2, 3])
    print("end", end[tree.num_leaves():])
    
    print("duration", compute_duration_attribute(tree, sits, [1, 2, 3])[tree.num_leaves():])
    
    hg.set_num_threads(1)
    # Benchmarking entre l'approche naïve et l'approche optimisée
    t1, t2 = benchmark_equal(compute_begin_attribute, compute_begin_attribute_v2, (tree, sits, [1, 2, 3]), n_iter=100)
    print("begin", t1, t2)
    
    t1, t2 = benchmark_equal(compute_end_attribute, compute_end_attribute_v2, (tree, sits, [1, 2, 3]), n_iter=100)
    print("end", t1, t2)   
    
    t1, t2 = benchmark_equal(compute_duration_attribute, compute_duration_attribute_v2, (tree, sits, [1, 2, 3]), n_iter=100)
    print("duration", t1, t2)   

    t1, t2 = benchmark_equal(compute_STcentroid_attribute, compute_STcentroid_attribute_v2, (tree, [1, 2, 3]), n_iter=100)
    print("st_centroid", t1, t2)
    
    t1, t2 = benchmark_equal(compute_STstability_attribute, compute_STstability_attribute_v2, (tree, [1, 2, 3]), n_iter=100)
    print("st_stability", t1, t2)

    #Comparer nos temps à un attribut simple de higra
    height_time = timeit.timeit(lambda: compute_height(tree, sits, [1, 2, 3]), number=100) / 100
    print("height", height_time)
    
    stability_time = timeit.timeit(lambda: compute_STstability_attribute(tree, [1, 2, 3]), number=100) / 100
    print("stability", stability_time)
    
    centroid_time = timeit.timeit(lambda: compute_STcentroid_attribute(tree, [1, 2, 3]), number=100) / 100
    print("centroid", centroid_time)
    """
    
    #st_centroid_v2 = compute_STcentroid_attribute_v2(tree, [1, 2, 3])
    #print("st_centroid_v2", st_centroid_v2[tree.num_leaves():])
    
    st_stability_v2 = compute_STstability_attribute_v2(tree, [1, 2, 3])
    print("st_stability_v2", st_stability_v2[tree.num_leaves():])
    
    compute_STstability_attribute_v3(tree, [1, 2, 3])
