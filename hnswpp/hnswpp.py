import numpy as np
import networkx as nx
from math import floor, log
from random import random
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


def compute_lid_mle(distances):
    k = len(distances)
    if k < 2 or distances[-1] == 0:
        return np.nan
    ratios = distances[-1] / distances[:-1]
    ratios = ratios[ratios > 0]
    if len(ratios) < (k - 1):
        return np.nan
    lid = (np.mean(np.log(ratios))) ** -1
    return lid


def compute_gid(data, k):
    n_samples = data.shape[0]
    data_reshaped = data.reshape(n_samples, -1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data_reshaped)
    distances, _ = nbrs.kneighbors(data_reshaped)
    
    # Calculate average distance
    avg_distance = np.mean(distances[:, 1:])  # Exclude the first column (distance to self)
    
    lids = np.array([compute_lid_mle(distances[i][1:k+1]) for i in range(n_samples)])
    gid = np.nanmean(lids)
    
    return gid, lids, avg_distance


def normalize_lids(lids):
    min_lid, max_lid = np.min(lids), np.max(lids)
    return (lids - min_lid) / (max_lid - min_lid)


def assign_layer(topL: int, mL: float, normalized_LIDs: np.ndarray) -> np.ndarray:
    n = len(normalized_LIDs)
    world_0_size = (n + 1) // 2  # World 0 gets the extra node if n is odd
    world_1_size = n // 2
    
    # Calculate expected layer sizes for each world
    expected_layer_size = [np.zeros(topL, dtype=int) for _ in range(2)]
    for world in range(2):
        world_size = world_0_size if world == 0 else world_1_size
        for _ in range(world_size):
            layer_i = max(min(floor(-1 * log(random()) * mL), topL - 1), 0)
            expected_layer_size[world][layer_i] += 1
    
    # Sort indices by normalized LID values in descending order
    sorted_indices = np.argsort(normalized_LIDs)[::-1]
    
    # Initialize assigned_layers array with layer and world information
    assigned_layers = np.zeros((n, 2), dtype=int)
    current_layer_size = [np.zeros(topL, dtype=int) for _ in range(2)]
    
    # Assign nodes to worlds and layers
    world_counts = [0, 0]
    current_world = 0  # Start with World 0
    for idx in sorted_indices:
        # Alternate between worlds
        world = current_world
        
        # Assign layer
        for layer in range(topL - 1, -1, -1):
            if current_layer_size[world][layer] < expected_layer_size[world][layer]:
                assigned_layers[idx] = [layer, world]
                current_layer_size[world][layer] += 1
                world_counts[world] += 1
                break
        
        # Switch to the other world for the next iteration
        current_world = 1 - current_world
        
        # If one world is full, assign remaining nodes to the other world
        if world_counts[current_world] == (world_0_size if current_world == 0 else world_1_size):
            current_world = 1 - current_world
    
    return assigned_layers


def Insert(HNSW: dict, q: int, maxk: int, efConstruction: int, mL: float, matrix: np.ndarray):
    W = [set(), set()]  # Separate W for each world
    eP = [HNSW["entrance1"], HNSW["entrance2"]]
    topL = TopLayer(HNSW)

    # Add the new node to the HNSW structure
    HNSW["nodes"][q] = matrix

    # Determine the world to insert into based on node ID parity
    layer_i, world = HNSW["assigned_layers"][q]

    # print(f"Node {q}: Normalized LID = {HNSW['normalized_lids'][q]:.4f}, Assigned Layer: {layer_i}, World: {world}")

    for lc in range(topL, layer_i, -1):
        layer = GetLayer(HNSW, lc, world)
        W[world - 1],_ = SearchLayer(HNSW, lc, q, eP[world - 1], ef=1, world=world-1)  # No frequency count in construction
        eP[world - 1] = Nearest(HNSW, W[world - 1], HNSW["nodes"][q])

    for lc in range(layer_i, -1, -1):
        layer = GetLayer(HNSW, lc, world)
        W[world - 1],_ = SearchLayer(HNSW, lc, q, eP[world - 1], efConstruction, world=world-1)  # No frequency count in construction
        neighbors = SelectNeighbors(HNSW, q, W[world - 1], maxk)
        AddEdges(layer, q, neighbors)

        for e in neighbors:
            e_neighbors = Neighborhood(layer, e)
            if len(e_neighbors) > maxk:
                e_new_edges = SelectNeighbors(HNSW, e, e_neighbors, maxk)
                ShrinkEdges(layer, e, e_new_edges)

        eP[world - 1] = Nearest(HNSW, W[world - 1], HNSW["nodes"][q])

    HNSW["layer_nodes"][world-1][layer_i].add(q)

    if layer_i > topL:
        HNSW[f"entrance{world}"] = eP[world - 1]

 
def SearchLayer(HNSW: dict, layer_i: int, q: int, ep: int, ef: int, world: int, lid_threshold: float = None, exclude_set: set = None) -> set:
    q_matrix = HNSW["nodes"][q]
    layer = GetLayer(HNSW, layer_i, world)

    # Handle case where ep is in exclude_set
    if exclude_set and ep in exclude_set:
        # Find a new entry point that's not in exclude_set
        possible_eps = set()
        for node in layer:
            if node not in exclude_set:
                possible_eps.add(node)
        
        if not possible_eps:
            return set(), False  # Return empty set if no valid entry points
        ep = min(possible_eps, key=lambda x: Distance(HNSW["nodes"][x], q_matrix))

    visited = {ep}
    candidates = {ep}
    nearest_neighbors = {ep}

    # Precompute distances to avoid redundant calculations
    dist_cache = {}

    def distance_cached(node1, node2):
        if (node1, node2) not in dist_cache:
            dist_cache[(node1, node2)] = Distance(HNSW["nodes"][node1], HNSW["nodes"][node2])
        return dist_cache[(node1, node2)]

    while candidates:
        c = min(candidates, key=lambda x: distance_cached(x, q))
        candidates.remove(c)

        f = max(nearest_neighbors, key=lambda x: distance_cached(x, q))

        if distance_cached(c, q) > distance_cached(f, q):
            break

        neighbors = Neighborhood(layer, c)

        for e in neighbors:
            if e not in visited and (exclude_set is None or e not in exclude_set):
                visited.add(e)
                
                e_distance = distance_cached(e, q)
                
                if len(nearest_neighbors) < ef:
                    nearest_neighbors.add(e)
                    candidates.add(e)
                else:
                    f = max(nearest_neighbors, key=lambda x: distance_cached(x, q))
                    f_distance = distance_cached(f, q)
                    
                    if e_distance < f_distance:
                        nearest_neighbors.remove(f)
                        nearest_neighbors.add(e)
                        candidates.add(e)

    skip = False
    if lid_threshold:
        nearest_neighbor = min(nearest_neighbors, key=lambda x: distance_cached(x, q))
        nearest_neighbor_distance = distance_cached(nearest_neighbor, q)

        if nearest_neighbor_distance <= HNSW["average"] and HNSW['normalized_lids'][nearest_neighbor] >= lid_threshold:
            skip = True

    return nearest_neighbors, skip



def SelectNeighbors(HNSW: dict, q: int, cands: set, M: int) -> set:
    return set(sorted(cands, key=lambda x: Distance(HNSW["nodes"][x], HNSW["nodes"][q]))[:M])


def GetLayer(HNSW: dict, lc: int, world: int) -> nx.Graph:
    return HNSW["layers"][world-1][lc]


def EntrancePoint(HNSW: dict, world: int) -> int:
    return HNSW[f"entrance{world}"]


def SetEntrancePoint(HNSW: dict, eP: int, world: int):
    HNSW[f"entrance{world}"] = eP


def TopLayer(HNSW: dict) -> int:
    return len(HNSW["layers"][0]) - 1


def Distance(u: np.ndarray, v: np.ndarray) -> float:    
    return np.linalg.norm(u - v)


def Furthest(HNSW: dict, W: set, q: np.ndarray) -> int:
    return max(W, key=lambda w: Distance(HNSW["nodes"][w], q))


def Nearest(HNSW: dict, W: set, q: np.ndarray) -> int:
    return min(W, key=lambda w: Distance(HNSW["nodes"][w], q))


def Neighborhood(layer: nx.Graph, u: int) -> set:
    return set(layer[u]) if u in layer else set()


def AddEdges(layer: nx.Graph, u: int, neighbors: set):
    for n in neighbors:
        layer.add_edge(u, n)


def ShrinkEdges(layer: nx.Graph, u: int, new_edges: set):
    removes = [(u, n) for n in layer[u] if n not in new_edges]
    layer.remove_edges_from(removes)


def ConstructHNSW(layers: int, maxk: int, matrices: np.ndarray, efConstruction: int, mL: float) -> dict:
    gid, real_lids, average = compute_gid(matrices, k=10)
    normalized_lids = normalize_lids(real_lids)
    
    # Pre-assign layers and worlds for all nodes
    assigned_layers = assign_layer(layers-1, mL, normalized_lids)

    # Calculate and print layer statistics
    layer_stats = [defaultdict(list) for _ in range(2)]  # One dict for each world
    for node, (layer, world) in enumerate(assigned_layers):
        layer_stats[world][layer].append(normalized_lids[node])

    print("Layer statistics before insertion:")
    for world in range(2):
        print(f"World {world + 1}:")
        for layer in range(layers):
            nodes_in_layer = len(layer_stats[world][layer])
            avg_normalized_lid = np.mean(layer_stats[world][layer]) if nodes_in_layer > 0 else 0
            print(f"  Layer {layer}: {nodes_in_layer} nodes, Average Normalized LID: {avg_normalized_lid:.4f}")
    
    # Initialize HNSW structure
    shared_layer_0 = nx.Graph()
    HNSW = {
        "entrance1": 0,
        "entrance2": 0,
        "layers": [
            [shared_layer_0] + [nx.Graph() for _ in range(layers - 1)],
            [shared_layer_0] + [nx.Graph() for _ in range(layers - 1)]
        ],
        "nodes": {},
        "GID_full": gid,
        "normalized_lids": normalized_lids,
        "assigned_layers": assigned_layers,
        "total_N": len(matrices),
        "layer_nodes": [defaultdict(set) for _ in range(2)],
        "average": average,
        "distances": [],
        "count": 0
    }

    total_matrices = len(matrices)
    progress_bar = tqdm(total=total_matrices, desc='Inserting matrices', unit='matrix', ncols=80)

    # Iterate and insert matrices
    for i, matrix in enumerate(matrices):
        Insert(HNSW, i, maxk, efConstruction, mL, matrix)
        progress_bar.update(1)

    progress_bar.close()
    
    return HNSW