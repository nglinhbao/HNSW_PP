from hnswpp.hnswpp import *
from concurrent.futures import ThreadPoolExecutor
# import time
# import csv

def Search(HNSW: dict, q: int, efSearch: int, k: int, lid_threshold: float) -> set:
    W1 = set([HNSW["entrance1"]])
    W2 = set([HNSW["entrance2"]])
    layer_i1 = TopLayer(HNSW)
    layer_i2 = TopLayer(HNSW)
    skip_count = 0
    exclude_set1 = set()
    exclude_set2 = set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        while layer_i1 >= 0 or layer_i2 >= 0:
            futures = []
            world_order = []

            if layer_i1 == 0 and layer_i2 == 0:
                # Handle base layer separately
                ef1 = efSearch // 2 + (efSearch % 2)  # W1 gets the extra if efSearch is odd
                W1, skip1 = SearchLayer(HNSW, 0, q, Nearest(HNSW, W1, HNSW["nodes"][q]), ef1, world=0, lid_threshold=lid_threshold, exclude_set=exclude_set1)
                
                exclude_set2.update(W1)
                
                ef2 = efSearch // 2
                W2, skip2 = SearchLayer(HNSW, 0, q, Nearest(HNSW, W2, HNSW["nodes"][q]), ef2, world=1, lid_threshold=lid_threshold, exclude_set=exclude_set2)
                
                break
            
            if layer_i1 == 0:
                ef1 = efSearch // 2 + (efSearch % 2)
                futures.append(executor.submit(SearchLayer, HNSW, layer_i1, q, Nearest(HNSW, W1, HNSW["nodes"][q]), ef1, world=0, lid_threshold=lid_threshold, exclude_set=exclude_set1))
                world_order.append(0)
            elif layer_i1 > 0:
                ef1 = 1
                futures.append(executor.submit(SearchLayer, HNSW, layer_i1, q, Nearest(HNSW, W1, HNSW["nodes"][q]), ef1, world=0, lid_threshold=lid_threshold))
                world_order.append(0)
            
            if layer_i2 == 0:
                ef2 = efSearch // 2
                futures.append(executor.submit(SearchLayer, HNSW, layer_i2, q, Nearest(HNSW, W2, HNSW["nodes"][q]), ef2, world=1, lid_threshold=lid_threshold, exclude_set=exclude_set2))
                world_order.append(1)
            elif layer_i2 > 0:
                ef2 = 1
                futures.append(executor.submit(SearchLayer, HNSW, layer_i2, q, Nearest(HNSW, W2, HNSW["nodes"][q]), ef2, world=1, lid_threshold=lid_threshold))
                world_order.append(1)
            
            results = [future.result() for future in futures]
            
            for world, (W, skip) in zip(world_order, results):
                if world == 0:  # World 1
                    W1 = W
                    if skip and layer_i1 != 0:
                        layer_i1 = 0
                        skip_count += 1
                    else:
                        if layer_i1 == 0:
                            exclude_set2.update(W1)
                        layer_i1 -= 1
                else:  # World 2
                    W2 = W
                    if skip and layer_i2 != 0:
                        layer_i2 = 0
                        skip_count += 1
                    else:
                        if layer_i2 == 0:
                            exclude_set1.update(W2)
                        layer_i2 -= 1
                        
    W = W1.union(W2)
    return set(sorted([(x, Distance(HNSW["nodes"][x], HNSW["nodes"][q])) for x in W], key=lambda x: x[1])[:k]), skip_count