import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd 
import os
import fire
from hnswpp.hnswpp import ConstructHNSW as ConstructHNSW_hnswpp
from hnswpp.hnswpp_search import Search as SearchHNSW_hnswpp
from utils.read_dataset import *

def brute_force_search(data, query, k):
    flattened_data = [matrix.flatten() for matrix in data]
    flattened_query = query.flatten()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(flattened_data)
    distances, indices = nbrs.kneighbors([flattened_query])
    return set(zip(indices[0], distances[0]))

def save_results_to_csv(results, dataset, layers, maxk, efConstruction, mL, efSearch, k, num_samples, num_queries):
    results_dir = f'results/{dataset}'
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/hnswpp_results.csv"

    config_data = {
        'Metric': ['Dataset', 'Layers', 'Max K', 'efConstruction', 'mL', 'efSearch', 'k', 'Num Samples', 'Num Queries'],
        'Value': [dataset, layers, maxk, efConstruction, mL, efSearch, k, num_samples, num_queries]
    }

    metrics_data = {
        'Metric': ['Construction Time (s)', 'Average Query Time (s)', 'Average Recall', 'Average Precision', 
                   'Average F1 Score', 'Average Accuracy', 'Total Skip Count'],
        'Value': [
            results['construction_time'],
            np.mean(results['query_times']),
            results['avg_recall'],
            results['avg_precision'],
            results['avg_f1_score'],
            results['avg_accuracy'],
            results['total_skips']
        ]
    }

    config_df = pd.DataFrame(config_data)
    metrics_df = pd.DataFrame(metrics_data)
    final_df = pd.concat([config_df, metrics_df], ignore_index=True)
    final_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def test_construction_speed(data, layers, maxk, efConstruction, mL):
    start_time = time.time()
    HNSW = ConstructHNSW_hnswpp(layers, maxk, data, efConstruction, mL)
    end_time = time.time()
    return end_time - start_time, HNSW

def test_inference_speed(HNSW, query, efSearch, k):
    query_element = len(HNSW["nodes"])
    HNSW["nodes"][query_element] = query
    start_time = time.time()
    neighbors, skip_count = SearchHNSW_hnswpp(HNSW, query_element, efSearch, k, lid_threshold=0.5)
    end_time = time.time()
    del HNSW["nodes"][query_element]
    return end_time - start_time, neighbors, skip_count

def calculate_metrics(true_neighbors, hnsw_neighbors):
    true_indices = set(x[0] for x in true_neighbors)
    hnsw_indices = set(x[0] for x in hnsw_neighbors)
    tp = len(true_indices.intersection(hnsw_indices))
    fp = len(hnsw_indices - true_indices)
    fn = len(true_indices - hnsw_indices)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return recall, precision, f1_score, accuracy

def run_experiment(base_vectors, query_vectors, layers, maxk, efConstruction, mL, efSearch, k):
    construction_time, HNSW = test_construction_speed(base_vectors, layers, maxk, efConstruction, mL)

    total_recall = total_precision = total_f1_score = total_accuracy = 0
    query_times = []
    total_skips = 0

    num_queries = query_vectors.shape[0]

    for search_count in range(num_queries):
        query = query_vectors[search_count]
        
        inference_time, hnsw_neighbors, skip_count = test_inference_speed(HNSW, query, efSearch, k)
        total_skips += skip_count
        query_times.append(inference_time)

        true_neighbors = brute_force_search(base_vectors, query, k)
        
        recall, precision, f1_score, accuracy = calculate_metrics(true_neighbors, hnsw_neighbors)
        total_recall += recall
        total_precision += precision
        total_f1_score += f1_score
        total_accuracy += accuracy

    results = {
        "construction_time": construction_time,
        "query_times": query_times,
        "avg_recall": total_recall / num_queries,
        "avg_precision": total_precision / num_queries,
        "avg_f1_score": total_f1_score / num_queries,
        "avg_accuracy": total_accuracy / num_queries,
        "total_skips": total_skips,
    }

    return results

def main(dataset, layers=5, maxk=16, ef_construction=128, ml=0.36, 
         ef_search=100, k=50, train_count=10000, test_count=1000):
    if dataset == "siftsmall":
        base_vectors, query_vectors = read_siftsmall(train_count=train_count, test_count=test_count)
    elif dataset == "glove":
        base_vectors, query_vectors = read_glove(train_count=train_count, test_count=test_count)
    elif dataset == "deep":
        base_vectors, query_vectors = read_deep(train_count=train_count, test_count=test_count)
    elif dataset == "mnist":
        base_vectors, query_vectors = read_mnist(train_count=train_count, test_count=test_count)
    elif dataset == "random":
        base_vectors, query_vectors = generate_random_dataset(num_base_vectors=train_count, num_query_vectors=test_count)
    elif dataset == "gaussian":
        base_vectors, query_vectors = generate_gaussian_dataset(num_base_vectors=train_count, num_query_vectors=test_count)
    elif dataset == "yout":
        base_vectors, query_vectors = read_yout(train_count=train_count, test_count=test_count)
    elif dataset == "gist":
        base_vectors, query_vectors = read_gist(train_count=train_count, test_count=test_count)

    results = run_experiment(base_vectors, query_vectors, layers, maxk, ef_construction, ml, ef_search, k)
    save_results_to_csv(results, dataset, layers, maxk, ef_construction, ml, ef_search, k, train_count, test_count)

    print("\nHNSW++ Results:")
    print(f"Construction Time: {results['construction_time']:.4f} seconds")
    print(f"Average Query Time: {np.mean(results['query_times']):.4f} seconds")
    print(f"Average Recall: {results['avg_recall']:.4f}")
    print(f"Average Precision: {results['avg_precision']:.4f}")
    print(f"Average F1 Score: {results['avg_f1_score']:.4f}")
    print(f"Average Accuracy: {results['avg_accuracy']:.4f}")
    print(f"Total Skips: {results['total_skips']}")

if __name__ == '__main__':
    fire.Fire(main)