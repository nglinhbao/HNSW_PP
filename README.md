# HNSW++ Experiment Framework

HNSW++ is an algorithm developed based on Hierarchical Navigable Small Worlds (HNSW) with added features including:
* Skipping layer method
* LID-based layer assignment
* Multi-branch HNSW

This framework is designed to construct and evaluate Hierarchical Navigable Small World (HNSW++) graphs for approximate nearest neighbor search. The code allows running experiments on different datasets, evaluates the search performance, and records various metrics such as construction time, average query time, recall, precision, and accuracy. Additionally, it tracks the number of skipped nodes during inference, a feature specific to HNSW++ optimization.

## Dependencies

Ensure the following Python libraries are installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `os`
- `fire`

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Code Structure

- `main.py`: The entry point for running the HNSW++ experiments.
- `hnswpp/`: Contains the core HNSW++ implementation and search functionality.
  - `hnswpp.py`: Code for constructing the HNSW++ graph.
  - `hnswpp_search.py`: Code for searching the HNSW++ graph.
- `utils/`: Utility scripts for dataset handling.
  - `read_dataset.py`: Functions to read and preprocess various datasets.
  - `download_dataset_*.py`: Scripts to download datasets like SIFT, GLOVE, MNIST, etc.
- `plots/`: Contains images generated from the experiments for inclusion in research papers. Subfolders include:
  - `ablation`: Results from ablation studies.
  - `datasets_overview`: Overview of datasets used.
  - `peer`: Peer comparison graphs.
  - `skip_exp`: Experiments highlighting the skip feature in HNSW++.
- `results/`: This folder stores the experiment results as CSV files.

## Usage

Run experiments using the `main.py` script. The script uses the `fire` library to expose a command-line interface for specifying experiment parameters.

### Command Format
```bash
python main.py dataset=<dataset_name> layers=<num_layers> maxk=<max_k_neighbors> ef_construction=<ef_construction> ml=<ml_value> ef_search=<ef_search> k=<k_value> train_count=<train_count> test_count=<test_count>
```

### Parameters

- `dataset`: The name of the dataset to use for the experiment. Supported datasets include:
  - `siftsmall`, `glove`, `deep`, `mnist`, `random`, `gaussian`, `yout`, `gist`
- `layers`: Number of layers in the HNSW++ graph (default: 5).
- `maxk`: Maximum number of neighbors in the graph (default: 16).
- `ef_construction`: Construction parameter controlling trade-off between construction time and accuracy (default: 128).
- `ml`: A scaling factor used during graph construction (default: 0.36).
- `ef_search`: Search parameter controlling trade-off between search time and accuracy (default: 100).
- `k`: Number of neighbors to retrieve during search (default: 50).
- `train_count`: Number of training vectors (default: 10000).
- `test_count`: Number of query vectors (default: 1000).

### Example

Run the experiment on the GLOVE dataset with default parameters:
```bash
python main.py dataset=glove
```

### Output

After running the experiment, the results will be saved in a CSV file inside the `results/` directory. The file will include:

- Configuration parameters (e.g., dataset, number of layers, maxK).
- Performance metrics such as:
  - Construction Time
  - Average Query Time
  - Average Recall
  - Average Precision
  - Average F1 Score
  - Average Accuracy
  - Total Skip Count (for HNSW++ optimization)

The results will also be printed in the terminal.

## Metrics Calculation

The following performance metrics are calculated:

- **Recall**: The fraction of true neighbors that were retrieved.
- **Precision**: The fraction of retrieved neighbors that are true neighbors.
- **F1 Score**: The harmonic mean of precision and recall.
- **Accuracy**: The fraction of correct neighbors retrieved in total.
- **Query Time**: The average time taken to perform the nearest neighbor search.
- **Construction Time**: The time taken to build the HNSW++ graph.

## Dataset Handling

Dataset-specific preprocessing functions are available in the `read_dataset.py` file. The supported datasets include:
- SIFT
- GLOVE
- MNIST
- Deep features
- GIST

For random datasets, functions are also provided to generate random vectors or Gaussian-distributed vectors.

## Plots

The `plots/` folder contains visualizations of the experiments conducted, which can be directly used in research papers or presentations. The subfolders are organized as follows:
- `ablation`: Results from ablation studies evaluating various aspects of the model.
- `datasets_overview`: Overview and distribution of datasets used in experiments.
- `peer`: Comparison of HNSW++ with peer methods.
- `skip_exp`: Experiments focused on the impact of the skip connections in HNSW++.

## Extending the Framework

You can add more datasets by extending the `read_dataset.py` file and creating corresponding `download_dataset_<dataset>.py` scripts if required. Ensure the data is formatted as vectors for compatibility with the experiment functions.

## Citations

If you use this code or reference any of the included datasets or methods, please cite the following papers:

- Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. [https://doi.org/10.1109/TPAMI.2018.2889473].
  
- Wolf, L., Hassner, T., & Maoz, I. (2011). Face Recognition in Unconstrained Videos with Matched Background Similarity. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

- Jégou, H., Douze, M., & Schmid, C. (2011). Product Quantization for Nearest Neighbor Search. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

- Babenko, A., & Lempitsky, V. (2016). Efficient Indexing of Billion-Scale Datasets of Deep Descriptors. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*.
