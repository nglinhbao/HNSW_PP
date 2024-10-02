from datasets import load_dataset
import numpy as np
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import os

def fvecs_read(filename, bounds=None, max_count=10000):
    with open(filename, 'rb') as f:
        # Read the dimension from the file
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Read all data as float32 and reshape accordingly
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32).reshape(-1, dim + 1)
        vectors = data[:, 1:]  # Skip the first column, which contains the dimensions

        # Apply bounds if specified
        if bounds is not None:
            start, end = bounds
            vectors = vectors[start-1:end]

        # Restrict vectors to max_count if specified
        if max_count is not None and vectors.shape[0] > max_count:
            vectors = vectors[:max_count]

        return vectors

# def ivecs_read(filename, max_count=10000):
#     with open(filename, 'rb') as f:
#         data = np.fromfile(f, dtype=np.int32)
#         dim = data[0]
#         vectors = data.reshape(-1, dim + 1)
#         vectors = vectors[:, 1:]
        
#         # Shuffle and split the data
#         np.random.shuffle(vectors)
#         train_vectors = vectors[:max_count]
#         test_vectors = vectors[max_count:2*max_count]
        
#         return train_vectors, test_vectors

def read_mnist(dataset_path="ylecun/mnist", train_count=10000, test_count=3000):
    """
    Read and split the MNIST dataset into train and test sets, with images normalized.

    Args:
    dataset_path (str): The path to the MNIST dataset.
    train_count (int): Number of samples for the training set.
    test_count (int): Number of samples for the test set.

    Returns:
    tuple: (train_data, test_data), where train_data and test_data are arrays of normalized image data.
    """
    # Load the MNIST dataset
    dataset = load_dataset(dataset_path)
    all_data = dataset['train'].select(range(len(dataset['train'])))
    
    # Shuffle and split the data
    all_indices = list(range(len(all_data)))
    random.shuffle(all_indices)
    
    train_indices = all_indices[:train_count]
    test_indices = all_indices[train_count:train_count + test_count]
    
    # Select the images only, ignoring the labels, and normalize them
    train_data = np.array([np.array(all_data[i]['image']) / 255.0 for i in train_indices])
    test_data = np.array([np.array(all_data[i]['image']) / 255.0 for i in test_indices])
    
    return train_data, test_data

def read_glove(file_path="datasets/glove.twitter.27B.100d.txt", train_count=10000, test_count=3000):
    embeddings = []
    expected_dim = 100

    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            embedding = split_line[1:]
            
            if len(embedding) != expected_dim:
                continue
            
            embeddings.append([float(val) for val in embedding])
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Shuffle and split the data
    np.random.shuffle(embeddings_array)
    train_embeddings = embeddings_array[:train_count]
    test_embeddings = embeddings_array[train_count:train_count+test_count]
    
    return train_embeddings, test_embeddings

def generate_gaussian_dataset(num_base_vectors=10000, num_query_vectors=3000, num_clusters=1000, 
                              dimensions=12, space_range=10, cluster_std=1):
    """
    Generate a dataset of vectors based on Gaussian distribution.

    Args:
    num_base_vectors (int): Number of base vectors to generate.
    num_query_vectors (int): Number of query vectors to generate.
    num_clusters (int): Number of cluster centers.
    dimensions (int): Number of dimensions for each vector.
    space_range (float): Range of the space for cluster centers [0, space_range].
    cluster_std (float): Standard deviation for the Gaussian distribution of each cluster.

    Returns:
    tuple: (base_vectors, query_vectors)
        base_vectors: numpy array of shape (num_base_vectors, dimensions)
        query_vectors: numpy array of shape (num_query_vectors, dimensions)
    """
    # Generate cluster centers
    cluster_centers = np.random.uniform(0, space_range, size=(num_clusters, dimensions))

    # Generate base vectors
    base_vectors = []
    for _ in range(num_base_vectors):
        # Randomly choose a cluster center
        center = cluster_centers[np.random.randint(num_clusters)]
        # Generate a vector following Gaussian distribution around the center
        vector = np.random.normal(center, cluster_std, dimensions)
        base_vectors.append(vector)

    # Generate query vectors (same process as base vectors)
    query_vectors = []
    for _ in range(num_query_vectors):
        center = cluster_centers[np.random.randint(num_clusters)]
        vector = np.random.normal(center, cluster_std, dimensions)
        query_vectors.append(vector)

    return np.array(base_vectors), np.array(query_vectors)

def read_siftsmall(data_filename="datasets/siftsmall/siftsmall_base.fvecs", query_filename="datasets/siftsmall/siftsmall_query.fvecs", train_count=10000, test_count=3000):    
    # Read data and query points
    data_points = fvecs_read(data_filename, max_count=train_count)
    query_points = fvecs_read(query_filename, max_count=test_count)
    
    return data_points, query_points

def read_gist(data_filename="datasets/gist/gist_base.fvecs", query_filename="datasets/gist/gist_query.fvecs", train_count=10000, test_count=3000):    
    # Read data and query points
    data_points = fvecs_read(data_filename, max_count=train_count)
    query_points = fvecs_read(query_filename, max_count=test_count)
    
    return data_points, query_points

def read_deep(data_filename="datasets/deep/deep1M_base.fvecs", query_filename="datasets/deep/deep1B_queries.fvecs", train_count=10000, test_count=3000):    
    # Read data and query points
    data_points = fvecs_read(data_filename, max_count=train_count)
    query_points = fvecs_read(query_filename, max_count=test_count)
    
    return data_points, query_points

def read_gist(data_filename="datasets/gist/gist_base.fvecs", query_filename="datasets/gist/gist_query.fvecs", train_count=10000, test_count=3000):    
    # Read data and query points
    data_points = fvecs_read(data_filename, max_count=train_count)
    query_points = fvecs_read(query_filename, max_count=test_count)
    
    return data_points, query_points

def generate_random_dataset(num_base_vectors=10000, num_query_vectors=3000, dimensions=100):
    # Generate random base vectors with shape (num_base_vectors, dimensions)
    data_points = np.random.rand(num_base_vectors, dimensions)
    
    # Generate random query vectors with shape (num_query_vectors, dimensions)
    query_points = np.random.rand(num_query_vectors, dimensions)
    
    return data_points, query_points

def read_yout(directory="datasets/aligned_images_DB", train_count=10000, test_count=3000, image_size=(24, 24)):
    """
    Read and split the YouTube Faces dataset into train and test sets.

    Args:
    directory (str): The path to the dataset directory.
    train_count (int): Number of samples for the training set.
    test_count (int): Number of samples for the test set.
    image_size (tuple): Size to which images should be resized.

    Returns:
    tuple: (X_train, X_test), where X_train and X_test are arrays of image data
    """
    image_paths = []
    
    # Collect image paths and labels
    for label_folder in os.listdir(directory):
        label_folder_path = os.path.join(directory, label_folder)
        if os.path.isdir(label_folder_path):
            for video_folder in os.listdir(label_folder_path):
                video_folder_path = os.path.join(label_folder_path, video_folder)
                if os.path.isdir(video_folder_path):
                    for image_file in os.listdir(video_folder_path):
                        image_path = os.path.join(video_folder_path, image_file)
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(image_path)

    # Shuffle and split data
    image_paths = np.array(image_paths)
    if len(image_paths) < train_count + test_count:
        raise ValueError("Not enough images to meet the requested train and test counts.")
    
    train_paths, test_paths = train_test_split(image_paths, train_size=train_count, test_size=test_count, shuffle=True)

    def load_images(image_paths):
        images = []
        for path in image_paths:
            with Image.open(path) as img:
                img = img.resize(image_size)
                # Convert the image to a NumPy array
                img_array = np.array(img)
                # Normalize the image array by scaling pixel values to [0, 1]
                img_array = img_array / 255.0
                images.append(img_array)
        return np.array(images)

    # Load training and testing images
    X_train = load_images(train_paths)
    X_test = load_images(test_paths)

    return X_train, X_test