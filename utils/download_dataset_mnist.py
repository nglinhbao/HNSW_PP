from datasets import load_dataset
import os

def download_mnist(dataset_name='ylecun/mnist', data_dir='./datasets'):
    """Download the MNIST dataset from Hugging Face and save it to the specified directory."""
    print(f"Downloading the dataset '{dataset_name}' from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset(dataset_name, cache_dir=data_dir)

    print(f"Dataset '{dataset_name}' downloaded and stored in '{data_dir}'")

# Create the datasets directory if it doesn't exist
os.makedirs('./datasets', exist_ok=True)

# Download the MNIST dataset
download_mnist()
