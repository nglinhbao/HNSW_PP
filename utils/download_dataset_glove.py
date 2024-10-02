import urllib.request
import zipfile
import os

def download_file(url, local_filename):
    """Download a file from the specified URL and save it to the local path."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, local_filename)
    print(f"Downloaded to {local_filename}")

def extract_zip(file_path, extract_to='./datasets'):
    """Extract a .zip file to the specified directory."""
    print(f"Extracting {file_path}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

# URL of the GloVe Twitter dataset
url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
local_filename = './datasets/glove.twitter.27B.zip'

# Download the dataset
download_file(url, local_filename)

# Extract the dataset
extract_zip(local_filename, './datasets')

# Optionally, remove the zip file after extraction
os.remove(local_filename)
print(f"Removed {local_filename} after extraction")
