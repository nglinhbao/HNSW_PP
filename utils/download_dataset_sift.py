import urllib.request
import tarfile
import os

def download_file(url, local_filename):
    """Download a file from the specified URL and save it to the local path."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, local_filename)
    print(f"Downloaded to {local_filename}")

def extract_tar_gz(file_path, extract_to='./datasets'):
    """Extract a .tar.gz file to the specified directory."""
    print(f"Extracting {file_path}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted to {extract_to}")

# URL of the dataset
url = 'ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz'
local_filename = './datasets/siftsmall.tar.gz'

# Download the dataset
download_file(url, local_filename)

# Extract the dataset
extract_tar_gz(local_filename, './datasets')

# Optionally, remove the tar.gz file after extraction
os.remove(local_filename)
print(f"Removed {local_filename} after extraction")
