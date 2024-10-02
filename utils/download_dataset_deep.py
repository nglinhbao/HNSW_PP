import urllib.request
import os

def download_file(url, local_filename):
    """Download a file from the specified URL and save it to the local path."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, local_filename)
    print(f"Downloaded to {local_filename}")

# URL of the new dataset
url = 'https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin'
local_filename = './datasets/base.1M.fbin'

# Ensure the datasets directory exists
os.makedirs('./datasets', exist_ok=True)

# Download the dataset
download_file(url, local_filename)

# Optionally, remove the file if needed after processing
# os.remove(local_filename)
# print(f"Removed {local_filename} after processing")
