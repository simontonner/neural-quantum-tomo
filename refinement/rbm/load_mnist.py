import os
import gzip
import requests
import numpy as np
from tqdm import tqdm


def download_mnist_if_needed(root="./data", train_only=False):
    mnist_base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    mnist_file_name_dict = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
    }
    mnist_test_file_name_dict = {
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    if not train_only:
        mnist_file_name_dict.update(mnist_test_file_name_dict)

    mnist_raw_dir = os.path.join(root, "MNIST", "raw")
    os.makedirs(mnist_raw_dir, exist_ok=True)

    for file_key, file_name in mnist_file_name_dict.items():
        path = os.path.join(mnist_raw_dir, file_name)

        if os.path.exists(path):
            continue

        response = requests.get(mnist_base_url + file_name, stream=True)

        download_desc = f"Downloading {file_name}"
        tqdm_bar = tqdm(desc=download_desc, total=int(response.headers.get('content-length', 0)), unit='B', unit_scale=True)
        with open(path, 'wb') as f, tqdm_bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                tqdm_bar.update(len(chunk))

    file_path_dict = {file_key: os.path.join(mnist_raw_dir, file_name) for file_key, file_name in mnist_file_name_dict.items()}

    return file_path_dict


def load_images(path):
    with gzip.open(path, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 28, 28)


def load_labels(path):
    with gzip.open(path, 'rb') as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data