import os
import shutil
import pickle
import hashlib
import tarfile
import subprocess
import urllib.request
import importlib.util
from typing import Tuple

import yaml
import numpy as np
from jinja2 import Template
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .plot import plot_cifar10


def load_module(src_dir: str, module_name: str) -> object:
    """Dynamically load a Python module from a specified directory.

    Args:
        src_dir (str): The directory containing the module file.
        module_name (str): The name of the module to be loaded (without the .py extension).

    Returns:
        object: The loaded module object.

    Note:
        This function requires that the module file exists in the specified directory.
    """
    module_path = f"{src_dir}/{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_config(config_path: str, args: dict) -> dict:
    """Load and render a YAML configuration file with template variables.

    Args:
        config_path (str): The path to the YAML configuration file.
        args (dict): A dictionary of arguments to render the template.

    Returns:
        dict: A dictionary representation of the loaded and rendered configuration.

    Note:
        The configuration file can contain Jinja2 template syntax, which will be
        replaced with the values provided in the `args` dictionary.
    """
    with open(config_path) as file:
        template = Template(file.read())
    config_text = template.render(**args)
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    return config


CIFAR10_URLS = (
    # Byte-identical copy of the official archive; much faster than the original server.
    "https://huggingface.co/datasets/liangnanying/cifar-10-python/resolve/main/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
)
# MD5 of the official archive. The loader unpickles these files, so never skip this check.
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


def _download_with_progress(url: str, destination: str) -> None:
    """Downloads a file and shows a progress bar."""
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="Download") as progress:

        def report(block_count: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                progress.total = total_size
            progress.update(block_count * block_size - progress.n)

        urllib.request.urlretrieve(url, destination, reporthook=report)


def _md5(file: str) -> str:
    digest = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_cifar10(directory: str) -> None:
    """Downloads and extracts the CIFAR-10 dataset into the specified directory.

    This function checks if the CIFAR-10 dataset is already present in the specified directory.
    If not, it downloads the archive with a progress bar, trying the mirrors in `CIFAR10_URLS`
    in order, verifies its checksum, extracts it, and moves the extracted files to the directory.
    Temporary files are kept next to the target directory.

    Args:
        directory (str): The directory where the dataset will be downloaded and extracted.

    Returns:
        None
    """

    if os.path.exists(directory):
        print("CIFAR-10 dataset already exists")
        return

    parent = os.path.dirname(os.path.abspath(directory))
    os.makedirs(parent, exist_ok=True)
    tar = os.path.join(parent, "cifar-10-python.tar.gz")
    extracted_folder = os.path.join(parent, "cifar-10-batches-py")

    # Remove leftovers from an interrupted run
    if os.path.exists(extracted_folder):
        shutil.rmtree(extracted_folder)

    for url in CIFAR10_URLS:
        if os.path.exists(tar):
            os.remove(tar)
        print(f"Downloading CIFAR-10 (~170 MB) from {url}")
        try:
            _download_with_progress(url, tar)
        except KeyboardInterrupt:
            if os.path.exists(tar):
                os.remove(tar)
            raise
        except Exception as e:
            print(f"Download failed: {e}")
            continue
        if _md5(tar) == CIFAR10_MD5:
            break
        print("The downloaded file is corrupted (checksum mismatch).")
    else:
        if os.path.exists(tar):
            os.remove(tar)
        raise RuntimeError(
            "Could not download CIFAR-10. Check your internet connection, or download "
            f"cifar-10-python.tar.gz manually from {CIFAR10_URLS[-1]} and extract it to {directory}."
        )

    print("Extracting archive...")
    with tarfile.open(tar, "r:gz") as tar_ref:
        tar_ref.extractall(parent)
    shutil.move(extracted_folder, directory)
    os.remove(tar)
    print(f"CIFAR-10 is ready in {directory}")


def load_cifar10_bach_file(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a single batch file from the CIFAR-10 dataset.

    Args:
        file (str): The path to the batch file to be loaded.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - An array containing the image data of shape (10000, 32, 32, 3).
            - An array containing the corresponding labels of shape (10000,).
    """

    with open(file, "rb") as f:
        datadict = pickle.load(f, encoding="latin1")
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_cifar10(directory: str, visualize_samples: bool = False) -> Tuple:
    """Loads the CIFAR-10 dataset from the specified directory and splits it into training, validation, and test sets.

    Args:
        directory (str): The directory where the CIFAR-10 dataset is located.
        visualize_samples (bool, optional): If True, visualizes a sample of images from the training set. Default is False.

    Returns:
        Tuple containing:
            - X_train (numpy.ndarray): An array of training images.
            - y_train (numpy.ndarray): An array of training labels.
            - X_val (numpy.ndarray): An array of validation images.
            - y_val (numpy.ndarray): An array of validation labels.
            - X_test (numpy.ndarray): An array of test images.
            - y_test (numpy.ndarray): An array of test labels.
    """

    # Download the CIFAR-10 dataset if it does not exist
    download_cifar10(directory)

    # Load the training and test data
    X_train, y_train = [], []
    for i in range(1, 6):
        file = os.path.join(directory, f"data_batch_{i}")
        X, y = load_cifar10_bach_file(file)
        X_train.append(X)
        y_train.append(y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_cifar10_bach_file(os.path.join(directory, "test_batch"))

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    if visualize_samples:
        plot_cifar10(X_train, y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_cifar10_subset(
    directory: str,
    num_train: int,
    num_val: int,
    num_test: int,
    visualize_samples: bool = False,
) -> Tuple:
    """Loads a balanced subset of the CIFAR-10 dataset with specified numbers of training, validation, and test samples.

    Args:
        directory (str): The directory where the CIFAR-10 dataset is located.
        num_train (int): The number of training samples per class.
        num_val (int): The number of validation samples per class.
        num_test (int): The number of test samples per class.
        visualize_samples (bool, optional): If True, visualizes a sample of images from the training set. Default is False.

    Returns:
        Tuple containing:
            - X_train (numpy.ndarray): An array of training images.
            - y_train (numpy.ndarray): An array of training labels.
            - X_val (numpy.ndarray): An array of validation images.
            - y_val (numpy.ndarray): An array of validation labels.
            - X_test (numpy.ndarray): An array of test images.
            - y_test (numpy.ndarray): An array of test labels.
    """

    # Load the full CIFAR-10 dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(directory)

    # Get the number of classes in the dataset
    num_classes = len(np.unique(y_train))

    # Select a subset of the data and make sure the classes are balanced
    X_train = np.vstack([X_train[y_train == i][:num_train] for i in range(num_classes)])
    y_train = np.hstack(
        [y_train[y_train == i][:num_train] for i in range(num_classes)]
    ).astype(np.int32)

    X_val = np.vstack([X_val[y_val == i][:num_val] for i in range(num_classes)])
    y_val = np.hstack([y_val[y_val == i][:num_val] for i in range(num_classes)]).astype(
        np.int32
    )

    X_test = np.vstack([X_test[y_test == i][:num_test] for i in range(num_classes)])
    y_test = np.hstack(
        [y_test[y_test == i][:num_test] for i in range(num_classes)]
    ).astype(np.int32)

    if visualize_samples:
        plot_cifar10(X_train, y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test
