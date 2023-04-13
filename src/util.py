"""This module provies utilities for using/manipulating the dataset."""

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image
from sklearn.utils import Bunch

_MODULE_PATH = Path(__file__).parent
"""Parent directory of this python module."""


def __load_class_training(container_path: Path, dataset: dict, target: int) -> None:
    """Helper function to load all files of a single classification, in-place.

    Args:
        container_path (Path): Path to directory containing the data files.
        dataset (dict): Dataset instance to append data to.
        target (int): The target classification of the data being loaded.
    """
    i = 0
    for file in container_path.iterdir():
        if i == 500:
            break
        i = i + 1
        image: Image.Image = Image.open(file).convert("L")
        data: ArrayLike = np.asarray(image).flatten() / 255.0

        dataset["filename"].append(file.name)
        dataset["data"].append(data)
        dataset["target"].append(target)

def __load_class_test(container_path: Path, dataset: dict, target: int) -> None:
    """Helper function to load all files of a single classification, in-place.

    Args:
        container_path (Path): Path to directory containing the data files.
        dataset (dict): Dataset instance to append data to.
        target (int): The target classification of the data being loaded.
    """
    i = -1
    for file in container_path.iterdir():
        i = i + 1
        if i < 500:
            continue
        image: Image.Image = Image.open(file).convert("L")
        data: ArrayLike = np.asarray(image).flatten() / 255.0

        dataset["filename"].append(file.name)
        dataset["data"].append(data)
        dataset["target"].append(target)


def load_star_galaxy_dataset() -> (Bunch, Bunch):
    """Loads and returns the star-galaxy dataset (classification).

    Returns:
        data: Dictionary-like object, with the following attributes.

            DESCR (str): Description of this dataset.
            filename (np.array): Name of image file of data.
            data (np.array): Image file represented as array.
            target (np.array): Target classification for data.
            target_names (np.array): The names of target classes.
    """

    # Set up base dataset without the data
    dataset_train: dict = {}
    dataset_train["DESCR"] = (
        "This is a simple dataset consisting of ~3000 64x64 images of stars "
        "and ~1000 images of galaxies. The images were captured by the "
        "in-house 1.3m telescope of the observatory situated in Devasthal, "
        "Nainital, India."
    )
    dataset_train["filename"] = []
    dataset_train["data"] = []
    dataset_train["target"] = []
    dataset_train["target_names"] = np.asarray(["star", "galaxy"])

    # Load each class of data into the dataset
    dataset_path: Path = _MODULE_PATH / "dataset"
    for target, target_name in enumerate(dataset_train["target_names"]):
        __load_class_training(
            container_path=(dataset_path / target_name),
            dataset=dataset_train,
            target=target,
        )

    # Convert dataset attributes to numpy arrays
    dataset_train["filename"] = np.asarray(dataset_train["filename"])
    dataset_train["data"] = np.asarray(dataset_train["data"])
    dataset_train["target"] = np.asarray(dataset_train["target"])





    dataset_test: dict = {}
    dataset_test["DESCR"] = (
        "This is a simple dataset consisting of ~3000 64x64 images of stars "
        "and ~1000 images of galaxies. The images were captured by the "
        "in-house 1.3m telescope of the observatory situated in Devasthal, "
        "Nainital, India."
    )
    dataset_test["filename"] = []
    dataset_test["data"] = []
    dataset_test["target"] = []
    dataset_test["target_names"] = np.asarray(["star", "galaxy"])

    # Load each class of data into the dataset
    dataset_path: Path = _MODULE_PATH / "dataset"
    for target, target_name in enumerate(dataset_test["target_names"]):
        __load_class_test(
            container_path=(dataset_path / target_name),
            dataset=dataset_test,
            target=target,
        )

    # Convert dataset attributes to numpy arrays
    dataset_test["filename"] = np.asarray(dataset_test["filename"])
    dataset_test["data"] = np.asarray(dataset_test["data"])
    dataset_test["target"] = np.asarray(dataset_test["target"])

    return (Bunch(**dataset_train), Bunch(**dataset_test))

def load_star_gal_training() -> Bunch:

    dataset: dict = {}
    dataset["DESCR"] = (
        "This is a simple dataset consisting of ~3000 64x64 images of stars "
        "and ~1000 images of galaxies. The images were captured by the "
        "in-house 1.3m telescope of the observatory situated in Devasthal, "
        "Nainital, India."
    )
    dataset["filename"] = []
    dataset["data"] = []
    dataset["target"] = []
    dataset["target_names"] = np.asarray(["star", "galaxy"])

    # Load each class of data into the dataset
    dataset_path: Path = _MODULE_PATH / "dataset"
    for target, target_name in enumerate(dataset["target_names"]):
        # __load_class_training(
        container_path=(dataset_path / target_name),
        #     dataset=dataset,
        #     target=target,
        # )
        i = 0
        for file in container_path.iterdir():
            if i == 500:
                break;
            image: Image.Image = Image.open(file).convert("L")
            data: ArrayLike = np.asarray(image).flatten() / 255.0

            dataset["filename"].append(file.name)
            dataset["data"].append(data)
            dataset["target"].append(target)

    # Convert dataset attributes to numpy arrays
    dataset["filename"] = np.asarray(dataset["filename"])
    dataset["data"] = np.asarray(dataset["data"])
    dataset["target"] = np.asarray(dataset["target"])

    return Bunch(**dataset)