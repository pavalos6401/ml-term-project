"""This module provies utilities for using/manipulating the dataset."""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split as _train_test_split
from sklearn.utils import Bunch

_MODULE_PATH = Path(__file__).parent
"""Parent directory of this python module."""


def __load_class(container_path: Path, dataset: dict, target: int) -> None:
    """Helper function to load all files of a single classification, in-place.

    Args:
        container_path (Path): Path to directory containing the data files.
        dataset (dict): Dataset instance to append data to.
        target (int): The target classification of the data being loaded.
    """

    for file in container_path.iterdir():
        image: np.ndarray = np.asarray(Image.open(file).convert("L"))
        data: np.ndarray = image.flatten() / 255.0

        dataset["filename"].append(file.name)
        dataset["image"].append(image)
        dataset["data"].append(data)
        dataset["target"].append(target)


def load_star_galaxy_dataset(return_X_y: bool = False) -> Union[Bunch, tuple]:
    """Loads and returns the star-galaxy dataset (classification).

    Args:
        return_X_y: Return just the data and target if True. Default: `False`.

    Returns:
        data: Dictionary-like object, with the following attributes.

            DESCR (str): Description of this dataset.
            filename (np.array): Name of image file of data.
            image (np.array): Image file.
            data (np.array): Image file represented as array.
            target (np.array): Target classification for data.
            target_names (np.array): The names of target classes.

        X, y: If `return_X_y` is True, returns the data and target.
    """

    # Set up base dataset without the data
    dataset: dict = {}
    dataset["DESCR"] = (
        "This is a simple dataset consisting of ~3000 64x64 images of stars "
        "and ~1000 images of galaxies. The images were captured by the "
        "in-house 1.3m telescope of the observatory situated in Devasthal, "
        "Nainital, India."
    )
    dataset["filename"] = []
    dataset["image"] = []
    dataset["data"] = []
    dataset["target"] = []
    dataset["target_names"] = np.asarray(["star", "galaxy"])

    # Load each class of data into the dataset
    dataset_path: Path = _MODULE_PATH / "dataset"
    for target, target_name in enumerate(dataset["target_names"]):
        __load_class(
            container_path=(dataset_path / target_name),
            dataset=dataset,
            target=target,
        )

    # Convert dataset attributes to numpy arrays
    dataset["filename"] = np.asarray(dataset["filename"])
    dataset["image"] = np.asarray(dataset["image"])
    dataset["data"] = np.asarray(dataset["data"])
    dataset["target"] = np.asarray(dataset["target"])

    if return_X_y:
        return dataset["data"], dataset["target"]

    return Bunch(**dataset)


def train_test_split(
    X,
    y,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle: bool = True,
    stratify=None,
) -> list:
    """Wraps sklearn's :func:`train_test_split` function.

    This ensures the training set has a more fair share of stars to galaxies.

    Args:
        *arrays: arrays
        test_size: optional
        train_size: optional
        random_state: optional
        shuffle: optional
        stratify: optional

    Returns:
        splitting: list
    """

    # TODO: Ensure train has a more even number of stars?

    return _train_test_split(
        X,
        y,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )
