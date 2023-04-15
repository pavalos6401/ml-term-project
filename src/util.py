"""This module provies utilities for using/manipulating the dataset."""

import random
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

_MODULE_PATH: Path = Path(__file__).parent
"""Parent directory of this python module."""

__MEMO_DATASET: Union[None, Bunch] = None
"""Memoized dataset for performance."""

_ARRAY = Union[list, np.ndarray]
"""Type definition for anything array-like."""


def train_val_test_split(
    x: _ARRAY,
    y: _ARRAY,
    train_size: float = 0.8,
    test_size: float = 0.5,
) -> tuple[_ARRAY, _ARRAY, _ARRAY, _ARRAY, _ARRAY, _ARRAY]:
    """Split the given dataset into train, validation, and test subsets.

    Args:
       x: Data points.
       y: Targets.
       train_size: Size of the training subset. Proportional to the entire
           dataset. Default: 0.8
       test_size: Size of the test subset. Proportional to the remainder after
           splitting to find the training subset. Default: 0.5

    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test: The splits of the dataset.
    """

    x_train, x_rest, y_train, y_rest = train_test_split(x, y, train_size=train_size)
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=test_size)
    return x_train, x_val, x_test, y_train, y_val, y_test


def load_star_galaxy_dataset(even: bool = False) -> Bunch:
    """Loads and returns the star-galaxy dataset (classification).

    Args:
        even: Create a dataset with an even number of stars and galaxies.

    Returns:
        data: Dictionary-like object, with the following attributes.

            DESCR (str): Description of this dataset.
            filename (np.array): Name of image file of data.
            image (np.array): Image file.
            data (np.array): Image file represented as array.
            target (np.array): Target classification for data.
            target_names (np.array): The names of target classes.
    """

    # Check if dataset has been memoized
    global __MEMO_DATASET
    if __MEMO_DATASET is None:
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

        # Memoize the dataset
        __MEMO_DATASET = Bunch(**dataset)

    # If an even dataset is requested create a new dataset to return,
    # don't modify the memoized dataset
    if even:
        image, data, target = __even_data(__MEMO_DATASET)
        return Bunch(
            DESCR=__MEMO_DATASET["DESCR"],
            filename=__MEMO_DATASET["filename"].copy(),
            image=image,
            data=data,
            target=target,
            target_names=__MEMO_DATASET["target_names"].copy(),
        )

    # Otherwise return the memoized dataset
    return __MEMO_DATASET


def __load_class(container_path: Path, dataset: dict, target: int) -> None:
    """Helper function to load all files of a single classification, in-place.

    Args:
        container_path (Path): Path to directory containing the data files.
        dataset (dict): Dataset instance to append data to.
        target (int): The target classification of the data being loaded.
    """

    for file in container_path.iterdir():
        image: np.ndarray = np.asarray(Image.open(file).convert("L")) / 255
        data: np.ndarray = image.flatten()

        dataset["filename"].append(file.name)
        dataset["image"].append(image)
        dataset["data"].append(data)
        dataset["target"].append(target)


def __even_data(dataset: Bunch) -> tuple[_ARRAY, _ARRAY, _ARRAY]:
    """Creates a dataset with an even number of star and galaxy images.

    Returns:
        image, data, target: Images, flat array data, and targets.
    """

    # Handy-dandy variables
    STAR: int = np.where(dataset.target_names == "star")
    GALAXY: int = np.where(dataset.target_names == "galaxy")

    # Get all the galaxies in the dataset
    galaxies: np.ndarray = np.asarray(
        [im.copy() for i, im in enumerate(dataset.image) if dataset.target[i] == GALAXY]
    )
    galaxies_num: int = len(galaxies)

    # Sample the same number of stars from the dataset
    stars: np.ndarray = np.asarray(
        random.sample(
            [
                im.copy()
                for i, im in enumerate(dataset.image)
                if dataset.target[i] == STAR
            ],
            k=galaxies_num,
        )
    )

    # Create the custom subset of the dataset
    image: np.ndarray = np.concatenate((stars, galaxies), axis=0)
    data: np.ndarray = np.asarray([im.flatten() for im in image])
    target: np.ndarray = np.concatenate(
        (np.full(galaxies_num, STAR), np.full(galaxies_num, GALAXY)), axis=0
    )
    return image, data, target
