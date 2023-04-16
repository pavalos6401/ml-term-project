"""This module provies utilities for using/manipulating the dataset."""

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

STAR = 0
"""Target value of a star."""

GALAXY = 1
"""Target value of a galaxy."""


def star_galaxy_split(x: _ARRAY, y: _ARRAY) -> tuple[_ARRAY, _ARRAY]:
    """Split the given dataset into stars and galaxies.

    Args:
        x: Data points.
        y: Targets.

    Returns:
        stars, galaxies: Subsets of data containing only the corresponding class.
    """

    stars, galaxies = [], []
    for i, im in enumerate(x):
        if y[i] == STAR:
            stars.append(im.copy())
        elif y[i] == GALAXY:
            galaxies.append(im.copy())
    return np.asarray(stars), np.asarray(galaxies)


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
           dataset. Default: `0.8`.
       test_size: Size of the test subset. Proportional to the remainder after
           splitting to find the training subset. Default: `0.5`.

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
            Default: `False`.

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
        for y, target_name in enumerate(dataset["target_names"]):
            for file in (dataset_path / target_name).iterdir():
                im = np.asarray(Image.open(file).convert("L")) / 255
                x = im.flatten()

                dataset["filename"].append(file.name)
                dataset["image"].append(im)
                dataset["data"].append(x)
                dataset["target"].append(y)

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


def __even_data(dataset: Bunch) -> tuple[_ARRAY, _ARRAY, _ARRAY]:
    """Creates a dataset with an even number of star and galaxy images.

    Returns:
        image, data, target: Images, flat array data, and targets.
    """

    # Get the stars and galaxies subsets
    stars, galaxies = star_galaxy_split(dataset.image, dataset.target)
    size = len(galaxies)

    # Make the stars subset a random sample of the same size as the galaxies
    stars = stars[np.random.choice(len(stars), size=size, replace=False)]

    # Create the custom subset of the dataset
    image = np.concatenate((stars, galaxies), axis=0)
    data = np.asarray([im.flatten() for im in image])
    target = np.concatenate((np.full(size, STAR), np.full(size, GALAXY)), axis=0)
    return image, data, target
