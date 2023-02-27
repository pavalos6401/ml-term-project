#!/usr/bin/env python3
"""This module provies utilities for using/manipulating the dataset."""

from pathlib import Path

import numpy as np
from PIL import Image as Im
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

    for f in container_path.iterdir():
        dataset["filename"].append(f.name)
        dataset["data"].append(np.asarray(Im.open(f).convert("L")).flatten())
        dataset["target"].append(target)


def load_star_galaxy_dataset() -> Bunch:
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
        __load_class(
            container_path=(dataset_path / target_name),
            dataset=dataset,
            target=target,
        )

    # Convert dataset attributes to numpy arrays
    dataset["filename"] = np.asarray(dataset["filename"])
    dataset["data"] = np.asarray(dataset["data"])
    dataset["target"] = np.asarray(dataset["target"])

    return Bunch(**dataset)
