#!/usr/bin/env python3
"""This module provies utilities for using/manipulating the dataset."""

from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image
from sklearn.utils import Bunch


def __load_class(path: Path, dataset: dict, classification: int) -> None:
    """Helper function to load a single classification.

    Args:
        path (Path): Path to directory containing the data files.
        dataset (dict): Dataset instance to append data to.
        classification (int): The target classification of the data being loaded.
    """

    for f in path.iterdir():
        dataset["filename"].append(f.name)
        dataset["data"].append(np.array(Image.open(f)))
        dataset["classification"].append(classification)


def load_star_galaxy_dataset(
    as_bunch: bool = False,
) -> Union[dict[str, Any], Bunch]:
    """Loads and returns the star-galaxy dataset (classification).

    Args:
        as_bunch (bool): Optional. Return the dataset as an sklearn Bunch object.

    Returns:
        data: Dictionary-like object.

        Keys for the dataset:
            "filename": Name of image file of data (str).
            "data": Image file represented as array (np.array).
            "classification": Target classification for data (0: star, 1: galaxy).
    """

    dataset: dict[str, Any] = {}
    dataset["description"] = (
        "This is a simple dataset consisting of ~3000 64x64 images of stars"
        "and ~1000 images of galaxies. The images were captured by the"
        "in-house 1.3m telescope of the observatory situated in Devasthal,"
        "Nainital, India."
    )
    dataset["filename"] = []
    dataset["data"] = []
    dataset["classification"] = []

    __load_class(path=Path("./dataset/star/"), dataset=dataset, classification=0)
    __load_class(path=Path("./dataset/galaxy/"), dataset=dataset, classification=1)

    return Bunch(**dataset) if as_bunch else dataset
