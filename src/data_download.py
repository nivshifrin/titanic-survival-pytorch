"""Fetch the Titanic dataset (Kaggle's train.csv) directly from Kaggle.

``ensure_train_csv(path)`` returns a path to ``train.csv``. If the file is not there
yet, it is downloaded from the Kaggle "titanic" competition with ``kagglehub``
(Kaggle's official download library) and saved at ``path``; later runs reuse that
local copy.

Kaggle needs credentials and an accepted competition before it allows a download:

1. Create an API token on https://www.kaggle.com/settings (section "API").
   Either save the downloaded ``kaggle.json`` as ``~/.kaggle/kaggle.json``, or set
   the environment variables ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.
2. Accept the competition rules once on https://www.kaggle.com/competitions/titanic/rules.

Only ``train.csv`` is downloaded; Kaggle's ``test.csv`` and ``gender_submission.csv``
are not used in this project.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import kagglehub
import requests
from kagglehub.exceptions import KaggleApiHTTPError, KaggleEnvironmentError, UnauthenticatedError

KAGGLE_COMPETITION = "titanic"
KAGGLE_FILE = "train.csv"

logger = logging.getLogger(__name__)


class DatasetDownloadError(RuntimeError):
    """Raised when train.csv is not available locally and cannot be downloaded from Kaggle."""


def download_train_csv(destination: Path) -> Path:
    """Download Kaggle's Titanic ``train.csv`` and save it at ``destination``.

    Args:
        destination: Where to save the file (its folder is created if missing).

    Returns:
        ``destination``.

    Raises:
        DatasetDownloadError: If the download fails (no credentials, competition
            rules not accepted, no network). The message says how to fix it.
    """
    try:
        # kagglehub downloads into its own cache folder and returns the path of the file there.
        downloaded = Path(kagglehub.competition_download(KAGGLE_COMPETITION, path=KAGGLE_FILE))
    except (UnauthenticatedError, KaggleEnvironmentError, KaggleApiHTTPError, requests.RequestException) as error:
        raise DatasetDownloadError(
            f"Could not download {KAGGLE_FILE} from the Kaggle '{KAGGLE_COMPETITION}' competition ({error}).\n"
            "To fix it, either:\n"
            "  1. set up Kaggle credentials (~/.kaggle/kaggle.json, or the environment variables "
            "KAGGLE_USERNAME and KAGGLE_KEY; create a token on https://www.kaggle.com/settings) and accept the "
            "competition rules on https://www.kaggle.com/competitions/titanic/rules, then run again; or\n"
            f"  2. download train.csv by hand from https://www.kaggle.com/competitions/titanic/data "
            f"and place it at {destination}."
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, destination)
    return destination


def ensure_train_csv(path: str | Path) -> Path:
    """Return the path to ``train.csv``, downloading it from Kaggle first if it is missing.

    Args:
        path: Where ``train.csv`` is expected (for example ``data/train.csv``).

    Returns:
        The path to the file.

    Raises:
        DatasetDownloadError: If the file is missing and the download fails.
    """
    path = Path(path)
    if path.is_file():
        logger.info("Using the local copy of the dataset: %s", path)
        return path
    logger.info("%s not found; downloading it from Kaggle (competition '%s')...", path, KAGGLE_COMPETITION)
    download_train_csv(path)
    logger.info("Downloaded the dataset from Kaggle to %s", path)
    return path
