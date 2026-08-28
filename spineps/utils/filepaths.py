"""File-path helpers for locating the SPINEPS model weights directory and individual model folders."""

from __future__ import annotations

import os
import warnings
from itertools import chain
from pathlib import Path

from TPTBox import No_Logger

logger = No_Logger(prefix="filepaths")

spineps_environment_path_override = None  # Path(
#    "/DATA/NAS/ongoing_projects/hendrik/mri_usage/models/"
# )  # None  # You can put an absolute path to the model weights here instead of using environment variable
# EDIT this to use this instead of the environment variable. Created on demand by
# get_mri_segmentor_models_dir(), never at import time (that would write into site-packages).
spineps_environment_path_backup = Path(__file__).parent.parent.joinpath("models")


def get_mri_segmentor_models_dir() -> Path:
    """Returns the path to the models weight directory, reading from environment variable, specified override or backup

    Returns:
        Path: Path to the overall models folder

    Raises:
        RuntimeError: If no models directory could be determined, or the fallback directory cannot be created.
        FileNotFoundError: If the directory named by 'SPINEPS_SEGMENTOR_MODELS' does not exist.
    """
    if spineps_environment_path_override is not None and spineps_environment_path_override.exists():
        return spineps_environment_path_override

    from_env = os.environ.get("SPINEPS_SEGMENTOR_MODELS")
    if from_env is not None:
        folder_path = Path(from_env)
        if not folder_path.exists():
            raise FileNotFoundError(f"Environment variable 'SPINEPS_SEGMENTOR_MODELS' = {folder_path} does not exist")
        return folder_path

    if spineps_environment_path_backup is None:
        raise RuntimeError(
            "Environment variable 'SPINEPS_SEGMENTOR_MODELS' is not defined. Setup the environment variable as stated "
            "in the readme or set the override in utils.filepaths.py"
        )
    try:
        spineps_environment_path_backup.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Environment variable 'SPINEPS_SEGMENTOR_MODELS' is not defined and the fallback models directory "
            f"{spineps_environment_path_backup} could not be created ({e}). Set the environment variable as stated "
            "in the readme."
        ) from e
    return spineps_environment_path_backup


def filepath_model(model_folder_name: str, model_dir: str | Path | None = None) -> Path:
    """Returns the path to a model folder with specified model id name

    Args:
        model_folder_name (str): Name of the model (corresponds to its folder name)
        model_dir (str | Path | None, optional): Base path to the models directory. If none, will calculate that itself. Defaults to None.

    Returns:
        Path: Path to the model specified by name
    """
    if model_dir is None:
        model_dir = get_mri_segmentor_models_dir()

    if isinstance(model_dir, str):
        model_dir = Path(model_dir)

    path = model_dir.joinpath(model_folder_name)
    if not path.exists():
        paths = search_path(Path(model_dir), query=f"**/{model_folder_name}")
        if len(paths) == 1:
            return paths[0]
    return model_dir.joinpath(model_folder_name)


def search_path(basepath: str | Path, query: str, verbose: bool = False, suppress: bool = False) -> list[Path]:
    """Searches from basepath with query

    Args:
        basepath: ground path to look into
        query: search query, can contain wildcards like *.npz or **/*.npz
        verbose:
        suppress: if true, will not throwing warnings if nothing is found

    Returns:
        All found paths
    """
    basepath = str(basepath)
    if not basepath.endswith("/"):
        basepath += "/"
    logger.print(f"search_path: in {basepath}{query}", verbose=verbose)
    paths = sorted(chain(list(Path(f"{basepath}").glob(f"{query}"))))
    if len(paths) == 0 and not suppress:
        warnings.warn(f"did not find any paths in {basepath}{query}", UserWarning, stacklevel=1)
    return paths


if __name__ == "__main__":
    print(filepath_model("highres"))
