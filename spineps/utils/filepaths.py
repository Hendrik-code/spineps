from __future__ import annotations

import os
import warnings
from itertools import chain
from pathlib import Path

spineps_environment_path_override = None  # Path(
#    "/DATA/NAS/ongoing_projects/hendrik/mri_usage/models/"
# )  # None  # You can put an absolute path to the model weights here instead of using environment variable
spineps_environment_path_backup = Path(__file__).parent.parent.joinpath("models")  # EDIT this to use this instead of environment variable


def get_mri_segmentor_models_dir() -> Path:
    """Returns the path to the models weight directory, reading from environment variable, specified override or backup

    Returns:
        Path: Path to the overall models folder
    """
    folder_path = (
        os.environ.get("SPINEPS_SEGMENTOR_MODELS")
        if spineps_environment_path_override is None or not spineps_environment_path_override.exists()
        else spineps_environment_path_override
    )
    if folder_path is None and spineps_environment_path_backup is not None:
        folder_path = spineps_environment_path_backup

    assert (
        folder_path is not None
    ), "Environment variable 'SPINEPS_SEGMENTOR_MODELS' is not defined. Setup the environment variable as stated in the readme or set the override in utils.filepaths.py"
    folder_path = Path(folder_path)
    assert folder_path.exists(), f"Environment variable 'SPINEPS_SEGMENTOR_MODELS' = {folder_path} does not exist"
    return folder_path


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
    print(f"search_path: in {basepath}{query}") if verbose else None
    paths = sorted(chain(list(Path(f"{basepath}").glob(f"{query}"))))
    if len(paths) == 0 and not suppress:
        warnings.warn(f"did not find any paths in {basepath}{query}", UserWarning, stacklevel=1)
    return paths


if __name__ == "__main__":
    print(filepath_model("highres"))
