"""Automatic download and extraction of pretrained SPINEPS model weights from the GitHub releases."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from TPTBox import Print_Logger
from tqdm import tqdm

from spineps.seg_enums import SpinepsPhase
from spineps.utils.filepaths import get_mri_segmentor_models_dir

link = "https://github.com/Hendrik-code/spineps/releases/download/"
current_highest_version = "v1.0.9"
current_instance_highest_version = "v1.2.0"
current_labeling_highest_version = "v1.4.0"
current_highest_ct_version = "v2.0.0"


phase_to_version: dict[str, str] = {
    SpinepsPhase.SEMANTIC.name: current_highest_version,
    SpinepsPhase.INSTANCE.name: current_instance_highest_version,
    SpinepsPhase.LABELING.name: current_labeling_highest_version,
    SpinepsPhase.SEMANTIC.name + "_ct": current_highest_ct_version,
    SpinepsPhase.INSTANCE.name + "_ct_instance": current_highest_ct_version,
}

instances: dict[str, Path | str] = {
    "instance": link + current_instance_highest_version + "/instance.zip",
    "ct_instance": link + current_highest_ct_version + "/CT_instance.zip",
}
semantic: dict[str, Path | str] = {
    "t2w": link + current_highest_version + "/t2w.zip",
    "t1w": link + current_highest_version + "/t1w.zip",
    "vibe": link + current_highest_version + "/vibe.zip",
    "ct": link + current_highest_ct_version + "/ct.zip",
}
# Semantic models are also accepted with an explicit "_semantic" suffix: "t2w_semantic" names the same
# weights as "t2w". The alias has to resolve to the same download name and release version as the
# canonical id, which is what canonical_model_key() below is for.
SEMANTIC_ALIAS_SUFFIX = "_semantic"
for i, j in semantic.copy().items():
    semantic[i + SEMANTIC_ALIAS_SUFFIX] = j

labeling: dict[str, Path | str] = {
    "t2w_labeling": link + current_labeling_highest_version + "/labeling.zip",
    "ct_labeling": link + current_labeling_highest_version + "/ct_labeling.zip",
}

download_names = {
    "instance": "instance_sagittal",
    "ct_instance": "CT_instance",
    "t2w": "T2w_semantic",
    "t1w": "T1w_semantic",
    "vibe": "Vibe_semantic",
    "ct": "CT_semantic",
    "t2w_labeling": "T2w_labeling",
    "ct_labeling": "CT_labeling",
}


def canonical_model_key(key: str) -> str:
    """Maps a model id alias onto the id its weights are registered under.

    ``"t2w_semantic"`` and ``"t2w"`` are the same model, but only the canonical id has entries in
    ``download_names`` and ``phase_to_version``. Without this, asking for an ``*_semantic`` id raised
    ``KeyError: 't2w_semantic'`` on first use -- even though those ids are listed as available models.

    Args:
        key (str): Model id as the user (or the model registry) spelled it.

    Returns:
        str: The canonical model id.
    """
    if key.endswith(SEMANTIC_ALIAS_SUFFIX):
        canonical = key[: -len(SEMANTIC_ALIAS_SUFFIX)]
        if canonical in download_names:
            return canonical
    return key


def download_if_missing(key: str, url: Path | str, phase: SpinepsPhase) -> Path:
    """Return the local model folder for a model, downloading and extracting its weights if absent.

    The target folder name combines the model's download name with the version resolved for its phase (and the
    phase/key-specific override when one exists, e.g. CT models).

    Args:
        key: Model key identifying the model within its phase (e.g. ``"t2w"``, ``"instance"``).
        url: Release URL of the model's weights zip archive.
        phase (SpinepsPhase): Pipeline phase the model belongs to.

    Returns:
        Path: Path to the local model folder containing the (possibly just downloaded) weights.

    Raises:
        KeyError: If the model id is unknown to the download registry.
        RuntimeError: If the weights could not be downloaded or extracted.
    """
    key = canonical_model_key(key)
    if key not in download_names:
        raise KeyError(f"no download is registered for model '{key}', known models are {sorted(download_names)}")
    version = phase_to_version.get(f"{phase.name}_{key}", phase_to_version[phase.name])
    out_path = Path(get_mri_segmentor_models_dir(), download_names[key] + "_" + version)
    if not out_path.exists():
        download_weights(url, out_path)

    return out_path


def download_weights(weights_url: Path | str, out_path: Path | str) -> None:
    """Download a weights zip archive and extract it into ``out_path``.

    Shows a progress bar during download. Everything happens inside a temporary folder next to
    ``out_path``, which is moved into place only once the archive extracted and actually contains an
    ``inference_config.json``; on any failure the temporary folder is removed and the error is raised.
    That matters because the caller treats "``out_path`` exists" as "the model is installed": a download
    that failed halfway used to leave a partial folder behind that was never retried and never worked,
    and a failure before the download even started was swallowed entirely, surfacing much later as a
    confusing "config not found" from the model loader.

    Args:
        weights_url: URL of the weights zip archive to download.
        out_path: Destination folder for the extracted weights.

    Raises:
        RuntimeError: If the archive could not be downloaded or extracted, or if what was extracted does
            not look like a model folder.
    """
    out_path = Path(out_path)
    logger = Print_Logger()
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # A unique staging folder, so that several SPINEPS processes (e.g. a job array) downloading the same
    # model at the same time cannot overwrite each other's partial files.
    staging = Path(tempfile.mkdtemp(prefix=out_path.name + ".download.", dir=out_path.parent))
    try:
        zip_path = staging.joinpath("weights.zip")
        extracted = staging.joinpath("extracted")
        try:
            with urllib.request.urlopen(str(weights_url)) as response:
                file_size = int(response.info().get("Content-Length", -1))
            logger.print("Downloading pretrained weights...")
            with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc=Path(weights_url).name) as pbar:

                def update_progress(block_num: int, block_size: int, total_size: int) -> None:
                    if pbar.total != total_size:
                        pbar.total = total_size
                    pbar.update(block_num * block_size - pbar.n)

                urllib.request.urlretrieve(str(weights_url), zip_path, reporthook=update_progress)
        except Exception as e:
            raise RuntimeError(
                f"could not download the model weights from {weights_url} ({e}). Check your internet "
                "connection, or download the weights manually from "
                "https://github.com/Hendrik-code/spineps/releases and extract them into "
                f"{out_path.parent} (see the README)."
            ) from e

        logger.print("Extracting pretrained weights...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extracted)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"the downloaded weights archive from {weights_url} is corrupt ({e}); try again.") from e

        model_folder = _model_folder_in(extracted)
        if model_folder is None:
            contents = sorted(p.name for p in extracted.rglob("*"))[:10]
            raise RuntimeError(f"the archive from {weights_url} contains no inference_config.json, found {contents}")
        # Another process may have finished the same download while this one was busy; theirs is as good.
        if not out_path.exists():
            shutil.move(str(model_folder), str(out_path))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _model_folder_in(extracted: Path) -> Path | None:
    """Picks the folder to install as the model folder out of a freshly extracted archive.

    Archives either hold the model at the top level or wrap it in one extra folder. The config itself may
    sit deeper still (nnU-Net plan subfolders), which the model loader resolves by globbing, so only its
    presence somewhere below is verified here.

    Args:
        extracted (Path): Folder the archive was extracted into.

    Returns:
        Path | None: The folder to move into place, or None if this is not a SPINEPS model archive.
    """
    if next(extracted.rglob("inference_config.json"), None) is None:
        return None
    if extracted.joinpath("inference_config.json").is_file():
        return extracted
    children = sorted(extracted.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]  # the archive wrapped the model in one extra folder
    return extracted
