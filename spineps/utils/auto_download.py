from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Union

from TPTBox import Print_Logger
from tqdm import tqdm

from spineps.seg_enums import SpinepsPhase
from spineps.utils.filepaths import get_mri_segmentor_models_dir

link = "https://github.com/Hendrik-code/spineps/releases/download/"
current_highest_version = "v1.0.9"
current_instance_highest_version = "v1.2.0"
current_labeling_highest_version = "v1.4.0"

phase_to_version: dict[SpinepsPhase, str] = {
    SpinepsPhase.SEMANTIC: current_highest_version,
    SpinepsPhase.INSTANCE: current_instance_highest_version,
    SpinepsPhase.LABELING: current_labeling_highest_version,
}

instances: dict[str, Union[Path, str]] = {"instance": link + current_instance_highest_version + "/instance.zip"}
semantic: dict[str, Union[Path, str]] = {
    "t2w": link + current_highest_version + "/t2w.zip",
    "t1w": link + current_highest_version + "/t1w.zip",
    "vibe": link + current_highest_version + "/vibe.zip",
}
labeling: dict[str, Union[Path, str]] = {"t2w_labeling": link + current_labeling_highest_version + "/labeling.zip"}

download_names = {
    "instance": "instance_sagittal",
    "t2w": "T2w_semantic",
    "t1w": "T1w_semantic",
    "vibe": "Vibe_semantic",
    "t2w_labeling": "T2w_labeling",
}


def download_if_missing(key, url, phase: SpinepsPhase):
    version = phase_to_version[phase]
    out_path = Path(get_mri_segmentor_models_dir(), download_names[key] + "_" + version)
    if not out_path.exists():
        download_weights(url, out_path)

    return out_path


def download_weights(weights_url, out_path) -> None:
    out_path = Path(out_path)
    logger = Print_Logger()
    try:
        # Retrieve file size
        with urllib.request.urlopen(str(weights_url)) as response:
            file_size = int(response.info().get("Content-Length", -1))
    except Exception:
        logger.on_fail("Download attempt failed:", weights_url)
        return
    logger.print("Downloading pretrained weights...")

    with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc=Path(weights_url).name) as pbar:

        def update_progress(block_num: int, block_size: int, total_size: int) -> None:
            if pbar.total != total_size:
                pbar.total = total_size
            pbar.update(block_num * block_size - pbar.n)

        zip_path = Path(str(out_path) + ".zip")
        # Download the file
        urllib.request.urlretrieve(str(weights_url), zip_path, reporthook=update_progress)

    logger.print("Extracting pretrained weights...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_path)
    # Test if there is an additional folder and move the content on up.
    if not Path(out_path, "inference_config.json").exists():
        source = next(out_path.iterdir())
        assert source.is_dir()
        for i in source.iterdir():
            shutil.move(i, out_path)

    zip_path.unlink()
