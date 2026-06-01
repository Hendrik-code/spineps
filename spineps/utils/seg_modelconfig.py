from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from TPTBox import AX_CODES, ZOOMS, Location, Log_Type, Logger_Interface, v_name2idx

from spineps.seg_enums import Acquisition, InputType, Modality, ModelType

# Number of spatial dimensions of a volumetric (3D) image.
SPATIAL_DIMS = 3

# Default voxel geometry and post-processing cleaning thresholds. The voxel-count
# thresholds are multiplied by the resolution scaling factor at runtime.
DEFAULT_CUTOUT_SIZE = (248, 304, 64)
DEFAULT_SACRUM_IDS = (26,)
DEFAULT_CORPUS_SIZE_CLEANING = 100  # minimum corpus component size in voxels
DEFAULT_CORPUS_BORDER_THRESHOLD = 10
DEFAULT_VERT_SIZE_THRESHOLD = 250  # minimum vertebra size in voxels

# Default remapping of raw model label ids onto canonical SPINEPS label ids.
DEFAULT_LABEL_MAPPING = {41: 1, 42: 2, 43: 3, 44: 4, 45: 5, 46: 6, 47: 7, 48: 8, 49: 9, 50: 9, Location.Dens_axis.value: 9, 26: 0}


class Segmentation_Inference_Config:
    """Bucket for saving Inference Config data"""

    def __init__(
        self,
        logger: Logger_Interface | None,
        log_name: str,
        modality: str | tuple[str],
        acquisition: str,
        modeltype: str,
        model_expected_orientation: AX_CODES,
        available_folds: int | str | tuple[str] | tuple[int],
        inference_augmentation: bool,
        resolution_range: ZOOMS | tuple[ZOOMS, ZOOMS],
        default_step_size: float,
        labels: dict,
        expected_inputs: list[InputType | str] = [InputType.img],  # noqa: B006
        has_c1=False,
        needs_corp=False,
        sacrum_ids=DEFAULT_SACRUM_IDS,
        cutout_size=DEFAULT_CUTOUT_SIZE,
        corpus_size_cleaning=DEFAULT_CORPUS_SIZE_CLEANING,
        corpus_border_threshold=DEFAULT_CORPUS_BORDER_THRESHOLD,
        vert_size_threshold=DEFAULT_VERT_SIZE_THRESHOLD,
        mapping=None,
        **kwargs,
    ):
        scaling_factor = np.prod(resolution_range) if len(resolution_range) == SPATIAL_DIMS else np.prod(resolution_range[0])
        if mapping is None:
            mapping = dict(DEFAULT_LABEL_MAPPING)
        if not isinstance(modality, (list, tuple)):
            modality = [modality]

        self.log_name: str = log_name
        self.modalities: list[Modality] = [Modality[m] for m in modality]
        self.acquisition: Acquisition = Acquisition[acquisition]
        self.modeltype: ModelType = ModelType[modeltype]
        self.model_expected_orientation: AX_CODES = tuple(model_expected_orientation)  # type:ignore
        self.resolution_range = resolution_range
        self.available_folds: int | str | tuple[str] | tuple[int] = available_folds
        self.inference_augmentation: bool = inference_augmentation
        self.default_step_size = float(default_step_size)
        self.expected_inputs = [InputType[i] if isinstance(i, str) else i for i in expected_inputs]  # type: ignore
        self.has_c1 = has_c1
        self.needs_corp = needs_corp
        self.sacrum_ids = sacrum_ids
        self.cutout_size = cutout_size
        self.corpus_size_cleaning = corpus_size_cleaning * scaling_factor  # voxel threshold * resolution
        self.corpus_border_threshold = corpus_border_threshold
        self.vert_size_threshold = vert_size_threshold * scaling_factor  # voxel threshold * resolution
        self.mapping = mapping
        names = [member.name for member in Location]
        try:
            self.segmentation_labels = {
                int(k): Location[v].value if v in names else v_name2idx[v] if v in v_name2idx else int(v) for k, v in labels.items()
            }
        except KeyError as e:
            if logger is not None:
                logger.print("not a valid label!", Log_Type.FAIL)
            raise e  # noqa: TRY201

        if logger is not None:
            logger.prefix = self.log_name
            for k in kwargs:
                logger.print(f"Ignored inference config argument {k}", Log_Type.STRANGE)

    def str_representation(self, short: bool = False):
        to_print = self.__dict__ if not short else ["modalities", "acquisition", "resolution_range"]
        sb = []
        for key in self.__dict__:
            if key == "log_name" or key not in to_print:
                continue
            val = self.__dict__[key]
            val = str(val) if not isinstance(val, list) else [str(e) for e in val]
            sb.append(f"'{key!s}'={val}")

        return ", ".join(sb)

    def __str__(self):
        return self.str_representation()

    def __repr__(self):
        return self.str_representation(short=True)


def load_inference_config(json_dir: str | Path, logger: Logger_Interface | None = None):
    with open(str(json_dir), encoding="utf-8") as json_file:
        inference_config = json.load(json_file)
    return Segmentation_Inference_Config(**inference_config, logger=logger)
