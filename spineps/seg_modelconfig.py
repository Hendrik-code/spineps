import json
from pathlib import Path

from TPTBox import Ax_Codes, Location, Log_Type, Logger_Interface, Zooms, v_name2idx

from spineps.seg_enums import Acquisition, InputType, Modality, ModelType


class Segmentation_Inference_Config:
    """Bucket for saving Inference Config data"""

    def __init__(
        self,
        logger: Logger_Interface | None,
        log_name: str,
        modality: str | list[str],
        acquisition: str,
        modeltype: str,
        model_expected_orientation: Ax_Codes,
        available_folds: int,
        inference_augmentation: bool,
        resolution_range: Zooms | tuple[Zooms, Zooms],
        default_step_size: float,
        labels: dict,
        expected_inputs: list[InputType | str] = [InputType.img],  # noqa: B006
        **kwargs,
    ):
        if not isinstance(modality, list):
            modality = [modality]

        self.log_name: str = log_name
        self.modalities: list[Modality] = [Modality[m] for m in modality]
        self.acquisition: Acquisition = Acquisition[acquisition]
        self.modeltype: ModelType = ModelType[modeltype]
        self.model_expected_orientation: Ax_Codes = tuple(model_expected_orientation)  # type:ignore
        self.resolution_range = resolution_range
        self.available_folds: int = int(available_folds)
        self.inference_augmentation: bool = inference_augmentation
        self.default_step_size = float(default_step_size)
        self.expected_inputs = [InputType[i] if isinstance(i, str) else i for i in expected_inputs]  # type: ignore
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
            logger.override_prefix = self.log_name
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
