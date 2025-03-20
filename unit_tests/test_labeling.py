# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from TPTBox import NII, No_Logger
from TPTBox.tests.test_utils import get_test_mri
from typing_extensions import Self

from spineps.phase_labeling import VertLabelingClassifier
from spineps.phase_pre import preprocess_input
from spineps.phase_semantic import predict_semantic_mask
from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Inference_Config
from spineps.seg_utils import check_input_model_compatibility, check_model_modality_acquisition

logger = No_Logger()


class Test_Labeling_Model_Dummy(VertLabelingClassifier):
    def __init__(
        self,
        model_folder: str | Path = __file__,
        inference_config: Segmentation_Inference_Config | None = None,
        default_verbose: bool = False,
        default_allow_tqdm: bool = True,
    ):
        self.logger = No_Logger()
        inference_config = Segmentation_Inference_Config(
            logger=self.logger,
            modality=["T2w", "SEG", "T1w"],
            acquisition="sag",
            log_name="DummySegModel",
            modeltype="unet",
            model_expected_orientation=("P", "I", "R"),
            available_folds=1,
            inference_augmentation=False,
            resolution_range=[0.75, 0.75, 1.65],
            default_step_size=0.5,
            labels={1: 1},
        )
        super().__init__(model_folder, inference_config, default_verbose, default_allow_tqdm)

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        self.print("Model loaded from", self.model_folder, verbose=True)
        self.predictor = object()
        return self


class Test_Labeling_Read_Labels(unittest.TestCase):
    def test_simple_testing_case(self):
        from spineps.architectures.read_labels import (
            Objectives,
            Target,
            VertExact,
            VertExactClass,
            VertGroup,
            VertRegion,
            VertRel,
            VertT13,
            get_subject_info,
            get_vert_entry,
            vertgrp_sequence_to_class,
        )
        from spineps.phase_labeling import fpath_post_processing, is_valid_vertebra_sequence

        objectives = Objectives(
            [
                # Target.FULLYVISIBLE,
                Target.REGION,
                Target.VERTREL,
                Target.VERT,
                Target.VERTGRP,
                Target.VERTEX,
                Target.VT13,
            ],
            as_group=True,
        )

        entry_dict = {
            "vert_exact": VertExact.L1,
            "vert_region": VertRegion.LWS,
            "vert_rel": VertRel.FIRST_LWK,
            "vert_cut": True,
            "vert_group": VertGroup.L12,
            "vert_exact2": VertExactClass.T13,
            "vert_t13": VertT13.T13,
        }

        label = objectives(entry_dict)
        print(label)

        print()
        print()

        vertgrp = [VertGroup.C12, VertGroup.C345, VertGroup.C345, VertGroup.C345, VertGroup.L12]
        print(vertgrp)
        vert_sequ = vertgrp_sequence_to_class(vertgrp)
        vert_sequ = [i.value for i in vert_sequ]
        vert_sequ = fpath_post_processing(vert_sequ)
        print(vert_sequ)
        print(is_valid_vertebra_sequence(vert_sequ))

        print()
        print()

        vert_subfolders_int = [18, 19, 28, 20, 21, 22, 23, 24, 25]
        subject_info = get_subject_info(
            subject_name=1337,
            anomaly_dict={},
            vert_subfolders_int=vert_subfolders_int,
            anomaly_factor_condition=1,
        )

        for v in vert_subfolders_int:
            _, entry = get_vert_entry(v, subject_info)
            label = objectives(entry)
            print(v, label)
