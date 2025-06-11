# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest
from pathlib import Path

from TPTBox import No_Logger
from typing_extensions import Self

from spineps.seg_enums import Acquisition, InputType, Modality, ModelType
from spineps.seg_model import Segmentation_Inference_Config

logger = No_Logger()


class Test_ModelConfig(unittest.TestCase):
    def test_normal(self):
        config = Segmentation_Inference_Config(
            logger=No_Logger(),
            log_name="test",
            modality="CT",
            acquisition="ax",
            modeltype="nnunet",
            model_expected_orientation=["P", "I", "R"],
            available_folds=1,
            inference_augmentation=False,
            resolution_range=[(0.5, 0.5, 0.5)],
            default_step_size=0.5,
            labels={
                1: 0,
            },
            notfound_argument=0,
        )
        self.assertTrue(config is not None)

    def test_enumerror(self):
        with self.assertRaises(KeyError):
            config = Segmentation_Inference_Config(
                logger=No_Logger(),
                log_name="test",
                modality="CT",
                acquisition="axial_does_not_work",
                modeltype="nnunet",
                model_expected_orientation=["P", "I", "R"],
                available_folds=1,
                inference_augmentation=False,
                resolution_range=[(0.5, 0.5, 0.5)],
                default_step_size=0.5,
                labels={
                    1: 0,
                },
            )
            self.assertTrue(config is not None)

    def test_keyerror(self):
        with self.assertRaises(ValueError):
            config = Segmentation_Inference_Config(
                logger=No_Logger(),
                log_name="test",
                modality="CT",
                acquisition="ax",
                modeltype="nnunet",
                model_expected_orientation=["P", "I", "R"],
                available_folds=1,
                inference_augmentation=False,
                resolution_range=[(0.5, 0.5, 0.5)],
                default_step_size=0.5,
                labels={
                    1: "ERROR",
                },
            )
            self.assertTrue(config is not None)
