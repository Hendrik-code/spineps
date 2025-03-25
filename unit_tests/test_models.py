# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest
from pathlib import Path

from TPTBox import No_Logger
from typing_extensions import Self

from spineps.get_models import (
    check_available_models,
    get_actual_model,
    get_instance_model,
    get_labeling_model,
    get_mri_segmentor_models_dir,
    get_semantic_model,
)

logger = No_Logger()


class Test_GetModel(unittest.TestCase):
    def test_check_available_models(self):
        check_available_models(__file__)

    def test_get_unavailable_model(self):
        with self.assertRaises(KeyError):
            get_semantic_model("test")

        with self.assertRaises(KeyError):
            get_instance_model("test")

        with self.assertRaises(KeyError):
            get_labeling_model("test")

        with self.assertRaises(FileNotFoundError):
            get_actual_model(Path(__file__).parent)

        with self.assertRaises(NotADirectoryError):
            get_actual_model(__file__ + "/inference_config.json")
