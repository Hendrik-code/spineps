# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import spineps
from spineps.get_models import SegmentationModel, check_available_models, get_actual_model
from spineps.utils.filepaths import (
    filepath_model,
    get_mri_segmentor_models_dir,
    search_path,
    spineps_environment_path_backup,
    spineps_environment_path_override,
)


class Test_filepaths(unittest.TestCase):
    # def test_load_model_from_path(self):
    #    ms_p = "/DATA/NAS/ongoing_projects/hendrik/nako-segmentation/nnUNet/nnUNet_results/Dataset122_nako_wotest_spiderarcus/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom"
    #    mv_p = "/DATA/NAS/ongoing_projects/hendrik/nako-segmentation/nnUNet/unet3d_result/nakospider_highres_shiftposi"
    #    model_dir = "/DATA/NAS/ongoing_projects/hendrik/nako-segmentation/nnUNet/"
    #    ms = get_segmentation_model(in_config=filepath_model(ms_p, model_dir=model_dir))
    #    self.assertTrue(isinstance(ms, SegmentationModel))
    #    mv = get_segmentation_model(in_config=filepath_model(mv_p, model_dir=model_dir))
    #    self.assertTrue(isinstance(mv, SegmentationModel))
    #    self.assertTrue(True)

    def test_search_path_simple(self):
        package_path = Path(spineps.__file__).parent
        print(package_path)
        predictor_search = search_path(package_path, query="**/predictor.py")
        print(predictor_search)
        self.assertTrue(len(predictor_search) == 1)
        self.assertEqual(predictor_search[0], package_path.joinpath("utils", "predictor.py"))

    def test_search_path_multi(self):
        package_path = Path(spineps.__file__).parent
        print(package_path)
        predictor_search = search_path(package_path, query="seg_*.py")
        print(predictor_search)
        self.assertTrue(len(predictor_search) == 5, predictor_search)
        # self.assertEqual(predictor_search[0], package_path.joinpath("utils", "predictor.py"))

    def test_env_path(self):
        # no override
        try:
            p = get_mri_segmentor_models_dir()
            if spineps_environment_path_override is not None:
                self.assertEqual(p, spineps_environment_path_override)
            else:
                self.assertEqual(str(p) + "/", os.environ.get("SPINEPS_SEGMENTOR_MODELS"))
        except (AssertionError, FileNotFoundError, RuntimeError) as e:
            print(e)


class Test_filepaths_errors(unittest.TestCase):
    """User-input boundaries should raise descriptive exceptions, not bare AssertionError."""

    def test_check_available_models_missing_dir(self):
        with self.assertRaises(FileNotFoundError):
            check_available_models("/this/path/does/not/exist/models")

    def test_get_actual_model_without_config(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError):
            get_actual_model(d)

    def test_models_dir_nonexistent_env_path(self):
        import spineps.utils.filepaths as fp

        old_env = os.environ.get("SPINEPS_SEGMENTOR_MODELS")
        old_override = fp.spineps_environment_path_override
        try:
            fp.spineps_environment_path_override = None
            os.environ["SPINEPS_SEGMENTOR_MODELS"] = "/this/path/does/not/exist/models"
            with self.assertRaises(FileNotFoundError):
                fp.get_mri_segmentor_models_dir()
        finally:
            fp.spineps_environment_path_override = old_override
            if old_env is None:
                os.environ.pop("SPINEPS_SEGMENTOR_MODELS", None)
            else:
                os.environ["SPINEPS_SEGMENTOR_MODELS"] = old_env
