# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

from spineps.models import Segmentation_Model, get_segmentation_model
from spineps.utils.filepaths import (
    filepath_model,
    get_mri_segmentor_models_dir,
    spineps_environment_path_backup,
    spineps_environment_path_override,
)


class Test_filepaths(unittest.TestCase):
    def test_load_model_from_path(self):
        ms_p = "/DATA/NAS/ongoing_projects/hendrik/nako-segmentation/nnUNet/nnUNet_results/Dataset122_nako_wotest_spiderarcus/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom"
        mv_p = "/DATA/NAS/ongoing_projects/hendrik/nako-segmentation/nnUNet/unet3d_result/nakospider_highres_shiftposi"
        model_dir = "/DATA/NAS/ongoing_projects/hendrik/nako-segmentation/nnUNet/"
        ms = get_segmentation_model(in_config=filepath_model(ms_p, model_dir=model_dir))
        self.assertTrue(isinstance(ms, Segmentation_Model))
        mv = get_segmentation_model(in_config=filepath_model(mv_p, model_dir=model_dir))
        self.assertTrue(isinstance(mv, Segmentation_Model))
        self.assertTrue(True)

    def test_env_path(self):
        # no override
        p = get_mri_segmentor_models_dir()
        if spineps_environment_path_override is not None:
            self.assertEqual(p, spineps_environment_path_override)
        else:
            self.assertEqual(str(p) + "/", os.environ.get("SPINEPS_SEGMENTOR_MODELS"))
