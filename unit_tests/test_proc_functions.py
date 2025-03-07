# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import os
import unittest
from pathlib import Path

from TPTBox import Log_Type, No_Logger
from TPTBox.tests.test_utils import get_test_mri

import spineps
from spineps.get_models import Segmentation_Model, get_actual_model
from spineps.utils.compat import zip_strict
from spineps.utils.proc_functions import clean_cc_artifacts, connected_components_3d, n4_bias

logger = No_Logger()


class Test_proc_functions(unittest.TestCase):
    def test_n4_bias(self):
        mri, subreg, vert, label = get_test_mri()
        mri.normalize_to_range_()
        mri_min = mri.min()
        mri_max = mri.max()
        self.assertEqual(mri_min, 0)
        self.assertEqual(mri_max, 387)
        mri_n4biased, mask = n4_bias(mri)
        mri_min = mri_n4biased.min()
        mri_max = mri_n4biased.max()
        self.assertEqual(mri_min, 0)
        self.assertEqual(mri_max, 252)

    def test_clean_artifacts(self):
        mri, subreg, vert, label = get_test_mri()
        l3 = vert.extract_label(label)
        l3 = subreg.apply_mask(l3)
        l3_volumes = l3.volumes()
        l3_cleaned = clean_cc_artifacts(l3, logger=logger, labels=[41, 42, 43, 44, 45, 46, 47, 48, 49])
        l3_cleaned = l3.set_array(l3_cleaned)
        l3_cleaned_volumes = l3_cleaned.volumes()
        for a, b in zip_strict(l3_volumes.values(), l3_cleaned_volumes.values()):
            self.assertEqual(a, b)
