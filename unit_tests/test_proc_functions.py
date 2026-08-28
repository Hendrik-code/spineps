# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
from TPTBox import Log_Type, No_Logger
from TPTBox.tests.test_utils import get_test_mri

import spineps
from spineps.get_models import SegmentationModel, get_actual_model
from spineps.utils.compat import zip_strict
from spineps.utils.proc_functions import clean_cc_artifacts, connected_components_3d, n4_bias

logger = No_Logger()


class Test_proc_functions(unittest.TestCase):
    def test_n4_bias(self):
        mri, _subreg, _vert, _label = get_test_mri()
        mri.normalize_to_range_()
        mri_min = mri.min()
        mri_max = mri.max()
        self.assertEqual(mri_min, 0)
        self.assertEqual(mri_max, 387)
        mri_n4biased, _mask = n4_bias(mri)
        mri_min = mri_n4biased.min()
        mri_max = mri_n4biased.max()
        self.assertEqual(mri_min, 0)
        self.assertEqual(mri_max, 252)

    def test_clean_artifacts(self):
        _mri, subreg, vert, label = get_test_mri()
        l3 = vert.extract_label(label)
        l3 = subreg.apply_mask(l3)
        l3_volumes = l3.volumes()
        l3_cleaned = clean_cc_artifacts(l3, logger=logger, labels=[41, 42, 43, 44, 45, 46, 47, 48, 49])
        l3_cleaned = l3.set_array(l3_cleaned)
        l3_cleaned_volumes = l3_cleaned.volumes()
        for a, b in zip_strict(l3_volumes.values(), l3_cleaned_volumes.values()):
            self.assertEqual(a, b)

    def test_clean_artifacts_no_zeros(self):
        _mri, _subreg, vert, label = get_test_mri()
        l3 = vert.extract_label(label)
        l3[l3 == 0] = 1
        l3_volumes = l3.volumes()
        l3_cleaned = clean_cc_artifacts(l3, logger=logger, labels=[41, 42, 43, 44, 45, 46, 47, 48, 49])
        l3_cleaned = l3.set_array(l3_cleaned)
        l3_cleaned_volumes = l3_cleaned.volumes()
        for a, b in zip_strict(l3_volumes.values(), l3_cleaned_volumes.values()):
            self.assertEqual(a, b)

    def test_clean_artifacts_zeros(self):
        _mri, subreg, vert, label = get_test_mri()
        l3 = vert.extract_label(label)
        l3 = subreg.apply_mask(l3) * 0
        l3_volumes = l3.volumes()

        for ignore_missing_labels in [False, True]:
            if ignore_missing_labels:
                l3_cleaned = clean_cc_artifacts(
                    l3, logger=logger, labels=[41, 42, 43, 44, 45, 46, 47, 48, 49], ignore_missing_labels=ignore_missing_labels
                )
                l3_cleaned = l3.set_array(l3_cleaned)
                l3_cleaned_volumes = l3_cleaned.volumes()
                for a, b in zip_strict(l3_volumes.values(), l3_cleaned_volumes.values()):
                    self.assertEqual(a, b)
            else:
                with self.assertRaises(AssertionError):
                    l3_cleaned = clean_cc_artifacts(
                        l3, logger=logger, labels=[41, 42, 43, 44, 45, 46, 47, 48, 49], ignore_missing_labels=ignore_missing_labels
                    )


class Test_Clean_CC_Artifacts_Branches(unittest.TestCase):
    """Pin both cleaning branches: a small component next to a big one is relabeled, an isolated one deleted.

    ``clean_cc_artifacts`` now works inside each component's padded bounding box instead of over the whole
    volume, so the neighbourhood dilation and the majority vote need to keep giving the same answers.
    """

    @staticmethod
    def _mask() -> np.ndarray:
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[2:12, 2:12, 2:12] = 1  # big label-1 body
        arr[12:14, 5:7, 5:7] = 2  # small label-2 speck glued to it -> majority vote says 1
        arr[17:19, 17:19, 17:19] = 2  # isolated label-2 speck -> deleted
        return arr

    def test_relabel_and_delete(self):
        arr = self._mask()
        out = clean_cc_artifacts(arr, logger=logger, labels=[2], cc_size_threshold=100, only_delete=False, verbose=False)
        self.assertTrue(np.all(out[12:14, 5:7, 5:7] == 1), "the attached speck should inherit its neighbour's label")
        self.assertTrue(np.all(out[17:19, 17:19, 17:19] == 0), "the isolated speck should be deleted")
        self.assertTrue(np.all(out[2:12, 2:12, 2:12] == 1), "the big component must be untouched")

    def test_only_delete_removes_both(self):
        arr = self._mask()
        out = clean_cc_artifacts(arr, logger=logger, labels=[2], cc_size_threshold=100, only_delete=True, verbose=False)
        self.assertEqual(int((out == 2).sum()), 0)
        self.assertTrue(np.all(out[2:12, 2:12, 2:12] == 1))

    def test_component_touching_the_volume_edge(self):
        """The bounding-box crop must clamp at the array bounds exactly like the full-volume code did."""
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[2:12, 2:12, 2:12] = 1
        arr[0:2, 0:2, 0:2] = 2  # in the corner, so the padded bbox is clipped
        out = clean_cc_artifacts(arr, logger=logger, labels=[2], cc_size_threshold=100, only_delete=False, verbose=False)
        self.assertEqual(int((out == 2).sum()), 0)
