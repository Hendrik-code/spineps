# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest

from TPTBox import No_Logger
from TPTBox.tests.test_utils import get_test_mri

from spineps.phase_post import phase_postprocess_combined
from spineps.seg_pipeline import predict_centroids_from_both

logger = No_Logger()


class Test_Post_Processing(unittest.TestCase):
    def test_phase_postprocess(self):
        mri, subreg, vert, _label = get_test_mri()
        print(vert.unique())
        subreg_cleaned, vert_cleaned = phase_postprocess_combined(mri, subreg, vert, model_labeling=None, debug_data={})
        self.assertTrue(subreg_cleaned.assert_affine(other=vert_cleaned))

        vert_labels = vert_cleaned.unique()
        print(vert_labels)
        for i in [1, 2, 3, 101, 102, 201, 202, 203]:
            self.assertTrue(i in vert_labels)
        self.assertEqual(len(vert_labels), 8)

    def test_calc_centroids(self):
        _mri, subreg, vert, _label = get_test_mri()

        poi = predict_centroids_from_both(vert, subreg, models=[], parameter={"TEST": "TEST"})
        self.assertTrue(poi is not None)
