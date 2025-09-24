# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
from TPTBox.tests.test_utils import get_test_mri

from spineps.architectures.read_labels import Objectives, Target, VertExact, VertRegion, VertRel
from spineps.phase_labeling import VertLabelingClassifier, perform_labeling_step
from spineps.utils.find_min_cost_path import find_most_probably_sequence


class VertLabelingClassifierDummy(VertLabelingClassifier):
    def __init__(self):
        pass


class Test_PathLabeling(unittest.TestCase):
    def test_search_path_simple(self):
        cost = np.array(
            [
                # colum lables
                # rows predictions
                [0, 10, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=int,
        )
        rel_cost = np.array(
            [
                # nothing, last0, first1, last2
                [0, 0, 0, 0],
                [0, 10, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 11, 0],
            ],
            dtype=int,
        )
        rel_cost = -rel_cost
        fcost, fpath, _min_costs_path = find_most_probably_sequence(
            cost,
            region_rel_cost=rel_cost,
            regions=[0, 3],
            softmax_temp=0,
            softmax_cost=False,
        )
        fcost_avg = fcost / len(fpath)
        print()
        print("Path cost", round(fcost, 3))
        print("Path", fpath)
        self.assertEqual(round(fcost_avg, 3), -5.0)
        self.assertEqual(fpath, [1, 2, 3, 4])

    def test_search_path_relativeonly(self):
        self.skipTest("Notimplemented")

    def test_search_path_complex(self):
        self.skipTest("Notimplemented")

    def test_objective(self):
        objectives = Objectives(
            [
                Target.FULLYVISIBLE,
                Target.REGION,
                Target.VERTREL,
                Target.VERT,
            ],
            as_group=True,
        )

        entry_dict = {
            "vert_exact": VertExact.L1,
            "vert_region": VertRegion.LWS,
            "vert_rel": VertRel.FIRST_LWK,
            "vert_cut": True,
        }

        label = objectives(entry_dict)
        print(label)
        self.assertEqual(label["FULLYVISIBLE"], [1, 0])
        self.assertEqual(label["REGION"], [0, 0, 1])
        self.assertEqual(label["VERTREL"], [0, 0, 0, 0, 1, 0])

    def test_labeling_easy(self):
        self.skipTest("Notimplemented")
        # mri, subreg, vert, label = get_test_mri()
        # model = VertLabelingClassifierDummy()
        # l = vert.unique()
        # model.run_all_seg_instances = MagicMock(return_value=l)
        # perform_labeling_step(model, mri, vert)
