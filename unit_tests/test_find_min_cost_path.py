# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest

import numpy as np

from spineps.utils.find_min_cost_path import (
    argmin,
    c_to_region_idx,
    find_most_probably_sequence,
    internal_to_real_path,
    softmax_T,
)


class Test_Argmin(unittest.TestCase):
    def test_normal_case(self):
        idx, val = argmin([3, 1, 2])
        self.assertEqual(idx, 1)
        self.assertEqual(val, 1)

    def test_min_at_start_and_end(self):
        self.assertEqual(argmin([0, 5, 9]), (0, 0))
        self.assertEqual(argmin([9, 5, 0]), (2, 0))

    def test_tie_returns_first_index(self):
        # When the minimum value appears multiple times the first index wins.
        idx, val = argmin([1, 0, 0, 2])
        self.assertEqual(idx, 1)
        self.assertEqual(val, 0)

    def test_single_element(self):
        self.assertEqual(argmin([5]), (0, 5))

    def test_negative_values(self):
        idx, val = argmin([2.0, -3.5, -1.0, 4.0])
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(val, -3.5)


class Test_SoftmaxT(unittest.TestCase):
    def test_columns_sum_to_one(self):
        x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        s = softmax_T(x, 1.0)
        self.assertEqual(s.shape, (3, 2))
        col_sums = s.sum(axis=0)
        for v in col_sums:
            self.assertAlmostEqual(v, 1.0)

    def test_ordering_preserved(self):
        # softmax is monotone, so the largest score keeps the largest probability.
        x = np.array([[1.0], [2.0], [3.0]])
        s = softmax_T(x, 1.0)
        col = s[:, 0]
        self.assertTrue(np.all(np.diff(col) > 0))
        # Largest input (row 2) maps to the largest output.
        self.assertEqual(int(np.argmax(col)), 2)
        self.assertEqual(int(np.argmin(col)), 0)

    def test_all_probabilities_in_unit_interval(self):
        x = np.array([[0.0, 5.0], [5.0, 0.0], [2.5, 2.5]])
        s = softmax_T(x, 0.5)
        self.assertTrue(np.all(s > 0.0))
        self.assertTrue(np.all(s < 1.0))

    def test_lower_temperature_sharpens(self):
        # A lower temperature pushes mass toward the max -> higher peak probability.
        x = np.array([[1.0], [2.0], [4.0]])
        hot = softmax_T(x, 2.0)
        cold = softmax_T(x, 0.25)
        self.assertGreater(cold[2, 0], hot[2, 0])


class Test_CToRegionIdx(unittest.TestCase):
    def test_default_region_starts(self):
        regions = [0, 7, 19]
        # cervical region (0)
        self.assertEqual(c_to_region_idx(0, regions), 0)
        self.assertEqual(c_to_region_idx(6, regions), 0)
        # thoracic region (1) begins at 7
        self.assertEqual(c_to_region_idx(7, regions), 1)
        self.assertEqual(c_to_region_idx(18, regions), 1)
        # lumbar region (2) begins at 19
        self.assertEqual(c_to_region_idx(19, regions), 2)
        self.assertEqual(c_to_region_idx(25, regions), 2)

    def test_boundaries(self):
        regions = [0, 3, 6]
        # Just below / at / above each boundary.
        self.assertEqual(c_to_region_idx(2, regions), 0)
        self.assertEqual(c_to_region_idx(3, regions), 1)
        self.assertEqual(c_to_region_idx(5, regions), 1)
        self.assertEqual(c_to_region_idx(6, regions), 2)

    def test_class_below_first_start_returns_minus_one(self):
        # If the first region start is > 0 then classes before it resolve to -1.
        self.assertEqual(c_to_region_idx(0, [2, 5]), -1)
        self.assertEqual(c_to_region_idx(1, [2, 5]), -1)
        self.assertEqual(c_to_region_idx(2, [2, 5]), 0)


class Test_InternalToRealPath(unittest.TestCase):
    def test_sorts_by_row_and_returns_classes(self):
        p = [(2, "c"), (0, "a"), (1, "b")]
        self.assertEqual(internal_to_real_path(p), ["a", "b", "c"])

    def test_numeric_classes(self):
        p = [(3, 30), (1, 10), (0, 5), (2, 20)]
        self.assertEqual(internal_to_real_path(p), [5, 10, 20, 30])

    def test_already_sorted_is_unchanged(self):
        p = [(0, 9), (1, 8), (2, 7)]
        self.assertEqual(internal_to_real_path(p), [9, 8, 7])

    def test_single_node(self):
        self.assertEqual(internal_to_real_path([(0, 42)]), [42])


class Test_FindMostProbableSequence(unittest.TestCase):
    @staticmethod
    def _strong_diagonal(n_rows: int, n_cols: int, value: float = 10.0) -> np.ndarray:
        """Build a cost matrix whose obvious optimum is the main diagonal."""
        cost = np.zeros((n_rows, n_cols), dtype=float)
        for i in range(n_rows):
            cost[i, i] = value
        return cost

    def test_strong_diagonal_invert_cost(self):
        # With invert_cost the highest scores are preferred -> follow the diagonal.
        cost = self._strong_diagonal(4, 6, 10.0)
        fcost, fpath, min_costs_path = find_most_probably_sequence(
            cost,
            invert_cost=True,
            allow_skip_at_region=[],
        )
        self.assertEqual(fpath, [0, 1, 2, 3])
        # Each of the four chosen diagonal cells contributes -10 after inversion.
        self.assertAlmostEqual(fcost, -40.0)
        # Path covers every row exactly once.
        self.assertEqual(len(fpath), cost.shape[0])
        # The returned memo table mirrors the matrix shape.
        self.assertEqual(len(min_costs_path), cost.shape[0])
        self.assertEqual(len(min_costs_path[0]), cost.shape[1])

    def test_no_invert_prefers_low_cost(self):
        # Without inversion the solver minimises raw cost: 0 on the diagonal, 10 elsewhere.
        cost = np.full((4, 6), 10.0)
        for i in range(4):
            cost[i, i] = 0.0
        fcost, fpath, _ = find_most_probably_sequence(
            cost,
            invert_cost=False,
            allow_skip_at_region=[],
        )
        self.assertEqual(fpath, [0, 1, 2, 3])
        self.assertAlmostEqual(fcost, 0.0)

    def test_invert_symmetry(self):
        # invert_cost=True on a matrix equals invert_cost=False on its negation.
        cost = self._strong_diagonal(4, 6, 7.0)
        fcost_a, fpath_a, _ = find_most_probably_sequence(cost, invert_cost=True, allow_skip_at_region=[])
        fcost_b, fpath_b, _ = find_most_probably_sequence(-cost, invert_cost=False, allow_skip_at_region=[])
        self.assertEqual(fpath_a, fpath_b)
        self.assertAlmostEqual(fcost_a, fcost_b)

    def test_min_start_class_shifts_path(self):
        # The diagonal of high scores starts at column 2; min_start_class must allow it.
        cost = np.zeros((4, 8), dtype=float)
        for i in range(4):
            cost[i, i + 2] = 5.0
        fcost, fpath, _ = find_most_probably_sequence(
            cost,
            invert_cost=True,
            allow_skip_at_region=[],
            min_start_class=2,
        )
        self.assertEqual(fpath, [2, 3, 4, 5])
        self.assertAlmostEqual(fcost, -20.0)

    def test_first_label_respects_min_start_class(self):
        # Even when an earlier column looks attractive, the path may not start before min_start_class.
        cost = np.zeros((4, 8), dtype=float)
        cost[0, 0] = 100.0  # very attractive but forbidden as a start
        for i in range(4):
            cost[i, i + 3] = 5.0
        fcost, fpath, _ = find_most_probably_sequence(
            cost,
            invert_cost=True,
            allow_skip_at_region=[],
            min_start_class=3,
        )
        self.assertGreaterEqual(fpath[0], 3)
        self.assertEqual(len(fpath), cost.shape[0])
        self.assertTrue(np.isfinite(fcost))

    def test_list_input_is_accepted(self):
        # A plain nested list must be handled identically to an ndarray.
        cost = [
            [7, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 0, 0],
            [0, 0, 7, 0, 0, 0],
            [0, 0, 0, 7, 0, 0],
        ]
        fcost, fpath, _ = find_most_probably_sequence(cost, invert_cost=True, allow_skip_at_region=[])
        self.assertEqual(fpath, [0, 1, 2, 3])
        self.assertAlmostEqual(fcost, -28.0)

    def test_allow_multiple_at_class_enables_repeat(self):
        # Class 1 is strongly preferred for two consecutive rows; repeating it captures both.
        cost = np.zeros((4, 5), dtype=float)
        cost[0, 0] = 10.0
        cost[1, 1] = 10.0
        cost[2, 1] = 10.0  # second consecutive class-1 vertebra
        cost[3, 2] = 10.0
        fcost, fpath, _ = find_most_probably_sequence(
            cost,
            invert_cost=True,
            allow_skip_at_region=[],
            allow_multiple_at_class=[1],
            punish_multiple_sequence=0.0,
        )
        self.assertEqual(fpath, [0, 1, 1, 2])
        self.assertAlmostEqual(fcost, -40.0)

    def test_without_multiple_forces_diagonal(self):
        # The same matrix, but repeats disallowed -> must advance every step.
        cost = np.zeros((4, 5), dtype=float)
        cost[0, 0] = 10.0
        cost[1, 1] = 10.0
        cost[2, 1] = 10.0
        cost[3, 2] = 10.0
        fcost, fpath, _ = find_most_probably_sequence(
            cost,
            invert_cost=True,
            allow_skip_at_region=[],
            allow_multiple_at_class=[],
            punish_multiple_sequence=0.0,
        )
        self.assertEqual(fpath, [0, 1, 2, 3])
        # Only rows 0 and 1 hit their high-score cells.
        self.assertAlmostEqual(fcost, -20.0)

    def test_repeat_capped_when_penalty_high(self):
        # A large repeat penalty makes the diagonal cheaper than repeating class 1.
        cost = np.zeros((4, 5), dtype=float)
        cost[0, 0] = 10.0
        cost[1, 1] = 10.0
        cost[2, 1] = 10.0
        cost[3, 2] = 10.0
        _, fpath, _ = find_most_probably_sequence(
            cost,
            invert_cost=True,
            allow_skip_at_region=[],
            allow_multiple_at_class=[1],
            punish_multiple_sequence=100.0,
        )
        self.assertEqual(fpath, [0, 1, 2, 3])

    def test_allow_skip_at_class_enables_jump(self):
        # After class 0 we may skip class 1 and land on class 2.
        cost = np.zeros((4, 6), dtype=float)
        cost[0, 0] = 10.0
        cost[1, 2] = 10.0  # reached by skipping class 1
        cost[2, 3] = 10.0
        cost[3, 4] = 10.0
        fcost, fpath, _ = find_most_probably_sequence(
            cost,
            invert_cost=True,
            allow_skip_at_region=[],
            allow_skip_at_class=[0],
            punish_skip_sequence=0.0,
        )
        self.assertEqual(fpath, [0, 2, 3, 4])
        self.assertAlmostEqual(fcost, -40.0)

    def test_without_skip_no_jump(self):
        # The same matrix, but skipping disallowed -> the path shifts to a pure diagonal.
        cost = np.zeros((4, 6), dtype=float)
        cost[0, 0] = 10.0
        cost[1, 2] = 10.0
        cost[2, 3] = 10.0
        cost[3, 4] = 10.0
        fcost, fpath, _ = find_most_probably_sequence(
            cost,
            invert_cost=True,
            allow_skip_at_region=[],
            allow_skip_at_class=[],
            punish_skip_sequence=0.0,
        )
        self.assertEqual(fpath, [1, 2, 3, 4])
        self.assertAlmostEqual(fcost, -30.0)

    def test_region_rel_cost_pulls_region_start(self):
        # An all-zero cost matrix means only the region transition cost matters.
        # Region 1 starts at class index 3; making "first of region 1" attractive at
        # vertebra 2 forces vertebra 2 onto class 3 -> path [1, 2, 3, 4].
        cost = np.zeros((4, 5), dtype=float)
        rel = np.zeros((4, 4), dtype=float)  # columns: nothing, last0, first1, last2
        rel[2, 2] = -8.0  # "first of region 1" reward at vertebra 2
        fcost, fpath, _ = find_most_probably_sequence(
            cost,
            region_rel_cost=rel,
            regions=[0, 3],
            invert_cost=True,
        )
        self.assertEqual(fpath, [1, 2, 3, 4])
        self.assertAlmostEqual(fcost, -8.0)

    def test_region_rel_cost_matches_existing_simple_case(self):
        # Mirrors the documented simple scenario: a strong column plus region rewards.
        cost = np.array(
            [
                [0, 10, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        rel = -np.array(
            [
                [0, 0, 0, 0],
                [0, 10, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 11, 0],
            ],
            dtype=float,
        )
        fcost, fpath, _ = find_most_probably_sequence(
            cost,
            region_rel_cost=rel,
            regions=[0, 3],
        )
        self.assertEqual(fpath, [1, 2, 3, 4])
        # Mean cost per vertebra, as asserted in the existing path test.
        self.assertAlmostEqual(fcost / len(fpath), -5.0)

    def test_region_rel_cost_wrong_shape_raises(self):
        # region_rel_cost must have (n_regions * 2) columns for the given regions.
        cost = np.zeros((4, 5), dtype=float)
        bad_rel = np.zeros((4, 3), dtype=float)  # should be 4 columns for regions [0, 3]
        with self.assertRaises(AssertionError):
            find_most_probably_sequence(cost, region_rel_cost=bad_rel, regions=[0, 3])

    def test_min_start_class_out_of_range_raises(self):
        cost = np.zeros((4, 5), dtype=float)
        with self.assertRaises(AssertionError):
            find_most_probably_sequence(cost, min_start_class=5, allow_skip_at_region=[])

    def test_path_properties_on_random_matrix(self):
        # Robust structural properties that must hold for any valid solution.
        rng = np.random.default_rng(0)
        cost = rng.random((5, 8))
        fcost, fpath, min_costs_path = find_most_probably_sequence(
            cost,
            invert_cost=True,
            allow_skip_at_region=[],
        )
        # One label per vertebra (row).
        self.assertEqual(len(fpath), cost.shape[0])
        # Cost is finite.
        self.assertTrue(np.isfinite(fcost))
        # Labels are valid column indices and weakly increasing (monotone path).
        for c in fpath:
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, cost.shape[1])
        self.assertTrue(all(fpath[i] <= fpath[i + 1] for i in range(len(fpath) - 1)))
        # Memo table shape mirrors the cost matrix.
        self.assertEqual(len(min_costs_path), cost.shape[0])
        self.assertEqual(len(min_costs_path[0]), cost.shape[1])


if __name__ == "__main__":
    unittest.main()
