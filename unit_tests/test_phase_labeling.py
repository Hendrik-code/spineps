# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest

import numpy as np

from spineps.architectures.read_labels import VertExact, vert_group_idx_to_exact_idx_dict
from spineps.phase_labeling import (
    CERV,
    LUMB,
    T13_LABEL,
    THOR,
    VERT_CLASSES,
    fpath_post_processing,
    is_valid_vertebra_sequence,
    prepare_region,
    prepare_vert,
    prepare_vertgrp,
    prepare_vertrel,
    prepare_vertrel_columns,
    prepare_vertt13_columns,
    prepare_visible,
    region_to_vert,
)


class Test_region_to_vert(unittest.TestCase):
    def test_shape_and_broadcast(self):
        out = region_to_vert(np.array([0.2, 0.5, 0.3]))
        self.assertEqual(out.shape, (VERT_CLASSES,))
        # cervical slice all == region[0]
        self.assertTrue(np.allclose(out[CERV], 0.2))
        # thoracic slice all == region[1]
        self.assertTrue(np.allclose(out[THOR], 0.5))
        # lumbar slice all == region[2]
        self.assertTrue(np.allclose(out[LUMB], 0.3))

    def test_slice_lengths(self):
        out = region_to_vert(np.array([1.0, 2.0, 3.0]))
        # CERV = slice(None, 7) -> 7 classes
        self.assertEqual(out[CERV].shape[0], 7)
        # THOR = slice(7, 19) -> 12 classes
        self.assertEqual(out[THOR].shape[0], 12)
        # LUMB = slice(19, None) -> 5 classes
        self.assertEqual(out[LUMB].shape[0], 5)
        # every entry filled (no zeros remain for nonzero regions)
        self.assertEqual(np.count_nonzero(out), VERT_CLASSES)

    def test_zero_region_leaves_zeros(self):
        out = region_to_vert(np.array([0.0, 1.0, 0.0]))
        self.assertTrue(np.allclose(out[CERV], 0.0))
        self.assertTrue(np.allclose(out[LUMB], 0.0))
        self.assertTrue(np.allclose(out[THOR], 1.0))


class Test_prepare_vert(unittest.TestCase):
    def test_no_smoothing_normalizes(self):
        v = np.zeros(VERT_CLASSES)
        v[3] = 2.0
        v[10] = 1.0
        out = prepare_vert(v, gaussian_sigma=0.0)
        self.assertEqual(out.shape, (VERT_CLASSES,))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
        # with no smoothing the ratio of the two peaks is preserved (2:1)
        self.assertAlmostEqual(float(out[3]), 2.0 / 3.0, places=5)
        self.assertAlmostEqual(float(out[10]), 1.0 / 3.0, places=5)

    def test_does_not_mutate_input(self):
        v = np.zeros(VERT_CLASSES)
        v[3] = 2.0
        _ = prepare_vert(v, gaussian_sigma=0.0)
        self.assertEqual(float(v[3]), 2.0)

    def test_smoothing_regionwise_sums_to_one(self):
        v = np.zeros(VERT_CLASSES)
        v[3] = 1.0
        out = prepare_vert(v, gaussian_sigma=0.85, gaussian_radius=2, gaussian_regionwise=True)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
        # smoothing spreads the single peak across neighbouring cervical classes
        self.assertGreater(np.count_nonzero(out > 1e-6), 1)

    def test_smoothing_regionwise_does_not_leak_across_regions(self):
        # peak at last cervical class (index 6); region-wise smoothing must not
        # leak probability into the thoracic region (index >= 7).
        v = np.zeros(VERT_CLASSES)
        v[6] = 1.0
        out = prepare_vert(v, gaussian_sigma=0.85, gaussian_radius=2, gaussian_regionwise=True)
        self.assertTrue(np.allclose(out[THOR], 0.0))
        self.assertTrue(np.allclose(out[LUMB], 0.0))

    def test_smoothing_global_can_leak_across_regions(self):
        # global smoothing of the same peak DOES leak into the thoracic region.
        v = np.zeros(VERT_CLASSES)
        v[6] = 1.0
        out = prepare_vert(v, gaussian_sigma=0.85, gaussian_radius=2, gaussian_regionwise=False)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
        self.assertGreater(float(out[7]), 0.0)


class Test_prepare_vertgrp(unittest.TestCase):
    def test_group_expanded_to_member_classes(self):
        # group index 0 (C12) maps to exact classes [0, 1]
        g = np.zeros(len(vert_group_idx_to_exact_idx_dict))
        g[0] = 1.0
        out = prepare_vertgrp(g, gaussian_sigma=0.0)
        self.assertEqual(out.shape, (VERT_CLASSES,))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
        nonzero = set(np.nonzero(out)[0].tolist())
        self.assertEqual(nonzero, set(vert_group_idx_to_exact_idx_dict[0]))
        # the group value is copied onto each member class, then the whole vector
        # is normalized; with one group of two members each ends up at 0.5.
        self.assertAlmostEqual(float(out[0]), 0.5, places=5)
        self.assertAlmostEqual(float(out[1]), 0.5, places=5)

    def test_multiple_groups_normalize(self):
        g = np.zeros(len(vert_group_idx_to_exact_idx_dict))
        g[0] = 1.0  # group C12 -> two member classes [0, 1], each gets value 1.0
        g[11] = 3.0  # group L56 -> single exact class [23], gets value 3.0
        out = prepare_vertgrp(g, gaussian_sigma=0.0)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
        # raw assigned mass before normalization is 1.0 + 1.0 + 3.0 = 5.0
        # group 11's single member carries 3/5 of the mass
        self.assertAlmostEqual(float(out[vert_group_idx_to_exact_idx_dict[11][0]]), 3.0 / 5.0, places=5)
        # each member of group 0 carries 1/5 of the mass
        self.assertAlmostEqual(float(out[0]), 1.0 / 5.0, places=5)
        self.assertAlmostEqual(float(out[1]), 1.0 / 5.0, places=5)

    def test_smoothing_global_branch_sums_to_one(self):
        g = np.zeros(len(vert_group_idx_to_exact_idx_dict))
        g[5] = 1.0
        out = prepare_vertgrp(g, gaussian_sigma=0.85, gaussian_regionwise=False)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)


class Test_prepare_region(unittest.TestCase):
    def test_no_smoothing_normalizes(self):
        out = prepare_region(np.array([0.2, 0.5, 0.3]), gaussian_sigma=0.0)
        self.assertEqual(out.shape, (VERT_CLASSES,))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)

    def test_smoothing_normalizes(self):
        out = prepare_region(np.array([0.2, 0.5, 0.3]), gaussian_sigma=0.75, gaussian_radius=1)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)

    def test_all_zero_input_stays_zero(self):
        # guard `np.sum(...) > 0` skips smoothing and division leaves zeros
        out = prepare_region(np.array([0.0, 0.0, 0.0]), gaussian_sigma=0.75)
        self.assertEqual(float(out.sum()), 0.0)


class Test_prepare_vertrel(unittest.TestCase):
    def test_no_smoothing_returns_copy_unchanged(self):
        vr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = prepare_vertrel(vr, gaussian_sigma=0.0)
        self.assertTrue(np.allclose(out, vr))
        # returned object is a copy, not the same array
        self.assertIsNot(out, vr)

    def test_smoothing_preserves_shape(self):
        vr = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        out = prepare_vertrel(vr, gaussian_sigma=0.75, gaussian_radius=1)
        self.assertEqual(out.shape, vr.shape)
        # smoothing spreads the single peak to neighbours
        self.assertGreater(np.count_nonzero(out > 1e-6), 1)


class Test_prepare_vertrel_columns(unittest.TestCase):
    def test_column_zero_untouched(self):
        m = np.zeros((3, 6))
        m[:, 0] = [5.0, 5.0, 5.0]
        out = prepare_vertrel_columns(m.copy(), gaussian_sigma=0.0)
        # the first column (NOTHING) is skipped by the loop -> unchanged
        self.assertTrue(np.allclose(out[:, 0], [5.0, 5.0, 5.0]))

    def test_column_sum_greater_one_normalizes_to_one(self):
        m = np.zeros((3, 6))
        m[:, 1] = [0.5, 0.5, 0.5]  # sum 1.5 > 1 -> divide by sum
        out = prepare_vertrel_columns(m.copy(), gaussian_sigma=0.0)
        self.assertAlmostEqual(float(out[:, 1].sum()), 1.0, places=5)

    def test_column_sum_less_one_divides_by_one_plus_sum(self):
        m = np.zeros((3, 6))
        m[:, 2] = [0.1, 0.1, 0.1]  # sum 0.3 < 1 -> divide by (1 + sum)
        out = prepare_vertrel_columns(m.copy(), gaussian_sigma=0.0)
        expected = 0.1 / (1.0 + 0.3)
        self.assertTrue(np.allclose(out[:, 2], expected, atol=1e-6))
        # the resulting column sum is below 1
        self.assertLess(float(out[:, 2].sum()), 1.0)

    def test_returns_same_array_object(self):
        m = np.zeros((3, 6))
        m[:, 1] = [0.5, 0.5, 0.5]
        out = prepare_vertrel_columns(m, gaussian_sigma=0.0)
        self.assertIs(out, m)


class Test_prepare_vertt13_columns(unittest.TestCase):
    def test_column_zero_untouched_rest_normalized(self):
        m = np.array([[0.9, 0.1], [0.8, 0.4], [0.7, 0.5]], dtype=float)
        out = prepare_vertt13_columns(m.copy())
        # first column untouched
        self.assertTrue(np.allclose(out[:, 0], [0.9, 0.8, 0.7]))
        # second column normalized to sum 1
        self.assertAlmostEqual(float(out[:, 1].sum()), 1.0, places=5)
        # relative proportions preserved within the normalized column
        self.assertAlmostEqual(float(out[0, 1]), 0.1 / 1.0, places=5)

    def test_returns_same_array_object(self):
        m = np.array([[0.9, 0.1], [0.8, 0.4]], dtype=float)
        out = prepare_vertt13_columns(m)
        self.assertIs(out, m)


class Test_prepare_visible(unittest.TestCase):
    def _make_preds(self, visible_pairs):
        return {idx: {"soft": {"FULLYVISIBLE": pair, "VERT": [0.0] * 24}} for idx, pair in enumerate(visible_pairs)}

    def test_fullyvisible_present_weight_one(self):
        preds = self._make_preds([[0.1, 0.9], [0.2, 0.8]])
        # with visible_w=1 and no smoothing, weight == FULLYVISIBLE[1]
        out = prepare_visible(preds, visible_w=1.0, gaussian_sigma=0.0)
        self.assertEqual(out.shape, (2,))
        self.assertTrue(np.allclose(out, [0.9, 0.8]))
        # weights are clipped into [0, 1]
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))

    def test_fullyvisible_absent_returns_ones(self):
        preds = {0: {"soft": {"VERT": [0.0] * 24}}, 1: {"soft": {"VERT": [0.0] * 24}}}
        out = prepare_visible(preds, visible_w=1.0, gaussian_sigma=0.0)
        self.assertTrue(np.allclose(out, [1.0, 1.0]))

    def test_visible_weight_zero_disables_downweighting(self):
        preds = self._make_preds([[0.6, 0.4], [0.7, 0.3]])
        # visible_w=0 -> 1 - (1 - x) * 0 = 1 for every instance
        out = prepare_visible(preds, visible_w=0.0, gaussian_sigma=0.0)
        self.assertTrue(np.allclose(out, [1.0, 1.0]))

    def test_partial_weight_between(self):
        preds = self._make_preds([[0.5, 0.5]])
        # weight = 1 - (1 - 0.5) * 0.5 = 0.75
        out = prepare_visible(preds, visible_w=0.5, gaussian_sigma=0.0)
        self.assertAlmostEqual(float(out[0]), 0.75, places=3)


class Test_fpath_post_processing(unittest.TestCase):
    def test_plus_one_offset(self):
        # plain class indices simply shift by +1 (0-based -> 1-based)
        self.assertEqual(fpath_post_processing([0, 1, 2]), [1, 2, 3])

    def test_returns_new_list(self):
        src = [0, 1, 2]
        out = fpath_post_processing(src)
        self.assertEqual(src, [0, 1, 2])
        self.assertIsNot(out, src)

    def test_double_t12_second_becomes_t13(self):
        # [T11, T12, T12] -> the second T12 turns into the special T13 label (28),
        # the rest is offset by +1. T13_LABEL is left untouched by the offset.
        out = fpath_post_processing([17, VertExact.T12.value, VertExact.T12.value])
        self.assertEqual(out, [18, 19, T13_LABEL])
        self.assertIn(T13_LABEL, out)

    def test_double_t12_at_start(self):
        # T12 at index 0 then T12 -> next index becomes T13
        out = fpath_post_processing([VertExact.T12.value, VertExact.T12.value, 19])
        self.assertEqual(out, [19, T13_LABEL, 20])

    def test_trailing_double_l5_becomes_l5_l6(self):
        # trailing [..., L5, L5] -> the last L5 is bumped by +1 (to L6 slot 24),
        # then everything offsets by +1.
        out = fpath_post_processing([21, 22, VertExact.L5.value, VertExact.L5.value])
        self.assertEqual(out, [22, 23, 24, 25])

    def test_single_trailing_l5_just_offsets(self):
        out = fpath_post_processing([22, VertExact.L5.value])
        self.assertEqual(out, [23, 24])


class Test_is_valid_vertebra_sequence(unittest.TestCase):
    def test_valid_consecutive_ints(self):
        self.assertTrue(is_valid_vertebra_sequence([1, 2, 3, 4]))

    def test_invalid_with_gap(self):
        self.assertFalse(is_valid_vertebra_sequence([1, 2, 5]))

    def test_valid_t13_to_l1_jump(self):
        # special allowed jump: T13 (28) -> L1 (20)
        self.assertTrue(is_valid_vertebra_sequence([28, 20]))

    def test_valid_t12_to_l1_jump(self):
        # special allowed jump: T12 (18) -> L1 (20)
        self.assertTrue(is_valid_vertebra_sequence([18, 20]))

    def test_invalid_backwards(self):
        self.assertFalse(is_valid_vertebra_sequence([5, 4, 3]))

    def test_vertexact_input_valid(self):
        self.assertTrue(is_valid_vertebra_sequence([VertExact.C1, VertExact.C2, VertExact.C3]))

    def test_vertexact_input_invalid_skip(self):
        self.assertFalse(is_valid_vertebra_sequence([VertExact.L1, VertExact.L3]))


if __name__ == "__main__":
    unittest.main()
