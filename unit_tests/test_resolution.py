# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
"""Tests for the resolution-aware mm<->voxel threshold helpers."""

from __future__ import annotations

import unittest

from spineps.utils.resolution import (
    REFERENCE_VOXEL_VOLUME_MM3,
    REFERENCE_ZOOM,
    isotropic_area_to_voxels,
    mm3_to_voxels,
    mm_to_voxels,
    mm_to_voxels_axis,
)

CT_ZOOM = (0.8, 0.8, 0.8)
COARSE_ZOOM = (1.5, 1.5, 1.5)


class Test_Resolution_Helpers(unittest.TestCase):
    def test_reference_volume_roundtrips(self):
        # A threshold defined as N voxels at the reference resolution must convert back to N voxels there.
        for n in (1, 30, 40, 100):
            mm3 = n * REFERENCE_VOXEL_VOLUME_MM3
            self.assertEqual(mm3_to_voxels(mm3, REFERENCE_ZOOM), n)

    def test_reference_distance_roundtrips(self):
        for n in (2, 4, 5, 25):
            mm = n * min(REFERENCE_ZOOM)
            self.assertEqual(mm_to_voxels(mm, REFERENCE_ZOOM), n)

    def test_reference_axis_roundtrips(self):
        # Distance along the inferior axis (1.65 mm) reproduces the voxel count there.
        for n in (5, 10, 64):
            mm = n * REFERENCE_ZOOM[1]
            self.assertAlmostEqual(mm_to_voxels_axis(mm, REFERENCE_ZOOM, 1), n)

    def test_reference_area_roundtrips(self):
        for n in (10, 20, 50):
            mm2 = n * REFERENCE_VOXEL_VOLUME_MM3 ** (2.0 / 3.0)
            self.assertEqual(isotropic_area_to_voxels(mm2, REFERENCE_ZOOM), n)

    def test_finer_resolution_needs_more_voxels(self):
        # The same physical volume spans more voxels at a finer (smaller) spacing.
        mm3 = 30 * REFERENCE_VOXEL_VOLUME_MM3
        at_ct = mm3_to_voxels(mm3, CT_ZOOM)  # 0.8 mm iso -> finer than 1.65 axis
        at_coarse = mm3_to_voxels(mm3, COARSE_ZOOM)  # 1.5 mm iso -> coarser
        self.assertGreater(at_ct, 30)
        self.assertLess(at_coarse, 30)

    def test_ct_volume_value(self):
        # 30 voxels at the reference is ~27.84 mm^3 -> 54 voxels at 0.8 mm iso (0.512 mm^3/voxel).
        mm3 = 30 * REFERENCE_VOXEL_VOLUME_MM3
        self.assertEqual(mm3_to_voxels(mm3, CT_ZOOM), 54)

    def test_minimum_floor(self):
        self.assertEqual(mm3_to_voxels(0.0, REFERENCE_ZOOM, minimum=1), 1)
        self.assertEqual(mm3_to_voxels(0.0, REFERENCE_ZOOM, minimum=5), 5)
        self.assertEqual(mm_to_voxels(0.0, REFERENCE_ZOOM, minimum=0), 0)
        self.assertEqual(isotropic_area_to_voxels(0.0, REFERENCE_ZOOM, minimum=1), 1)

    def test_return_types_are_int(self):
        self.assertIsInstance(mm3_to_voxels(50.0, CT_ZOOM), int)
        self.assertIsInstance(mm_to_voxels(5.0, CT_ZOOM), int)
        self.assertIsInstance(isotropic_area_to_voxels(20.0, CT_ZOOM), int)


if __name__ == "__main__":
    unittest.main()
