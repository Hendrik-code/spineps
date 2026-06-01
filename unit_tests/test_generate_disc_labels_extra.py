# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from spineps.utils.generate_disc_labels import (
    DISCS_MAP,
    closest_point_seg_to_line,
    default_name_discs,
    extract_centroids_3d,
    project_point_on_line,
)


class Test_ProjectPointOnLine(unittest.TestCase):
    def test_point_on_axis_aligned_line(self):
        # Line along the x-axis; an off-line point projects to the nearest vertex.
        line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        point = np.array([1.4, 5.0, 0.0])
        closest, dist = project_point_on_line(point, line)
        # Nearest vertex is x=1 (1.4 rounds toward 1), distance^2 = 0.4^2 + 5.0^2 = 25.16
        npt.assert_allclose(closest, np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(dist, 25.16)

    def test_point_exactly_on_vertex(self):
        line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        point = np.array([2.0, 0.0, 0.0])
        closest, dist = project_point_on_line(point, line)
        npt.assert_allclose(closest, np.array([2.0, 0.0, 0.0]))
        self.assertAlmostEqual(dist, 0.0)

    def test_snaps_to_nearest_vertex_not_interpolated(self):
        # The function returns the closest *vertex*, it does not interpolate along the segment.
        line = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]])
        point = np.array([0.0, 1.6, 0.0])
        closest, dist = project_point_on_line(point, line)
        # 1.6 is closer to vertex 2 than vertex 1 -> [0, 2, 0], distance^2 = 0.4^2 = 0.16
        npt.assert_allclose(closest, np.array([0.0, 2.0, 0.0]))
        self.assertAlmostEqual(dist, 0.16)

    def test_returns_squared_distance(self):
        # Verify the returned distance is the squared euclidean distance (not the root).
        line = np.array([[0.0, 0.0, 0.0]])
        point = np.array([3.0, 4.0, 0.0])
        closest, dist = project_point_on_line(point, line)
        npt.assert_allclose(closest, np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(dist, 25.0)  # 3^2 + 4^2, not 5


class Test_ExtractCentroids3d(unittest.TestCase):
    def test_two_components_sorted_by_vertical_axis(self):
        arr = np.zeros((10, 10, 10), dtype=int)
        arr[1:3, 1:3, 1:3] = 5  # lower along axis 1 (S-I axis in RSP)
        arr[1:3, 6:8, 1:3] = 7  # higher along axis 1
        centroids, _bounding_boxes = extract_centroids_3d(arr)

        self.assertEqual(len(centroids), 2)
        # Centroid of a 2x2x2 block starting at (1,1,1) is (1,1,1) after int truncation; same for (1,6,1).
        npt.assert_array_equal(centroids, np.array([[1, 1, 1], [1, 6, 1]]))
        # Sorted ascending along axis 1.
        self.assertTrue(np.all(np.diff(centroids[:, 1]) >= 0))
        # Integer dtype is guaranteed by the implementation.
        self.assertTrue(np.issubdtype(centroids.dtype, np.integer))

    def test_background_component_removed(self):
        arr = np.zeros((6, 6, 6), dtype=int)
        arr[2:4, 2:4, 2:4] = 3
        centroids, bounding_boxes = extract_centroids_3d(arr)
        # Only one foreground component, background (label 0) must be dropped.
        self.assertEqual(len(centroids), 1)
        self.assertEqual(len(bounding_boxes), 1)
        npt.assert_array_equal(centroids[0], np.array([2, 2, 2]))

    def test_sorting_independent_of_insertion_order(self):
        arr = np.zeros((12, 12, 4), dtype=int)
        arr[0:2, 8:10, 0:2] = 1  # high axis 1
        arr[0:2, 0:2, 0:2] = 2  # low axis 1
        arr[0:2, 4:6, 0:2] = 3  # mid axis 1
        centroids, _ = extract_centroids_3d(arr)
        # Regardless of which label was written first, output is sorted by axis-1 coord.
        npt.assert_array_equal(centroids[:, 1], np.array([0, 4, 8]))

    def test_bounding_boxes_match_components(self):
        arr = np.zeros((8, 8, 8), dtype=int)
        arr[1:3, 1:3, 1:3] = 4
        _, bounding_boxes = extract_centroids_3d(arr)
        self.assertEqual(len(bounding_boxes), 1)
        # cc3d returns slice tuples; the bounding box must contain exactly the block we placed.
        bb = bounding_boxes[0]
        sub = arr[bb[0], bb[1], bb[2]]
        self.assertEqual(sub.shape, (2, 2, 2))
        self.assertTrue(np.all(sub == 4))


class Test_ClosestPointSegToLine(unittest.TestCase):
    def test_picks_closest_voxel_and_preserves_label(self):
        arr = np.zeros((10, 10, 10), dtype=int)
        arr[1:3, 1:3, 1:3] = 5
        arr[1:3, 6:8, 1:3] = 7
        _, bounding_boxes = extract_centroids_3d(arr)

        # Centerline far in the +z direction -> nearest voxel of each disc is the one with max z.
        centerline = np.array([[1.0, 1.0, 100.0], [1.0, 6.0, 100.0]])
        result = closest_point_seg_to_line(arr, centerline, bounding_boxes)

        self.assertEqual(result.shape, (2, 4))
        # Each row is [x, y, z, disc_value]; z must be 2 (top of the [1,3) z-range), labels preserved.
        npt.assert_array_equal(result, np.array([[1, 1, 2, 5], [1, 6, 2, 7]]))

    def test_label_value_in_last_column(self):
        arr = np.zeros((6, 6, 6), dtype=int)
        arr[2:4, 2:4, 2:4] = 9
        _, bounding_boxes = extract_centroids_3d(arr)
        centerline = np.array([[0.0, 0.0, 0.0]])
        result = closest_point_seg_to_line(arr, centerline, bounding_boxes)
        self.assertEqual(result.shape, (1, 4))
        # Closest voxel to the origin within the [2,4) block is (2,2,2), value 9.
        npt.assert_array_equal(result[0], np.array([2, 2, 2, 9]))

    def test_single_voxel_disc(self):
        arr = np.zeros((5, 5, 5), dtype=int)
        arr[3, 1, 4] = 11  # a single labelled voxel
        _, bounding_boxes = extract_centroids_3d(arr)
        centerline = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        result = closest_point_seg_to_line(arr, centerline, bounding_boxes)
        npt.assert_array_equal(result, np.array([[3, 1, 4, 11]]))


class Test_DefaultNameDiscs(unittest.TestCase):
    def test_default_suffix_with_compound_extension(self):
        out = default_name_discs("/data/sub-amu_T2w_dseg.nii.gz")
        self.assertEqual(out, Path("/data/sub-amu_T2w_dseg_label-discs_dlabel.nii.gz"))

    def test_custom_suffix(self):
        out = default_name_discs(Path("/data/foo.nii.gz"), suffix="_disc")
        self.assertEqual(out, Path("/data/foo_disc.nii.gz"))

    def test_single_extension(self):
        out = default_name_discs("/data/foo.mha")
        self.assertEqual(out, Path("/data/foo_label-discs_dlabel.mha"))

    def test_accepts_path_object_input(self):
        out = default_name_discs(Path("/data/scan.nii"))
        self.assertIsInstance(out, Path)
        self.assertEqual(out.name, "scan_label-discs_dlabel.nii")


class Test_DiscsMap(unittest.TestCase):
    def test_mapping_known_values(self):
        # Spot-check the static vertebra->disc remapping table.
        self.assertEqual(DISCS_MAP[2], 1)
        self.assertEqual(DISCS_MAP[102], 3)
        self.assertEqual(DISCS_MAP[124], 25)

    def test_mapping_is_consecutive_for_thoracolumbar_block(self):
        # Keys 102..124 map to consecutive disc values 3..25.
        block_keys = list(range(102, 125))
        values = [DISCS_MAP[k] for k in block_keys]
        self.assertEqual(values, list(range(3, 26)))

    def test_disc_value_2_is_not_directly_mapped(self):
        # Disc 2 is inserted between 1 and 3 by extract_discs_label, so it is absent from the map values.
        self.assertNotIn(2, DISCS_MAP.values())
        self.assertEqual(len(DISCS_MAP), 24)


if __name__ == "__main__":
    unittest.main()
