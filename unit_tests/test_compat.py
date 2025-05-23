# Call "python -m unittest" on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest

from spineps.utils.compat import zip_strict


class TestZipStrict(unittest.TestCase):
    def test_equal_length_lists(self):
        a = [1, 2, 3]
        b = ["a", "b", "c"]
        expected = [(1, "a"), (2, "b"), (3, "c")]
        result = list(zip_strict(a, b))
        self.assertEqual(result, expected)

    def test_unequal_length_lists(self):
        a = [1, 2, 3]
        b = ["a", "b"]
        with self.assertRaises(ValueError) as context:
            list(zip_strict(a, b))
        self.assertIn("Length mismatch", str(context.exception))

    def test_empty_iterables(self):
        a = []
        b = []
        result = list(zip_strict(a, b))
        self.assertEqual(result, [])

    def test_multiple_iterables(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        c = [7, 8, 9]
        expected = [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
        result = list(zip_strict(a, b, c))
        self.assertEqual(result, expected)

    def test_generator_iterables(self):
        a = (x for x in range(3))
        b = (chr(97 + x) for x in range(3))
        expected = [(0, "a"), (1, "b"), (2, "c")]
        result = list(zip_strict(a, b))
        self.assertEqual(result, expected)
