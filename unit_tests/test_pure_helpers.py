# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest

from spineps.phase_post import find_nearest_lower
from spineps.phase_semantic import overlap_slice
from spineps.seg_enums import Acquisition, Modality, ModelType
from spineps.seg_utils import add_ignore_text


class Test_Modality_format_keys(unittest.TestCase):
    def test_single_modalities(self):
        self.assertEqual(Modality.format_keys(Modality.CT), ["CT", "ct"])
        self.assertEqual(Modality.format_keys(Modality.SEG), ["msk", "seg"])
        self.assertEqual(Modality.format_keys(Modality.T1w), ["T1w", "t1", "T1", "T1c"])
        self.assertEqual(
            Modality.format_keys(Modality.T2w),
            ["T2w", "dixon", "mr", "t2", "T2", "T2c"],
        )
        self.assertEqual(
            Modality.format_keys(Modality.Vibe),
            ["t1dixon", "vibe", "mevibe", "GRE"],
        )
        self.assertEqual(Modality.format_keys(Modality.MPR), ["mpr", "MPR", "Mpr"])

    def test_list_of_modalities(self):
        # A list of modalities concatenates the per-modality keys in order.
        self.assertEqual(
            Modality.format_keys([Modality.CT, Modality.SEG]),
            ["CT", "ct", "msk", "seg"],
        )
        self.assertEqual(
            Modality.format_keys([Modality.T1w, Modality.MPR]),
            ["T1w", "t1", "T1", "T1c", "mpr", "MPR", "Mpr"],
        )

    def test_single_equals_singleton_list(self):
        # Passing a single member is equivalent to passing a one-element list.
        self.assertEqual(
            Modality.format_keys(Modality.Vibe),
            Modality.format_keys([Modality.Vibe]),
        )

    def test_not_implemented_modalities(self):
        with self.assertRaises(NotImplementedError):
            Modality.format_keys(Modality.PD)
        with self.assertRaises(NotImplementedError):
            Modality.format_keys(Modality.FLAIR)
        # Also raises when an unsupported modality appears inside a list.
        with self.assertRaises(NotImplementedError):
            Modality.format_keys([Modality.CT, Modality.PD])


class Test_Acquisition_format_keys(unittest.TestCase):
    def test_defined_acquisitions(self):
        self.assertEqual(Acquisition.format_keys(Acquisition.sag), ["sagittal", "sag"])
        self.assertEqual(Acquisition.format_keys(Acquisition.cor), ["coronal", "cor"])
        self.assertEqual(Acquisition.format_keys(Acquisition.ax), ["axial", "ax", "axl"])
        self.assertEqual(Acquisition.format_keys(Acquisition.iso), ["iso", "ISO"])

    def test_all_four_defined_do_not_raise(self):
        # NotImplementedError is not applicable for the 4 defined members.
        for acq in (Acquisition.sag, Acquisition.cor, Acquisition.ax, Acquisition.iso):
            keys = Acquisition.format_keys(acq)
            self.assertIsInstance(keys, list)
            self.assertGreater(len(keys), 0)


class Test_Enum_Compare(unittest.TestCase):
    def test_equality_to_string_name(self):
        self.assertEqual(ModelType.nnunet, "nnunet")
        self.assertEqual(ModelType.unet, "unet")
        self.assertEqual(ModelType.classifier, "classifier")
        self.assertEqual(ModelType.nnunet.name, "nnunet")

    def test_equality_to_self(self):
        self.assertEqual(ModelType.nnunet, ModelType.nnunet)

    def test_inequality_across_members(self):
        self.assertNotEqual(ModelType.nnunet, ModelType.unet)
        self.assertNotEqual(ModelType.nnunet, "unet")
        self.assertNotEqual(ModelType.unet, "nope")
        # Comparison against an unrelated type is not equal.
        self.assertNotEqual(ModelType.nnunet, 5.0)

    def test_hash_usable_as_dict_key(self):
        # hash() works and members are usable as dict keys.
        self.assertEqual(hash(ModelType.nnunet), ModelType.nnunet.value)
        mapping = {
            ModelType.nnunet: "a",
            ModelType.unet: "b",
            ModelType.classifier: "c",
        }
        self.assertEqual(len(mapping), 3)
        self.assertEqual(mapping[ModelType.unet], "b")

    def test_str_and_repr_format(self):
        self.assertEqual(str(ModelType.nnunet), "ModelType.nnunet")
        self.assertEqual(repr(ModelType.nnunet), "ModelType.nnunet")
        self.assertEqual(str(Modality.T2w), "Modality.T2w")
        self.assertEqual(repr(Acquisition.sag), "Acquisition.sag")


class Test_MetaEnum_membership(unittest.TestCase):
    def test_member_name_in_enum(self):
        self.assertTrue("nnunet" in ModelType)
        self.assertTrue("unet" in ModelType)
        self.assertTrue("classifier" in ModelType)

    def test_non_member_membership(self):
        # MetaEnum.__contains__ returns False for names that are not members
        # (it catches both KeyError and ValueError from the member lookup).
        self.assertFalse("nope" in ModelType)
        self.assertFalse("NNUNET" in ModelType)  # case-sensitive: not a member


class Test_overlap_slice(unittest.TestCase):
    def test_overlapping(self):
        self.assertTrue(overlap_slice(slice(0, 10), slice(5, 15)))
        self.assertTrue(overlap_slice(slice(5, 15), slice(0, 10)))
        # One range fully contained in the other.
        self.assertTrue(overlap_slice(slice(0, 20), slice(5, 10)))

    def test_touching_at_border(self):
        # Borders are inclusive, so touching at a single point counts as overlap.
        self.assertTrue(overlap_slice(slice(0, 10), slice(10, 20)))
        self.assertTrue(overlap_slice(slice(10, 20), slice(0, 10)))

    def test_disjoint(self):
        self.assertFalse(overlap_slice(slice(0, 5), slice(10, 15)))
        self.assertFalse(overlap_slice(slice(10, 15), slice(0, 5)))


class Test_find_nearest_lower(unittest.TestCase):
    def test_returns_largest_element_below_x(self):
        self.assertEqual(find_nearest_lower([1, 5, 10, 20], 12), 10)
        self.assertEqual(find_nearest_lower([1, 5, 10, 20], 20), 10)
        self.assertEqual(find_nearest_lower([3, 1, 2], 3), 2)

    def test_returns_min_when_none_lower(self):
        # No element strictly below x -> falls back to min(seq).
        self.assertEqual(find_nearest_lower([10, 20, 30], 5), 10)
        self.assertEqual(find_nearest_lower([10, 20, 30], 10), 10)


class Test_add_ignore_text(unittest.TestCase):
    def test_mutates_last_element(self):
        texts = ["First.", "Second."]
        add_ignore_text(texts)
        # Last char dropped from the last entry, then the ignore marker appended.
        self.assertEqual(texts[-1], "Second (IGNORED).")
        # Earlier entries are untouched.
        self.assertEqual(texts[0], "First.")

    def test_returns_none_and_mutates_in_place(self):
        texts = ["only."]
        result = add_ignore_text(texts)
        self.assertIsNone(result)
        self.assertEqual(texts[0], "only (IGNORED).")


if __name__ == "__main__":
    unittest.main()
