# Call 'python -m unittest' on this folder  # noqa: INP001
"""Tests for the high-level spineps.segment API, config dataclasses and result wrapping."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from TPTBox import BIDS_FILE
from TPTBox.tests.test_utils import get_test_mri

from spineps import InstanceConfig, LabelingConfig, PostConfig, SemanticConfig, segment
from spineps.api import SpinepsPipeline, _as_bids_file, _to_result
from spineps.seg_enums import ErrCode


class Test_Config_To_Kwargs(unittest.TestCase):
    def test_semantic(self):
        kw = SemanticConfig(crop_input=False, n4_bias_correction=False).to_kwargs()
        self.assertFalse(kw["proc_sem_crop_input"])
        self.assertFalse(kw["proc_sem_n4_bias_correction"])

    def test_instance(self):
        kw = InstanceConfig(batch_size=8, largest_k_cc=3).to_kwargs()
        self.assertEqual(kw["proc_inst_batch_size"], 8)
        self.assertEqual(kw["proc_inst_largest_k_cc"], 3)
        self.assertEqual(kw["vertebra_instance_labeling_offset"], 2)

    def test_labeling_and_post(self):
        self.assertTrue(LabelingConfig(force_no_tl_anomaly=True).to_kwargs()["proc_lab_force_no_tl_anomaly"])
        self.assertFalse(PostConfig(fill_3d_holes=False).to_kwargs()["proc_fill_3d_holes"])


class Test_To_Result(unittest.TestCase):
    def test_save_mode_tuple(self):
        r = _to_result(({"out_spine": Path("/tmp/x")}, ErrCode.OK))
        self.assertTrue(r.success)
        self.assertIsNotNone(r.output_paths)
        self.assertIsNone(r.semantic)

    def test_in_memory_tuple(self):
        a, b, c = object(), object(), object()
        r = _to_result((a, b, c, ErrCode.OK))
        self.assertIs(r.semantic, a)
        self.assertIs(r.vertebra, b)
        self.assertIs(r.centroids, c)
        self.assertIsNone(r.output_paths)

    def test_failure_errcode_not_success(self):
        self.assertFalse(_to_result(({}, ErrCode.COMPATIBILITY)).success)


class Test_As_Bids_File(unittest.TestCase):
    def test_wraps_existing_path(self):
        mri, *_ = get_test_mri()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub-x_T2w.nii.gz"
            mri.save(path, verbose=False)
            with _as_bids_file(str(path)) as bf:
                self.assertIsInstance(bf, BIDS_FILE)

    def test_wraps_in_memory_nii(self):
        mri, *_ = get_test_mri()
        with _as_bids_file(mri) as bf:
            self.assertIsInstance(bf, BIDS_FILE)

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError), _as_bids_file("/no/such/file.nii.gz"):
            pass

    def test_non_niigz_raises(self):
        with self.assertRaises(ValueError), _as_bids_file("/tmp/not_an_image.png"):
            pass


class Test_Segment_Api(unittest.TestCase):
    def test_segment_forwards_config_kwargs(self):
        captured: dict = {}

        def fake_segment_image(**kwargs):
            captured.update(kwargs)
            return ({"out_spine": Path("/tmp/x")}, ErrCode.OK)

        mri, *_ = get_test_mri()
        with (
            mock.patch("spineps.api._resolve_model", side_effect=lambda m, _getter, _cpu: m or "model"),
            mock.patch("spineps.api.segment_image", side_effect=fake_segment_image),
        ):
            res = segment(mri, instance=InstanceConfig(batch_size=8), semantic=SemanticConfig(crop_input=False))
        self.assertEqual(res.errcode, ErrCode.OK)
        self.assertEqual(captured["proc_inst_batch_size"], 8)
        self.assertFalse(captured["proc_sem_crop_input"])
        # an in-memory NII input forces in-memory output
        self.assertTrue(captured["return_output_instead_of_save"])

    def test_pipeline_resolves_models_once(self):
        calls = {"n": 0}

        def fake_resolve(model, _getter, _cpu):
            calls["n"] += 1
            return model or "model"

        with (
            mock.patch("spineps.api._resolve_model", side_effect=fake_resolve),
            mock.patch("spineps.api.segment_image", return_value=({"x": Path("/tmp/x")}, ErrCode.OK)),
        ):
            pipe = SpinepsPipeline("t2w", "instance", "t2w_labeling")
            mri, *_ = get_test_mri()
            pipe.segment(mri)
            pipe.segment(mri)
        # 3 models resolved once in __init__, never again per segment() call
        self.assertEqual(calls["n"], 3)


if __name__ == "__main__":
    unittest.main()
