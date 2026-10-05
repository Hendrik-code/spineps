# Call 'python -m unittest' on this folder
"""A `spineps` run that produced nothing must not look like a success.

`run_sample` and `run_dataset` threw their result away and returned a constant, and `entry_point`
returned nothing at all, so the `spineps` console script exited 0 no matter what happened. A scan skipped
for a compatibility mismatch, an empty mask, or a dataset path where no scan was found all ended in
"Sample took: 12.3 seconds" and exit status 0 -- invisible to any shell script or job array.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from TPTBox import No_Logger

from spineps import entrypoint
from spineps.seg_enums import ErrCode

logger = No_Logger()


def _sample_opt(input_path: str) -> argparse.Namespace:
    """The `sample` namespace with every flag at its parser default."""
    parser = argparse.ArgumentParser()
    entrypoint.parser_arguments(parser)
    opt = parser.parse_args([])
    opt.input = input_path
    opt.model_semantic = "t2w"
    opt.model_instance = "instance"
    opt.model_labeling = "none"
    opt.cmd = "sample"
    return opt


def _dataset_opt(directory: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    entrypoint.parser_arguments(parser)
    opt = parser.parse_args([])
    opt.directory = directory
    opt.rawdata_name = "rawdata"
    opt.model_semantic = "t2w"
    opt.model_instance = "instance"
    opt.model_labeling = "none"
    opt.ignore_bids_filter = False
    opt.ignore_model_compatibility = False
    opt.save_log = False
    opt.save_snaps_folder = False
    opt.cmd = "dataset"
    return opt


class Test_Run_Sample_Exit_Status(unittest.TestCase):
    def _run(self, errcode: ErrCode) -> int:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub-01_T2w.nii.gz"
            path.write_bytes(b"")  # never opened: segment_image is mocked
            opt = _sample_opt(str(path))
            with (
                mock.patch.object(entrypoint, "get_semantic_model"),
                mock.patch.object(entrypoint, "get_instance_model"),
                mock.patch.object(entrypoint, "segment_image", return_value=({"out_spine": path}, errcode)),
            ):
                return entrypoint.run_sample(opt)

    def test_ok_exits_zero(self):
        self.assertEqual(self._run(ErrCode.OK), entrypoint.EXIT_OK)

    def test_already_done_exits_zero(self):
        self.assertEqual(self._run(ErrCode.ALL_DONE), entrypoint.EXIT_OK)

    def test_compatibility_skip_exits_nonzero(self):
        self.assertEqual(self._run(ErrCode.COMPATIBILITY), entrypoint.EXIT_FAILED)

    def test_empty_mask_exits_nonzero(self):
        self.assertEqual(self._run(ErrCode.EMPTY), entrypoint.EXIT_FAILED)


class Test_Run_Dataset_Exit_Status(unittest.TestCase):
    def _run(self, summary: dict) -> int:
        with tempfile.TemporaryDirectory() as td:
            opt = _dataset_opt(td)
            with (
                mock.patch.object(entrypoint, "get_semantic_model"),
                mock.patch.object(entrypoint, "get_instance_model"),
                mock.patch.object(entrypoint, "process_dataset", return_value=summary),
            ):
                return entrypoint.run_dataset(opt)

    def test_all_processed_exits_zero(self):
        summary = {"seen": 3, "processed": 3, "already_done": 0, "failed": 0, "failures": []}
        self.assertEqual(self._run(summary), entrypoint.EXIT_OK)

    def test_no_scan_found_exits_nonzero(self):
        summary = {"seen": 0, "processed": 0, "already_done": 0, "failed": 0, "failures": []}
        self.assertEqual(self._run(summary), entrypoint.EXIT_FAILED)

    def test_partial_failure_exits_nonzero(self):
        summary = {
            "seen": 2,
            "processed": 1,
            "already_done": 0,
            "failed": 1,
            "failures": [(ErrCode.COMPATIBILITY, "/data/sub-02_T2w.nii.gz")],
        }
        self.assertEqual(self._run(summary), entrypoint.EXIT_FAILED)


class Test_Entry_Point_Propagates(unittest.TestCase):
    def test_entry_point_returns_the_subcommand_status(self):
        with (
            mock.patch("sys.argv", ["spineps", "sample", "-i", "x.nii.gz", "-ms", "t2w"]),
            mock.patch.object(entrypoint, "run_sample", return_value=entrypoint.EXIT_FAILED) as run_sample,
        ):
            self.assertEqual(entrypoint.entry_point(), entrypoint.EXIT_FAILED)
        run_sample.assert_called_once()

    def test_explanations_are_actionable(self):
        for errcode in (ErrCode.COMPATIBILITY, ErrCode.EMPTY, ErrCode.SHAPE, ErrCode.UNKNOWN):
            text = entrypoint.explain_errcode(errcode)
            self.assertTrue(text.endswith(("?", ".")), text)
            self.assertGreater(len(text), 20, text)
        # unknown codes still produce something printable instead of a KeyError
        self.assertIn("ALL_DONE", entrypoint.explain_errcode(ErrCode.ALL_DONE))


class Test_Process_Dataset_Summary(unittest.TestCase):
    def test_empty_dataset_reports_zero_seen(self):
        from spineps.seg_run import process_dataset

        model = mock.MagicMock()
        model.modelid.return_value = "dummy"
        with tempfile.TemporaryDirectory() as td:
            summary = process_dataset(
                dataset_path=Path(td),
                model_instance=model,
                model_semantic=model,
                save_log_data=False,
                ignore_model_compatibility=True,
            )
        self.assertEqual(summary["seen"], 0)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["failures"], [])


if __name__ == "__main__":
    unittest.main()
