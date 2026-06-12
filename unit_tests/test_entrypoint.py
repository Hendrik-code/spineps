# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import unittest
from unittest import mock

from TPTBox import No_Logger

from spineps import entrypoint

logger = No_Logger()


class Test_EntryPoint(unittest.TestCase):
    def test_normal(self):
        entrypoint.parser_arguments(argparse.ArgumentParser())

    def test_shared_flag_defaults_and_negation(self):
        # positive-polarity boolean flags + the new batch-size knob
        p = argparse.ArgumentParser()
        entrypoint.parser_arguments(p)
        defaults = p.parse_args([])
        self.assertTrue(defaults.crop_input)
        self.assertTrue(defaults.n4)
        self.assertFalse(defaults.enforce_12_thoracic)
        self.assertEqual(defaults.batch_size, 4)
        negated = p.parse_args(["--no-crop", "--no-n4", "--enforce-12-thoracic", "--batch-size", "8"])
        self.assertFalse(negated.crop_input)
        self.assertFalse(negated.n4)
        self.assertTrue(negated.enforce_12_thoracic)
        self.assertEqual(negated.batch_size, 8)

    def test_speed_knob_flags(self):
        # --amp / --step-size / --tta tri-state
        p = argparse.ArgumentParser()
        entrypoint.parser_arguments(p)
        defaults = p.parse_args([])
        self.assertFalse(defaults.amp)
        self.assertIsNone(defaults.step_size)
        self.assertIsNone(defaults.tta)  # None = use the model's configured setting
        opts = p.parse_args(["--amp", "--step-size", "0.7", "--no-tta"])
        self.assertTrue(opts.amp)
        self.assertEqual(opts.step_size, 0.7)
        self.assertFalse(opts.tta)
        self.assertTrue(p.parse_args(["--tta"]).tta)

    def test_subparser_help_builds(self):
        # Regression: empty metavar="" used to crash `spineps <sub> -h` in argparse usage formatting.
        for sub in ("sample", "dataset"):
            with (
                self.assertRaises(SystemExit) as cm,
                mock.patch.object(sys, "argv", ["spineps", sub, "-h"]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                entrypoint.entry_point()
            self.assertEqual(cm.exception.code, 0)

    def test_run_sample_missing_parent_raises(self):
        # parent directory of the input does not exist -> FileNotFoundError (not a bare AssertionError)
        opt = argparse.Namespace(input="/this/path/does/not/exist/scan.nii.gz")
        with self.assertRaises(FileNotFoundError):
            entrypoint.run_sample(opt)

    def test_run_dataset_missing_directory_raises(self):
        opt = argparse.Namespace(directory="/this/path/does/not/exist")
        with self.assertRaises(FileNotFoundError):
            entrypoint.run_dataset(opt)

    def test_run_dataset_file_not_directory_raises(self):
        # an existing path that is a file, not a directory -> NotADirectoryError
        opt = argparse.Namespace(directory=__file__)
        with self.assertRaises(NotADirectoryError):
            entrypoint.run_dataset(opt)
