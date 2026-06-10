# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import argparse
import unittest

from TPTBox import No_Logger

from spineps import entrypoint

logger = No_Logger()


class Test_EntryPoint(unittest.TestCase):
    def test_normal(self):
        entrypoint.parser_arguments(argparse.ArgumentParser())

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
