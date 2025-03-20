# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
import argparse
import unittest
from pathlib import Path

from TPTBox import No_Logger
from typing_extensions import Self

from spineps import entrypoint

logger = No_Logger()


class Test_EntryPoint(unittest.TestCase):
    def test_normal(self):
        entrypoint.parser_arguments(argparse.ArgumentParser())
