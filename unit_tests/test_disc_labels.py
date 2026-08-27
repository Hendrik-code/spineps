# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from TPTBox import No_Logger
from typing_extensions import Self

from spineps.utils.generate_disc_labels import Image, main

logger = No_Logger()


class Test_DiscLabels(unittest.TestCase):
    def test_main_without_args(self):
        # --path-vert is required; omitting it must exit via argparse, not proceed.
        with (
            self.assertRaises(SystemExit) as cm,
            mock.patch.object(sys, "argv", ["generate_disc_labels"]),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            main()
        self.assertEqual(cm.exception.code, 2)

    def test_image(self):
        img = Image(param=np.array([0, 0, 0, 0]))

        img.dim  # noqa: B018
        img.orientation  # noqa: B018
        img.absolutepath  # noqa: B018
        img.copy()
        img.change_type(np.uint8)
