# Call 'python -m unittest' on this folder
"""Every user-facing message must name a flag that exists and a cause the user can act on.

The hints told users to set flags that do not exist (`-override_subreg`, `-override_vert`,
`-override_ctd`, `-model_instance`; the real ones are `--override-semantic`, `--override-instance`,
`--override-ctd`, `--model-instance`), and `--input scan.nii` -- an uncompressed NIfTI, which SPINEPS
cannot read -- silently became `scan.nii.nii.gz` and then reported *"-input does not exist or is not a
file"* about a path the user never typed.
"""

from __future__ import annotations

import argparse
import inspect
import re
import tempfile
import unittest
from pathlib import Path

from spineps import entrypoint, seg_run


def _all_flags() -> set[str]:
    """Every option string the CLI actually accepts, including the --no-* forms argparse generates."""
    parser = argparse.ArgumentParser()
    entrypoint.parser_arguments(parser)
    flags: set[str] = set()
    for action in parser._actions:
        flags.update(action.option_strings)
        if isinstance(action, argparse.BooleanOptionalAction):
            flags.update("--no-" + s[2:] for s in action.option_strings if s.startswith("--"))
    # the per-subcommand options, which parser_arguments does not register
    flags.update(
        [
            *("--input", "-i", "--directory", "-d", "--rawdata-name", "-rn", "--model-semantic", "-ms"),
            *("--model-instance", "-mi", "-mv", "--model-labeling", "-ml", "--ignore-bids-filter", "-ibf"),
            *("--ignore-model-compatibility", "-imc", "--save-log", "-sl", "--save-snaps-folder", "-ssf"),
        ]
    )
    return flags


class Test_Messages_Name_Real_Flags(unittest.TestCase):
    """Any `-flag` mentioned in a message or exception must be one the parser knows."""

    FLAG_PATTERN = re.compile(r"(?<![\w-])--?[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*")

    def _check(self, module) -> None:
        flags = _all_flags()
        source = inspect.getsource(module)
        # only user-facing strings, i.e. logger.print(...) / raise ...(f"...")
        for line in source.splitlines():
            if "logger.print(" not in line and "raise " not in line and 'f"' not in line:
                continue
            for quoted in re.findall(r'"([^"]*)"', line):
                for match in self.FLAG_PATTERN.finditer(quoted):
                    candidate = match.group()
                    # "--model-{kind}" is assembled at runtime, so the literal "--model" prefix in front
                    # of the placeholder is not a flag on its own.
                    rest = quoted[match.end() :]
                    if rest.startswith("{") or (rest[:1] in "-_" and rest[1:2] == "{"):
                        continue
                    if "_" in candidate or candidate.startswith("--"):
                        with self.subTest(module=module.__name__, flag=candidate, line=line.strip()[:90]):
                            self.assertIn(candidate, flags, f"message mentions unknown flag {candidate}")

    def test_entrypoint_messages(self):
        self._check(entrypoint)

    def test_seg_run_messages(self):
        self._check(seg_run)


class Test_Input_Validation_Messages(unittest.TestCase):
    @staticmethod
    def _opt(input_path: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        entrypoint.parser_arguments(parser)
        opt = parser.parse_args([])
        opt.input = input_path
        opt.model_semantic = "t2w"
        opt.model_instance = "instance"
        opt.model_labeling = "none"
        return opt

    def test_uncompressed_nifti_says_so(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "scan.nii"
            path.write_bytes(b"")
            with self.assertRaises(ValueError) as cm:
                entrypoint.run_sample(self._opt(str(path)))
        message = str(cm.exception)
        self.assertIn("scan.nii", message)
        self.assertIn(".nii.gz", message)
        # and it must not invent a path the user never typed
        self.assertNotIn("scan.nii.nii.gz", message)

    def test_missing_file_names_the_path_given(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "sub-01_T2w.nii.gz"
            with self.assertRaises(FileNotFoundError) as cm:
                entrypoint.run_sample(self._opt(str(missing)))
        self.assertIn(str(missing), str(cm.exception))

    def test_missing_folder_names_the_folder(self):
        with tempfile.TemporaryDirectory() as td:
            # a native path, so the assertion holds on Windows too
            missing_folder = Path(td) / "does" / "not" / "exist"
            with self.assertRaises(FileNotFoundError) as cm:
                entrypoint.run_sample(self._opt(str(missing_folder / "scan.nii.gz")))
            self.assertIn(str(missing_folder), str(cm.exception))

    def test_extension_may_still_be_omitted(self):
        # `--input sub-01_T2w` keeps working: the extension is appended only if nothing is there already.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub-01_T2w.nii.gz"
            path.write_bytes(b"")
            with self.assertRaises(Exception) as cm:  # fails later, in model loading
                entrypoint.run_sample(self._opt(str(path)[: -len(".nii.gz")]))
        self.assertNotIsInstance(cm.exception, (FileNotFoundError, ValueError))


if __name__ == "__main__":
    unittest.main()
