# Call 'python -m unittest' on this folder
"""Every console script declared in pyproject.toml must point at something that exists.

`spineps_ = 'spineps.entrypoint:entrypoint_no_checks'` shipped for over a year although
`entrypoint_no_checks` was never written, so running `spineps_` after `pip install spineps` died with an
ImportError from the generated wrapper. Nothing caught it because the declaration is only resolved at
runtime, by the installed wrapper script.
"""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _declared_scripts() -> dict[str, str]:
    """Parses the ``[tool.poetry.scripts]`` table of the project's pyproject.toml.

    Raises:
        unittest.SkipTest: On Python < 3.11 without ``tomli``, where no TOML parser is available.
    """
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:  # pragma: no cover - only on 3.9/3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            raise unittest.SkipTest("no TOML parser available (needs Python >= 3.11 or tomli)") from None

    with PYPROJECT.open("rb") as f:
        return tomllib.load(f).get("tool", {}).get("poetry", {}).get("scripts", {})


class Test_Console_Scripts(unittest.TestCase):
    def test_pyproject_is_readable(self):
        self.assertTrue(PYPROJECT.is_file(), PYPROJECT)

    def test_every_entry_point_resolves(self):
        scripts = _declared_scripts()
        self.assertIn("spineps", scripts, "the main console script disappeared")
        for name, target in scripts.items():
            module_name, _, attribute = target.partition(":")
            with self.subTest(script=name, target=target):
                module = importlib.import_module(module_name)
                self.assertTrue(
                    hasattr(module, attribute),
                    f"console script '{name}' points at '{target}', but {module_name} has no '{attribute}'",
                )
                self.assertTrue(callable(getattr(module, attribute)), f"'{target}' is not callable")


if __name__ == "__main__":
    unittest.main()
