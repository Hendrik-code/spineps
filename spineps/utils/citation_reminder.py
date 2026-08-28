"""Citation reminder utilities that prompt users to cite SPINEPS when the package is used."""

from __future__ import annotations

import atexit
import functools
import os

from rich.console import Console

GITHUB_LINK = "https://github.com/Hendrik-code/spineps"

ARXIV_LINK = "https://arxiv.org/abs/2402.16368"

# Set this environment variable to any of the values below to silence the reminder entirely.
OPT_OUT_ENV_VAR = "SPINEPS_NO_CITATION_REMINDER"
_OPT_OUT_VALUES = frozenset({"1", "true", "yes", "on"})

has_reminded_citation = False


def reminder_disabled() -> bool:
    """Return whether the user opted out of the citation reminder via the environment."""
    return os.environ.get(OPT_OUT_ENV_VAR, "").strip().lower() in _OPT_OUT_VALUES


def citation_reminder(func):
    """Decorator to remind users to cite SPINEPS."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global has_reminded_citation  # noqa: PLW0603
        if not has_reminded_citation and not reminder_disabled():
            print_citation_reminder()
            has_reminded_citation = True
        return func(*args, **kwargs)

    return wrapper


def print_citation_reminder():
    """Print a formatted reminder with the SPINEPS GitHub and ArXiv links asking users to cite the work."""
    console = Console()
    console.rule("Thank you for using [bold]SPINEPS[/bold]")
    console.print(
        "Please support our development by citing",
        justify="center",
    )
    console.print(
        f"GitHub: {GITHUB_LINK}\nArXiv: {ARXIV_LINK}\n Thank you!",
        justify="center",
    )
    console.rule()
    console.line()


def _print_citation_reminder_at_exit() -> None:
    """Repeat the reminder on interpreter exit, but only if SPINEPS actually ran and the user did not opt out.

    Registering ``print_citation_reminder`` directly made merely importing ``spineps`` print the banner, and
    ignored the opt-out environment variable entirely.
    """
    if has_reminded_citation and not reminder_disabled():
        print_citation_reminder()


atexit.register(_print_citation_reminder_at_exit)
