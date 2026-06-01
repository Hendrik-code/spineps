"""Citation reminder utilities that prompt users to cite SPINEPS when the package is used."""

from __future__ import annotations

import atexit
import os

from rich.console import Console

GITHUB_LINK = "https://github.com/Hendrik-code/spineps"

ARXIV_LINK = "https://arxiv.org/abs/2402.16368"

has_reminded_citation = False


def citation_reminder(func):
    """Decorator to remind users to cite SPINEPS."""

    def wrapper(*args, **kwargs):
        global has_reminded_citation  # noqa: PLW0603
        if not has_reminded_citation and os.environ.get("SPINEPS_TURN_OF_CITATION_REMINDER", "FALSE") != "TRUE":
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


atexit.register(print_citation_reminder)
