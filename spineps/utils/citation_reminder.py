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
        if not has_reminded_citation:
            print_citation_reminder()
            has_reminded_citation = True
        func_result = func(*args, **kwargs)
        return func_result

    return wrapper


def print_citation_reminder():
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
