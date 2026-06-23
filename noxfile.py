from __future__ import annotations

from pathlib import Path

import nox


@nox.session(python=["3.12", "3.13", "3.14"])
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("python", "-m", "pytest", *session.posargs)


@nox.session(python="3.13")
def lint(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "src", "tests")
    session.run("ruff", "format", "--check", "src", "tests")
    session.run("mypy", "src", "tests")
    session.run("check-manifest")


@nox.session(python="3.13")
def build(session: nox.Session) -> None:
    session.install("build", "twine")
    session.run("python", "-m", "build")
    distributions = sorted(str(path) for path in Path("dist").iterdir() if path.is_file())
    if not distributions:
        session.error("No distributions were created in dist/.")
    session.run("python", "-m", "twine", "check", *distributions)
