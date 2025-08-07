# ruff: noqa: D103
"""Nox configuration file."""

import nox

nox.options.sessions = ("tests", "notebooks", "doctests")


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session):
    session.install(".[dev]", "h5netcdf", "netCDF4")
    session.run("pytest")


@nox.session
def notebooks(session):
    session.install(".[dev]")
    session.run("pytest", "--nbval", "docs/notebooks")


@nox.session
def doctests(session):
    session.install(".[dev]")
    session.run("pytest", "--doctest-modules", "src/lsapy")


@nox.session
def lint(session):
    session.install(".[dev]")
    # run pre-commit hooks manually to bypass no-commit-to-branch
    # leading to a failure in CI
    session.run("pre-commit", "run", "check-json", "-a")
    session.run("pre-commit", "run", "check-merge-conflict", "-a")
    session.run("pre-commit", "run", "check-toml", "-a")
    session.run("pre-commit", "run", "check-yaml", "-a")
    session.run("pre-commit", "run", "end-of-file-fixer", "-a")
    session.run("pre-commit", "run", "name-tests-test", "-a")
    session.run("pre-commit", "run", "pretty-format-json", "-a")
    session.run("pre-commit", "run", "trailing-whitespace", "-a")
    session.run("pre-commit", "run", "ruff-check", "-a")
    session.run("pre-commit", "run", "ruff-format", "-a")
    session.run("pre-commit", "run", "codespell", "-a")
    session.run("pre-commit", "run", "numpydoc-validation", "-a")


@nox.session
def docs(session):
    session.install(".[docs]")
    session.chdir("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
