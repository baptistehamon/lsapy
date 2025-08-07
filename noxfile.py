# ruff: noqa: D103
"""Nox configuration file."""

import nox

nox.options.sessions = ("tests", "notebooks", "doctests")


@nox.session(venv_backend="conda", python=["3.10", "3.11", "3.12", "3.13"])
def tests(session):
    session.install(".[dev]", "h5netcdf", "netCDF4")
    session.run("pytest")


@nox.session
def notebooks(session):
    session.install(".[docs]")
    session.run("pytest", "--nbval", "docs/notebooks")


@nox.session
def doctests(session):
    session.install(".[docs]")
    session.run("pytest", "--doctest-modules", "src/lsapy")


@nox.session
def lint(session):
    session.install(".[dev]")
    session.run("pre-commit", "run", "-a")


@nox.session
def docs(session):
    session.install(".[docs]")
    session.chdir("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
