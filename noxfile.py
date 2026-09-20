# ruff: noqa: D103
"""Nox configuration file."""

import nox

nox.options.sessions = ("tests", "notebooks", "doctests")


def _get_group_dependencies(*groups):
    """Get dependencies from the pyproject.toml file for the given dependency groups."""
    pyproject = nox.project.load_toml("pyproject.toml")
    return nox.project.dependency_groups(pyproject, *groups)


@nox.session(python=["3.10", "3.11", "3.12", "3.13", "3.14"])
def tests(session):
    session.install(".[all]", *_get_group_dependencies("test"))
    session.run("pytest")


@nox.session
def coverage(session):
    session.install(".[all]", *_get_group_dependencies("test"), "coveralls")
    session.run("pytest", "--cov=lsapy", "--cov-report=term-missing")
    session.run("coveralls")


@nox.session
def notebooks(session):
    session.install(".[all]", *_get_group_dependencies("test", "notebooks"))
    session.run("pytest", "--nbval", "docs/notebooks")


@nox.session
def doctests(session):
    session.install(".[all]", *_get_group_dependencies("test"))
    session.run("pytest", "--doctest-modules", "src/lsapy")


@nox.session
def lint(session):
    session.install(".[all]", *_get_group_dependencies("lint"), "pre-commit")
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
    session.run("pre-commit", "run", "yamllint", "-a")
    session.run("pre-commit", "run", "ruff-check", "-a")
    session.run("pre-commit", "run", "ruff-format", "-a")
    session.run("pre-commit", "run", "codespell", "-a")
    session.run("pre-commit", "run", "numpydoc-validation", "-a")
    session.run("pre-commit", "run", "vulture", "-a")
    session.run("pre-commit", "run", "nbstripout", "-a")
    session.run("pre-commit", "run", "python-check-blanket-type-ignore", "-a")
    session.run("pre-commit", "run", "python-no-eval", "-a")
    session.run("pre-commit", "run", "python-no-log-warn", "-a")
    session.run("pre-commit", "run", "python-use-type-annotations", "-a")
    session.run("pre-commit", "run", "rst-directive-colons", "-a")
    session.run("pre-commit", "run", "rst-inline-touching-normal", "-a")
    session.run("pre-commit", "run", "text-unicode-replacement-char", "-a")
    session.run("pre-commit", "run", "mdformat", "-a")
    session.run("pre-commit", "run", "formatbibtex", "-a")


@nox.session
def docs(session):
    session.install(".[all]", *_get_group_dependencies("docs"))
    session.chdir("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
