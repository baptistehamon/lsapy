"""Run public examples in a fresh interpreter to catch import side effects."""

import os
import subprocess  # noqa: S404
import sys
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.mark.parametrize("document", ["README.rst", "docs/notebooks/chunked.rst"])
def test_public_example(document):
    root = Path(__file__).resolve().parents[1]
    text = (root / document).read_text(encoding="utf-8")
    lines = text.split(".. code-block:: python\n", 1)[1].splitlines()
    block = []
    for line in lines:
        if line and not line.startswith(" "):
            break
        block.append(line)
    code = dedent("\n".join(block))
    code += "\nassert result.suitability.shape == temperature.shape\n"
    code += "assert bool(result.suitability.notnull().all().compute())\n"
    # A subprocess avoids conftest.py populating the registry before this test.
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, env=env, timeout=60
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
