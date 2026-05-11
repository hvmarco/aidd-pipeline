"""Tiny helpers used by every ``_build_<name>.py`` notebook builder.

The builders use these to emit cells in narrative order, e.g.::

    nb = notebook(
        markdown('# Title'),
        code('import sys'),
        markdown('## Section 1'),
        ...
    )
    save(nb, NOTEBOOK_PATH)

Constant ``AUTORELOAD_SNIPPET`` provides a portable replacement for the
``%load_ext autoreload`` magic that breaks on Python 3.12 + older IPython
(Colab's current default, as of 2026-05). Paste it at the top of any setup
code cell that wants autoreload behaviour.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def markdown(source: str) -> nbf.NotebookNode:
    """Create a markdown cell from a multi-line string.

    Leading/trailing blank lines are stripped so triple-quoted strings render
    cleanly in Jupyter regardless of indentation in the builder.
    """
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    """Create a code cell from a multi-line string."""
    return nbf.v4.new_code_cell(source.strip("\n"))


def notebook(
    *cells: nbf.NotebookNode,
    kernel_display_name: str = "aidd",
    python_version: str = "3.11",
) -> nbf.NotebookNode:
    """Assemble cells into a notebook with our standard metadata."""
    nb = nbf.v4.new_notebook()
    nb.cells = list(cells)
    nb.metadata = {
        "kernelspec": {
            "display_name": kernel_display_name,
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": python_version},
    }
    return nb


def save(nb: nbf.NotebookNode, path: Path) -> None:
    """Write the notebook and print a one-line summary."""
    nbf.write(nb, path)
    print(f"Wrote {path.name} ({len(nb.cells)} cells)")


# Portable autoreload loader. Works on any IPython, on any Python version.
# Background: %load_ext autoreload imports `imp`, which was removed in Python
# 3.12. Colab's current Python 3.12 + bundled IPython hit this. The wrapper
# below tries the magic and silently degrades when it can't.
AUTORELOAD_SNIPPET = """\
# Auto-reload edits made to src/aidd/ without restarting the kernel.
# Tolerant wrapper — Colab's Python-3.12 + older IPython removed the `imp` module
# that the autoreload magic used to import. We silently skip if it can't load.
from IPython import get_ipython
_ip = get_ipython()
if _ip is not None:
    try:
        _ip.run_line_magic("load_ext", "autoreload")
        _ip.run_line_magic("autoreload", "2")
    except Exception as _e:
        print(f"autoreload unavailable (skip): {_e}")
"""
