"""Tiny helpers used by every ``_build_<name>.py`` notebook builder.

The builders use these to emit cells in narrative order, e.g.::

    nb = notebook(
        markdown('# Title'),
        code('import sys'),
        markdown('## Section 1'),
        ...
    )
    save(nb, NOTEBOOK_PATH)
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
