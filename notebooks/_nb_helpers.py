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


def code(source: str, title: str | None = None) -> nbf.NotebookNode:
    """Create a code cell from a multi-line string.

    If ``title`` is provided, a ``#@title <title>`` comment is prepended. On
    Colab this turns the cell into a labelled collapsible block visible in the
    notebook outline; on JupyterLab / VS Code it is just a plain comment. Use
    short titles ("Setup", "Inputs", "Run ColabFold", etc.) so the outline
    reads as a table of contents.
    """
    body = source.strip("\n")
    if title is not None:
        body = f"#@title {title}\n{body}"
    return nbf.v4.new_code_cell(body)


def notebook(
    *cells: nbf.NotebookNode,
    kernel_display_name: str = "aidd",
    python_version: str = "3.11",
    accelerator: str | None = None,
    gpu_type: str | None = None,
) -> nbf.NotebookNode:
    """Assemble cells into a notebook with our standard metadata.

    ``accelerator`` and ``gpu_type`` let GPU-specific notebooks request a
    Colab runtime by default. Set ``accelerator="GPU"`` and ``gpu_type="T4"``
    on the folding / Boltz-2 / docking notebooks; leave both unset for the
    CPU-only notebooks. The fields mirror ColabFold's AlphaFold2.ipynb
    metadata so the runtime is pre-selected the moment the notebook opens
    on Colab.
    """
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
    if accelerator is not None:
        nb.metadata["accelerator"] = accelerator
    if gpu_type is not None:
        nb.metadata.setdefault("colab", {})["gpuType"] = gpu_type
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
