"""Path conventions and Drive-mount helpers for the aidd pipeline.

Persistent artefacts (anything under ``data/derived/``) need to outlive a
single Colab runtime — when the runtime dies (idle timeout, browser close,
network drop), ``/content/`` is wiped. Without persistence, every reconnect
means re-running the expensive stages from scratch.

The fix is to put ``data/derived/`` on Google Drive when on Colab, and leave
it under the repo locally. Reference data that ships with the repo
(``data/structures/``, ``data/ligands/``, ``data/compounds/``) is unaffected
— it's already in git and gets cloned fresh into ``/content/`` every runtime.

Public API
----------

- :func:`is_colab` — True iff running inside Colab.
- :func:`mount_drive_if_colab` — the one entry point each notebook calls.
  Optionally mounts Drive (Colab only, opt-out via ``use_drive=False``),
  and returns the appropriate ``data/derived/`` root for the environment
  (``DATA_ROOT``). The intended call pattern in every setup cell is::

      USE_DRIVE = IS_COLAB   # flip to False to opt out of Drive persistence
      DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)

  and every later ``REPO_ROOT / "data" / "derived" / ...`` reference becomes
  ``DATA_ROOT / ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

# Where this project's derived data lives on Drive. Mirrors the local
# ``data/derived/`` layout so users can find files in the Drive UI without
# learning a separate naming scheme.
DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/aidd-pipeline")
DRIVE_MOUNT_POINT = Path("/content/drive")


def is_colab() -> bool:
    """True if running inside Google Colab."""
    return "google.colab" in sys.modules


def mount_drive_if_colab(
    repo_root: PathLike,
    *,
    use_drive: bool | None = None,
) -> Path:
    """Set up the ``data/derived/`` root, mounting Drive on Colab if asked.

    Despite its name, this function returns the appropriate ``data/derived/``
    root path (``DATA_ROOT``) for the current environment; the Drive mount
    is a side effect when applicable. It's intended to be called once in
    each notebook's setup cell::

        USE_DRIVE = IS_COLAB
        DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)

    Parameters
    ----------
    repo_root
        The path to the repo's root (the directory containing ``src/`` and
        ``data/``). Used for the local-disk fallback path.
    use_drive
        Three-way switch:

        - ``None`` (default) — auto: ``True`` on Colab, ``False`` locally.
          The sensible behaviour for almost every caller.
        - ``True`` — mount Drive and return the Drive-backed root. Only valid
          on Colab; raises ``RuntimeError`` elsewhere.
        - ``False`` — skip the Drive mount and return the local ``data/derived/``
          under ``repo_root``. Useful for one-off Colab testing without an
          auth prompt, or to explicitly disable persistence.

    Returns
    -------
    Path
        On Colab + ``use_drive=True``: ``/content/drive/MyDrive/aidd-pipeline/data/derived/``.
        Otherwise: ``<repo_root>/data/derived/``.

    Notes
    -----
    Drive is mounted with ``drive.mount('/content/drive')`` on the first
    call per runtime; this opens an OAuth dialog the user must click through.
    Subsequent calls in the same runtime are no-ops (Drive already mounted).
    """
    if use_drive is None:
        use_drive = is_colab()

    if use_drive:
        if not is_colab():
            raise RuntimeError(
                "use_drive=True is only valid on Colab. Set use_drive=False "
                "(or leave it as None, the default) on local runs."
            )
        if not (DRIVE_MOUNT_POINT / "MyDrive").exists():
            from google.colab import drive  # type: ignore[import-not-found]

            drive.mount(str(DRIVE_MOUNT_POINT))
        root = DRIVE_PROJECT_ROOT / "data" / "derived"
        root.mkdir(parents=True, exist_ok=True)
        return root

    # Ephemeral / local-disk path. Works on Colab when opted out, and on
    # every non-Colab platform.
    return Path(repo_root) / "data" / "derived"
