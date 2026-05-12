"""gnina docking + PoseBusters QC.

Thin wrappers around the ``gnina`` CLI plus utilities to read its SDF output
into a tidy DataFrame, compute redock RMSDs against a crystallographic pose,
and run PoseBusters QC on the resulting poses.

The classical "interpretable lane" of the screening pipeline. The complementary
fast lane (Boltz-2 co-folding + affinity) lives in a separate module / notebook.

Cross-platform notes
--------------------
- gnina is **Linux-native**. On Windows / macOS the helpers refuse loudly with
  a one-line install hint. The notebook that uses this module is therefore
  Colab-first; locally it works only under WSL2 or a Linux container.
- On Colab the static binary from gnina's GitHub Releases is the simplest
  install. See ``GNINA_RELEASE_URL`` below and the install cell in
  ``notebooks/_build_03_dock_gnina.py``.

Adapted, in spirit, from the workflow in
``_archive/Week_3_Monday_Docking_and_Scoring.ipynb`` (which used PLANTS); the
binding-site geometry for the ERK2/4FV7 example is taken verbatim from the
archived PLANTS config ``_archive/configs/plants_4fv7.conf``.

Module API note
---------------
The original step-9 brief asked for a ``prepare_receptor(pdb) -> Path`` helper
that would convert PDB to PDBQT. This was elided after verification: gnina
(unlike AutoDock Vina) reads PDB receptors directly and does its own atom
typing internally, so no upfront conversion step is needed. PDBQT prep
remains relevant for the Vina / AutoDock4 fallback path, which lives in
a separate module if/when it is needed.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, Union

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

# Silence RDKit's per-molecule logger for the rest of the session. PoseBusters,
# parse_poses_sdf, and the 3-D viewer cells in notebook 03 all read gnina-output
# SDFs through SDMolSupplier; those SDFs don't carry the "3D" flag in their
# header, so RDKit emits one ``molecule is tagged as 2D, but at least one Z
# coordinate is not zero`` warning per pose. On a 100-compound × 9-pose run
# that's ~900 lines of noise drowning every cell output that touches an SDF.
# The warning is cosmetic — RDKit parses the molecule correctly. Silencing
# globally at import time (rather than scoped to one function call) matches
# the same pattern aidd.ligands uses and avoids the issue where a scoped
# silencer re-enables on exit and undoes the silence set by other modules.
RDLogger.DisableLog("rdApp.*")

PathLike = Union[str, Path]
logger = logging.getLogger("aidd.docking")

GNINA_RELEASE_URL = (
    "https://github.com/gnina/gnina/releases/download/v1.3/gnina"
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def gnina_path() -> Path | None:
    """Return the gnina executable on ``PATH``, or ``None`` if not installed."""
    p = shutil.which("gnina")
    return Path(p) if p else None


def gnina_available() -> bool:
    return gnina_path() is not None


def describe_environment() -> dict:
    """Report the docking-relevant environment state for sanity printing."""
    g = gnina_path()
    info: dict = {
        "platform": platform.system(),
        "gnina_path": str(g) if g else None,
        "gnina_version": None,
    }
    if g is not None:
        try:
            out = subprocess.run(
                [str(g), "--version"], capture_output=True, text=True, timeout=10
            )
            line = (out.stdout or out.stderr).splitlines()
            info["gnina_version"] = line[0].strip() if line else None
        except Exception as exc:  # gnina absent / broken on a non-Linux host
            info["gnina_version"] = f"(version probe failed: {exc})"
    return info


def require_gnina() -> Path:
    """Return gnina's path or raise with a platform-appropriate install hint."""
    g = gnina_path()
    if g is not None:
        return g
    system = platform.system()
    if system == "Linux":
        msg = (
            "gnina not found on PATH. On Colab / Linux install with:\n"
            f"    wget -q -O /usr/local/bin/gnina {GNINA_RELEASE_URL}\n"
            "    chmod +x /usr/local/bin/gnina\n"
            "or via conda: `conda install -c bioconda gnina`."
        )
    elif system == "Darwin":
        msg = (
            "gnina has no native macOS build. Run this notebook on Google "
            "Colab (the setup cell installs gnina) or in a Linux container."
        )
    else:  # Windows
        msg = (
            "gnina is Linux-native and not available on Windows. Run this "
            "notebook on Google Colab (recommended — the setup cell installs "
            "gnina automatically), or under WSL2 with "
            "`conda install -c bioconda gnina`."
        )
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Binding-site geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BindingBox:
    """Cubic / cuboid search box passed to gnina via ``--center_* / --size_*``.

    ``center`` is in PDB coordinates (Å); ``size`` is the **edge length** of
    the box, not the half-width. Gnina samples poses anywhere inside the box.
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def gnina_args(self) -> list[str]:
        cx, cy, cz = self.center
        sx, sy, sz = self.size
        return [
            "--center_x", f"{cx:.4f}",
            "--center_y", f"{cy:.4f}",
            "--center_z", f"{cz:.4f}",
            "--size_x",   f"{sx:.4f}",
            "--size_y",   f"{sy:.4f}",
            "--size_z",   f"{sz:.4f}",
        ]


def box_from_center_radius(
    center: Sequence[float], radius: float, *, margin: float = 2.0
) -> BindingBox:
    """Cubic box covering a sphere of ``radius`` Å plus ``margin`` Å on each side."""
    side = 2.0 * radius + 2.0 * margin
    cx, cy, cz = (float(c) for c in center)
    return BindingBox(center=(cx, cy, cz), size=(side, side, side))


# ---------------------------------------------------------------------------
# Docking — library
# ---------------------------------------------------------------------------

_UNSAFE_ID_CHARS = set('/\\:*?"<>|\t\r\n')
OnFailure = Literal["raise", "skip", "skip_with_guard"]


def dock_library(
    receptor: PathLike,
    ligands_sdf: PathLike,
    out_dir: PathLike,
    box: BindingBox,
    *,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    cnn_scoring: str = "rescore",
    seed: int = 42,
    cpu: int | None = None,
    overwrite: bool = False,
    gnina_extra: Sequence[str] = (),
    progress: bool = True,
    on_failure: OnFailure = "skip_with_guard",
    failure_guard_n: int = 5,
) -> pd.DataFrame:
    """Dock every ligand in ``ligands_sdf`` into ``receptor`` with gnina.

    **Resumable per-compound caching.** Each compound is docked in its own
    gnina subprocess and its outputs land in ``out_dir/per_compound/<id>/``
    before the next compound starts. If the cell or runtime dies partway
    through (Colab disconnect, browser close, ``KeyboardInterrupt``), already-
    completed compounds remain on disk; re-running the function picks them up
    from cache and only dispatches gnina for the compounds that don't yet
    have a saved score row. After the loop, the aggregate
    ``out_dir/poses.sdf`` and ``out_dir/gnina_scores.csv`` are rebuilt from
    the per-compound store so downstream callers (notebook 03, notebook 04,
    notebook 06) see the same two files they always have.

    Backward-compatible legacy cache: if ``out_dir`` already holds
    ``poses.sdf`` + ``gnina_scores.csv`` from a pre-refactor (single-subprocess)
    run **and** has no ``per_compound/`` sub-directory, the legacy aggregate
    is honoured as-is and returned unchanged. This is the path that picks up
    the step-10 ERK2 labelled-subset cache on Google Drive.

    Parameters
    ----------
    cnn_scoring
        ``'rescore'`` (default, fast — Vina docks, CNN rescore the final pose),
        ``'refinement'`` (slower, CNN-guided refinement after Vina), or
        ``'all'`` (CNN at every step; slowest, highest accuracy when GPU is
        available). ``'none'`` disables CNN and gives Vina/smina behaviour.
    cpu
        Number of CPU threads for gnina. Default ``None`` lets gnina pick.
    overwrite
        When True, redock every compound even if a per-compound cache exists.
    gnina_extra
        Extra raw CLI flags appended after the standard arguments.
    progress
        Show a tqdm bar over compounds. Off-by-default-friendly for unit tests.
    on_failure
        How to react when one compound's gnina exits non-zero:

        - ``"raise"``       — raise immediately on the first failing compound
                              (the old fail-fast behaviour). Use when you want
                              an unfamiliar pipeline to error loudly on the
                              first sign of trouble.
        - ``"skip"``        — log the failure, write a ``<id>.FAILED`` marker
                              under ``per_compound/``, keep going. Cached failed
                              compounds are not retried on subsequent runs.
        - ``"skip_with_guard"`` (default) — like ``"skip"``, but raise if
                              ``failure_guard_n`` consecutive *fresh* failures
                              occur in a row (cached failures from earlier
                              runs do not count). Catches systemic issues
                              (broken install, GPU dead, wrong receptor) early
                              without sacrificing robustness to one-off
                              compound-specific failures further into the run.
    failure_guard_n
        Streak threshold for ``on_failure="skip_with_guard"``. Default 5.

    Outputs in ``out_dir/``
    ----------------------
    - ``poses.sdf``           — concatenated successful poses.
    - ``gnina_scores.csv``    — tidy scores table (one row per pose, successes only).
    - ``failures.csv``        — one row per failed compound (compound_id,
                                error_class, error_summary, gnina_log_tail).
                                Written even when empty.
    - ``per_compound/<id>/``  — success: ``pose.sdf`` + ``scores.csv`` + ``gnina.log``.
    - ``per_compound/<id>.FAILED`` — JSON marker for a failed compound. Its
                                presence tells the cache check "don't retry".

    Notes
    -----
    The ``on_failure`` parameter shape is mirrored on
    :func:`aidd.co_folding.predict_library` so both lanes have the same
    failure-handling contract.

    To force a retry of a previously-failed compound, delete its
    ``per_compound/<id>.FAILED`` marker (and the matching ``per_compound/<id>/``
    directory if you want a clean log) and re-run; the cache check will see
    no record and re-dock it.
    """
    if on_failure not in {"raise", "skip", "skip_with_guard"}:
        raise ValueError(
            f"on_failure must be 'raise', 'skip', or 'skip_with_guard'; got {on_failure!r}."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses_sdf = out_dir / "poses.sdf"
    scores_csv = out_dir / "gnina_scores.csv"
    failures_csv = out_dir / "failures.csv"
    per_compound_root = out_dir / "per_compound"
    inputs_dir = per_compound_root / "_inputs"

    if (
        not overwrite
        and poses_sdf.exists()
        and scores_csv.exists()
        and not per_compound_root.exists()
    ):
        logger.info(
            "Legacy aggregate docking cache found at %s (no per-compound "
            "store); returning it unchanged.",
            out_dir,
        )
        return pd.read_csv(scores_csv)

    per_compound_root.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    splits = _split_input_sdf(Path(ligands_sdf), inputs_dir)
    if not splits:
        raise ValueError(f"No molecules parsed from {ligands_sdf}.")

    # Defer require_gnina() until we actually need to dispatch a docking call.
    # When every compound is already cached (success or failed marker), the
    # function still has to walk the scope to rebuild aggregates — and doing
    # that on Windows after a Colab run on Drive is a supported flow. Calling
    # require_gnina() unconditionally would break it.
    g: Path | None = None

    iterator = _maybe_tqdm(splits, total=len(splits), desc="gnina dock", enable=progress)
    n_cached_ok = 0
    n_cached_failed = 0
    n_fresh_ok = 0
    n_fresh_failed = 0
    consecutive_fresh_failures = 0
    for compound_id, single_input_sdf in iterator:
        cdir = per_compound_root / compound_id
        cpose = cdir / "pose.sdf"
        cscores = cdir / "scores.csv"
        clog = cdir / "gnina.log"
        failed_marker = per_compound_root / f"{compound_id}.FAILED"

        if not overwrite and cscores.exists() and cpose.exists():
            n_cached_ok += 1
            consecutive_fresh_failures = 0
            continue
        if not overwrite and failed_marker.exists():
            n_cached_failed += 1
            continue

        if g is None:
            g = require_gnina()
        cdir.mkdir(parents=True, exist_ok=True)
        cmd: list[str] = [
            str(g),
            "-r", str(receptor),
            "-l", str(single_input_sdf),
            "-o", str(cpose),
            "--cnn_scoring", cnn_scoring,
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--seed", str(seed),
            *box.gnina_args(),
        ]
        if cpu is not None:
            cmd += ["--cpu", str(cpu)]
        cmd += list(gnina_extra)

        with clog.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            err_msg = _format_gnina_error(proc.returncode, clog)
            if on_failure == "raise":
                raise RuntimeError(err_msg)
            _persist_failure(
                failed_marker,
                compound_id=compound_id,
                error_class="GninaNonZeroExit",
                error_summary=f"gnina exited with status {proc.returncode}",
                gnina_log_tail=_read_log_tail(clog, n=30),
            )
            n_fresh_failed += 1
            consecutive_fresh_failures += 1
            logger.warning(
                "gnina failed on compound %s (status %d) — skipping; marker at %s",
                compound_id, proc.returncode, failed_marker,
            )
            if (
                on_failure == "skip_with_guard"
                and consecutive_fresh_failures >= failure_guard_n
            ):
                raise RuntimeError(
                    f"gnina has failed on {consecutive_fresh_failures} consecutive "
                    f"compounds (guard threshold = {failure_guard_n}). This usually "
                    f"means a systemic problem (broken install, GPU dead, wrong "
                    f"receptor, missing CUDA library) rather than per-compound "
                    f"chemistry. Last error:\n\n{err_msg}\n\n"
                    f"Per-compound markers in {per_compound_root}/*.FAILED; "
                    f"resolve the root cause, delete the marker files for the "
                    f"compounds you want to retry, and re-run."
                )
            continue

        per_df = parse_poses_sdf(cpose)
        per_df.to_csv(cscores, index=False)
        n_fresh_ok += 1
        consecutive_fresh_failures = 0

    logger.info(
        "dock_library: %d cached-success, %d cached-failed, %d fresh-success, "
        "%d fresh-failed (total %d compounds).",
        n_cached_ok, n_cached_failed, n_fresh_ok, n_fresh_failed, len(splits),
    )

    df = _aggregate_per_compound(
        per_compound_root,
        poses_sdf,
        scores_csv,
        scope_ids=[cid for cid, _ in splits],
    )
    _aggregate_failures(
        per_compound_root,
        failures_csv,
        scope_ids=[cid for cid, _ in splits],
    )
    return df


def _persist_failure(
    marker_path: Path,
    *,
    compound_id: str,
    error_class: str,
    error_summary: str,
    gnina_log_tail: str,
) -> None:
    payload = {
        "compound_id": compound_id,
        "error_class": error_class,
        "error_summary": error_summary,
        "gnina_log_tail": gnina_log_tail,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    marker_path.write_text(json.dumps(payload, indent=2))


def _read_log_tail(log_path: Path, *, n: int = 30) -> str:
    if not log_path.exists():
        return ""
    try:
        with log_path.open() as fh:
            lines = fh.readlines()
        return "".join(lines[-n:]).rstrip()
    except OSError:
        return ""


def _split_input_sdf(ligands_sdf: Path, work_dir: Path) -> list[tuple[str, Path]]:
    """Split a multi-compound SDF into one per-compound SDF under ``work_dir``.

    Returns ``[(compound_id, single_input_sdf_path), ...]``. Idempotent:
    pre-existing per-compound input files are reused. Compound IDs must be
    non-empty and free of filesystem-unsafe characters; the per-compound store
    uses ``compound_id`` directly as a sub-directory name, so the IDs become
    part of the on-disk layout.
    """
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    sup = Chem.SDMolSupplier(str(ligands_sdf), removeHs=False)
    for i, mol in enumerate(sup):
        if mol is None:
            continue
        cid = (mol.GetProp("_Name") if mol.HasProp("_Name") else "").strip()
        if not cid:
            raise ValueError(
                f"Molecule {i} in {ligands_sdf} has no SDF title (_Name). "
                "dock_library uses _Name as the compound_id everywhere "
                "downstream (cache directory, scores row, join key); set a "
                "unique non-empty title on every input molecule."
            )
        bad = sorted({c for c in cid if c in _UNSAFE_ID_CHARS})
        if bad:
            raise ValueError(
                f"compound_id {cid!r} contains filesystem-unsafe characters "
                f"{bad}. Rename the input compound or sanitise IDs upstream — "
                "dock_library writes a per-compound subdirectory whose name "
                "is the compound_id."
            )
        if cid in seen:
            raise ValueError(
                f"Duplicate compound_id {cid!r} in {ligands_sdf}. IDs must be "
                "unique; downstream code joins on this column."
            )
        seen.add(cid)
        single = work_dir / f"{cid}.sdf"
        if not single.exists():
            w = Chem.SDWriter(str(single))
            w.write(mol)
            w.close()
        out.append((cid, single))
    return out


def _aggregate_per_compound(
    per_compound_root: Path,
    poses_sdf: Path,
    scores_csv: Path,
    *,
    scope_ids: Sequence[str],
) -> pd.DataFrame:
    """Rebuild aggregate ``poses.sdf`` + ``gnina_scores.csv`` from the per-compound store.

    Filters to ``scope_ids`` so the aggregate reflects the compounds passed
    into the *current* ``dock_library`` call, not the union of every
    compound that has ever been docked into this ``out_dir`` (which would
    surprise downstream callers like notebook 04 reading
    ``gnina_scores.csv`` directly).

    Concatenation is at the byte level — each per-compound SDF already ends
    with the ``$$$$\\n`` terminator that RDKit's SDWriter emits, so the
    resulting file is a valid multi-compound SDF.
    """
    score_frames: list[pd.DataFrame] = []
    pose_paths: list[Path] = []
    for cid in scope_ids:
        cdir = per_compound_root / cid
        cscores = cdir / "scores.csv"
        cpose = cdir / "pose.sdf"
        if cscores.exists():
            score_frames.append(pd.read_csv(cscores))
        if cpose.exists():
            pose_paths.append(cpose)

    if not score_frames:
        empty = pd.DataFrame(columns=[
            "compound_id", "pose_rank", "affinity", "cnn_score",
            "cnn_affinity", "cnn_vs", "minimized_rmsd",
        ])
        empty.to_csv(scores_csv, index=False)
        poses_sdf.write_bytes(b"")
        return empty

    df = pd.concat(score_frames, ignore_index=True)
    df.to_csv(scores_csv, index=False)

    with poses_sdf.open("wb") as out:
        for p in pose_paths:
            out.write(p.read_bytes())
    return df


def _aggregate_failures(
    per_compound_root: Path,
    failures_csv: Path,
    *,
    scope_ids: Sequence[str],
) -> pd.DataFrame:
    """Collect every ``<id>.FAILED`` marker under ``per_compound_root`` for
    compounds in ``scope_ids`` into a single tidy CSV.

    The output is rebuilt on every call (no append semantics) so it reflects
    the current scope, not the union of every run that ever wrote into this
    directory. An empty ``failures.csv`` is still written when no compound
    failed — a missing file would be ambiguous (did the run not happen, or
    did everything succeed?).
    """
    rows: list[dict] = []
    for cid in scope_ids:
        marker = per_compound_root / f"{cid}.FAILED"
        if not marker.exists():
            continue
        try:
            payload = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            payload = {
                "compound_id": cid,
                "error_class": "MarkerParseError",
                "error_summary": f"Could not parse failure marker: {exc}",
                "gnina_log_tail": "",
                "timestamp": "",
            }
        rows.append({
            "compound_id":    payload.get("compound_id", cid),
            "error_class":    payload.get("error_class", ""),
            "error_summary":  payload.get("error_summary", ""),
            "gnina_log_tail": payload.get("gnina_log_tail", ""),
            "timestamp":      payload.get("timestamp", ""),
        })
    df = pd.DataFrame(
        rows,
        columns=["compound_id", "error_class", "error_summary", "gnina_log_tail", "timestamp"],
    )
    df.to_csv(failures_csv, index=False)
    return df


def _maybe_tqdm(it, *, total: int, desc: str, enable: bool):
    if not enable:
        return it
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return it
    return tqdm(it, total=total, desc=desc)


# ---------------------------------------------------------------------------
# Docking — single-ligand redock (sanity check)
# ---------------------------------------------------------------------------

def redock_reference(
    receptor: PathLike,
    ref_ligand: PathLike,
    out_dir: PathLike,
    *,
    autobox_ligand: PathLike | None = None,
    autobox_add: float = 4.0,
    exhaustiveness: int = 16,
    num_modes: int = 9,
    cnn_scoring: str = "rescore",
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    """Re-dock a known crystallographic ligand back into its receptor.

    Returns a dict with the top-pose gnina scores, the heavy-atom RMSD between
    the predicted top pose and the crystal pose (``rmsd_to_crystal``), and the
    SDF path of the redocked poses. A redock RMSD < 2 Å is the standard
    methodological sanity check.

    ``autobox_ligand`` defaults to ``ref_ligand`` — gnina derives the search
    box from the ligand's own coordinates plus ``autobox_add`` Å margin.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    redock_sdf = out_dir / "redock_poses.sdf"
    log_path = out_dir / "redock_gnina.log"

    if overwrite or not redock_sdf.exists():
        g = require_gnina()
        autobox = autobox_ligand or ref_ligand
        cmd: list[str] = [
            str(g),
            "-r", str(receptor),
            "-l", str(ref_ligand),
            "-o", str(redock_sdf),
            "--autobox_ligand", str(autobox),
            "--autobox_add", str(autobox_add),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--cnn_scoring", cnn_scoring,
            "--seed", str(seed),
        ]
        logger.info("gnina redock: %s", " ".join(cmd))
        with log_path.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(_format_gnina_error(proc.returncode, log_path))

    poses_df = parse_poses_sdf(redock_sdf)
    if poses_df.empty:
        raise RuntimeError(
            f"gnina produced no poses. See {log_path} for what went wrong."
        )

    top = poses_df.iloc[0].to_dict()
    top["rmsd_to_crystal"] = pose_rmsd(redock_sdf, ref_ligand, pose_index=0)
    top["poses_sdf"] = str(redock_sdf)
    return top


# ---------------------------------------------------------------------------
# SDF parsing + RMSD
# ---------------------------------------------------------------------------

_GNINA_PROP_KEYS = ("minimizedAffinity", "CNNscore", "CNNaffinity", "CNN_VS", "minimizedRMSD")


def parse_poses_sdf(path: PathLike) -> pd.DataFrame:
    """Read a gnina-output SDF; one row per pose.

    gnina writes SD tags on every pose. The ones we extract:

    - ``minimizedAffinity``  — Vina/smina affinity in kcal/mol (lower = stronger).
                               Always present.
    - ``CNNscore``           — CNN classification score, 0–1 (higher = better).
                               Always present when ``--cnn_scoring`` ≠ ``none``.
    - ``CNNaffinity``        — CNN-predicted affinity in pK_d (higher = stronger).
                               Always present when ``--cnn_scoring`` ≠ ``none``.
    - ``CNN_VS``             — combined virtual-screening score (higher = better).
                               Always present when ``--cnn_scoring`` ≠ ``none``.
    - ``minimizedRMSD``      — RMSD between the input pose and gnina's
                               iteratively-minimised pose. **Only emitted under
                               ``--cnn_scoring refinement`` or ``--cnn_scoring all``
                               (the CNN-guided iterative-refinement modes). Under
                               the default ``--cnn_scoring rescore`` (Vina docks +
                               CNN scores once at the end, no refinement), gnina
                               does not write this tag and the column is None for
                               every row.** Not a bug — a consequence of the speed /
                               accuracy trade-off we chose.

    The returned DataFrame has the columns above (renamed to snake_case),
    plus ``compound_id`` (the SDF ``_Name``) and ``pose_rank`` (1-based, per
    compound, matching the order gnina writes them).
    """
    rows: list[dict] = []
    counts: dict[str, int] = {}
    sup = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in sup:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        counts[name] = counts.get(name, 0) + 1
        rows.append(
            {
                "compound_id":    name,
                "pose_rank":      counts[name],
                "affinity":       _float_prop(mol, "minimizedAffinity"),
                "cnn_score":      _float_prop(mol, "CNNscore"),
                "cnn_affinity":   _float_prop(mol, "CNNaffinity"),
                "cnn_vs":         _float_prop(mol, "CNN_VS"),
                "minimized_rmsd": _float_prop(mol, "minimizedRMSD"),
            }
        )
    return pd.DataFrame(rows)


def _format_gnina_error(returncode: int, log_path: Path) -> str:
    """Build a debug-friendly message from a non-zero gnina exit.

    Exit code 127 from an existing executable typically means a missing shared
    library at dynamic-load time (e.g. ``libcuda.so.1`` on a CPU-only runtime);
    we call that out explicitly. For other non-zero exits we tail the log so
    the user sees the actual gnina stderr without having to navigate to it.
    """
    tail = ""
    try:
        with log_path.open() as fh:
            lines = fh.readlines()
        tail = "".join(lines[-30:]).rstrip()
    except OSError:
        pass

    hint = ""
    if returncode == 127:
        hint = (
            "\n\nExit code 127 usually means gnina is present but a shared "
            "library it links against is missing at load time. Most common "
            "cause: gnina v1.3+ is CUDA-linked (PyTorch backend) and the "
            "Colab runtime is CPU-only. Switch the runtime to a T4 GPU "
            "(Runtime → Change runtime type → T4 GPU) and re-run."
        )
    return (
        f"gnina exited with status {returncode}. Full log at {log_path}. "
        f"Last lines:\n{tail}{hint}"
    )


def _float_prop(mol: Chem.Mol, key: str) -> float | None:
    if not mol.HasProp(key):
        return None
    try:
        return float(mol.GetProp(key))
    except ValueError:
        return None


def pose_rmsd(
    pose_path: PathLike,
    ref_path: PathLike,
    *,
    pose_index: int = 0,
) -> float:
    """Heavy-atom symmetry-corrected RMSD between a docked pose and a reference.

    Both files must hold the same chemical structure (matching heavy-atom
    composition) — typical for a redock where the input ligand IS the
    reference. Hydrogens are stripped before RMSD (heavy-atom RMSD is the
    convention in docking literature).

    Cross-format robustness: PDB-derived molecules have *heuristic* bond
    orders that often disagree with the SDF-derived pose's bond orders, even
    when the underlying chemistry is identical. ``rdMolAlign.CalcRMS``
    requires substructure matching including bond orders, so the naive call
    fails with "no sub-structure match" across SDF↔PDB. We work around it in
    two layers:

    1. **Transfer bond orders.** Use ``AllChem.AssignBondOrdersFromTemplate``
       to copy the SDF pose's bond orders onto the PDB reference. The
       reference now matches the pose's connectivity and ``CalcRMS`` works.
    2. **Element-graph fallback.** If the template assignment fails (e.g.
       heavy-atom counts differ), normalise both molecules to single bonds
       + neutral charges (preserving 3-D coordinates) so the substructure
       match relies purely on the element graph, then call ``CalcRMS``.

    Coordinates are not rigid-body-realigned before the RMSD — gnina already
    writes poses in the receptor's coordinate frame, which is the same frame
    the reference ligand is in.
    """
    pose_mol = _read_first_mol(Path(pose_path), index=pose_index)
    ref_mol = _read_first_mol(Path(ref_path), index=0)
    pose_mol = Chem.RemoveHs(pose_mol)
    ref_mol = Chem.RemoveHs(ref_mol)

    try:
        ref_with_pose_bonds = AllChem.AssignBondOrdersFromTemplate(pose_mol, ref_mol)
        return float(rdMolAlign.CalcRMS(pose_mol, ref_with_pose_bonds))
    except (ValueError, RuntimeError):
        logger.info(
            "pose_rmsd: AssignBondOrdersFromTemplate failed; falling back to "
            "element-graph matching."
        )
        return float(
            rdMolAlign.CalcRMS(_element_skeleton(pose_mol), _element_skeleton(ref_mol))
        )


def _element_skeleton(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of ``mol`` with all bond orders set to single and charges
    cleared, so substructure matching depends only on the heavy-atom element
    graph. 3-D coordinates are preserved untouched.
    """
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    skel = rw.GetMol()
    skel.UpdatePropertyCache(strict=False)
    return skel


def _read_first_mol(path: Path, *, index: int = 0) -> Chem.Mol:
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".sd"}:
        sup = Chem.SDMolSupplier(str(path), removeHs=False)
        mols = [m for m in sup if m is not None]
        if not mols:
            raise ValueError(f"No molecules in {path}")
        if index >= len(mols):
            raise IndexError(f"Pose index {index} out of range ({len(mols)} poses in {path})")
        return mols[index]
    if suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}")
    if mol is None:
        raise ValueError(f"RDKit could not parse {path}")
    return mol


# ---------------------------------------------------------------------------
# PoseBusters QC
# ---------------------------------------------------------------------------

def run_posebusters(
    poses_sdf: PathLike,
    receptor: PathLike | None = None,
    *,
    mode: str = "dock",
    full_report: bool = False,
) -> pd.DataFrame:
    """Run PoseBusters on a docking output SDF; return one row per pose.

    PoseBusters runs a battery of stereochemistry / geometry / clash checks
    on each pose. We add two convenience columns:

    - ``pb_passes_all``  — True iff every individual check passed.
    - ``first_failing``  — name of the first failing check (or ``""``).

    ``mode='dock'`` is the standard "no crystallographic ground-truth, just
    the receptor" mode. ``mode='redock'`` adds RMSD-against-true checks but
    needs a known native pose. ``mode='mol'`` skips receptor-dependent checks
    (used for ligand-only screens).

    RDKit's per-molecule SDF-parsing warnings are silenced at the module
    level (see top of file) so the per-pose noise from PoseBusters' internal
    SDMolSupplier calls doesn't drown the cell output.
    """
    from posebusters import PoseBusters  # imported lazily; pip-installed

    pb = PoseBusters(config=mode)
    if receptor is not None and mode != "mol":
        df = pb.bust(mol_pred=str(poses_sdf), mol_cond=str(receptor), full_report=full_report)
    else:
        df = pb.bust(mol_pred=str(poses_sdf), full_report=full_report)

    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    if bool_cols:
        df = df.copy()
        df["pb_passes_all"] = df[bool_cols].all(axis=1)
        df["first_failing"] = df[bool_cols].apply(
            lambda row: next((c for c in bool_cols if not row[c]), ""),
            axis=1,
        )
    return df
