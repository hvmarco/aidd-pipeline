"""Boltz-2 co-folding + affinity prediction (the fast lane).

Wraps the ``boltz`` CLI to predict a protein-ligand complex and a learned
affinity for one (sequence, SMILES) pair at a time. The library-level helper
loops over compounds with **per-compound caching** so a Colab disconnect
mid-screen does not cost the run; re-running picks up cached compounds and
only dispatches Boltz for the rest.

The complementary "fast lane" to ``aidd.docking`` (the gnina-driven
interpretable lane). The aggregate ``affinity.csv`` emitted at the end of a
``predict_library`` run is **join-compatible** with the docking lane's
``scored_poses.parquet`` on ``compound_id``, which is the contract step 12
(consensus) relies on.

Failure policy and the on-disk layout deliberately mirror
:func:`aidd.docking.dock_library` so both lanes have the same operational
shape (resumability, ``<id>.FAILED`` markers, ``failures.csv``,
``on_failure="raise"|"skip"|"skip_with_guard"``). Error-class buckets are
Boltz-2-specific: ``boltz_oom``, ``boltz_msa_failed``,
``boltz_co_fold_diverged``, ``boltz_install_error``, and a generic
``boltz_nonzero_exit``.

Cross-platform notes
--------------------
- Boltz-2 is GPU-mandatory (~5 GB checkpoint). The helpers refuse on
  non-Linux hosts with a Colab-install hint; the consuming notebook is
  therefore Colab-first.
- Per-compound inference dominates wall time. Never quote a runtime in
  notebook markdown without measuring on the actual runtime tier first
  (see ``feedback_measure_before_rationale.md``).

Upstream
--------
- Repo:  https://github.com/jwohlwend/boltz
- Paper: Wohlwend et al. 2025
- This wrapper assumes Boltz-2 v2.x default output layout (one MMCIF complex
  + ``affinity_*.json`` + ``confidence_*.json`` per prediction).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence, Union

import pandas as pd
from rdkit import Chem

PathLike = Union[str, Path]
logger = logging.getLogger("aidd.co_folding")

# Boltz-2 PyPI version pin. PyPI releases are immutable, so this is
# reproducible across re-installs. The notebook's setup cell installs from
# PyPI by this version; tighten to a git commit pin if a tagged release ever
# proves unreliable upstream (the same escape hatch the ColabFold install
# uses — see feedback_colabfold_install_mirror.md).
BOLTZ2_VERSION = "2.2.1"
BOLTZ2_LAST_VERIFIED: str | None = None  # bumped after each successful Colab smoke-test

PROTEIN_CHAIN_ID = "A"
LIGAND_CHAIN_ID = "B"

OnFailure = Literal["raise", "skip", "skip_with_guard"]
_UNSAFE_ID_CHARS = set('/\\:*?"<>|\t\r\n')


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoltzResult:
    """Affinity + confidence numbers parsed out of a Boltz-2 prediction.

    Field meanings (per Boltz-2's published output spec):

    - ``affinity``              — ``affinity_pred_value`` from Boltz; the
                                  learned affinity score. Higher = stronger
                                  predicted binder, in pIC50-ish units.
    - ``affinity_probability``  — ``affinity_probability_binary`` (0..1).
                                  Boltz-2's "is this molecule a binder"
                                  classifier head.
    - ``confidence``            — ``confidence_score`` for the complex.
    - ``iptm``                  — interface predicted-TM. Whole-complex.
    - ``ligand_iptm``           — iPTM restricted to the ligand interface
                                  (often more discriminating than overall
                                  iptm for small-molecule binding).
    - ``complex_cif``           — path to the predicted complex MMCIF on disk.
    """

    compound_id: str
    affinity: float
    affinity_probability: float
    confidence: float
    iptm: float
    ligand_iptm: float
    complex_cif: Path

    def as_row(self) -> dict:
        return {
            "compound_id":                 self.compound_id,
            "boltz_affinity":              self.affinity,
            "boltz_affinity_probability":  self.affinity_probability,
            "boltz_confidence":            self.confidence,
            "boltz_iptm":                  self.iptm,
            "boltz_ligand_iptm":           self.ligand_iptm,
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def boltz_path() -> Path | None:
    """Return the ``boltz`` CLI on PATH, or ``None`` if not installed."""
    p = shutil.which("boltz")
    return Path(p) if p else None


def boltz_available() -> bool:
    return boltz_path() is not None


def describe_environment() -> dict:
    """Report the Boltz-2-relevant environment state for sanity printing."""
    b = boltz_path()
    info: dict = {
        "platform":         platform.system(),
        "boltz_path":       str(b) if b else None,
        "boltz_version":    None,
        "expected_version": BOLTZ2_VERSION,
    }
    if b is not None:
        try:
            out = subprocess.run(
                [str(b), "--version"], capture_output=True, text=True, timeout=30
            )
            text = out.stdout or out.stderr
            info["boltz_version"] = text.strip().splitlines()[0] if text else None
        except Exception as exc:
            info["boltz_version"] = f"(probe failed: {exc})"
    return info


def require_boltz() -> Path:
    """Return the boltz CLI or raise with a platform-appropriate install hint."""
    b = boltz_path()
    if b is not None:
        return b
    system = platform.system()
    if system == "Linux":
        msg = (
            "boltz CLI not found on PATH. On Colab / Linux install with:\n"
            f"    pip install boltz=={BOLTZ2_VERSION}\n"
            "and run on a GPU runtime."
        )
    elif system == "Darwin":
        msg = (
            "Boltz-2 is GPU-mandatory and not practical on macOS (no CUDA). "
            "Run this notebook on Google Colab — the setup cell installs Boltz-2."
        )
    else:  # Windows
        msg = (
            "Boltz-2 is GPU-mandatory and not natively supported on Windows. "
            "Run this notebook on Google Colab (recommended) or under WSL2 + CUDA "
            f"with `pip install boltz=={BOLTZ2_VERSION}`."
        )
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------------

def _build_boltz_yaml(sequence: str, smiles: str, *, use_msa_server: bool) -> str:
    """Return a Boltz-2 input YAML for one protein + one ligand + affinity head.

    The ligand chain ID is fixed to :data:`LIGAND_CHAIN_ID` so the
    ligand-extraction helper knows where to look in the predicted CIF.
    """
    msa_line = "" if use_msa_server else f"      msa: empty\n"
    return (
        "version: 1\n"
        "sequences:\n"
        f"  - protein:\n"
        f"      id: {PROTEIN_CHAIN_ID}\n"
        f"      sequence: {sequence}\n"
        f"{msa_line}"
        f"  - ligand:\n"
        f"      id: {LIGAND_CHAIN_ID}\n"
        f"      smiles: '{smiles}'\n"
        "properties:\n"
        "  - affinity:\n"
        f"      binder: {LIGAND_CHAIN_ID}\n"
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _find_prediction_dir(boltz_out_root: Path) -> Path:
    """Locate Boltz-2's prediction sub-directory under its output root.

    Boltz writes to
    ``<out>/boltz_results_<input_stem>/predictions/<input_stem>/``. We use
    one input YAML per compound so there is exactly one prediction dir
    under ``boltz_out_root`` — anything else indicates stale state.
    """
    candidates = list(boltz_out_root.glob("boltz_results_*/predictions/*"))
    if not candidates:
        raise FileNotFoundError(
            f"No Boltz-2 prediction directory under {boltz_out_root}. "
            "The CLI may have failed silently; check the per-compound boltz.log."
        )
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Expected one prediction directory under {boltz_out_root}, found "
            f"{len(candidates)}. Stale state from a previous run."
        )
    return candidates[0]


def _parse_boltz_outputs(prediction_dir: Path) -> dict:
    """Read affinity + confidence + complex CIF from a Boltz-2 prediction dir."""
    aff_files = sorted(prediction_dir.glob("affinity_*.json"))
    if not aff_files:
        raise FileNotFoundError(f"No affinity_*.json under {prediction_dir}.")
    conf_files = sorted(prediction_dir.glob("confidence_*.json"))
    if not conf_files:
        raise FileNotFoundError(f"No confidence_*.json under {prediction_dir}.")
    cif_files = sorted(prediction_dir.glob("*_model_0.cif"))
    if not cif_files:
        cif_files = sorted(prediction_dir.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No predicted CIF under {prediction_dir}.")

    aff = json.loads(aff_files[0].read_text())
    conf = json.loads(conf_files[0].read_text())
    return {
        "affinity":             float(aff.get("affinity_pred_value", float("nan"))),
        "affinity_probability": float(aff.get("affinity_probability_binary", float("nan"))),
        "confidence":           float(conf.get("confidence_score", float("nan"))),
        "iptm":                 float(conf.get("iptm", float("nan"))),
        "ligand_iptm":          float(conf.get("ligand_iptm", float("nan"))),
        "complex_plddt":        float(conf.get("complex_plddt", float("nan"))),
        "complex_cif":          cif_files[0],
        "affinity_json":        aff_files[0],
        "confidence_json":      conf_files[0],
    }


def _classify_error(returncode: int, log_tail: str) -> str:
    """Bucket a Boltz-2 failure into one of the per-failure taxonomy classes.

    The taxonomy starts conservative; widen the buckets as real failures from
    Colab runs surface patterns that aren't covered here.

    Buckets, in priority order:

    - ``boltz_input_invalid``    — Boltz's RDKit pipeline rejected the SMILES
                                   (kekulization, valence, fragment-chooser).
                                   The CLI prints ``Failed to process … Skipping``
                                   and exits 0, which is why we explicitly look
                                   for this pattern: returncode alone won't catch it.
    - ``boltz_oom``              — out-of-memory; OOM kill (137) or "CUDA out of memory".
    - ``boltz_msa_failed``       — MSA backend failure.
    - ``boltz_co_fold_diverged`` — NaN / inf / model divergence.
    - ``boltz_install_error``    — weight load failure, missing CUDA libs at startup.
    - ``boltz_nonzero_exit``     — generic catch-all.
    """
    t = log_tail.lower()
    if "failed to process" in t or "skipping. error" in t or "kekuliz" in t:
        return "boltz_input_invalid"
    if returncode == 137 or "out of memory" in t or "cuda oom" in t:
        return "boltz_oom"
    if "mmseqs" in t or "msa server" in t or "alphafold msa" in t:
        return "boltz_msa_failed"
    if "nan" in t and ("loss" in t or "logit" in t):
        return "boltz_co_fold_diverged"
    if (
        "could not load" in t
        or "cuda library" in t
        or ("weight" in t and "load" in t)
    ):
        return "boltz_install_error"
    return "boltz_nonzero_exit"


def _format_boltz_error(returncode: int, log_path: Path) -> str:
    tail = _read_log_tail(log_path, n=30)
    hint = ""
    if returncode == 137:
        hint = (
            "\n\nExit code 137 usually means the process was killed by the OS "
            "for using too much memory (CUDA OOM on the GPU, or RAM exhaustion). "
            "Try a larger GPU runtime tier, or a smaller protein / ligand."
        )
    return (
        f"boltz exited with status {returncode}. Full log at {log_path}. "
        f"Last lines:\n{tail}{hint}"
    )


def _format_boltz_input_error(log_text: str, log_path: Path) -> str:
    """Build a debug-friendly message for a silently-skipped Boltz input.

    Boltz-2's CLI exits with status 0 even when it logs ``Failed to process
    … Skipping. Error: <reason>`` for an input its RDKit pipeline rejected.
    Find that line in the log and surface it as the error summary so the
    failure marker carries the real cause.
    """
    for line in log_text.splitlines():
        s = line.strip()
        if "Failed to process" in s or "Skipping. Error:" in s:
            return f"Boltz silently skipped the input (status 0). {s} Full log at {log_path}."
    return f"Boltz silently skipped the input (status 0, no clear cause). Full log at {log_path}."


# ---------------------------------------------------------------------------
# Public API — single compound
# ---------------------------------------------------------------------------

def predict_complex(
    sequence: str,
    smiles: str,
    output_dir: PathLike,
    *,
    compound_id: str | None = None,
    use_msa_server: bool = True,
    cache: bool = True,
    boltz_extra: Sequence[str] = (),
) -> BoltzResult:
    """Co-fold one (protein, ligand) pair with Boltz-2.

    ``output_dir`` is the *compound-specific* directory (typically
    ``per_compound/<id>/`` under a library run). On a cache hit — when
    ``metadata.json`` and ``complex.cif`` exist and the saved (sequence,
    SMILES) hashes match the current request — the cached result is returned
    without invoking the CLI.

    Parameters
    ----------
    compound_id
        Used as the input YAML stem (Boltz-2 echoes it into output filenames).
        Defaults to the first 16 hex chars of SHA-256(smiles).
    use_msa_server
        When True (default), Boltz-2 queries its remote MSA backend. Set to
        False for single-sequence mode (faster, less accurate for protein
        modelling but unaffected for ligand placement on a well-folded target).
    cache
        When True (default), an existing matching cache short-circuits.
    boltz_extra
        Extra raw CLI flags appended after the standard arguments.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    affinity_path = output_dir / "affinity.json"
    confidence_path = output_dir / "confidence.json"
    complex_path = output_dir / "complex.cif"
    boltz_log = output_dir / "boltz.log"
    cid = compound_id or hashlib.sha256(smiles.encode()).hexdigest()[:16]

    cur_hashes = _hash_inputs(sequence, smiles)

    if cache and metadata_path.exists() and complex_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            if (
                meta.get("sequence_sha256") == cur_hashes["sequence_sha256"]
                and meta.get("smiles_sha256") == cur_hashes["smiles_sha256"]
            ):
                return BoltzResult(
                    compound_id=cid,
                    affinity=float(meta["affinity"]),
                    affinity_probability=float(meta["affinity_probability"]),
                    confidence=float(meta["confidence"]),
                    iptm=float(meta["iptm"]),
                    ligand_iptm=float(meta["ligand_iptm"]),
                    complex_cif=complex_path,
                )
        except (OSError, json.JSONDecodeError, KeyError):
            pass  # corrupt metadata -> fall through and re-run

    b = require_boltz()

    # Use mkdtemp (not TemporaryDirectory) so the tempdir survives on failure
    # for forensic inspection -- a silent Boltz skip writes nothing to the
    # output dir except boltz.log, and the tempdir contents may carry the
    # only on-disk evidence of what went wrong. Only clean up on success.
    tmp_path = Path(tempfile.mkdtemp(prefix="boltz_in_", dir=output_dir))
    cleanup_tmp = False
    try:
        input_yaml = tmp_path / f"{cid}.yaml"
        input_yaml.write_text(_build_boltz_yaml(sequence, smiles, use_msa_server=use_msa_server))
        boltz_out_root = tmp_path / "out"
        boltz_out_root.mkdir(exist_ok=True)

        cmd: list[str] = [
            str(b), "predict", str(input_yaml),
            "--out_dir", str(boltz_out_root),
        ]
        if use_msa_server:
            cmd.append("--use_msa_server")
        cmd += list(boltz_extra)

        with boltz_log.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(_format_boltz_error(proc.returncode, boltz_log))

        # Boltz-2's CLI exits 0 even when it skips an input it can't process
        # ("Failed to process … Skipping. Error: <reason>"). Catch this here
        # so the outer loop's on_failure path classifies it cleanly instead
        # of letting _find_prediction_dir raise FileNotFoundError.
        log_text = boltz_log.read_text() if boltz_log.exists() else ""
        if "Failed to process" in log_text or "Skipping. Error:" in log_text:
            raise RuntimeError(_format_boltz_input_error(log_text, boltz_log))

        pred_dir = _find_prediction_dir(boltz_out_root)
        parsed = _parse_boltz_outputs(pred_dir)

        shutil.copy2(parsed["complex_cif"], complex_path)
        shutil.copy2(parsed["affinity_json"], affinity_path)
        shutil.copy2(parsed["confidence_json"], confidence_path)
        cleanup_tmp = True
    finally:
        if cleanup_tmp:
            shutil.rmtree(tmp_path, ignore_errors=True)

    metadata = {
        **cur_hashes,
        "compound_id":          cid,
        "boltz_version":        BOLTZ2_VERSION,
        "use_msa_server":       use_msa_server,
        "affinity":             parsed["affinity"],
        "affinity_probability": parsed["affinity_probability"],
        "confidence":           parsed["confidence"],
        "iptm":                 parsed["iptm"],
        "ligand_iptm":          parsed["ligand_iptm"],
        "complex_plddt":        parsed["complex_plddt"],
        "timestamp":            _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return BoltzResult(
        compound_id=cid,
        affinity=parsed["affinity"],
        affinity_probability=parsed["affinity_probability"],
        confidence=parsed["confidence"],
        iptm=parsed["iptm"],
        ligand_iptm=parsed["ligand_iptm"],
        complex_cif=complex_path,
    )


# ---------------------------------------------------------------------------
# Public API — library
# ---------------------------------------------------------------------------

def predict_library(
    sequence: str,
    ligands: Union[pd.DataFrame, PathLike],
    out_dir: PathLike,
    *,
    smiles_col: str = "smiles",
    id_col: str = "compound_id",
    use_msa_server: bool = True,
    overwrite: bool = False,
    progress: bool = True,
    on_failure: OnFailure = "skip_with_guard",
    failure_guard_n: int = 5,
    sleep_between: float = 0.0,
    boltz_extra: Sequence[str] = (),
) -> pd.DataFrame:
    """Co-fold every (sequence, ligand) pair in ``ligands`` with Boltz-2.

    **Resumable per-compound.** Each compound writes its outputs to
    ``out_dir/per_compound/<id>/`` before the next compound starts; a
    runtime disconnect does not cost the run. Re-running picks up cached
    compounds and only dispatches Boltz for those without a saved record.

    Outputs in ``out_dir/``
    ----------------------
    - ``affinity.csv``        — the contract step 12 (consensus) joins
                                against ``scored_poses.parquet`` on
                                ``compound_id``. Columns: ``compound_id,
                                boltz_affinity, boltz_affinity_probability,
                                boltz_confidence, boltz_iptm,
                                boltz_ligand_iptm``.
    - ``poses.sdf``           — predicted ligand poses, best-effort
                                extracted from each ``complex.cif`` by
                                filtering to chain ``B``. Skipped compounds
                                are silently absent; the per-compound CIF
                                is always available for direct 3D viewing.
    - ``failures.csv``        — one row per failed compound (``compound_id,
                                error_class, error_summary, boltz_log_tail,
                                timestamp``). Written even when empty.
    - ``run.log``             — per-compound timing summary, one line each.
    - ``per_compound/<id>/``  — success: ``complex.cif``, ``affinity.json``,
                                ``confidence.json``, ``metadata.json``,
                                ``boltz.log``.
    - ``per_compound/<id>.FAILED`` — JSON marker for a failed compound.
                                Its presence tells the cache check
                                "don't retry". To force a retry, delete
                                the marker and re-run.

    Parameters
    ----------
    ligands
        Either a DataFrame with at least ``id_col`` and ``smiles_col``
        columns, or a path to a prepared SDF (each molecule's ``_Name`` is
        the compound_id; the SMILES is recovered with ``MolToSmiles``).
    on_failure, failure_guard_n
        Same semantics and defaults as :func:`aidd.docking.dock_library`.
        Boltz-2's error taxonomy: ``boltz_oom``, ``boltz_msa_failed``,
        ``boltz_co_fold_diverged``, ``boltz_install_error``,
        ``boltz_nonzero_exit``. The classifier inspects the per-compound
        ``boltz.log`` tail.
    sleep_between
        Seconds to pause between compounds. Useful for rate-limit-friendly
        runs when ``use_msa_server=True`` (the MSA backend is shared
        infrastructure). Default 0.

    Notes
    -----
    To force a retry of a previously-failed compound, delete its
    ``per_compound/<id>.FAILED`` marker and re-run; the cache check will
    see no record and re-run Boltz-2 for it.
    """
    if on_failure not in {"raise", "skip", "skip_with_guard"}:
        raise ValueError(
            f"on_failure must be 'raise', 'skip', or 'skip_with_guard'; got {on_failure!r}."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_compound_root = out_dir / "per_compound"
    per_compound_root.mkdir(exist_ok=True)
    affinity_csv = out_dir / "affinity.csv"
    poses_sdf = out_dir / "poses.sdf"
    failures_csv = out_dir / "failures.csv"
    run_log = out_dir / "run.log"

    rows = list(_iter_ligands(ligands, smiles_col=smiles_col, id_col=id_col))
    if not rows:
        raise ValueError(f"No (compound_id, smiles) pairs parsed from {ligands!r}.")

    # Defer require_boltz() until we actually need it — re-running against a
    # fully-cached output_dir on Windows must work without the boltz CLI.
    boltz_ready = False
    n_cached_ok = n_cached_failed = n_fresh_ok = n_fresh_failed = 0
    consecutive_fresh_failures = 0
    run_log_lines: list[str] = []

    def _flush_run_log() -> None:
        """Persist per-compound timing after every iteration so partial
        progress survives any uncategorised crash (the timing data we lost
        on Colab attempt #3 when an uncaught FileNotFoundError broke the
        loop before the end-of-function write was reached)."""
        run_log.write_text("\n".join(run_log_lines) + ("\n" if run_log_lines else ""))

    iterator = _maybe_tqdm(rows, total=len(rows), desc="boltz2 cofold", enable=progress)
    for compound_id, smiles in iterator:
        _check_compound_id(compound_id)
        cdir = per_compound_root / compound_id
        failed_marker = per_compound_root / f"{compound_id}.FAILED"

        if not overwrite and failed_marker.exists():
            n_cached_failed += 1
            run_log_lines.append(f"{compound_id}\tcached_failed")
            _flush_run_log()
            continue
        if not overwrite and _has_cached_success(cdir, sequence=sequence, smiles=smiles):
            n_cached_ok += 1
            consecutive_fresh_failures = 0
            run_log_lines.append(f"{compound_id}\tcached_success")
            _flush_run_log()
            continue

        if not boltz_ready:
            require_boltz()
            boltz_ready = True

        t_start = _dt.datetime.now(_dt.timezone.utc)
        try:
            predict_complex(
                sequence,
                smiles,
                cdir,
                compound_id=compound_id,
                use_msa_server=use_msa_server,
                cache=False,
                boltz_extra=boltz_extra,
            )
        except RuntimeError as exc:
            dt = (_dt.datetime.now(_dt.timezone.utc) - t_start).total_seconds()
            run_log_lines.append(f"{compound_id}\tfresh_failed\t{dt:.1f}s")
            _flush_run_log()
            if on_failure == "raise":
                raise
            log_tail = _read_log_tail(cdir / "boltz.log", n=30)
            error_class = _classify_error(_extract_returncode(exc), log_tail)
            _persist_failure(
                failed_marker,
                compound_id=compound_id,
                error_class=error_class,
                error_summary=(str(exc).splitlines()[0] if str(exc) else "boltz failed"),
                boltz_log_tail=log_tail,
            )
            n_fresh_failed += 1
            consecutive_fresh_failures += 1
            logger.warning(
                "Boltz-2 failed on compound %s (%s) — skipping; marker at %s",
                compound_id, error_class, failed_marker,
            )
            if (
                on_failure == "skip_with_guard"
                and consecutive_fresh_failures >= failure_guard_n
            ):
                raise RuntimeError(
                    f"Boltz-2 has failed on {consecutive_fresh_failures} consecutive "
                    f"compounds (guard threshold = {failure_guard_n}). This usually "
                    f"means a systemic problem (broken install, GPU OOM at this batch "
                    f"size, MSA server down, missing CUDA library) rather than per-"
                    f"compound chemistry. Last error:\n\n{exc}\n\n"
                    f"Per-compound markers in {per_compound_root}/*.FAILED; resolve "
                    f"the root cause, delete the marker files for the compounds you "
                    f"want to retry, and re-run."
                )
            continue
        else:
            dt = (_dt.datetime.now(_dt.timezone.utc) - t_start).total_seconds()
            run_log_lines.append(f"{compound_id}\tfresh_success\t{dt:.1f}s")
            _flush_run_log()
            n_fresh_ok += 1
            consecutive_fresh_failures = 0

        if sleep_between > 0:
            import time as _time
            _time.sleep(sleep_between)

    logger.info(
        "predict_library: %d cached-success, %d cached-failed, %d fresh-success, "
        "%d fresh-failed (total %d compounds).",
        n_cached_ok, n_cached_failed, n_fresh_ok, n_fresh_failed, len(rows),
    )
    _flush_run_log()  # final flush (per-iteration flushes have it covered; redundancy is cheap)

    scope_ids = [cid for cid, _ in rows]
    df = _aggregate_affinity(per_compound_root, affinity_csv, scope_ids=scope_ids)
    _aggregate_failures(per_compound_root, failures_csv, scope_ids=scope_ids)
    _aggregate_poses(per_compound_root, poses_sdf, scope_ids=scope_ids)
    return df


def summarise_run(out_dir: PathLike) -> dict:
    """One-row summary of a predict_library run, for the notebook recap cell."""
    out_dir = Path(out_dir)
    affinity_csv = out_dir / "affinity.csv"
    failures_csv = out_dir / "failures.csv"
    n_predicted = len(pd.read_csv(affinity_csv)) if affinity_csv.exists() else 0
    by_class: dict[str, int] = {}
    n_failed = 0
    if failures_csv.exists():
        f = pd.read_csv(failures_csv)
        n_failed = len(f)
        if n_failed:
            by_class = f["error_class"].value_counts().to_dict()
    return {
        "out_dir":           str(out_dir),
        "n_predicted":       n_predicted,
        "n_failed":          n_failed,
        "failures_by_class": by_class,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_ligands(
    ligands: Union[pd.DataFrame, PathLike],
    *,
    smiles_col: str,
    id_col: str,
) -> Iterable[tuple[str, str]]:
    if isinstance(ligands, pd.DataFrame):
        missing = {smiles_col, id_col} - set(ligands.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}.")
        for _, row in ligands.iterrows():
            yield str(row[id_col]), str(row[smiles_col])
        return
    path = Path(ligands)
    if not path.exists():
        raise FileNotFoundError(path)
    sup = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in sup:
        if mol is None:
            continue
        cid = (mol.GetProp("_Name") if mol.HasProp("_Name") else "").strip()
        yield cid, Chem.MolToSmiles(mol)


def _has_cached_success(cdir: Path, *, sequence: str, smiles: str) -> bool:
    metadata = cdir / "metadata.json"
    if not metadata.exists() or not (cdir / "complex.cif").exists():
        return False
    try:
        meta = json.loads(metadata.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    cur = _hash_inputs(sequence, smiles)
    return (
        meta.get("sequence_sha256") == cur["sequence_sha256"]
        and meta.get("smiles_sha256") == cur["smiles_sha256"]
    )


def _hash_inputs(sequence: str, smiles: str) -> dict:
    return {
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "smiles_sha256":   hashlib.sha256(smiles.encode()).hexdigest(),
    }


def _check_compound_id(cid: str) -> None:
    if not cid:
        raise ValueError("compound_id must not be empty.")
    bad = sorted({c for c in cid if c in _UNSAFE_ID_CHARS})
    if bad:
        raise ValueError(
            f"compound_id {cid!r} contains filesystem-unsafe characters {bad}. "
            "predict_library writes a per-compound subdirectory named after "
            "the compound_id and needs IDs that are valid path components."
        )


def _persist_failure(
    marker_path: Path,
    *,
    compound_id: str,
    error_class: str,
    error_summary: str,
    boltz_log_tail: str,
) -> None:
    payload = {
        "compound_id":    compound_id,
        "error_class":    error_class,
        "error_summary":  error_summary,
        "boltz_log_tail": boltz_log_tail,
        "timestamp":      _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
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


def _extract_returncode(exc: BaseException) -> int:
    """Best-effort recovery of the boltz returncode from the exception text."""
    m = re.search(r"status (-?\d+)", str(exc))
    return int(m.group(1)) if m else -1


def _aggregate_affinity(
    per_compound_root: Path,
    affinity_csv: Path,
    *,
    scope_ids: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for cid in scope_ids:
        meta_path = per_compound_root / cid / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({
            "compound_id":                cid,
            "boltz_affinity":             meta.get("affinity", float("nan")),
            "boltz_affinity_probability": meta.get("affinity_probability", float("nan")),
            "boltz_confidence":           meta.get("confidence", float("nan")),
            "boltz_iptm":                 meta.get("iptm", float("nan")),
            "boltz_ligand_iptm":          meta.get("ligand_iptm", float("nan")),
        })
    df = pd.DataFrame(
        rows,
        columns=[
            "compound_id", "boltz_affinity", "boltz_affinity_probability",
            "boltz_confidence", "boltz_iptm", "boltz_ligand_iptm",
        ],
    )
    df.to_csv(affinity_csv, index=False)
    return df


def _aggregate_failures(
    per_compound_root: Path,
    failures_csv: Path,
    *,
    scope_ids: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for cid in scope_ids:
        marker = per_compound_root / f"{cid}.FAILED"
        if not marker.exists():
            continue
        try:
            payload = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            payload = {
                "compound_id": cid, "error_class": "MarkerParseError",
                "error_summary": str(exc), "boltz_log_tail": "", "timestamp": "",
            }
        rows.append({
            "compound_id":    payload.get("compound_id", cid),
            "error_class":    payload.get("error_class", ""),
            "error_summary":  payload.get("error_summary", ""),
            "boltz_log_tail": payload.get("boltz_log_tail", ""),
            "timestamp":      payload.get("timestamp", ""),
        })
    df = pd.DataFrame(
        rows,
        columns=["compound_id", "error_class", "error_summary", "boltz_log_tail", "timestamp"],
    )
    df.to_csv(failures_csv, index=False)
    return df


def _aggregate_poses(
    per_compound_root: Path,
    poses_sdf: Path,
    *,
    scope_ids: Sequence[str],
) -> None:
    """Extract ligand atoms from each per_compound complex.cif and write one
    multi-compound SDF. Best-effort — compounds whose extraction fails are
    skipped with a logged warning; the 3D-viewer cell in the notebook can
    always fall back to the full complex.cif under per_compound/<id>/.
    """
    ligands: list[Chem.Mol] = []
    for cid in scope_ids:
        cif = per_compound_root / cid / "complex.cif"
        if not cif.exists():
            continue
        mol = _extract_ligand_from_cif(cif, compound_id=cid)
        if mol is None:
            logger.warning("predict_library: could not extract ligand from %s", cif)
            continue
        ligands.append(mol)
    w = Chem.SDWriter(str(poses_sdf))
    for m in ligands:
        w.write(m)
    w.close()


def _extract_ligand_from_cif(cif_path: Path, *, compound_id: str) -> Chem.Mol | None:
    """Read a Boltz-2 predicted complex CIF and return only the ligand atoms.

    We control the input YAML and always put the ligand under chain
    :data:`LIGAND_CHAIN_ID`; Boltz preserves chain IDs in the output CIF,
    so chain filtering is deterministic.
    """
    mol = None
    if hasattr(Chem, "MolFromMmcifFile"):
        mol = Chem.MolFromMmcifFile(str(cif_path), removeHs=False)
    if mol is None and hasattr(Chem, "MolFromMmcif"):
        mol = Chem.MolFromMmcif(str(cif_path))
    if mol is None:
        return None

    rw = Chem.RWMol(mol)
    to_remove: list[int] = []
    for atom in rw.GetAtoms():
        info = atom.GetPDBResidueInfo()
        chain = info.GetChainId() if info is not None else ""
        if chain != LIGAND_CHAIN_ID:
            to_remove.append(atom.GetIdx())
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    ligand = rw.GetMol()
    if ligand.GetNumAtoms() == 0:
        return None
    ligand.SetProp("_Name", compound_id)
    return ligand


def _maybe_tqdm(it, *, total: int, desc: str, enable: bool):
    if not enable:
        return it
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return it
    return tqdm(it, total=total, desc=desc)
