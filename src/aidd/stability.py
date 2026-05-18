"""RaSP variant-stability lookup (lookup-only; no model inference).

RaSP (Blaabjerg et al., eLife 2023; "Rapid Stability Predictions") is an
ML-based predictor of variant stability change (ΔΔG, kcal/mol) for
single-residue protein substitutions. Positive values indicate
destabilising mutations.

The published RaSP install is currently unmaintained: Python 3.6 +
PyTorch 1.2.0 + DSSP / Reduce build deps; the upstream README states
*"Colab no longer supports the dependencies of RaSP and there is
currently no solution in the pipeline."* We sidestep the install
entirely by looking up against the upstream's *precomputed*, saturated
single-residue predictions for the human proteome — the exact same
architectural pattern as :mod:`aidd.variants`'s AlphaMissense lookup.

Helper
------

- :func:`rasp_ddg`(uniprot_id, position, wt_aa, mut_aa) -> float | None
  ΔΔG prediction (kcal/mol). Positive = destabilising; negative =
  stabilising. Returns ``None`` when the variant is not in the cache
  (see *Limits* below).

Two cache files, auto-detected (step 15)
----------------------------------------

The module supports two upstream RaSP data sources. Both produce a small
parquet under ``<cache_dir>/`` filtered to the seven demo-set genes; the
public API is identical for either source. The cache loader prefers the
full-proteome file when present.

- ``full_proteome_demo_set.parquet`` — built by :func:`_build_rasp_full_cache`
  from the upstream's 9 GB AlphaFold-based archive
  (``rasp_preds_alphafold_UP000005640_9606_HUMAN_v2_prism_dir.zip``;
  the ``prism_dir`` suffix in the upstream filename is misleading
  nomenclature — entries are ``.csv`` files, not PRISM ``.txt``;
  see :func:`_parse_rasp_csv_text` for the actual format).
  Covers all 23,391 human SwissProt proteins, so all seven demo genes
  resolve. **One-off operator step per machine**: download the zip,
  pass the path to :func:`_build_rasp_full_cache`, write the small
  resulting parquet to your Drive cache. Once-and-done; subsequent
  notebook runs read the parquet only.
- ``demo_set.parquet`` — built automatically on first call by
  :func:`_build_rasp_demo_cache` from the upstream's 414 MB
  experimental-structures CSV (``rasp_preds_exp_strucs_gnomad_clinvar.csv``).
  Saturated predictions on every human protein with at least one PDB
  structure — but the upstream's selection scope means only NAT2 is
  covered of our seven demo genes (DPYD, CYP2D6, UGT1A1, KRAS, BRCA1,
  and ESR1 all return zero rows; the CSV name is misleading). This
  file is the legacy fallback and is preserved for backwards
  compatibility.

Step 15 added the full-proteome cache and the cache-loader auto-detect.
Step 17 corrected the parser's format assumption — upstream archive ships
CSV per protein (despite the ``prism_dir`` zip name), not PRISM YAML +
columns text. Public API (:func:`rasp_ddg`, :func:`rasp_covers`,
:func:`_build_rasp_full_cache`) and the parquet schema are unchanged.
Old callers see no API change.

Coverage of the seven demo genes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``full_proteome_demo_set.parquet`` present (operator step run
once), all seven demo genes are covered. Without it, only NAT2 is
covered via the legacy ``demo_set.parquet``. The :func:`rasp_covers`
helper exposes coverage programmatically so downstream callers can
branch cleanly.

Honesty / limits
----------------

- **Lookup-only.** Cannot predict ΔΔG for variants on a user-supplied
  custom PDB (e.g., a mutant fold from notebook 01). Predictions are
  fixed to the canonical structures used by the upstream's saturation
  run — AlphaFold models in the full-proteome cache, experimental
  crystals in the legacy demo cache.
- **Position numbering.** RaSP predictions use the residue numbering of
  the structure they were computed on. For the AlphaFold dataset this
  is canonical UniProt numbering by construction. For the legacy
  experimental dataset, UniProt and PDB numbering can differ when the
  structure lacks the N-terminal methionine, has a cleaved signal
  peptide, or has gaps. In both cases the helper requires the caller's
  ``wt_aa`` to match the residue recorded for that position; a
  mismatch returns ``None`` so numbering drift surfaces cleanly rather
  than silently returning the wrong residue's prediction.
- **Multi-row averaging.** When the cache has multiple rows for the
  same (uniprot, position, wt_aa, mut_aa) — e.g., one per chain or per
  crystal structure in the experimental dataset — the helper returns
  the mean. Spread across rows is small in practice (RaSP is
  sequence-conditioned with only mild structural-context dependency at
  most positions).
- **AlphaFold-vs-crystal predictions can differ numerically.** RaSP
  predictions on a protein's AlphaFold model and on its crystal
  structure are similar but not identical — small differences in
  side-chain orientation propagate to small ΔΔG shifts. NAT2 lookups
  served from the full-proteome cache may therefore differ slightly
  from the same lookups served from the legacy cache.
- **The upstream RaSP install is unmaintained.** Re-running the
  saturation prediction locally is not in v1's scope; it would
  require non-trivial dependency rescue work upstream.

Upstream
--------

- RaSP — Blaabjerg et al., eLife 2023 (DOI: 10.7554/eLife.82593).
- Code: https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg
- Predictions: https://sid.erda.dk/sharelink/fFPJWflLeE
"""

from __future__ import annotations

import functools
import logging
import re
import zipfile
from pathlib import Path
from typing import Union

import pandas as pd
import requests

from .variants import DEMO_SET_UNIPROT_IDS, _resolve_cache_dir

PathLike = Union[str, Path]
logger = logging.getLogger("aidd.stability")


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

RASP_URL = (
    "https://sid.erda.dk/share_redirect/fFPJWflLeE/"
    "rasp_preds_exp_strucs_gnomad_clinvar.csv"
)
RASP_DATA_VERSION = "exp_strucs_gnomad_clinvar"

RASP_AF_PRISM_URL = (
    "https://sid.erda.dk/share_redirect/fFPJWflLeE/"
    "rasp_preds_alphafold_UP000005640_9606_HUMAN_v2_prism_dir.zip"
)
RASP_AF_DATA_VERSION = "alphafold_UP000005640_9606_HUMAN_v2_prism_dir"
# Bumped after each successful demo-set cache rebuild on a fresh host.
RASP_LAST_VERIFIED: str | None = None


# ---------------------------------------------------------------------------
# Cache file names
# ---------------------------------------------------------------------------

# Preferred: built by ``_build_rasp_full_cache`` from the AlphaFold archive.
# Covers all seven demo genes. Loader uses this file when present.
FULL_PROTEOME_PARQUET = "full_proteome_demo_set.parquet"
# Legacy: built by ``_build_rasp_demo_cache`` from the experimental CSV.
# Covers NAT2 only. Loader falls back to this when the full file is absent.
DEMO_SET_PARQUET = "demo_set.parquet"


# ===========================================================================
# Public API
# ===========================================================================

def rasp_ddg(
    uniprot_id: str,
    position: int,
    wt_aa: str,
    mut_aa: str,
    *,
    cache_dir: PathLike | None = None,
) -> float | None:
    """ΔΔG (kcal/mol) for a missense substitution. Positive = destabilising.

    Returns ``None`` when:

    - The variant is absent from the precomputed table — most often because
      the protein has no human crystal structure (e.g. UGT1A1) and is
      therefore outside the experimental-structures upstream's coverage.
    - The provided ``wt_aa`` does not match the residue at that position
      in the crystal structure used for the prediction. This indicates
      numbering drift between UniProt and PDB indices; surfaces as ``None``
      rather than silently returning the wrong residue's prediction.
    - The variant is synonymous (``mut_aa == wt_aa``).

    Parameters
    ----------
    uniprot_id
        SwissProt accession (e.g. ``"P11245"`` for NAT2). Must be in
        :data:`aidd.variants.DEMO_SET_UNIPROT_IDS` for the v1 cache.
    position
        Residue position. Note the position-numbering caveat in the
        module docstring.
    wt_aa
        Wild-type amino acid as a 1-letter code (case-insensitive).
        Cross-checked against the cache to surface numbering drift.
    mut_aa
        Substituted amino acid as a 1-letter code (case-insensitive).
    cache_dir
        Override for the RaSP cache directory. When ``None``, defaults to
        ``<cwd>/data/cache/rasp/``. Notebook 07 sets this to a Drive-backed
        path on Colab so the (one-time) ~414 MB download survives runtime
        restarts.

    Raises
    ------
    ValueError
        If ``uniprot_id`` is not in
        :data:`aidd.variants.DEMO_SET_UNIPROT_IDS`. v1 ships a filtered
        cache for the seven demo-set genes only; extend the dict (and
        delete the cache) to extend coverage.
    """
    if uniprot_id not in DEMO_SET_UNIPROT_IDS:
        raise ValueError(
            f"UniProt {uniprot_id!r} is outside the v1 demo-set cache. "
            f"Add it to DEMO_SET_UNIPROT_IDS in src/aidd/variants.py and "
            f"delete the existing cache to trigger a rebuild."
        )
    wt = wt_aa.upper()
    mut = mut_aa.upper()
    if wt == mut:
        return None
    df = _load_rasp_cache(_resolve_cache_dir(cache_dir, "rasp"))
    rows = df[
        (df["uniprot_id"] == uniprot_id)
        & (df["position"] == int(position))
        & (df["wt_aa"] == wt)
        & (df["mut_aa"] == mut)
    ]
    if rows.empty:
        return None
    return float(rows["rasp_ddg"].mean())


def rasp_covers(uniprot_id: str, *, cache_dir: PathLike | None = None) -> bool:
    """Whether the cache has any RaSP predictions for ``uniprot_id``.

    Useful for callers that want to skip the prior cleanly when the
    protein has no experimental structure (e.g. UGT1A1). Cheap once the
    cache is loaded.
    """
    if uniprot_id not in DEMO_SET_UNIPROT_IDS:
        raise ValueError(
            f"UniProt {uniprot_id!r} is outside the v1 demo-set cache."
        )
    df = _load_rasp_cache(_resolve_cache_dir(cache_dir, "rasp"))
    return bool((df["uniprot_id"] == uniprot_id).any())


# ===========================================================================
# Cache loader + builder
# ===========================================================================

@functools.lru_cache(maxsize=4)
def _load_rasp_cache(cache_dir: Path) -> pd.DataFrame:
    """Load (and build-if-missing) the RaSP demo-set parquet.

    Prefers ``full_proteome_demo_set.parquet`` when present (covers all
    seven demo genes via the AlphaFold dataset). Falls back to
    ``demo_set.parquet``, building it from the 414 MB experimental CSV
    on first call (covers NAT2 only).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    full_path = cache_dir / FULL_PROTEOME_PARQUET
    if full_path.exists():
        return pd.read_parquet(full_path)
    parquet_path = cache_dir / DEMO_SET_PARQUET
    if not parquet_path.exists():
        logger.info(
            "RaSP demo cache not found at %s; downloading + filtering from %s "
            "(this is a one-time ~414 MB stream-download with on-the-fly "
            "demo-set filtering; expect 1-3 min depending on connection).",
            parquet_path, RASP_URL,
        )
        _build_rasp_demo_cache(parquet_path)
    return pd.read_parquet(parquet_path)


def _build_rasp_demo_cache(parquet_path: Path) -> None:
    """Stream the RaSP CSV, filter to demo-set, write parquet.

    Memory bounded — only rows whose ``uniprot_id`` is in the demo set
    survive the filter (~tens of thousands of rows total across the
    six covered proteins).
    """
    keep_uniprots = set(DEMO_SET_UNIPROT_IDS.keys())
    rows: list[tuple[str, str, int, str, str, float]] = []

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    with requests.get(RASP_URL, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) or None
        pbar = (
            tqdm(total=total, unit="B", unit_scale=True, desc="RaSP")
            if tqdm is not None else None
        )
        try:
            buffer = b""
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
                if not chunk:
                    continue
                if pbar is not None:
                    pbar.update(len(chunk))
                buffer += chunk
                while b"\n" in buffer:
                    line_b, buffer = buffer.split(b"\n", 1)
                    row = _parse_rasp_csv_line(
                        line_b.decode("utf-8", errors="replace"), keep_uniprots,
                    )
                    if row is not None:
                        rows.append(row)
            if buffer:
                row = _parse_rasp_csv_line(
                    buffer.decode("utf-8", errors="replace"), keep_uniprots,
                )
                if row is not None:
                    rows.append(row)
        finally:
            if pbar is not None:
                pbar.close()

    df = pd.DataFrame(
        rows,
        columns=["uniprot_id", "pdb", "position", "wt_aa", "mut_aa", "rasp_ddg"],
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    covered_uniprots = sorted(df["uniprot_id"].unique())
    missing_uniprots = sorted(keep_uniprots - set(covered_uniprots))
    logger.info(
        "RaSP demo cache written to %s (%d rows across %d proteins). "
        "Covered: %s. Missing (no experimental structure in upstream): %s.",
        parquet_path, len(df), len(covered_uniprots),
        covered_uniprots, missing_uniprots,
    )


def _parse_rasp_csv_line(line: str, keep_uniprots: set[str]) -> tuple | None:
    """Parse one CSV row; return ``None`` to skip the line.

    Format (per upstream's column header):
    ``uniprot_id,pdb_chain,pdb,chain_id,variant,RaSP;score_ml,...``

    ``variant`` is encoded ``<wt><pos><alt>`` (e.g. ``S41A``) or
    ``<wt><pos>=`` for synonymous; we drop the latter.
    """
    if not line or line.startswith("uniprot_id,"):
        return None
    parts = line.split(",")
    if len(parts) < 6:
        return None
    uniprot = parts[0]
    if uniprot not in keep_uniprots:
        return None
    variant = parts[4]
    if not variant or variant.endswith("="):
        return None
    wt_aa = variant[:1]
    mut_aa = variant[-1:]
    try:
        pos = int(variant[1:-1])
        score = float(parts[5])
    except ValueError:
        return None
    return (uniprot, parts[2], pos, wt_aa, mut_aa, score)


# ===========================================================================
# Full-proteome cache builder (AlphaFold-RaSP PRISM archive)
# ===========================================================================

# Upstream RaSP per-UniProt CSV file format (verified 2026-05-18):
# - Header row (first line): ``uniprot,pdbid,chainid,sequence,variant,wt_nlf,
#   mt_nlf,score_ml_fermi,score_ml,b_factors``.
# - One row per (position, alt_aa) substitution; saturated 19-alt scan over
#   every position. ``variant`` is encoded ``<wt><pos><alt>`` (e.g. ``M1A``);
#   a literal ``WT`` row would be skipped (none observed in the AF archive).
# - ``score_ml`` is the raw ΔΔG in kcal/mol (positive = destabilising). This
#   is the value we cache. ``score_ml_fermi`` is the Fermi-Dirac-transformed
#   [0, 1] view and is NOT cached.
# - ``sequence`` is the full protein sequence (same per row, used only for
#   the saturated-scan yield sanity check below).
# - File-system layout inside the zip:
#   ``rasp_preds_alphafold_..._prism_dir/<U[0:2]>/<U[2:4]>/<U[4:6]>/rasp_pred_AF-<UNIPROT>-F1-model_v2A.csv``.
#   The ``prism_dir`` zip suffix is upstream naming for the directory shape;
#   the entries are CSV.

# Defensive sanity-check thresholds (used by ``_parse_rasp_csv_text``):
# A saturated single-residue scan produces 19 alts × N positions rows per
# protein. We accept ≥ 50 % of that lower bound as "ok_low" and emit a
# warning; below that fraction we flag the file as malformed.
RASP_VARIANT_OK_FRACTION = 0.5


def _build_rasp_full_cache(
    source: PathLike,
    *,
    cache_dir: PathLike | None = None,
    overwrite: bool = False,
) -> Path:
    """Build the full-proteome RaSP demo cache from a local AlphaFold-RaSP archive.

    One-off operator step per machine, **NOT** intended to run on every
    notebook startup. Walks the upstream's per-UniProt PRISM file tree
    (extracted directory or in-place inside the 9 GB zip), pulls out only
    the seven demo-set genes, parses each PRISM ``.txt``, writes a small
    parquet (~1 MB) at ``<cache_dir>/full_proteome_demo_set.parquet``.

    Once written, :func:`_load_rasp_cache` prefers this file over the
    legacy 414 MB-derived ``demo_set.parquet``. The public API of
    :func:`rasp_ddg` and :func:`rasp_covers` is unchanged.

    Parameters
    ----------
    source
        Path to either:

        - The upstream's PRISM-dir zip
          (``rasp_preds_alphafold_UP000005640_9606_HUMAN_v2_prism_dir.zip``).
          Members are read in-place via :mod:`zipfile`; the zip is never
          fully extracted to disk.
        - An already-extracted directory matching the same PRISM-dir
          layout (per-UniProt sharding ``<chars 0-1>/<chars 2-3>/<chars 4-5>/``
          under the archive root; e.g. ``P12345`` → ``P1/23/45/...``).

    cache_dir
        Override for the RaSP cache directory. When ``None``, defaults to
        ``<cwd>/data/cache/rasp/``. On Colab this is typically a
        Drive-backed path so the small parquet survives runtime restarts.
    overwrite
        If ``False`` (default) and the output parquet already exists, raise
        :class:`FileExistsError` rather than silently re-writing.

    Returns
    -------
    Path
        Absolute path to the written ``full_proteome_demo_set.parquet``.

    Raises
    ------
    FileNotFoundError
        ``source`` does not exist on disk.
    ValueError
        ``source`` is neither a directory nor a ``.zip`` file.
    FileExistsError
        Target parquet already exists and ``overwrite=False``.
    RuntimeError
        No demo-set proteins resolved from the archive (parser failed
        across all seven UniProts), or zero data rows were produced after
        parsing succeeded — both indicate the input archive's structure
        does not match the documented PRISM layout.

    Format assumptions (step 17, verified against the actual upstream
    archive — not the originally-documented PRISM format; mismatches
    surface as ``status: "error"`` rows in the per-protein coverage
    report this function prints)
    --------------------------------------------------------------------

    1. Per-UniProt files are sharded in the layout
       ``<archive_root>/<U[0:2]>/<U[2:4]>/<U[4:6]>/...`` under the wrapper
       directory ``rasp_preds_alphafold_..._prism_dir/``. File names match
       ``rasp_pred_AF-<UNIPROT>-F1-model_v2A.csv``. The ``prism_dir`` zip
       suffix is misleading nomenclature — entries are CSV, not PRISM
       ``.txt``.
    2. CSV header: ``uniprot,pdbid,chainid,sequence,variant,wt_nlf,mt_nlf,
       score_ml_fermi,score_ml,b_factors``. Required for caching:
       ``uniprot``, ``variant``, ``score_ml``. Missing columns surface a
       clear error rather than silently dropping data.
    3. The ``variant`` column uses ``<wt><pos><alt>`` encoding (e.g.
       ``M1A``); a literal ``WT`` row or any ``<pos>=`` / ``<pos>*`` rows
       are skipped.
    4. The ``score_ml`` column is the raw ΔΔG in kcal/mol (positive =
       destabilising). ``score_ml_fermi`` is the Fermi-Dirac-transformed
       [0, 1] view and is NOT cached.
    5. The parsed variant count is at least
       :data:`RASP_VARIANT_OK_FRACTION` × 19 × (sequence_length − 1) (the
       saturated single-residue scan upper bound). Below the threshold the
       per-protein status flips to ``ok_low`` and a WARNING is logged.
    """
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"source path does not exist: {source_path}")

    parquet_dir = _resolve_cache_dir(cache_dir, "rasp")
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / FULL_PROTEOME_PARQUET
    if parquet_path.exists() and not overwrite:
        raise FileExistsError(
            f"{parquet_path} already exists. Pass overwrite=True to replace "
            f"it (deletes the existing cache; you'll need to re-run this "
            f"function to rebuild)."
        )

    keep_uniprots = sorted(DEMO_SET_UNIPROT_IDS.keys())
    all_rows: list[tuple[str, str, int, str, str, float]] = []
    coverage: dict[str, dict] = {}

    if source_path.is_dir():
        logger.info("Reading PRISM files from directory %s", source_path)
        for uniprot in keep_uniprots:
            file_path = _find_prism_file_in_dir(source_path, uniprot)
            if file_path is None:
                coverage[uniprot] = {
                    "status": "missing",
                    "reason": "no PRISM .txt found at sharded path",
                }
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                coverage[uniprot] = {
                    "status": "error",
                    "reason": f"read failed: {exc}",
                }
                continue
            rows, report = _parse_rasp_csv_text(text, uniprot)
            all_rows.extend(rows)
            report["source"] = str(file_path.relative_to(source_path))
            coverage[uniprot] = report
    elif source_path.is_file() and source_path.suffix.lower() == ".zip":
        logger.info("Reading PRISM files in-place from zip %s", source_path)
        with zipfile.ZipFile(source_path) as zf:
            for uniprot in keep_uniprots:
                entry = _find_prism_entry_in_zip(zf, uniprot)
                if entry is None:
                    coverage[uniprot] = {
                        "status": "missing",
                        "reason": "no PRISM .txt found at sharded path inside zip",
                    }
                    continue
                try:
                    with zf.open(entry) as fh:
                        text = fh.read().decode("utf-8")
                except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
                    coverage[uniprot] = {
                        "status": "error",
                        "reason": f"zip read failed for {entry}: {exc}",
                    }
                    continue
                rows, report = _parse_rasp_csv_text(text, uniprot)
                all_rows.extend(rows)
                report["source"] = entry
                coverage[uniprot] = report
    else:
        raise ValueError(
            f"source must be a directory or a .zip file; got {source_path!r}"
        )

    if not all_rows:
        raise RuntimeError(
            "No demo-set RaSP rows were extracted from the archive. Coverage "
            f"report: {coverage!r}. Format assumption(s) likely violated — "
            f"see the per-UniProt 'reason' fields above and the format "
            f"assumptions documented in _build_rasp_full_cache.__doc__."
        )

    df = pd.DataFrame(
        all_rows,
        columns=["uniprot_id", "pdb", "position", "wt_aa", "mut_aa", "rasp_ddg"],
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    # Bust the lru_cache so subsequent rasp_ddg() / rasp_covers() calls
    # within the same Python session pick up the just-written parquet.
    _load_rasp_cache.cache_clear()

    covered = sorted(df["uniprot_id"].unique())
    missing = sorted(set(keep_uniprots) - set(covered))
    logger.info(
        "RaSP full-proteome cache written to %s (%d rows across %d proteins). "
        "Covered: %s. Missing: %s.",
        parquet_path, len(df), len(covered), covered, missing,
    )
    print("=" * 78)
    print("RaSP full-proteome cache — per-protein coverage report")
    print("=" * 78)
    print(f"Output parquet: {parquet_path}")
    print(f"Rows: {len(df)}   Proteins covered: {len(covered)} of {len(keep_uniprots)}")
    print("-" * 78)
    print(f"{'uniprot':>8s}  {'gene':>8s}  {'status':>8s}  {'n_rows':>7s}  "
          f"{'seq_len':>7s}  {'score_col':>14s}  reason / source")
    print("-" * 78)
    for uniprot in keep_uniprots:
        gene = DEMO_SET_UNIPROT_IDS.get(uniprot, "?")
        rep = coverage.get(uniprot, {"status": "missing", "reason": "not processed"})
        status = rep.get("status", "?")
        n_rows = rep.get("n_rows", "")
        seq_len = rep.get("seq_len", "")
        score_col = rep.get("score_col", "")
        detail = rep.get("reason") or rep.get("source") or ""
        print(
            f"{uniprot:>8s}  {gene:>8s}  {status:>8s}  "
            f"{str(n_rows):>7s}  {str(seq_len):>7s}  {str(score_col):>14s}  "
            f"{detail}"
        )
    print("=" * 78)
    return parquet_path


def _find_prism_file_in_dir(root: Path, uniprot: str) -> Path | None:
    """Return the first ``.csv`` under ``root/<shard>/...`` whose name contains ``uniprot``.

    Layout per upstream: ``P12345`` → ``P1/23/45/rasp_pred_AF-P12345-F1-model_v2A.csv``.
    Some extractions place the per-UniProt tree under a wrapper directory
    matching the zip's name (``rasp_preds_alphafold_..._prism_dir/``); we
    search one level down in that case rather than assuming a flat root.
    """
    if len(uniprot) != 6:
        return None
    shard = Path(uniprot[0:2]) / uniprot[2:4] / uniprot[4:6]
    shard_root = root / shard
    if not shard_root.exists():
        # Wrapper directory present (typical for the upstream zip extracted as-is).
        for child in root.iterdir():
            if child.is_dir():
                candidate = child / shard
                if candidate.exists():
                    shard_root = candidate
                    break
        else:
            return None
    matches = sorted(p for p in shard_root.rglob("*.csv") if uniprot in p.name)
    return matches[0] if matches else None


def _find_prism_entry_in_zip(zf: zipfile.ZipFile, uniprot: str) -> str | None:
    """Locate the first ``.csv`` entry inside ``zf`` matching the sharded layout.

    Sharded prefix may have a top-level wrapper directory before it (the
    upstream zip's ``rasp_preds_alphafold_..._prism_dir/`` wrapper); the
    substring check on ``shard_prefix`` matches either layout.
    """
    if len(uniprot) != 6:
        return None
    shard_prefix = f"{uniprot[0:2]}/{uniprot[2:4]}/{uniprot[4:6]}/"
    matches: list[str] = []
    for name in zf.namelist():
        if not name.endswith(".csv"):
            continue
        if shard_prefix in name and uniprot in Path(name).name:
            matches.append(name)
    matches.sort()
    return matches[0] if matches else None


def _parse_rasp_csv_text(text: str, expected_uniprot: str) -> tuple[list[tuple], dict]:
    """Parse one RaSP per-UniProt CSV body; return (rows, coverage_report).

    Upstream CSV schema (verified 2026-05-18; one row per (position, alt_aa)
    substitution; saturated 19-alt scan)::

        uniprot,pdbid,chainid,sequence,variant,wt_nlf,mt_nlf,score_ml_fermi,score_ml,b_factors

    The load-bearing columns are ``uniprot`` (must match
    ``expected_uniprot``), ``variant`` (``<wt><pos><alt>`` encoding), and
    ``score_ml`` (raw ΔΔG in kcal/mol; positive = destabilising).
    ``score_ml_fermi`` is the Fermi-Dirac-transformed [0, 1] view and is
    NOT cached. ``sequence`` is the full protein sequence (same per row,
    used only for the saturated-scan yield sanity check).

    ``rows`` are tuples shaped for the cache parquet:
    ``(uniprot_id, "AF", position, wt_aa, mut_aa, rasp_ddg)``. The ``pdb``
    column is set to the literal ``"AF"`` for all AlphaFold-sourced
    entries — schema match with the legacy experimental cache (where
    ``pdb`` is a PDB accession) without per-row inflation of the
    invariant AF-model ID. Per-row AF-model provenance lives in the
    upstream zip filename and the build-time :data:`RASP_LAST_VERIFIED`
    constant.

    ``coverage_report`` is a dict with at minimum a ``status`` field of:

    - ``"ok"`` — parse + sanity checks passed.
    - ``"ok_low"`` — parsed below :data:`RASP_VARIANT_OK_FRACTION` of the
      expected saturated variant count for this protein; suspect file
      truncation. Rows are still returned but the caller should inspect.
    - ``"error"`` — parser failed structurally (empty file, missing
      required column, no rows matched ``expected_uniprot``). No rows
      returned.

    Additional fields when parsing succeeds: ``n_rows``, ``seq_len``,
    ``score_col`` (always ``"score_ml"``), ``skipped_wt``, ``skipped_other``.
    """
    import csv as _csv
    import io as _io

    reader = _csv.DictReader(_io.StringIO(text))
    if reader.fieldnames is None:
        return [], {"status": "error", "reason": "empty CSV (no header row)"}

    required = {"uniprot", "variant", "score_ml"}
    missing = required - set(reader.fieldnames)
    if missing:
        return [], {
            "status": "error",
            "reason": (
                f"CSV missing required columns {sorted(missing)}; have "
                f"{reader.fieldnames}"
            ),
        }

    variant_re = re.compile(r"^([A-Za-z])(\d+)([A-Za-z*=~])$")
    rows: list[tuple[str, str, int, str, str, float]] = []
    skipped_wt = 0
    skipped_other = 0
    seq_len = 0
    saw_matching_uniprot = False

    for row in reader:
        if (row.get("uniprot") or "").strip() != expected_uniprot:
            continue
        saw_matching_uniprot = True
        seq = (row.get("sequence") or "").strip()
        if len(seq) > seq_len:
            seq_len = len(seq)
        variant = (row.get("variant") or "").strip()
        if variant == "WT":
            skipped_wt += 1
            continue
        m = variant_re.match(variant)
        if not m:
            skipped_other += 1
            continue
        wt_aa = m.group(1).upper()
        try:
            pos = int(m.group(2))
        except ValueError:
            skipped_other += 1
            continue
        mut_aa = m.group(3).upper()
        if mut_aa in ("*", "=", "~") or mut_aa == wt_aa:
            skipped_wt += 1
            continue
        try:
            score = float(row.get("score_ml") or "")
        except ValueError:
            skipped_other += 1
            continue
        rows.append((expected_uniprot, "AF", pos, wt_aa, mut_aa, score))

    if not saw_matching_uniprot:
        return [], {
            "status": "error",
            "reason": f"no rows with uniprot=={expected_uniprot!r}",
        }

    expected_n = max(1, 19 * max(seq_len - 1, 0))
    status = "ok"
    reason: str | None = None
    if not rows:
        status = "error"
        reason = "matched uniprot but produced zero valid variant rows"
    elif seq_len > 0 and len(rows) < RASP_VARIANT_OK_FRACTION * expected_n:
        status = "ok_low"
        reason = (
            f"parsed {len(rows)} variant rows; expected >= "
            f"{int(RASP_VARIANT_OK_FRACTION * expected_n)} for saturated scan "
            f"on a {seq_len}-residue protein"
        )
        logger.warning("RaSP low-yield for %s: %s", expected_uniprot, reason)

    report = {
        "status": status,
        "n_rows": len(rows),
        "seq_len": seq_len,
        "score_col": "score_ml",
        "skipped_wt": skipped_wt,
        "skipped_other": skipped_other,
    }
    if reason is not None:
        report["reason"] = reason
    return (rows if status != "error" else []), report
