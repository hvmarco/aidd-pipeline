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

Demo-set scope and data source (decision Q5, chat 2026-05-15)
-------------------------------------------------------------

Lookup-only against the upstream's precomputed RaSP predictions on
*experimental* crystal structures, filtered to the seven demo-set genes
on first run. Source:
``rasp_preds_exp_strucs_gnomad_clinvar.csv`` (414 MB) at
``https://sid.erda.dk/sharelink/fFPJWflLeE`` — saturated single-residue
predictions on every human protein with at least one PDB structure.

This file's name is mildly misleading: ``gnomad`` and ``clinvar`` are
*annotation columns*, not the variant filter. The variants are saturated
(every position × every alt AA), with extra columns flagging which ones
also appear in gnomAD or ClinVar.

Coverage of the seven demo genes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- NAT2, DPYD, CYP2D6, KRAS, BRCA1, ESR1 — covered (each has a human
  crystal structure).
- **UGT1A1 is NOT covered** (no human crystal exists; per
  ``_planning/MECHANISM_OF_ACTION_SCOPE.md``, the UGT1A1\\*28 demo in
  nb 09 was already flagged as needing AlphaFold-based handling for the
  ``fetch_pdb`` helper). For step 14 (calibration walkthrough) UGT1A1
  isn't in the calibration set, so this is acceptable.

When nb 09 (step 16) needs UGT1A1 RaSP predictions, swap the data
source in :func:`_build_rasp_demo_cache` to the AlphaFold-based 9 GB
upstream (``rasp_preds_alphafold_UP000005640_9606_HUMAN_v2.zip`` from
the same share link). The helper signature does not change.

Honesty / limits
----------------

- **Lookup-only.** Cannot predict ΔΔG for variants on a user-supplied
  custom PDB (e.g., a mutant fold from notebook 01). Predictions are
  fixed to the canonical experimental structures used by the upstream
  saturation run.
- **Position numbering.** RaSP predictions use *PDB residue numbering*
  of the structure they were computed on. UniProt numbering and PDB
  numbering can differ when the structure lacks the N-terminal
  methionine, has a cleaved signal peptide, or has gaps in the model.
  The helper requires the caller's ``wt_aa`` to match the residue
  recorded in the CSV's variant string; a mismatch returns ``None`` so
  numbering drift surfaces cleanly rather than silently returning the
  wrong residue's prediction.
- **Multi-PDB averaging.** When the CSV has multiple rows for the same
  (uniprot, position, wt_aa, mut_aa) — e.g., one per chain or per
  crystal structure — the helper returns the mean. Spread across rows
  is small in practice (RaSP is sequence-conditioned with only mild
  structural-context dependency at most positions).
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
from pathlib import Path
from typing import Union

import pandas as pd
import requests

from .variants import DEMO_SET_UNIPROT_IDS, _resolve_cache_dir

PathLike = Union[str, Path]
logger = logging.getLogger("aidd.stability")


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

RASP_URL = (
    "https://sid.erda.dk/share_redirect/fFPJWflLeE/"
    "rasp_preds_exp_strucs_gnomad_clinvar.csv"
)
RASP_DATA_VERSION = "exp_strucs_gnomad_clinvar"
# Bumped after each successful demo-set cache rebuild on a fresh host.
RASP_LAST_VERIFIED: str | None = None


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
    """Load (and build-if-missing) the RaSP demo-set parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / "demo_set.parquet"
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
