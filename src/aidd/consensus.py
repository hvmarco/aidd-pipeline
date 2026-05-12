"""Consensus ranking + shortlist building for the screening pipeline.

Joins the per-target classical-rescorer output (notebook ``04`` ->
``scored_poses.parquet``) with the Boltz-2 co-folded affinity output
(notebook ``05`` -> ``affinity.csv``), applies the *top-X% by both lanes*
intersection filter, and ranks survivors by **rank product** (geometric mean
of the two within-lane ranks; Wang & Wang 2001 *J. Chem. Inf. Comput. Sci.*
**41**, 1422; Houston & Walkinshaw 2013 *J. Chem. Inf. Model.* **53**, 384).

Public entry points:

- :func:`compute_consensus` -- joins the two score tables, computes ranks +
  rank-product, returns the inner-join shortlist plus two anti-join side
  tables for compounds that only one lane could score.
- :func:`write_shortlist` -- emits ``shortlist.sdf`` (one chosen pose per
  compound, picked from the gnina ``poses.sdf``) and ``shortlist.csv``
  (the same data minus 3-D coordinates).

Both are CPU-only and have no GPU / network dependency; the heavy upstream
computation happens in notebooks ``04`` and ``05``.

Expected input schemas (the implicit contract with the two upstream notebooks
-- if either side drifts, the join below will silently shrink or fail noisily,
so verify these columns exist before debugging the rest)::

    scored_poses.parquet  (notebook 04)
        compound_id          : str   -- join key
        rescorer_rf_proba    : float -- in [0, 1], higher = stronger (default
                                        ranking column; override via
                                        ``rescorer_col`` to use ``rescorer_xgb_proba``)
        Active               : int   -- 0/1 label, may be NaN for unlabelled
                                        compounds; used for the sanity-check
                                        recall in notebook 06 section 5
        gnina_cnn_affinity, gnina_affinity, in_random_test_fold,
        in_scaffold_test_fold, rescorer_xgb_proba  -- carried through

    affinity.csv          (notebook 05)
        compound_id                  : str   -- join key (must match)
        boltz_affinity               : float -- pIC50-ish, higher = stronger
                                                (default ranking column;
                                                override via ``boltz_col`` to use
                                                ``boltz_affinity_probability``)
        boltz_affinity_probability   : float -- in [0, 1]
        boltz_confidence, boltz_iptm, boltz_ligand_iptm  -- carried through
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from rdkit import Chem

PathLike = Union[str, Path]
logger = logging.getLogger("aidd.consensus")


# ---------------------------------------------------------------------------
# Ranking primitives
# ---------------------------------------------------------------------------

def _rank_descending(scores: pd.Series) -> pd.Series:
    """Return 1-based ranks, ``1`` = best (highest score). Ties: average rank."""
    return scores.rank(method="average", ascending=False)


def rank_product(rank_a, rank_b) -> np.ndarray:
    """Geometric mean of two rank vectors. Lower = better consensus rank.

    Used as the within-shortlist ordering rule for the consensus pipeline.
    Scale-free across the two input lanes (a key property when one lane
    outputs a 0-1 probability and the other outputs an unbounded affinity).
    """
    a = np.asarray(rank_a, dtype=float)
    b = np.asarray(rank_b, dtype=float)
    return np.sqrt(a * b)


# ---------------------------------------------------------------------------
# Consensus computation
# ---------------------------------------------------------------------------

def _dedupe_with_warning(df: pd.DataFrame, id_col: str, lane: str) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset=id_col, keep="first")
    n_after = len(df)
    if n_after < n_before:
        logger.warning(
            "%s table had %d duplicate %s rows; kept the first occurrence of each.",
            lane, n_before - n_after, id_col,
        )
    return df


def compute_consensus(
    scored_poses: pd.DataFrame,
    affinity: pd.DataFrame,
    *,
    top_fraction: float = 0.05,
    rescorer_col: str = "rescorer_rf_proba",
    boltz_col: str = "boltz_affinity",
    id_col: str = "compound_id",
) -> dict:
    """Join the two lanes, apply the consensus filter, return shortlist + side tables.

    Parameters
    ----------
    scored_poses
        Notebook ``04`` output. Must contain ``id_col`` and ``rescorer_col``;
        higher ``rescorer_col`` = stronger predicted activity.
    affinity
        Notebook ``05`` output. Must contain ``id_col`` and ``boltz_col``;
        higher ``boltz_col`` = stronger predicted binding.
    top_fraction
        Threshold per lane. A compound enters the shortlist iff it is in the
        top ``top_fraction`` of *both* lanes. Default 0.05 (top 5%) per
        ``_planning/PROJECT_PROPOSAL.md`` section 1.

    Returns
    -------
    dict with keys:
      ``"joined"`` -- inner-join DataFrame on ``id_col`` with both scores,
        within-lane ranks, top-X% flags, ``rank_product``, ``made_shortlist``.
      ``"shortlist"`` -- compounds with ``made_shortlist == True``, sorted
        by ``rank_product`` ascending. Has an added ``consensus_rank`` column
        (1-based, 1 = best).
      ``"rescorer_only"`` -- compounds top-X% by rescorer that the Boltz
        lane could not score (anti-join), ranked by ``rescorer_col``.
      ``"boltz_only"`` -- compounds top-X% by Boltz that the rescorer lane
        could not score (anti-join), ranked by ``boltz_col``.
      ``"summary"`` -- counts at every stage of the join + filter.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top_fraction must be in (0, 1]; got {top_fraction}")

    rescore = scored_poses.copy()
    rescore[id_col] = rescore[id_col].astype(str)
    if rescorer_col not in rescore.columns:
        raise KeyError(
            f"rescorer_col {rescorer_col!r} not in scored_poses columns "
            f"{list(rescore.columns)}"
        )
    rescore = _dedupe_with_warning(rescore, id_col, "rescorer (notebook 04)")

    boltz = affinity.copy()
    boltz[id_col] = boltz[id_col].astype(str)
    if boltz_col not in boltz.columns:
        raise KeyError(
            f"boltz_col {boltz_col!r} not in affinity columns "
            f"{list(boltz.columns)}"
        )
    boltz = _dedupe_with_warning(boltz, id_col, "Boltz-2 (notebook 05)")

    n_dropped_nan_rescorer = int(rescore[rescorer_col].isna().sum())
    n_dropped_nan_boltz = int(boltz[boltz_col].isna().sum())
    rescore = rescore.dropna(subset=[rescorer_col])
    boltz = boltz.dropna(subset=[boltz_col])

    rescore["rank_rescorer"] = _rank_descending(rescore[rescorer_col])
    boltz["rank_boltz"] = _rank_descending(boltz[boltz_col])

    n_rescorer = len(rescore)
    n_boltz = len(boltz)
    cut_rescorer = max(1, int(np.ceil(n_rescorer * top_fraction)))
    cut_boltz = max(1, int(np.ceil(n_boltz * top_fraction)))
    rescore["in_top_rescorer"] = rescore["rank_rescorer"] <= cut_rescorer
    boltz["in_top_boltz"] = boltz["rank_boltz"] <= cut_boltz

    joined = rescore.merge(boltz, on=id_col, how="inner")
    joined["rank_product"] = rank_product(joined["rank_rescorer"], joined["rank_boltz"])
    joined["made_shortlist"] = (
        joined["in_top_rescorer"].astype(bool) & joined["in_top_boltz"].astype(bool)
    )

    shortlist = (
        joined[joined["made_shortlist"]]
        .sort_values("rank_product", ascending=True)
        .reset_index(drop=True)
    )
    shortlist.insert(0, "consensus_rank", np.arange(1, len(shortlist) + 1))

    boltz_ids = set(boltz[id_col])
    rescorer_ids = set(rescore[id_col])

    rescorer_only = (
        rescore[rescore["in_top_rescorer"] & ~rescore[id_col].isin(boltz_ids)]
        .sort_values("rank_rescorer")
        .reset_index(drop=True)
    )
    boltz_only = (
        boltz[boltz["in_top_boltz"] & ~boltz[id_col].isin(rescorer_ids)]
        .sort_values("rank_boltz")
        .reset_index(drop=True)
    )

    summary = {
        "n_rescorer_total": int(n_rescorer),
        "n_boltz_total": int(n_boltz),
        "n_intersection": int(len(joined)),
        "n_rescorer_only_universe": int(len(rescorer_ids - boltz_ids)),
        "n_boltz_only_universe": int(len(boltz_ids - rescorer_ids)),
        "n_dropped_nan_rescorer": n_dropped_nan_rescorer,
        "n_dropped_nan_boltz": n_dropped_nan_boltz,
        "top_fraction": float(top_fraction),
        "cut_rescorer": int(cut_rescorer),
        "cut_boltz": int(cut_boltz),
        "n_top_rescorer": int(rescore["in_top_rescorer"].sum()),
        "n_top_boltz": int(boltz["in_top_boltz"].sum()),
        "n_shortlist": int(len(shortlist)),
        "n_rescorer_only_top": int(len(rescorer_only)),
        "n_boltz_only_top": int(len(boltz_only)),
        "rescorer_col": rescorer_col,
        "boltz_col": boltz_col,
    }

    return {
        "joined": joined,
        "shortlist": shortlist,
        "rescorer_only": rescorer_only,
        "boltz_only": boltz_only,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Shortlist file output
# ---------------------------------------------------------------------------

def write_shortlist(
    shortlist: pd.DataFrame,
    poses_sdf: PathLike,
    out_sdf: PathLike,
    out_csv: PathLike,
    *,
    id_col: str = "compound_id",
) -> dict:
    """Write the shortlist as both an SDF (poses + properties) and a CSV.

    For each compound in ``shortlist``, picks the **first** matching pose
    from ``poses_sdf`` (matched on ``_Name == compound_id``). Notebook
    ``03``/``04`` writes ``poses.sdf`` in pose-rank order, so "first match"
    is gnina's top-1 pose -- the same pose used for IFP feature extraction
    and the rescorer probability.

    Every column in ``shortlist`` (other than ``id_col``) is attached as an
    SD property on the chosen mol, so a wet-lab chemist opening the SDF in
    PyMOL / VS Code sees the consensus rank, the two lane scores, and the
    activity label (when known) alongside the 3-D pose.

    Compounds in ``shortlist`` for which no pose exists in ``poses_sdf``
    are still written to the CSV but skipped from the SDF, and a warning
    is logged with the ``compound_id``.

    Returns
    -------
    dict with ``n_sdf_written`` and ``n_csv_written`` counts.
    """
    poses_sdf = Path(poses_sdf)
    out_sdf = Path(out_sdf)
    out_csv = Path(out_csv)
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    short_ids = [str(x) for x in shortlist[id_col].tolist()]
    short_set = set(short_ids)

    chosen: dict[str, Chem.Mol] = {}
    if poses_sdf.exists():
        sup = Chem.SDMolSupplier(str(poses_sdf), removeHs=False)
        for mol in sup:
            if mol is None:
                continue
            cid = (mol.GetProp("_Name") if mol.HasProp("_Name") else "").strip()
            if cid in short_set and cid not in chosen:
                chosen[cid] = mol
    else:
        logger.warning(
            "poses_sdf does not exist (%s); shortlist.sdf will be empty but "
            "shortlist.csv will still be written.", poses_sdf,
        )

    prop_cols = [c for c in shortlist.columns if c != id_col]

    writer = Chem.SDWriter(str(out_sdf))
    n_sdf = 0
    csv_records = []
    try:
        for _, row in shortlist.iterrows():
            cid = str(row[id_col])
            csv_records.append({id_col: cid, **{c: row[c] for c in prop_cols}})
            mol = chosen.get(cid)
            if mol is None:
                logger.warning(
                    "No pose for compound_id=%s in %s; in CSV only.",
                    cid, poses_sdf.name,
                )
                continue
            mol = Chem.Mol(mol)
            mol.SetProp("_Name", cid)
            for c in prop_cols:
                v = row[c]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                mol.SetProp(str(c), str(v))
            writer.write(mol)
            n_sdf += 1
    finally:
        writer.close()

    csv_df = pd.DataFrame(csv_records)
    csv_df.to_csv(out_csv, index=False)

    return {"n_sdf_written": n_sdf, "n_csv_written": len(csv_df)}
