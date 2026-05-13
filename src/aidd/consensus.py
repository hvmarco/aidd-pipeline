"""Consensus ranking + shortlist building for the screening pipeline.

Joins the per-target classical-rescorer output (notebook ``04`` ->
``scored_poses.parquet``) with the Boltz-2 co-folded affinity output
(notebook ``05`` -> ``affinity.csv``) and applies one of two consensus
filters (selected via ``filter_method``):

- ``"intersection"`` (default) -- compounds passing top-X% in **both** lanes,
  ordered by rank product. Strict; correct when the two lanes agree on real
  signal (Spearman rho between lanes is in the [0.2, 0.5] band or higher).
  Empty by construction when lanes disagree.
- ``"rank_product_topk"`` -- top K = ceil(top_fraction x N_joined) compounds
  by rank product across the full joined cohort. No per-lane gating.
  Lenient; correct when lanes have low correlation (rho < ~0.15) and the
  intersection would be empty or near-random.

Both methods rank survivors by rank product (geometric mean of the two
within-lane ranks; Wang & Wang 2001 *J. Chem. Inf. Comput. Sci.* **41**,
1422; Houston & Walkinshaw 2013 *J. Chem. Inf. Model.* **53**, 384). Note
on rank-product behaviour worth knowing before relying on it: the geometric
mean does **not** penalise extreme cross-lane disagreement as much as you
might expect. A compound at rank 1 in lane A and rank N in lane B has
``rp = sqrt(N)``, which beats a compound at rank N/2 in both lanes
(``rp = N/2``) for any N > 4. Strong-in-one-lane compounds out-rank
moderate-in-both-lanes compounds. Whether this is desirable depends on
what you are triaging for.

Public entry points:

- :func:`compute_consensus` -- joins the two score tables, computes ranks +
  rank-product, returns the shortlist (per ``filter_method``) plus two
  anti-join side tables for compounds that only one lane could score.
- :func:`write_shortlist` -- emits ``shortlist.sdf`` (one chosen pose per
  compound, picked from the gnina ``poses.sdf``) and ``shortlist.csv``
  (the same data minus 3-D coordinates).

Both are CPU-only and have no GPU / network dependency; the heavy upstream
computation happens in notebooks ``04`` and ``05``.

Expected input schemas (the implicit contract with the two upstream notebooks
-- if either side drifts, the join below will silently shrink or fail noisily,
so verify these columns exist before debugging the rest). **Sign conventions
matter and differ between the two lanes** -- :func:`compute_consensus` handles
the negation internally via the ``boltz_lower_is_better`` /
``rescorer_lower_is_better`` flags; the on-disk files keep their upstream
convention so we never fight the producing tools::

    scored_poses.parquet  (notebook 04)
        compound_id              : str   -- join key
        rescorer_rf_proba_oof    : float -- in [0, 1], HIGHER = stronger.
                                            Out-of-fold predictions; default
                                            ranking column. Set by notebook
                                            04's CV step (see the "open
                                            rescorer leak" entry in
                                            _planning/KNOWN_ISSUES.md --
                                            until that fix lands, this column
                                            does not exist on disk and
                                            compute_consensus raises KeyError
                                            on purpose to prevent runs against
                                            the leaked alternative).
        rescorer_rf_proba        : float -- in [0, 1], data-leaked (RF
                                            trained on all rows, including
                                            its own test fold). Useful only
                                            for chemistry inspection, never
                                            for ranking. Pre-existing column.
        Active                   : int   -- 0/1 label, may be NaN for
                                            unlabelled compounds; used for
                                            the sanity-check recall in
                                            notebook 06 section 5.
        gnina_cnn_affinity, gnina_affinity, in_random_test_fold,
        in_scaffold_test_fold, rescorer_xgb_proba  -- carried through.

    affinity.csv          (notebook 05)
        compound_id                  : str   -- join key (must match).
        boltz_affinity               : float -- log10(IC50) µM. **LOWER is
                                                stronger** -- this is Boltz-2's
                                                native convention and is
                                                preserved on disk. Default
                                                ranking column;
                                                ``compute_consensus`` negates
                                                internally when
                                                ``boltz_lower_is_better=True``
                                                (the default). See
                                                feedback_boltz_affinity_sign.md
                                                in project memory.
        boltz_affinity_probability   : float -- in [0, 1], higher = more likely
                                                binder. If you switch
                                                ``boltz_col`` to this column,
                                                also pass
                                                ``boltz_lower_is_better=False``.
        boltz_confidence, boltz_iptm, boltz_ligand_iptm  -- carried through;
                                                all in [0, 1], higher = more
                                                confident. Not used for
                                                ranking by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Union

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
    rescorer_col: str = "rescorer_rf_proba_oof",
    boltz_col: str = "boltz_affinity",
    id_col: str = "compound_id",
    rescorer_lower_is_better: bool = False,
    boltz_lower_is_better: bool = True,
    filter_method: Literal["intersection", "rank_product_topk"] = "intersection",
) -> dict:
    """Join the two lanes, apply the consensus filter, return shortlist + side tables.

    Parameters
    ----------
    scored_poses
        Notebook ``04`` output. Must contain ``id_col`` and ``rescorer_col``.
    affinity
        Notebook ``05`` output. Must contain ``id_col`` and ``boltz_col``.
    top_fraction
        Threshold (per lane for ``"intersection"``; size fraction for
        ``"rank_product_topk"``). Default 0.05 (top 5%) per
        ``_planning/PROJECT_PROPOSAL.md`` section 1.
    rescorer_lower_is_better, boltz_lower_is_better
        Per-lane sign convention. The default (``False`` for the rescorer,
        ``True`` for Boltz) matches the on-disk convention of the two
        upstream notebooks: rescorer probabilities are higher-is-stronger,
        Boltz-2's ``boltz_affinity`` is log10(IC50) µM where lower is
        stronger. Internally we build a signed score per lane (negated when
        ``lower_is_better=True``) and rank descending, so rank 1 is always
        the strongest binder regardless of the column's native direction.
        Override when you swap ``boltz_col`` to ``boltz_affinity_probability``
        (already higher-is-better -> pass ``boltz_lower_is_better=False``).
    filter_method
        Which consensus rule to apply:

        - ``"intersection"`` (default, backward-compatible): a compound enters
          the shortlist iff it is in the top ``top_fraction`` by **both**
          lanes. Strict; correct when the two lanes agree on real signal
          (cross-lane Spearman rho in roughly the [0.2, 0.5] band or higher).
          Empty by construction when lanes disagree.
        - ``"rank_product_topk"``: take the top
          K = ``ceil(top_fraction * n_joined)`` compounds by ``rank_product``
          across the full joined cohort. No per-lane gating; the
          rank-product itself is the consensus score. Lenient; the right
          choice when lanes have low correlation (``rho`` near zero) and
          the intersection would be empty or near-random size. Note:
          rank product does not penalise extreme cross-lane disagreement
          as strongly as you might expect (see the module docstring).

    Returns
    -------
    dict with keys:
      ``"joined"`` -- inner-join DataFrame on ``id_col`` with both scores,
        within-lane ranks, per-lane top-X% flags, ``rank_product``,
        ``made_shortlist``.
      ``"shortlist"`` -- compounds with ``made_shortlist == True``, sorted
        by ``rank_product`` ascending. Has an added ``consensus_rank`` column
        (1-based, 1 = best).
      ``"rescorer_only"`` -- compounds top-X% by rescorer that the Boltz
        lane could not score (anti-join), ordered strongest-first. The
        per-lane top-X% definition is the same regardless of
        ``filter_method`` -- side files are diagnostic, not the headline.
      ``"boltz_only"`` -- symmetric.
      ``"summary"`` -- counts at every stage of the join + filter, plus the
        sign conventions and ``filter_method`` actually applied.
    """
    if filter_method not in ("intersection", "rank_product_topk"):
        raise ValueError(
            f"filter_method must be 'intersection' or 'rank_product_topk'; "
            f"got {filter_method!r}"
        )
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

    # Build a signed score per lane so rank 1 = strongest binder regardless
    # of the column's native direction. Boltz's `boltz_affinity` is
    # log10(IC50) µM (lower = stronger), so we negate before ranking; the
    # rescorer probability is already higher-is-stronger and is left alone.
    # See feedback_boltz_affinity_sign.md for the incident behind this.
    rescore_score = -rescore[rescorer_col] if rescorer_lower_is_better else rescore[rescorer_col]
    boltz_score   = -boltz[boltz_col]      if boltz_lower_is_better      else boltz[boltz_col]

    rescore["rank_rescorer"] = _rank_descending(rescore_score)
    boltz["rank_boltz"]      = _rank_descending(boltz_score)

    n_rescorer = len(rescore)
    n_boltz = len(boltz)
    cut_rescorer = max(1, int(np.ceil(n_rescorer * top_fraction)))
    cut_boltz = max(1, int(np.ceil(n_boltz * top_fraction)))
    rescore["in_top_rescorer"] = rescore["rank_rescorer"] <= cut_rescorer
    boltz["in_top_boltz"] = boltz["rank_boltz"] <= cut_boltz

    joined = rescore.merge(boltz, on=id_col, how="inner")
    joined["rank_product"] = rank_product(joined["rank_rescorer"], joined["rank_boltz"])

    if filter_method == "intersection":
        joined["made_shortlist"] = (
            joined["in_top_rescorer"].astype(bool) & joined["in_top_boltz"].astype(bool)
        )
        k_topk = None
    else:  # filter_method == "rank_product_topk"
        # Pick the K compounds with the lowest rank product across the full
        # joined cohort. Tie-breaking via rank(method="first") gives a stable
        # row count of exactly K (no surprise inflation when several compounds
        # share the threshold rank-product value).
        k_topk = max(1, int(np.ceil(len(joined) * top_fraction)))
        rp_position = joined["rank_product"].rank(method="first")
        joined["made_shortlist"] = rp_position <= k_topk

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
        "filter_method": str(filter_method),
        "top_fraction": float(top_fraction),
        "cut_rescorer": int(cut_rescorer),
        "cut_boltz": int(cut_boltz),
        "k_topk": int(k_topk) if k_topk is not None else None,
        "n_top_rescorer": int(rescore["in_top_rescorer"].sum()),
        "n_top_boltz": int(boltz["in_top_boltz"].sum()),
        "n_shortlist": int(len(shortlist)),
        "n_rescorer_only_top": int(len(rescorer_only)),
        "n_boltz_only_top": int(len(boltz_only)),
        "rescorer_col": rescorer_col,
        "boltz_col": boltz_col,
        "rescorer_lower_is_better": bool(rescorer_lower_is_better),
        "boltz_lower_is_better": bool(boltz_lower_is_better),
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
