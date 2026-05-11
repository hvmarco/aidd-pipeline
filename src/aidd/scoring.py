"""Per-target ML rescorer: gnina + ProLIF features → activity classifier.

Lifts the workflow of ``_archive/Week_3_Tuesday_Challenge_*.ipynb`` into a
small set of composable functions. Inputs are two per-pose tables (gnina
scores + ProLIF interaction fingerprints in wide format); the output is a
trained classifier whose per-compound score beats raw gnina ``CNN_affinity``
on top-N enrichment.

The "classical" rescorer that complements the Boltz-2 fast lane (notebook 06)
in the consensus shortlist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

PathLike = Union[str, Path]
logger = logging.getLogger("aidd.scoring")

GNINA_FEATURE_COLS: tuple[str, ...] = ("affinity", "cnn_score", "cnn_affinity", "cnn_vs")


# ---------------------------------------------------------------------------
# Pose selection
# ---------------------------------------------------------------------------

def select_top_pose(
    gnina_scores: pd.DataFrame,
    *,
    by: Literal["gnina_top1", "cnn_affinity", "affinity"] = "gnina_top1",
    only_pb_clean: bool = False,
    id_col: str = "compound_id",
) -> pd.DataFrame:
    """Reduce a per-pose gnina scores table to one row per compound.

    ``by='gnina_top1'``    — keep ``pose_rank == 1`` (gnina's own top choice).
    ``by='cnn_affinity'``  — keep the pose with the highest CNN affinity.
    ``by='affinity'``      — keep the pose with the most negative Vina affinity.

    When ``only_pb_clean`` is True, PoseBusters-failing poses are filtered out
    before selection. Compounds whose poses *all* fail PoseBusters fall back
    to their overall best pose so we do not silently drop labels: dropping
    compounds based on pose quality is a confound (it correlates with chemistry
    and would bias the rescorer).

    The returned DataFrame has an added ``_pose_row`` column carrying the
    original positional row index in the input ``gnina_scores`` table. It is
    used by :func:`align_ifp_to_top_pose` to pick the matching IFP rows.
    """
    df = gnina_scores.reset_index(drop=True).copy()
    df["_pose_row"] = np.arange(len(df))

    if only_pb_clean and "pb_passes_all" in df.columns:
        clean = df[df["pb_passes_all"].astype(bool)]
        kept_ids = set(clean[id_col].unique())
        missing = df[~df[id_col].isin(kept_ids)]
        df = pd.concat([clean, missing], ignore_index=True)

    if by == "gnina_top1":
        if "pose_rank" not in df.columns:
            raise ValueError("by='gnina_top1' needs a 'pose_rank' column")
        picked = df.sort_values([id_col, "pose_rank"]).drop_duplicates(id_col, keep="first")
    elif by == "cnn_affinity":
        picked = df.sort_values([id_col, "cnn_affinity"], ascending=[True, False]).drop_duplicates(id_col, keep="first")
    elif by == "affinity":
        picked = df.sort_values([id_col, "affinity"], ascending=[True, True]).drop_duplicates(id_col, keep="first")
    else:
        raise ValueError(f"unknown pose-selection rule: {by!r}")

    return picked.reset_index(drop=True)


def build_feature_table(
    gnina_top_pose: pd.DataFrame,
    ifp_wide_aligned: pd.DataFrame,
    *,
    id_col: str = "compound_id",
    gnina_cols: Sequence[str] = GNINA_FEATURE_COLS,
) -> pd.DataFrame:
    """Join gnina scores + IFP features into one feature row per compound.

    Both inputs must already be one-row-per-compound and aligned positionally
    (see :func:`select_top_pose` and :func:`align_ifp_to_top_pose`). Returns
    a DataFrame whose columns are ``id_col`` + ``gnina_cols`` + every IFP
    column (numeric 0/1 features).
    """
    base = gnina_top_pose[[id_col, *gnina_cols]].reset_index(drop=True).copy()
    ifp = ifp_wide_aligned.reset_index(drop=True).copy()
    if len(base) != len(ifp):
        raise ValueError(
            f"row count mismatch: gnina_top_pose has {len(base)} rows, "
            f"ifp_wide_aligned has {len(ifp)}. Run align_ifp_to_top_pose first."
        )
    return pd.concat([base, ifp], axis=1)


def align_ifp_to_top_pose(
    ifp_wide: pd.DataFrame,
    top_pose: pd.DataFrame,
    *,
    pose_row_col: str = "_pose_row",
) -> pd.DataFrame:
    """Pick the IFP rows that correspond to the selected top pose per compound.

    ``ifp_wide`` is the per-pose IFP table in the original ``poses.sdf`` row
    order — i.e. the same length and row-ordering as the gnina scores
    DataFrame passed into :func:`select_top_pose`. ``top_pose`` is that
    function's output; it carries a ``_pose_row`` column with the original
    row position, which we use to slice ``ifp_wide``.
    """
    if pose_row_col not in top_pose.columns:
        raise KeyError(
            f"top_pose is missing the {pose_row_col!r} column. Pass a DataFrame "
            "produced by select_top_pose() (which adds this column automatically)."
        )
    rows = top_pose[pose_row_col].to_numpy()
    if rows.max() >= len(ifp_wide):
        raise ValueError(
            f"top_pose references row {int(rows.max())} but ifp_wide only has "
            f"{len(ifp_wide)} rows. The IFP must be computed on the same poses.sdf "
            "that produced the gnina scores."
        )
    return ifp_wide.iloc[rows].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def enrichment_factor(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    fraction: float = 0.01,
) -> float:
    """Enrichment factor at the top ``fraction`` of a ranked list.

    EF_x = (actives_in_top_x / total_top_x) / (actives_total / total).

    EF = 1 is random; EF = 10 at fraction = 0.01 means actives are 10x more
    concentrated at the top of the rescorer's ranking than random — the
    classical virtual-screening metric. EF1% (fraction = 0.01) is the
    industry-standard early-recognition score.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * fraction)))
    top_idx = np.argsort(-y_score, kind="stable")[:k]
    actives_top = int(y_true[top_idx].sum())
    actives_total = int(y_true.sum())
    if actives_total == 0:
        return float("nan")
    return (actives_top / k) / (actives_total / n)


def evaluate_scores(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> dict:
    """Compute ROC-AUC, EF1%, EF5% on a single held-out fold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"auc": float("nan"), "ef1": float("nan"), "ef5": float("nan"),
                "n": int(len(y_true)), "n_actives": int(y_true.sum())}
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ef1": enrichment_factor(y_true, y_score, fraction=0.01),
        "ef5": enrichment_factor(y_true, y_score, fraction=0.05),
        "n":   int(len(y_true)),
        "n_actives": int(y_true.sum()),
    }


def roc_points(y_true: Sequence[int], y_score: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return FPR / TPR arrays for plotting a single ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    return fpr, tpr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_rescorer(
    kind: Literal["rf", "xgb"] = "rf",
    *,
    smote: bool = True,
    seed: int = 42,
    n_estimators: int = 300,
    rf_max_depth: int | None = None,
    xgb_max_depth: int = 5,
    xgb_learning_rate: float = 0.1,
):
    """Build an unfitted classifier with sensible imbalanced-class defaults.

    Returns an imbalanced-learn ``Pipeline`` when ``smote=True`` (SMOTE then
    classifier), or a plain sklearn estimator otherwise. Both share the same
    ``.fit(X, y)`` / ``.predict_proba(X)`` interface.

    ``kind='rf'``  — Random Forest. Interpretable (``feature_importances_``)
                     and robust on small datasets. ``class_weight='balanced'``
                     compensates for imbalance whether or not SMOTE is on.
    ``kind='xgb'`` — XGBoost. Usually the strongest classifier on tabular
                     IFP-style features. Uses ``scale_pos_weight`` for
                     imbalance when SMOTE is off.
    """
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=rf_max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif kind == "xgb":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            scale_pos_weight=1.0 if smote else None,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown kind: {kind!r}; expected 'rf' or 'xgb'")

    if not smote:
        return clf

    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    return ImbPipeline([
        ("smote", SMOTE(random_state=seed, k_neighbors=5)),
        ("clf", clf),
    ])


def feature_importances(model, feature_names: Sequence[str], *, top_n: int = 25) -> pd.DataFrame:
    """Extract feature importances from a fitted RF (or RF inside an imblearn pipeline)."""
    est = model
    if hasattr(model, "named_steps"):
        est = model.named_steps.get("clf", model)
    if not hasattr(est, "feature_importances_"):
        raise AttributeError(
            f"{type(est).__name__} has no feature_importances_ — only tree-based "
            "models expose feature importances out of the box."
        )
    imp = pd.DataFrame({"feature": list(feature_names), "importance": est.feature_importances_})
    return imp.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
