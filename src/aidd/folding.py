"""Wrapper utilities around ColabFold output.

ColabFold itself runs in the notebook (it needs a GPU and a special install
on Colab; we don't try to install it as part of the ``aidd`` conda env). The
helpers in this module read and summarise its **outputs**:

- ``parse_colabfold_ranking`` — collect the five (or N) predicted models with
  their pLDDT / pTM scores into a tidy DataFrame.
- ``best_model_path`` — return the rank-1 PDB.
- ``summarise_run`` — single-row report of the run.

ColabFold output directory layout (current as of v1.5):

    out_dir/
    ├── <name>_unrelaxed_rank_001_alphafold2_ptm_model_<i>_seed_<j>.pdb
    ├── <name>_unrelaxed_rank_002_...
    ├── ...
    ├── <name>_scores_rank_001_..._.json   (per-residue plddt + pae + max_pae + ptm)
    ├── <name>_scores_rank_002_...
    ├── ...
    └── <name>.a3m   (multiple-sequence alignment used)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

_PDB_RE = re.compile(r"^(?P<name>.+?)_unrelaxed_rank_(?P<rank>\d+)_.+\.pdb$")


def parse_colabfold_ranking(output_dir: PathLike, *, name: str | None = None) -> pd.DataFrame:
    """Collect the predicted models in ``output_dir`` into a DataFrame.

    Columns:
        rank          : 1 (best) .. N
        pdb_path      : absolute Path to the PDB
        mean_plddt    : mean per-residue pLDDT (0–100)
        ptm           : predicted TM-score (0–1)
        model_number  : the AF2 model number (1..5) the prediction came from

    Sorted by ``rank`` ascending. ColabFold itself ranks by pLDDT (default) or
    by ipTM for multimers — we trust its ordering.
    """
    output_dir = Path(output_dir)
    rows: list[dict] = []
    for pdb in sorted(output_dir.glob("*_unrelaxed_rank_*.pdb")):
        m = _PDB_RE.match(pdb.name)
        if not m:
            continue
        if name is not None and m.group("name") != name:
            continue
        rank = int(m.group("rank"))

        # Find the matching scores JSON: same name but ``_scores_`` instead of ``_unrelaxed_``
        scores_path = _find_scores_json(pdb)
        mean_plddt: float | None = None
        ptm: float | None = None
        model_number: int | None = None
        if scores_path is not None:
            with scores_path.open() as fh:
                scores = json.load(fh)
            plddt = scores.get("plddt") or scores.get("plddt_per_residue") or []
            if plddt:
                mean_plddt = float(sum(plddt) / len(plddt))
            ptm = float(scores["ptm"]) if "ptm" in scores else None
        mm = re.search(r"model_(\d+)", pdb.name)
        if mm:
            model_number = int(mm.group(1))

        rows.append(
            {
                "rank": rank,
                "pdb_path": pdb.resolve(),
                "mean_plddt": mean_plddt,
                "ptm": ptm,
                "model_number": model_number,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No ColabFold output PDBs found under {output_dir}")
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def best_model_path(output_dir: PathLike, *, name: str | None = None) -> Path:
    """Return the path to the rank-1 (best) predicted PDB."""
    df = parse_colabfold_ranking(output_dir, name=name)
    return Path(df.iloc[0]["pdb_path"])


def summarise_run(output_dir: PathLike, *, name: str | None = None) -> dict:
    """One-row summary of a ColabFold run.

    Returns a dict with ``n_models``, top-model pLDDT/pTM, and the path
    to the best model. Use for quick logging or for the notebook recap.
    """
    df = parse_colabfold_ranking(output_dir, name=name)
    top = df.iloc[0]
    return {
        "n_models": int(len(df)),
        "best_rank": int(top["rank"]),
        "best_mean_plddt": top["mean_plddt"],
        "best_ptm": top["ptm"],
        "best_pdb_path": Path(top["pdb_path"]),
    }


def _find_scores_json(pdb_path: Path) -> Path | None:
    candidate = pdb_path.parent / pdb_path.name.replace("_unrelaxed_", "_scores_").replace(
        ".pdb", ".json"
    )
    return candidate if candidate.exists() else None
