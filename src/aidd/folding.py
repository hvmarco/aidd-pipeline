"""Wrapper utilities around ColabFold output, plus an AF3-result parser.

ColabFold itself runs in the notebook (it needs a GPU and a special install
on Colab; we don't try to install it as part of the ``aidd`` conda env). The
helpers in this module read and summarise its **outputs**:

- ``parse_colabfold_ranking`` — collect the five (or N) predicted models with
  their pLDDT / pTM scores into a tidy DataFrame.
- ``best_model_path`` — return the rank-1 PDB.
- ``summarise_run`` — single-row report of the run.
- ``fold_with_af3_server`` — parser-only helper for AlphaFold Server (AF3)
  result zips. AF3 Server has no public programmatic API as of 2026-05; the
  operator submits the fold via the web UI under Google's academic-access
  terms, downloads the result zip, and points this helper at it. See the
  function docstring for the operator workflow.

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
import logging
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]
logger = logging.getLogger("aidd.folding")

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


# ===========================================================================
# AlphaFold Server (AF3) — parser-only helper
# ===========================================================================

def fold_with_af3_server(
    af3_result_zip: PathLike,
    output_dir: PathLike,
    *,
    target_name: str = "target",
) -> Path:
    """Parse a downloaded AlphaFold Server (AF3) result zip into the canonical PDB.

    **This is a parser, not an API client.** AlphaFold Server has no public
    programmatic submission API as of 2026-05; the operator submits the fold
    via the web UI at https://alphafoldserver.com under Google's academic
    non-commercial access terms (per-user application, ~30 jobs/day quota),
    downloads the result zip, and points this helper at it.

    Operator workflow
    -----------------
    1. Open https://alphafoldserver.com (signed in with the academic-access
       account).
    2. Submit a single-sequence protein-only job (paste the FASTA sequence
       into the visual builder, or upload a job-request JSON via the
       "Upload JSON" button).
    3. When the job completes, click "Download" to get a zip named
       ``fold_<job_name>.zip``.
    4. Pass the path to that zip into this helper. It unzips into a temporary
       directory, finds the rank-1 model CIF, converts to PDB via Biopython's
       MMCIF parser, and writes ``<output_dir>/<target_name>_best.pdb``.

    Cross-platform: ``zipfile`` is stdlib; Biopython is a conda-forge
    dependency on Win / Mac / Linux. No network, no auth, no GPU. The same
    helper works on Natallia's Windows env and on Colab CPU.

    Parameters
    ----------
    af3_result_zip
        Path to the AF3 Server result zip downloaded from the web UI.
    output_dir
        Directory to write the canonical ``<target_name>_best.pdb`` into.
        Created if missing.
    target_name
        Stem for the output filename. Default ``"target"``. Notebook 99 sets
        this to the per-target slug so the file matches the
        ``data/derived/<target>/fold/<target>_best.pdb`` convention used by
        the ColabFold branch.

    Returns
    -------
    Path
        Absolute path to the written PDB.

    Raises
    ------
    FileNotFoundError
        ``af3_result_zip`` does not exist on disk.
    ValueError
        ``af3_result_zip`` is not a ``.zip``, contains no ``.cif`` files, or
        the rank-1 CIF cannot be parsed by Biopython's MMCIF reader.

    Notes
    -----
    AF3 zip layout (current as of 2026-05): one CIF per ranked model named
    ``fold_<job>_model_<rank>.cif`` (0-indexed; ``model_0`` is the top-ranked
    by AF3's combined confidence). The helper prefers ``*model_0.cif``; on
    no match it falls back to the lexically-first ``.cif`` in the zip (AF3's
    naming sorts ``model_0`` first) and logs a WARNING so a future layout
    change surfaces visibly.

    Per the ``FOLD_PROVIDER`` toggle in notebook 99 (see
    ``_planning/PROJECT_PROPOSAL.md`` §9), both this branch and the ColabFold
    branch produce the same canonical ``<target_name>_best.pdb`` so
    downstream cells (docking, scoring, consensus) are agnostic to which
    folder was used. AF3 outputs are personal data per Google's academic
    terms; the project's ``.gitignore`` on ``data/derived/`` already
    prevents commits.
    """
    af3_result_zip = Path(af3_result_zip)
    output_dir = Path(output_dir)

    if not af3_result_zip.exists():
        raise FileNotFoundError(
            f"AF3 result zip not found: {af3_result_zip}. Download the result "
            f"from https://alphafoldserver.com and pass the path to the zip."
        )
    if af3_result_zip.suffix.lower() != ".zip":
        raise ValueError(
            f"Expected a .zip from AF3 Server; got {af3_result_zip!r}"
        )

    try:
        from Bio.PDB import PDBIO
        from Bio.PDB.MMCIFParser import MMCIFParser
    except ImportError as exc:
        raise ImportError(
            "fold_with_af3_server requires Biopython for CIF parsing. "
            "Install with `conda install -c conda-forge biopython` or "
            "`pip install biopython`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = output_dir / f"{target_name}_best.pdb"

    with zipfile.ZipFile(af3_result_zip) as zf:
        cif_names = [n for n in zf.namelist() if n.lower().endswith(".cif")]
        if not cif_names:
            raise ValueError(
                f"No .cif files inside {af3_result_zip}. AF3 Server result "
                f"zips contain one .cif per ranked model; this zip may be "
                f"corrupt or from a different tool."
            )
        rank1_matches = [n for n in cif_names if "model_0" in Path(n).name]
        if rank1_matches:
            rank1_name = sorted(rank1_matches)[0]
        else:
            rank1_name = sorted(cif_names)[0]
            logger.warning(
                "fold_with_af3_server: no *model_0.cif found in %s; falling "
                "back to lexically-first CIF %r. AF3 output layout may have "
                "changed -- verify the rank-1 model is correct.",
                af3_result_zip, rank1_name,
            )

        with tempfile.TemporaryDirectory(prefix="af3_unzip_") as tmpdir:
            zf.extract(rank1_name, tmpdir)
            cif_path = Path(tmpdir) / rank1_name
            try:
                structure = MMCIFParser(QUIET=True).get_structure("af3", str(cif_path))
            except Exception as exc:
                raise ValueError(
                    f"Biopython could not parse the AF3 CIF {rank1_name!r} "
                    f"inside {af3_result_zip}: {exc}"
                ) from exc
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(out_pdb))

    logger.info(
        "fold_with_af3_server: parsed %s (rank-1 CIF %s) -> %s",
        af3_result_zip, rank1_name, out_pdb,
    )
    return out_pdb
