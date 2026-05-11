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
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolAlign

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
) -> pd.DataFrame:
    """Dock every ligand in ``ligands_sdf`` into ``receptor`` with gnina.

    Outputs ``out_dir/poses.sdf`` (all poses, all ligands), ``out_dir/gnina.log``
    (raw stdout/stderr), and ``out_dir/gnina_scores.csv`` (tidy scores table).
    Returns the scores table.

    Idempotent: if both ``poses.sdf`` and ``gnina_scores.csv`` already exist
    under ``out_dir`` we read the cached scores and skip the run, unless
    ``overwrite=True``.

    Parameters
    ----------
    cnn_scoring
        ``'rescore'`` (default, fast — Vina docks, CNN rescore the final pose),
        ``'refinement'`` (slower, CNN-guided refinement after Vina), or
        ``'all'`` (CNN at every step; slowest, highest accuracy when GPU is
        available). ``'none'`` disables CNN and gives Vina/smina behaviour.
    cpu
        Number of CPU threads for gnina. Default ``None`` lets gnina pick.
    gnina_extra
        Extra raw CLI flags appended after the standard arguments.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses_sdf = out_dir / "poses.sdf"
    scores_csv = out_dir / "gnina_scores.csv"
    log_path = out_dir / "gnina.log"

    if not overwrite and poses_sdf.exists() and scores_csv.exists():
        logger.info("Cached docking found at %s; skipping (overwrite=True to redo).", out_dir)
        return pd.read_csv(scores_csv)

    g = require_gnina()
    cmd: list[str] = [
        str(g),
        "-r", str(receptor),
        "-l", str(ligands_sdf),
        "-o", str(poses_sdf),
        "--cnn_scoring", cnn_scoring,
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--seed", str(seed),
        *box.gnina_args(),
    ]
    if cpu is not None:
        cmd += ["--cpu", str(cpu)]
    cmd += list(gnina_extra)

    logger.info("gnina: %s", " ".join(cmd))
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gnina exited with status {proc.returncode}. See {log_path} for stderr."
        )

    df = parse_poses_sdf(poses_sdf)
    df.to_csv(scores_csv, index=False)
    return df


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
            raise RuntimeError(
                f"gnina exited with status {proc.returncode}. See {log_path}."
            )

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

    gnina writes the following SD tags on every pose:

    - ``minimizedAffinity``  — Vina/smina affinity in kcal/mol (lower = stronger).
    - ``CNNscore``           — CNN classification score, 0–1 (higher = better).
    - ``CNNaffinity``        — CNN-predicted affinity in pK_d (higher = stronger).
    - ``CNN_VS``             — combined virtual-screening score (higher = better).
    - ``minimizedRMSD``      — RMSD between the docked and the gnina-minimised pose.

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

    Both files must hold the same chemical structure (matching atoms) — typical
    for a redock where the input ligand IS the reference. Hydrogens are stripped
    before RMSD (heavy-atom RMSD is the convention in docking literature).

    Uses ``rdMolAlign.CalcRMS``, which solves the substructure-matching problem
    so that symmetric atoms (e.g. swapped ortho-positions of a phenyl ring) are
    matched optimally. Coordinates are taken as-is (no rigid-body realignment)
    because gnina already writes poses in the receptor's coordinate frame.
    """
    pose_mol = _read_first_mol(Path(pose_path), index=pose_index)
    ref_mol = _read_first_mol(Path(ref_path), index=0)
    pose_mol = Chem.RemoveHs(pose_mol)
    ref_mol = Chem.RemoveHs(ref_mol)
    return float(rdMolAlign.CalcRMS(pose_mol, ref_mol))


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
