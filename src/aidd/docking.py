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

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

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
        raise RuntimeError(_format_gnina_error(proc.returncode, log_path))

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
