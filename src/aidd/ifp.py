"""Interaction fingerprints (IFPs) via ProLIF.

Wraps the standard ProLIF workflow into one function: read a protein and one or
more ligand poses from disk, return a tidy pandas DataFrame of interactions.

Adapted from cells in _archive/Week_3_Monday_Docking_and_Scoring.ipynb,
updated for ProLIF >=2.0 and with the course's non-default HB-angle widening
removed (per _planning/CONSULTANT_REVIEW.md §6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import pandas as pd
import prolif as plf
from rdkit import Chem

PathLike = Union[str, Path]


def load_plf_molecule(pdb_path: PathLike) -> plf.Molecule:
    """Read a PDB file and wrap it as a ProLIF Molecule.

    Works for both protein receptors and single-pose ligands. For multi-pose
    docked ligands use mol2 or SDF via ``compute_ifp`` instead.
    """
    pdb_path = Path(pdb_path)
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse {pdb_path}")
    return plf.Molecule(mol)


def compute_ifp(
    protein: PathLike,
    ligands: Union[PathLike, Iterable[plf.Molecule]],
    *,
    progress: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute a protein-ligand interaction fingerprint.

    Parameters
    ----------
    protein
        Path to a protonated protein PDB file.
    ligands
        One of:
            * Path to a single-pose ligand PDB.
            * Path to a multi-pose mol2 file (uses ``plf.mol2_supplier``).
            * Path to a multi-pose SDF (uses ``plf.sdf_supplier``).
            * Pre-built iterable of ``plf.Molecule``.
    progress
        Display a progress bar; useful for multi-pose runs.
    n_jobs
        Worker processes. Default 1 to sidestep the Windows "spawn" trap that
        requires an ``if __name__ == '__main__':`` guard at the call site.
        Increase for large multi-pose runs from inside Jupyter / a guarded script.

    Returns
    -------
    pandas.DataFrame
        Rows = poses, columns = MultiIndex ``(ligand_resid, protein_resid, interaction_type)``.
    """
    prot = load_plf_molecule(protein)

    if isinstance(ligands, (str, Path)):
        pose_iter = _supplier_for(Path(ligands))
    else:
        pose_iter = ligands

    fp = plf.Fingerprint()
    fp.run_from_iterable(pose_iter, prot, progress=progress, n_jobs=n_jobs)
    return fp.to_dataframe()


def _supplier_for(path: Path) -> Iterable[plf.Molecule]:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return [load_plf_molecule(path)]
    if suffix == ".mol2":
        return plf.mol2_supplier(str(path))
    if suffix in {".sdf", ".sd"}:
        return plf.sdf_supplier(str(path))
    raise ValueError(
        f"Unsupported ligand format: {suffix!r}. Expected .pdb, .mol2, .sdf or .sd."
    )


def to_wide_features(ifp_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten an IFP DataFrame to a one-row-per-pose 0/1 feature matrix.

    Collapses the ligand-residue level (rarely useful for small molecules with
    a single residue) and joins ``protein_resid`` + ``interaction_type`` into a
    single column name (e.g. ``LEU107.A_Hydrophobic``). Suitable as input for
    sklearn / XGBoost rescorers.
    """
    flat = ifp_df.T.groupby(level=[1, 2]).any().T
    flat.columns = [f"{resid}_{interaction}" for resid, interaction in flat.columns]
    return flat.astype(int)
