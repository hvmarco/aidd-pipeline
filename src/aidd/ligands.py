"""Ligand preparation pipeline.

End-to-end flow for a SMILES library:

1. ``read_smiles``           — parse a tab/space-separated SMILES file
2. ``standardise``           — neutralise charges, desalt, canonicalise
3. ``compute_properties``    — MW, LogP, HBD, HBA, TPSA, RotBonds, QED, rings
4. ``check_drug_likeness``   — Lipinski (Ro5) + Veber rules
5. ``flag_pains``            — PAINS A/B/C structural-alert match
6. ``embed_3d``              — ETKDGv3 conformer + MMFF (UFF fallback) optimisation
7. ``prepare`` / ``prepare_library`` — orchestrate steps 2-6 per molecule
8. ``write_sdf``             — output ready for docking

Adapted from cells in ``_archive/descriptors_qsar_lab.ipynb``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Union

import datamol as dm
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

PathLike = Union[str, Path]

logger = logging.getLogger("aidd.ligands")
RDLogger.DisableLog("rdApp.*")  # silence the per-molecule SMILES parse warnings


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_smiles(
    path: PathLike,
    *,
    smiles_col: int = 0,
    name_col: int | None = 1,
    sep: str | None = None,
) -> pd.DataFrame:
    """Read a SMILES file.

    Returns a DataFrame with columns ``['smiles', 'name']``. ``name`` is the
    empty string when ``name_col`` is ``None`` or the file has no second column.
    Whitespace (``sep=None``) and tab are auto-detected.
    """
    df = pd.read_csv(path, sep=sep, header=None, engine="python", dtype=str)
    out = pd.DataFrame({"smiles": df.iloc[:, smiles_col].astype(str).str.strip()})
    if name_col is not None and name_col < df.shape[1]:
        out["name"] = df.iloc[:, name_col].astype(str).str.strip()
    else:
        out["name"] = ""
    return out


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

def standardise(mol_or_smiles: Union[str, Chem.Mol]) -> Chem.Mol | None:
    """Neutralise, desalt, canonicalise; return ``None`` on failure."""
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            return None
    else:
        mol = mol_or_smiles
    try:
        return dm.standardize_mol(
            mol,
            disconnect_metals=False,
            normalize=True,
            reionize=True,
            uncharge=True,
            stereo=True,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def compute_properties(mol: Chem.Mol) -> dict[str, float]:
    """Common physicochemical descriptors + QED."""
    return {
        "mw":          Descriptors.MolWt(mol),
        "logp":        Descriptors.MolLogP(mol),
        "hbd":         Descriptors.NumHDonors(mol),
        "hba":         Descriptors.NumHAcceptors(mol),
        "tpsa":        Descriptors.TPSA(mol),
        "rotbonds":    Descriptors.NumRotatableBonds(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings":       Chem.rdMolDescriptors.CalcNumRings(mol),
        "qed":         QED.qed(mol),
    }


# ---------------------------------------------------------------------------
# Drug-likeness gates
# ---------------------------------------------------------------------------

LIPINSKI_RULES: dict[str, tuple[str, float]] = {
    "mw":   ("<=", 500.0),
    "logp": ("<=", 5.0),
    "hbd":  ("<=", 5.0),
    "hba":  ("<=", 10.0),
}

VEBER_RULES: dict[str, tuple[str, float]] = {
    "tpsa":     ("<=", 140.0),
    "rotbonds": ("<=", 10.0),
}


def _count_violations(props: dict, rules: dict) -> int:
    n = 0
    for col, (op, threshold) in rules.items():
        v = props[col]
        if (op == "<=" and v > threshold) or (op == ">=" and v < threshold):
            n += 1
    return n


def check_drug_likeness(props: dict, *, allow_lipinski_violations: int = 0) -> dict:
    """Flag drug-likeness vs Lipinski (Ro5) + Veber + QED."""
    lipinski_violations = _count_violations(props, LIPINSKI_RULES)
    veber_violations = _count_violations(props, VEBER_RULES)
    return {
        "lipinski_violations": lipinski_violations,
        "lipinski_pass":       lipinski_violations <= allow_lipinski_violations,
        "veber_violations":    veber_violations,
        "veber_pass":          veber_violations == 0,
        "qed_pass":            props["qed"] >= 0.5,
    }


# ---------------------------------------------------------------------------
# PAINS / structural alerts
# ---------------------------------------------------------------------------

_PAINS_CATALOG: FilterCatalog | None = None


def _pains_catalog() -> FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalogParams()
        for cat in (
            FilterCatalogParams.FilterCatalogs.PAINS_A,
            FilterCatalogParams.FilterCatalogs.PAINS_B,
            FilterCatalogParams.FilterCatalogs.PAINS_C,
        ):
            params.AddCatalog(cat)
        _PAINS_CATALOG = FilterCatalog(params)
    return _PAINS_CATALOG


def flag_pains(mol: Chem.Mol) -> str:
    """Return the description of the first matching PAINS rule, or '' if clean."""
    match = _pains_catalog().GetFirstMatch(mol)
    return match.GetDescription() if match is not None else ""


# ---------------------------------------------------------------------------
# 3D embedding
# ---------------------------------------------------------------------------

def embed_3d(
    mol: Chem.Mol,
    *,
    n_confs: int = 1,
    optimize: bool = True,
    seed: int = 42,
) -> Chem.Mol | None:
    """Generate 3D conformer(s) with ETKDGv3 + MMFF optimisation.

    Falls back to UFF if MMFF fails. Returns ``None`` if embedding fails.
    """
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if n_confs == 1:
            if AllChem.EmbedMolecule(mol, params) == -1:
                return None
        else:
            ids = AllChem.EmbedMultipleConfs(mol, n_confs, params)
            if len(ids) == 0:
                return None
        if optimize:
            try:
                AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
            except Exception:
                AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=200)
        return mol
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single-molecule end-to-end
# ---------------------------------------------------------------------------

def prepare(mol_or_smiles: Union[str, Chem.Mol], *, n_confs: int = 1) -> dict:
    """Standardise → properties → drug-likeness → PAINS → 3D embed for one molecule.

    Returns a dict with every intermediate result, plus:
        - ``mol``: 3D-embedded RDKit Mol (or ``None``)
        - ``ok``: bool, True iff every step succeeded
        - ``fail_reason``: which step failed when ``ok`` is False
    """
    out: dict = {"ok": False, "mol": None, "fail_reason": ""}
    std = standardise(mol_or_smiles)
    if std is None:
        out["fail_reason"] = "standardise"
        return out
    out["smiles_std"] = Chem.MolToSmiles(std)
    props = compute_properties(std)
    out.update(props)
    out.update(check_drug_likeness(props))
    out["pains"] = flag_pains(std)
    mol3d = embed_3d(std, n_confs=n_confs)
    if mol3d is None:
        out["fail_reason"] = "embed_3d"
        return out
    out["mol"] = mol3d
    out["ok"] = True
    return out


# ---------------------------------------------------------------------------
# Library-level driver
# ---------------------------------------------------------------------------

def prepare_library(
    df: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
    n_confs: int = 1,
    n_workers: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    """Run :func:`prepare` over a DataFrame of SMILES.

    Uses ``datamol.parallelized`` (joblib/loky) for parallel execution where
    worth it. ``n_workers=1`` for serial / debugging — recommended in Jupyter
    on Windows unless you've verified the spawn idiom works for you.
    """
    smiles_list = df[smiles_col].tolist()
    results = dm.parallelized(
        lambda smi: prepare(smi, n_confs=n_confs),
        smiles_list,
        n_jobs=n_workers,
        progress=progress,
        tqdm_kwargs={"desc": "prepare"},
    )
    results_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_sdf(
    df: pd.DataFrame,
    path: PathLike,
    *,
    mol_col: str = "mol",
    id_col: str = "name",
    props_to_write: Iterable[str] | None = None,
) -> int:
    """Write prepared molecules to an SDF. Returns the count actually written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(props_to_write) if props_to_write else []
    writer = Chem.SDWriter(str(path))
    n = 0
    try:
        for _, row in df.iterrows():
            mol = row.get(mol_col)
            if mol is None or not isinstance(mol, Chem.Mol):
                continue
            mol = Chem.Mol(mol)
            if id_col in row and row[id_col]:
                mol.SetProp("_Name", str(row[id_col]))
            for col in cols:
                if col in row and row[col] is not None and not (isinstance(row[col], float) and pd.isna(row[col])):
                    mol.SetProp(col, str(row[col]))
            writer.write(mol)
            n += 1
    finally:
        writer.close()
    return n
