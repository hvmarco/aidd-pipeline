"""Protein-structure utilities: PDB I/O, pLDDT extraction, distogram, alignment.

Adapted from cells in ``_archive/biopython.ipynb``. Used by the folding,
docking, and rescoring notebooks downstream of structure prediction.

AlphaFold / ColabFold convention: the per-residue confidence (pLDDT, 0–100) is
written into the ``B-factor`` column of the output PDB file. We expose
``extract_plddt`` to read it back, and ``distogram`` to summarise the geometry
as a Cα–Cα distance matrix (useful for visual fold QC and for comparing
predictions across models).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser, Superimposer

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def read_pdb(path: PathLike, *, structure_id: str | None = None):
    """Parse a PDB file with Biopython. Returns a ``Bio.PDB.Structure``."""
    path = Path(path)
    parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id or path.stem, str(path))


# ---------------------------------------------------------------------------
# pLDDT
# ---------------------------------------------------------------------------

def extract_plddt(path: PathLike, *, chain: str = "A") -> pd.Series:
    """Return per-residue pLDDT values from an AlphaFold / ColabFold PDB.

    pLDDT is stored in the B-factor column of the Cα atoms by convention.
    The returned Series is indexed by residue number; values are 0–100.
    """
    struct = read_pdb(path)
    rows: list[tuple[int, float]] = []
    for model in struct:
        for ch in model:
            if ch.id != chain:
                continue
            for residue in ch:
                if "CA" in residue:
                    rows.append((residue.id[1], float(residue["CA"].get_bfactor())))
        break
    if not rows:
        raise ValueError(f"No Cα atoms found in chain {chain!r} of {path}")
    return pd.Series(dict(rows), name="pLDDT")


def plddt_summary(plddt: pd.Series) -> dict:
    """Aggregate pLDDT statistics used to judge an AlphaFold model.

    Returns a dict with mean / median / per-band fractions. Bands follow the
    AlphaFold-conventional cut-offs:

    - ``very_high`` : pLDDT ≥ 90  (backbone trustworthy; side-chains often too)
    - ``confident`` : 70 ≤ pLDDT < 90
    - ``low``       : 50 ≤ pLDDT < 70
    - ``disordered``: pLDDT < 50  (likely flexible / unstructured)
    """
    return {
        "mean": float(plddt.mean()),
        "median": float(plddt.median()),
        "min": float(plddt.min()),
        "max": float(plddt.max()),
        "frac_very_high": float((plddt >= 90).mean()),
        "frac_confident": float(((plddt >= 70) & (plddt < 90)).mean()),
        "frac_low": float(((plddt >= 50) & (plddt < 70)).mean()),
        "frac_disordered": float((plddt < 50).mean()),
        "n_residues": int(len(plddt)),
    }


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def ca_coordinates(path: PathLike, *, chain: str = "A") -> np.ndarray:
    """Return Cα coordinates as an ``(N, 3)`` array, ordered by residue number."""
    struct = read_pdb(path)
    coords: list[tuple[int, np.ndarray]] = []
    for model in struct:
        for ch in model:
            if ch.id != chain:
                continue
            for residue in ch:
                if "CA" in residue:
                    coords.append((residue.id[1], residue["CA"].get_coord()))
        break
    if not coords:
        raise ValueError(f"No Cα atoms found in chain {chain!r} of {path}")
    coords.sort(key=lambda x: x[0])
    return np.vstack([c for _, c in coords])


def distogram(path: PathLike, *, chain: str = "A") -> np.ndarray:
    """Return the Cα–Cα distance matrix for ``chain`` of a PDB file.

    Useful for visual fold QC: a well-folded compact domain has a smooth
    block-diagonal pattern with dark off-diagonal blocks at long-range
    secondary-structure contacts. A noisy or disconnected distogram suggests a
    bad prediction.
    """
    ca = ca_coordinates(path, chain=chain)
    diff = ca[:, None, :] - ca[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


# ---------------------------------------------------------------------------
# Alignment + RMSD
# ---------------------------------------------------------------------------

def superpose_by_resnum(
    mobile_pdb: PathLike,
    reference_pdb: PathLike,
    out_pdb: PathLike,
    *,
    chain: str = "A",
    min_matched_residues: int = 10,
) -> dict:
    """Superpose a structure onto a reference by matched-residue-number Cα atoms.

    Writes the transformed mobile structure to ``out_pdb`` and returns a dict
    with the alignment RMSD plus the number of residues used to fit.

    The intended use case is bringing an AlphaFold model into the coordinate
    frame of a crystal structure so that crystal-frame binding-site
    coordinates apply to both (i.e. the pipeline can dock the AF model from
    notebook 01 using the binding-site center / radius derived from the
    crystal in the archived PLANTS config).

    The alignment uses Cα atoms whose residue numbers appear in *both*
    structures on the same chain. Crystal structures typically miss flexible
    loops; the matched-by-number subset is the well-resolved core, which is
    what you want anchoring the alignment. Side-chain conformations are not
    altered — only the rigid-body transform is applied.
    """
    parser = PDBParser(QUIET=True)
    ref = parser.get_structure("ref", str(reference_pdb))
    mob = parser.get_structure("mob", str(mobile_pdb))

    def ca_by_resnum(structure) -> dict[int, "object"]:
        for model in structure:
            for ch in model:
                if ch.id != chain:
                    continue
                return {r.id[1]: r["CA"] for r in ch if "CA" in r}
            break
        return {}

    ref_ca = ca_by_resnum(ref)
    mob_ca = ca_by_resnum(mob)
    common = sorted(set(ref_ca) & set(mob_ca))
    if len(common) < min_matched_residues:
        raise ValueError(
            f"Only {len(common)} residues match by number between "
            f"{Path(mobile_pdb).name} and {Path(reference_pdb).name} on chain "
            f"{chain!r} (need ≥ {min_matched_residues}). Cannot align."
        )

    sup = Superimposer()
    sup.set_atoms([ref_ca[n] for n in common], [mob_ca[n] for n in common])
    sup.apply(list(mob.get_atoms()))

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(mob)
    io.save(str(out_pdb))

    return {
        "rmsd": float(sup.rms),
        "n_matched_residues": len(common),
        "out_path": out_pdb,
    }


def ca_rmsd(ref_pdb: PathLike, target_pdb: PathLike, *, chain: str = "A") -> float:
    """Cα-RMSD between two structures of equal length, after rigid superposition.

    Both structures must have the same number of Cα atoms in ``chain``.
    Returns RMSD in Å. Useful for comparing a predicted model to a reference.
    """
    ref = read_pdb(ref_pdb)
    tgt = read_pdb(target_pdb)

    def _ca_atoms(structure):
        for model in structure:
            for ch in model:
                if ch.id != chain:
                    continue
                return [residue["CA"] for residue in ch if "CA" in residue]
        return []

    ref_atoms = _ca_atoms(ref)
    tgt_atoms = _ca_atoms(tgt)
    if len(ref_atoms) != len(tgt_atoms):
        raise ValueError(
            f"Cα count mismatch: ref has {len(ref_atoms)}, target has {len(tgt_atoms)}. "
            "Use sequence alignment before RMSD on different-length structures."
        )
    sup = Superimposer()
    sup.set_atoms(ref_atoms, tgt_atoms)
    return float(sup.rms)
