"""3D visualisation helpers for protein-ligand complexes using py3Dmol.

Composable workflow:

    view = show_protein_ligand("erk2_4fv7.pdb", "erk2_4fv7_ref.pdb",
                               ligand_resname="E94")
    show_binding_site(view, ligand_resname="E94")
    show_interactions_3d(view, fp, protein_plf, ligand_plf)
    view.zoomTo({"resn": "E94"})
    view.show()

Adapted from cells in _archive/Week_3_Monday_Docking_and_Scoring.ipynb.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import py3Dmol

PathLike = Union[str, Path]

INTERACTION_COLORS: dict[str, str] = {
    "Hydrophobic": "lime",
    "HBAcceptor": "red",
    "HBDonor": "blue",
    "PiStacking": "purple",
    "PiCation": "orange",
    "CationPi": "orange",
    "Anionic": "magenta",
    "Cationic": "cyan",
    "MetalDonor": "gray",
    "MetalAcceptor": "gray",
    "XBAcceptor": "brown",
    "XBDonor": "brown",
}


def show_protein_ligand(
    protein_pdb: PathLike,
    ligand_pdb: PathLike,
    *,
    ligand_resname: str | None = None,
    protein_color: str = "gold",
    ligand_colorscheme: str = "cyanCarbon",
    cartoon_opacity: float = 0.6,
    width: int = 600,
    height: int = 500,
) -> py3Dmol.view:
    """Create a py3Dmol view: protein as cartoon, ligand as sticks.

    Parameters
    ----------
    ligand_resname
        Three-letter residue name of the ligand (e.g. ``"E94"``). When given,
        styling is restricted to that residue; otherwise the entire second
        model is styled as sticks.
    """
    view = py3Dmol.view(width=width, height=height)
    view.removeAllModels()

    view.addModel(Path(protein_pdb).read_text(), format="pdb")
    view.setStyle({"cartoon": {"color": protein_color, "opacity": cartoon_opacity}})

    view.addModel(Path(ligand_pdb).read_text(), format="pdb")
    selector = {"resn": ligand_resname} if ligand_resname else {"model": 1}
    view.setStyle(selector, {"stick": {"colorscheme": ligand_colorscheme}})
    return view


def show_binding_site(
    view: py3Dmol.view,
    *,
    ligand_resname: str,
    radius: float = 5.0,
    colorscheme: str = "goldCarbon",
) -> py3Dmol.view:
    """Highlight protein residues within ``radius`` Å of the ligand as sticks."""
    selection = {"resn": ligand_resname, "byres": True, "expand": radius}
    view.addStyle(selection, {"stick": {"colorscheme": colorscheme}})
    return view


def show_interactions_3d(
    view: py3Dmol.view,
    fp,
    protein_plf,
    ligand_plf,
    *,
    pose_index: int = 0,
    radius: float = 0.15,
    dashed: bool = True,
    skip_types: tuple[str, ...] = ("VdWContact",),
) -> py3Dmol.view:
    """Overlay coloured interaction lines onto a py3Dmol view.

    Reads atom indices from ``fp.ifp[pose_index]`` and draws one cylinder per
    detected interaction, coloured by ``INTERACTION_COLORS``. Unknown
    interaction types are drawn in grey.

    Parameters
    ----------
    fp
        A fitted ``prolif.Fingerprint`` (i.e. ``fp.run_from_iterable`` was called).
    protein_plf, ligand_plf
        The ``plf.Molecule`` objects used to compute ``fp`` (needed to look up
        per-atom coordinates).
    pose_index
        Which pose to visualise when the ligand had multiple conformations.
    skip_types
        Interaction types to omit from the overlay. ``"VdWContact"`` is excluded
        by default because every binding-site residue typically has one, which
        clutters the view; pass ``skip_types=()`` to draw them.
    """
    pose = fp.ifp[pose_index]

    seen_residues: set[str] = set()
    for (lig_resid, prot_resid), interaction_dict in pose.items():
        residue_num = re.sub(r"\D", "", str(prot_resid))
        if residue_num and residue_num not in seen_residues:
            view.setStyle({"resi": residue_num}, {"stick": {"colorscheme": "goldCarbon"}})
            seen_residues.add(residue_num)

        protein_residue_conf = protein_plf[prot_resid].GetConformer()
        ligand_conf = ligand_plf.GetConformer()

        for interaction_type, entries in interaction_dict.items():
            if interaction_type in skip_types:
                continue
            color = INTERACTION_COLORS.get(interaction_type, "grey")
            for entry in entries:
                lig_idx = entry["indices"]["ligand"][0]
                prot_idx = entry["indices"]["protein"][0]
                p1 = ligand_conf.GetAtomPosition(lig_idx)
                p2 = protein_residue_conf.GetAtomPosition(prot_idx)
                view.addCylinder(
                    {
                        "start": {"x": p1.x, "y": p1.y, "z": p1.z},
                        "end": {"x": p2.x, "y": p2.y, "z": p2.z},
                        "color": color,
                        "radius": radius,
                        "dashed": dashed,
                        "fromCap": 1,
                        "toCap": 1,
                    }
                )
    return view


def show_interaction_network(fp, ligand_mol, *, kind: str = "frame", **kwargs):
    """Draw ProLIF's 2D LigNetwork diagram and return it for notebook display.

    Thin convenience wrapper around ``prolif.plotting.network.LigNetwork``; see
    that class for full kwargs. ``kind="frame"`` shows one pose,
    ``"aggregate"`` averages over all poses.

    Parameters
    ----------
    fp
        A fitted ``prolif.Fingerprint``.
    ligand_mol
        The ligand ``plf.Molecule`` used to compute ``fp`` (LigNetwork needs it
        for the 2D ligand drawing).
    """
    from prolif.plotting.network import LigNetwork

    return LigNetwork.from_fingerprint(fp, ligand_mol, kind=kind, **kwargs).display()
