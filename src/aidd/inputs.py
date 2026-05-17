"""Input-preparation helpers for the small-N MoA driver (notebook 09).

Three additive helpers - none touch existing modules:

- :func:`fetch_pdb`         - download a PDB by ID from rcsb.org with on-disk
  caching.
- :func:`prep_receptor`     - clean a raw rcsb PDB for docking via PDBFixer
  (waters, alt-confs, missing residues / sidechains); per-target cofactor
  presets available via ``target_name=`` against :data:`RECEPTOR_PREPS`.
- :func:`name_to_smiles`    - PubChem REST lookup; returns canonical SMILES of
  the parent compound (post-salt-dissociation).

The :data:`RECEPTOR_PREPS` dict ships sensible cofactor defaults for the five
notebook-09 demo targets (NAT2, CYP2D6, DPYD, UGT1A1, KRAS); callers can
override per-call with the explicit ``keep_cofactors=[...]`` kwarg.

Cross-platform: rcsb.org + PubChem are HTTPS endpoints; PDBFixer is conda-forge
+ pip-installable on Win / Mac / Linux. No GPU dependency.

Upstream
--------
- RCSB PDB    - ``https://files.rcsb.org/download/<pdb_id>.pdb``.
- PubChem PUG REST - ``https://pubchem.ncbi.nlm.nih.gov/rest/pug``. The
  legacy ``CanonicalSMILES`` property was deprecated in 2024-2025; the
  current property name is ``SMILES`` (stereo-aware) and the connectivity-
  only variant is ``ConnectivitySMILES``. We ask for ``SMILES`` because the
  five-demo set includes tamoxifen, whose (Z)-double-bond stereochemistry is
  clinically load-bearing.
- PDBFixer    - Eastman et al., OpenMM. https://github.com/openmm/pdbfixer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence, Union
from urllib.parse import quote

import requests

PathLike = Union[str, Path]
logger = logging.getLogger("aidd.inputs")


# ---------------------------------------------------------------------------
# Endpoints + defaults
# ---------------------------------------------------------------------------

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

PUBCHEM_NAME_TO_SMILES_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/SMILES/JSON"
)

DEFAULT_REQUEST_TIMEOUT = 30  # seconds


def _default_cache_dir(subdir: str) -> Path:
    """Default cache root under ``<cwd>/data/cache/<subdir>``.

    Mirrors :func:`aidd.variants._default_cache_dir`'s convention so
    notebook 09 can point the PDB cache at Google Drive on Colab via the
    same ``cache_dir=`` override pattern used by AlphaMissense / gnomAD.
    """
    return Path.cwd() / "data" / "cache" / subdir


# ---------------------------------------------------------------------------
# Per-target cofactor presets - see notebook 09 (small-N MoA driver).
# ---------------------------------------------------------------------------

# Each entry: target_name -> {"keep_cofactors": [<3-letter RCSB residue codes>]}.
# When :func:`prep_receptor` is called with ``target_name=<key>``, the preset
# ``keep_cofactors`` list is used UNLESS the caller passes an explicit
# ``keep_cofactors=[...]`` (explicit always wins).
#
# Codes are PDB ligand-residue 3-letter identifiers, NOT compound names:
#   HEM = heme (CYP2D6 prosthetic group)
#   FAD = flavin adenine dinucleotide (DPYD redox cofactor)
#   NAP = NADP+ / NADPH (DPYD reductant)
#   FMN = flavin mononucleotide (DPYD electron-transfer)
#   SF4 = [4Fe-4S] iron-sulfur cluster (DPYD electron-transfer)
#   GDP = guanosine diphosphate (KRAS substrate; sotorasib targets the
#         GDP-bound state. Sotorasib itself is the ligand we're docking,
#         not a cofactor - it gets stripped by the heterogen filter.)
#
# Limitation: a target's cofactor requirements can differ across PDB
# structures (apo vs holo crystals). For the five step-16 demo targets the
# single-key preset suffices. If a future agent needs (target_name, pdb_id)
# resolution, refactor this dict to a nested {target: {pdb_id: {...}}} form.
RECEPTOR_PREPS: dict[str, dict] = {
    "NAT2":   {"keep_cofactors": []},
    "CYP2D6": {"keep_cofactors": ["HEM"]},
    "DPYD":   {"keep_cofactors": ["FAD", "NAP", "FMN", "SF4"]},
    "UGT1A1": {"keep_cofactors": []},
    "KRAS":   {"keep_cofactors": ["GDP"]},
}


# ===========================================================================
# fetch_pdb
# ===========================================================================

def fetch_pdb(
    pdb_id: str,
    *,
    cache_dir: PathLike | None = None,
) -> Path:
    """Download a PDB by ID from rcsb.org with on-disk caching.

    Idempotent: returns the cached path if present, downloads if not.

    Parameters
    ----------
    pdb_id
        4-character RCSB PDB identifier (case-insensitive; cached as uppercase).
    cache_dir
        Directory to cache downloaded PDBs in. Defaults to
        ``<cwd>/data/cache/rcsb/``.

    Returns
    -------
    Path
        The cached PDB file.

    Raises
    ------
    ValueError
        If ``pdb_id`` is not a 4-character alphanumeric string.
    FileNotFoundError
        If rcsb.org returns HTTP 404 (PDB ID does not exist).
    ConnectionError
        On other network failures (timeout, HTTP 5xx, connection refused).
    """
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError(
            f"PDB ID must be 4 alphanumeric characters; got {pdb_id!r}."
        )

    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_dir("rcsb")
    cache_root.mkdir(parents=True, exist_ok=True)

    out_path = cache_root / f"{pdb_id}.pdb"
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("fetch_pdb: cache hit at %s", out_path)
        return out_path

    url = RCSB_PDB_URL.format(pdb_id=pdb_id)
    logger.info("fetch_pdb: downloading %s from %s", pdb_id, url)
    try:
        response = requests.get(url, timeout=DEFAULT_REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise ConnectionError(
            f"RCSB download failed for {pdb_id}: {exc}"
        ) from exc

    if response.status_code == 404:
        raise FileNotFoundError(
            f"RCSB PDB {pdb_id!r} not found (HTTP 404). Verify the PDB ID at "
            f"https://www.rcsb.org/structure/{pdb_id}. For targets without a "
            "human crystal structure (e.g. UGT1A1) use an AlphaFold model via "
            "notebook 01 instead."
        )
    if response.status_code != 200:
        raise ConnectionError(
            f"RCSB download failed for {pdb_id}: HTTP {response.status_code}"
        )

    out_path.write_bytes(response.content)
    logger.info(
        "fetch_pdb: cached %d bytes at %s", len(response.content), out_path,
    )
    return out_path


# ===========================================================================
# prep_receptor
# ===========================================================================

def prep_receptor(
    pdb_path: PathLike,
    *,
    keep_cofactors: Sequence[str] | None = None,
    target_name: str | None = None,
    drop_waters: bool = True,
    chain: Union[str, Sequence[str]] = "A",
    pick_alt_conf: str = "A",
    out_path: PathLike | None = None,
) -> Path:
    """Clean a raw rcsb PDB for docking.

    Operations (in order):

    1. Select the specified ``chain`` (default ``"A"``); accepts a single
       string for one chain or a sequence of strings for multi-chain
       complexes (e.g. ``chain=["A", "B"]`` for a homodimer).
    2. Pick alternative-conformation ``pick_alt_conf`` (default ``"A"``);
       atoms with altloc ``""``, ``" "``, or ``pick_alt_conf`` are kept.
    3. Remove crystallographic waters (``drop_waters=True``).
    4. Remove heterogens (cofactors, co-crystal ligands) except those in
       ``keep_cofactors`` (3-letter RCSB residue codes, e.g. ``["HEM"]``
       for the heme group). If ``target_name`` is given and matches a key
       in :data:`RECEPTOR_PREPS`, the preset list is used unless
       ``keep_cofactors`` is also passed explicitly (explicit always wins).
    5. Add missing residues + missing heavy atoms via PDBFixer.

    Hydrogens are NOT added - downstream docking tools (gnina, Boltz-2)
    handle protonation. Protonation-state assignment (Reduce / PROPKA) is
    out of scope here; if a downstream caller needs explicit pKa
    adjustment, run that tool on the output of this helper.

    Parameters
    ----------
    pdb_path
        Path to a raw rcsb PDB file (e.g. from :func:`fetch_pdb`).
    keep_cofactors
        Sequence of 3-letter RCSB residue codes for heterogens to retain.
        Resolution order: explicit ``keep_cofactors`` > ``RECEPTOR_PREPS``
        preset via ``target_name`` > empty list.
    target_name
        Optional preset key (e.g. ``"CYP2D6"``); see :data:`RECEPTOR_PREPS`.
    drop_waters
        Whether to remove water residues. Default ``True``.
    chain
        Chain identifier(s) to retain. Default ``"A"``.
    pick_alt_conf
        Alt-conformation identifier to retain. Default ``"A"``.
    out_path
        Output path. Defaults to ``<pdb_path-stem>.prepped.pdb`` next to
        the input.

    Returns
    -------
    Path
        The cleaned PDB.

    Notes
    -----
    Idempotent: if ``out_path`` already exists with non-zero size, it is
    returned without re-running PDBFixer.
    """
    from Bio.PDB import PDBIO, PDBParser, Select
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as exc:
        raise ImportError(
            "prep_receptor requires PDBFixer (and its openmm dependency). "
            "Install with `conda install -c conda-forge pdbfixer` or "
            "`pip install pdbfixer`."
        ) from exc

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {pdb_path}")

    # --- Resolve cofactor whitelist ---
    if keep_cofactors is None and target_name is not None:
        preset = RECEPTOR_PREPS.get(target_name)
        if preset is None:
            logger.warning(
                "prep_receptor: target_name=%r has no preset in RECEPTOR_PREPS "
                "(known: %s); proceeding with keep_cofactors=[]",
                target_name, sorted(RECEPTOR_PREPS),
            )
            keep_cofactors = []
        else:
            keep_cofactors = list(preset["keep_cofactors"])
    elif keep_cofactors is None:
        keep_cofactors = []
    keep_cofactors_set = {c.strip().upper() for c in keep_cofactors}

    # --- Normalise chain selector ---
    chain_set = {chain} if isinstance(chain, str) else set(chain)

    # --- Resolve output path ---
    out_path = (
        Path(out_path) if out_path is not None
        else pdb_path.with_suffix(".prepped.pdb")
    )
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("prep_receptor: cache hit at %s", out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: BioPython pre-filter (chain + alt-conf + selective heterogens) ---
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))

    class _PrepSelect(Select):
        def accept_chain(self, ch):
            return ch.id in chain_set

        def accept_residue(self, residue):
            hetflag, _resseq, _icode = residue.id
            if hetflag == " ":  # standard amino acid
                return True
            if hetflag == "W":  # water
                return not drop_waters
            return residue.get_resname().strip().upper() in keep_cofactors_set

        def accept_atom(self, atom):
            altloc = atom.get_altloc()
            return altloc in ("", " ", pick_alt_conf)

    intermediate_path = out_path.with_suffix(".filtered.pdb")
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(str(intermediate_path), _PrepSelect())

    # --- Stage 2: PDBFixer (missing residues + heavy atoms) ---
    # Skip findNonstandardResidues / replaceNonstandardResidues so kept cofactors
    # (HEM, FAD, etc.) are not accidentally rewritten.
    fixer = PDBFixer(filename=str(intermediate_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    with open(out_path, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

    intermediate_path.unlink(missing_ok=True)
    logger.info("prep_receptor: wrote cleaned PDB to %s", out_path)
    return out_path


# ===========================================================================
# name_to_smiles
# ===========================================================================

def name_to_smiles(name: str) -> str | None:
    """Look up the canonical SMILES of a compound by name via PubChem.

    Returns the canonical SMILES of the *parent* compound (post-salt-
    dissociation). Returns ``None`` if PubChem has no match.

    Handles brand names, generic names, IUPAC names, and synonyms. PubChem's
    salt dissociation means ``"tamoxifen citrate"`` and ``"tamoxifen"`` both
    return the SMILES of the parent compound (without the citrate counter-ion).

    Asks PubChem for the ``SMILES`` property (the current canonical name -
    the legacy ``CanonicalSMILES`` was deprecated in 2024-2025); ``SMILES``
    is stereo-aware, so E/Z and chiral information are preserved when
    available in PubChem's record.

    When RDKit detects stereo features (sp3 chiral centers or E/Z double
    bonds) in the returned SMILES, a warning is logged naming the compound
    and the feature counts. The clinically active form may not match
    PubChem's canonical SMILES (e.g. tamoxifen is sold as a racemate but
    the (Z)-isomer is active); the caller is expected to verify chirality
    where it matters.

    Parameters
    ----------
    name
        Compound name. URL-quoted internally; spaces and special characters
        are handled.

    Returns
    -------
    str | None
        Canonical SMILES if PubChem has a match, else ``None``.

    Raises
    ------
    ValueError
        If ``name`` is empty or whitespace-only.
    ConnectionError
        On network failures (timeout, HTTP 5xx, connection refused). HTTP
        404 is treated as "no match" and returns ``None``.
    """
    if not name or not name.strip():
        raise ValueError("name_to_smiles: name must be a non-empty string")

    name = name.strip()
    safe_name = quote(name, safe="")
    url = PUBCHEM_NAME_TO_SMILES_URL.format(name=safe_name)

    try:
        response = requests.get(url, timeout=DEFAULT_REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise ConnectionError(
            f"PubChem lookup failed for {name!r}: {exc}"
        ) from exc

    if response.status_code == 404:
        logger.info("name_to_smiles: PubChem has no match for %r", name)
        return None
    if response.status_code != 200:
        raise ConnectionError(
            f"PubChem lookup failed for {name!r}: HTTP {response.status_code}"
        )

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise ConnectionError(
            f"PubChem returned non-JSON response for {name!r}: {exc}"
        ) from exc

    props = payload.get("PropertyTable", {}).get("Properties", [])
    if not props:
        logger.info(
            "name_to_smiles: PubChem returned empty Properties for %r", name,
        )
        return None

    # Robust to API evolution: prefer SMILES (current), fall back to legacy
    # property names if a future PubChem update reshuffles the schema again.
    first = props[0]
    smiles = (
        first.get("SMILES")
        or first.get("IsomericSMILES")
        or first.get("CanonicalSMILES")
        or first.get("ConnectivitySMILES")
    )
    if not smiles:
        logger.info(
            "name_to_smiles: PubChem result for %r had no SMILES property "
            "(keys: %s)", name, sorted(first),
        )
        return None

    _warn_if_stereochemistry(name, smiles)
    return smiles


def _warn_if_stereochemistry(name: str, smiles: str) -> None:
    """Log a warning if RDKit detects stereo features in ``smiles``.

    Catches both sp3 chiral centers (via ``FindMolChiralCenters``) and E/Z
    double-bond stereo (via ``Bond.GetStereo``). Tamoxifen is the canonical
    case for the latter - PubChem returns its (Z)-isomer SMILES, but RDKit's
    chiral-center finder alone would not flag it.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_bond = sum(
        1 for b in mol.GetBonds()
        if b.GetStereo() != Chem.BondStereo.STEREONONE
    )
    if n_chiral == 0 and n_bond == 0:
        return
    parts: list[str] = []
    if n_chiral:
        parts.append(
            f"{n_chiral} stereocenter{'' if n_chiral == 1 else 's'}"
        )
    if n_bond:
        parts.append(
            f"{n_bond} E/Z double bond{'' if n_bond == 1 else 's'}"
        )
    feature_desc = " + ".join(parts)
    logger.warning(
        "name_to_smiles: %r has %s - verify clinically active form matches "
        "PubChem canonical SMILES (%s)",
        name, feature_desc, smiles,
    )


# ===========================================================================
# assert_wt_residue
# ===========================================================================

def assert_wt_residue(
    pdb_path: PathLike,
    chain: str,
    position: int,
    expected_wt_aa: str,
) -> None:
    """Assert that ``pdb_path`` has ``expected_wt_aa`` at ``(chain, position)``.

    Lightweight BioPython probe used pre-mutation to verify a stub fixture's
    source PDB is actually WT at the variant site. Codifies the pre-commit
    discipline that catches the class of mistake where a famous variant
    co-crystal is used as the WT reference -- the 6OIM = KRAS G12C +
    sotorasib case at step-16 commit 3, where residue 12 in 6OIM is already
    CYS rather than the expected GLY (one Colab iteration of cost when not
    caught locally).

    BioPython-only; does NOT require PDBFixer. Cross-platform.

    Parameters
    ----------
    pdb_path
        Path to a PDB file (raw rcsb download or :func:`prep_receptor`
        output -- either works, this helper only reads).
    chain
        Chain identifier (e.g. ``"A"``).
    position
        Residue number to check.
    expected_wt_aa
        1-letter amino acid code expected at ``(chain, position)``.

    Raises
    ------
    ValueError
        If ``expected_wt_aa`` is not a recognised 1-letter code.
    AssertionError
        On chain absence, residue absence, or residue-name mismatch.
        Error message names the actual residue + a hint about the most
        common mismatch class (variant co-crystal used as WT reference).
    """
    from Bio.PDB import PDBParser

    AA_1TO3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    AA_3TO1 = {v: k for k, v in AA_1TO3.items()}
    if expected_wt_aa not in AA_1TO3:
        raise ValueError(
            f"expected_wt_aa must be a 1-letter amino acid code; got "
            f"{expected_wt_aa!r}"
        )
    expected_3 = AA_1TO3[expected_wt_aa]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))

    for model in structure:
        chains_present = [c.id for c in model]
        if chain not in chains_present:
            raise AssertionError(
                f"chain {chain!r} not found in {pdb_path}; "
                f"chains present: {chains_present}"
            )
        ch = model[chain]
        try:
            residue = ch[position]
        except KeyError:
            raise AssertionError(
                f"residue {position} not found in chain {chain!r} of "
                f"{pdb_path}; possible numbering offset (signal-peptide "
                f"trimming, engineered crystallization construct, etc.)"
            )
        actual_3 = residue.get_resname().strip().upper()
        if actual_3 != expected_3:
            actual_1 = AA_3TO1.get(actual_3, "?")
            raise AssertionError(
                f"PDB-identity mismatch: {pdb_path} chain {chain!r} "
                f"residue {position} is {actual_3} ({actual_1}), expected "
                f"{expected_3} ({expected_wt_aa}). Most common cause: the "
                f"chosen PDB is the variant complex itself (e.g. 6OIM = "
                f"KRAS G12C+sotorasib has CYS at residue 12, not the GLY "
                f"that would be in a WT KRAS reference like 4OBE). Pick a "
                f"WT PDB for the structural-diff baseline."
            )
        return  # match; first model is enough
    raise AssertionError(f"no models found in {pdb_path}")


__all__ = [
    "RECEPTOR_PREPS",
    "assert_wt_residue",
    "fetch_pdb",
    "name_to_smiles",
    "prep_receptor",
]
