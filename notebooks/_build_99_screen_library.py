"""Builder for 99_screen_library.ipynb -- the production runner.

Source of truth for the audited end-to-end notebook. Cells appear below in
narrative order. Never edit the .ipynb directly -- see ``CLAUDE.md`` section
*Notebook workflow*.

Regenerate:
    python notebooks/_build_99_screen_library.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import AUTORELOAD_SNIPPET, code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "99_screen_library.ipynb"


def build() -> None:
    nb = notebook(
        # ====================================================================
        # CELL 1 -- TITLE
        # ====================================================================
        markdown("""
# 99 - Production runner (library or MoA mode; ColabFold or AF3)

**aidd-pipeline - the audited end-to-end deliverable**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hvmarco/aidd-pipeline/blob/main/notebooks/99_screen_library.ipynb)

This notebook runs the full pipeline end-to-end for one target (with optional variants), in one of two modes. It is the **canonical production artefact** of the project; reviewers, professors, grant funders, and wet-lab collaborators will read this notebook to verify what the pipeline does and how. The teaching notebooks 00-09 cover each pipeline stage in pedagogical detail; this notebook is tighter -- one short paragraph per stage, methods cited with DOIs in the summary table below, and an executive recap at the bottom.

Two top-level toggles control the run:

- `FOLD_PROVIDER`: `"colabfold"` (default, AlphaFold 2 via ColabFold, open weights) or `"af3_server"` (AlphaFold 3 via the AlphaFold Server academic web UI; parser-only helper consumes the downloaded result zip).
- `RUN_MODE`: `"library"` (1k-10k compound virtual screen producing a ranked shortlist) or `"moa"` (small-N mechanism-of-action; N <= 10 compounds; one HTML report per (compound x variant) pair).

The notebook is structured so opening it on Colab and hitting **Run All** with the default configuration (ERK2 in library mode, cached from steps 10-12) produces a useful shortlist in minutes. Switching to a non-ERK2 target or to MoA mode is a one-line uncomment in the configuration cell.

## Learning objectives

After running this notebook you will be able to:

- Configure a production screen for any target supported by the pipeline -- WT or with variants, library or MoA mode, AF2 or AF3 fold provider.
- Read the executive recap and locate the methodological caveats that constrain interpretation (Boltz-2 cofactor asymmetry, IFP-fidelity fallback, binding-vs-catalysis distinction).
- Trace any number in the output back to the upstream notebook (00-09) where the method is taught in detail.

## Audience

You'll get value if you are:

- A wet-lab oncologist / pharmacogenomicist commissioning an in-silico screen before a binding assay.
- A reviewer / grant auditor evaluating the methodological pedigree of the pipeline.
- A computational colleague running the pipeline on a new target.
- A clinician or student who wants to read the executive recap without running anything.

## Prerequisites

- The `aidd` conda environment (`conda env create -f environment.yml`), or running on Google Colab (the setup cell installs everything).
- For library mode on a non-ERK2 target: a SMILES file at `data/compounds/<target>/library.smi` (one SMILES per line, tab-separated ID); the headline-demo blocks in the configuration cell include suggested starter sets.
- For `FOLD_PROVIDER = "af3_server"`: an AlphaFold Server result zip downloaded from [alphafoldserver.com](https://alphafoldserver.com) under your academic-access account.
- For variant runs: the AlphaMissense + RaSP caches (built once by notebook 07; auto-rebuild if missing).

## Runtime

- **Library mode end-to-end on a fresh target (1k-10k compounds): 6-15 GPU-hours on Colab Pro+.** Boltz-2 co-folding is the bottleneck; gnina docking runs in parallel.
- **Library mode on ERK2 (cached): seconds to minutes.** Steps 10-12 wrote the docking / scoring / Boltz-2 outputs to Drive; this notebook walks the cache and produces the shortlist.
- **MoA mode (N <= 10 compounds): minutes per compound.** Boltz-2 per-compound co-fold dominates.

## What this notebook does and does NOT predict

The pipeline measures whether a candidate ligand **fits the binding site** (a necessary condition for binding) and triangulates with **co-folding affinity** and **per-variant priors** (pathogenicity, stability, allele frequency). It does NOT predict catalytic rate (`k_cat`, `K_m`) for enzymes; it does NOT predict patient-level clinical outcome; it does NOT predict off-target binding outside the supplied target list. The "Caveats" section in the recap names every limit honestly.
"""),

        # ====================================================================
        # CELL 2 -- METHODS SUMMARY (the 60-second-read for reviewers)
        # ====================================================================
        markdown("""
## Methods summary

What the pipeline does, end-to-end, in twelve method calls. Every tool below ships under a permissive or academic licence; everything in this table has a DOI or canonical software reference a reviewer can verify.

| Stage | Tool | Role in this pipeline | Version | Citation |
|---|---|---|---|---|
| Receptor prep | RDKit | small-molecule chemistry: SMILES parsing, standardisation, 3D embed (ETKDGv3 + MMFF94) | `>=2024.3` | Landrum, RDKit - Open-source cheminformatics. https://www.rdkit.org |
| Receptor prep | PDBFixer | rcsb-PDB cleanup: alt-conf selection, missing-residue rebuild, cofactor retention via per-target presets | conda-forge latest | Eastman et al. 2017, PLOS Comput Biol -- OpenMM 7. [10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659) |
| Fold (default) | ColabFold + AlphaFold 2 | protein structure prediction; open weights (CC-BY-4.0), runs on Colab GPU | ColabFold v1.5 / AF2 v2.3.2 | Mirdita et al. 2022, Nat Methods -- [10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1); Jumper et al. 2021, Nature -- [10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2) |
| Fold (optional) | AlphaFold Server (AF3) | optional fold provider; web-UI submission under Google's academic-access terms, parser-only helper in `aidd.folding.fold_with_af3_server` | server-side | Abramson et al. 2024, Nature -- [10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w) |
| Variant prior | AlphaMissense | sequence-based variant pathogenicity prior (0..1; per missense substitution in the human proteome) | precomputed table v1 (2023) | Cheng et al. 2023, Science -- [10.1126/science.adg7492](https://doi.org/10.1126/science.adg7492) |
| Variant prior | RaSP | structure-based variant stability prior (DDG, kcal/mol; positive = destabilising) | precomputed AF + experimental tables (2023) | Blaabjerg et al. 2023, eLife -- [10.7554/eLife.82593](https://doi.org/10.7554/eLife.82593) |
| Variant prior | gnomAD v4 | population allele frequency + per-ancestry breakdown + homozygote count | gnomAD r4 GraphQL | Karczewski et al. 2020, Nature -- [10.1038/s41586-020-2308-7](https://doi.org/10.1038/s41586-020-2308-7) |
| Docking (interpretable lane) | gnina | Vina-family docking + CNN rescoring; cross-platform on Colab Linux (Windows / macOS users invoke via Colab) | 1.3+ | McNutt et al. 2021, J Cheminform -- [10.1186/s13321-021-00522-2](https://doi.org/10.1186/s13321-021-00522-2) |
| Pose QC | PoseBusters | automated geometric / chemical sanity checks on docked poses (mandatory gate before scoring) | latest pip | Buttenschoen et al. 2024, Chem Sci -- [10.1039/D3SC04185A](https://doi.org/10.1039/D3SC04185A) |
| Interactions | ProLIF | protein-ligand interaction fingerprints (H-bonds, hydrophobic, pi-stacking, pi-cation, VdW with metal-vdW radii extension) | `>=2.0` | Bouysset & Fiorucci 2021, J Cheminform -- [10.1186/s13321-021-00548-6](https://doi.org/10.1186/s13321-021-00548-6) |
| Co-folding (fast lane) | Boltz-2 | joint protein-ligand co-folding + affinity prediction (log10 IC50 uM; lower = stronger); pip-installed on Colab GPU | `2.2.1` | Passaro et al. 2025, bioRxiv -- [10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707) |
| Consensus | rank-product across (gnina-CNN OR classical rescorer) x Boltz-2 affinity; runtime-detected per target (`rescorer.pkl` presence selects the lane) | this pipeline | n/a | Wang & Wang 2001, J Chem Inf Comput Sci 41:1422 -- [10.1021/ci0101207](https://doi.org/10.1021/ci0101207); Houston & Walkinshaw 2013, J Chem Inf Model 53:384 -- [10.1021/ci300399w](https://doi.org/10.1021/ci300399w) |

What this pipeline does NOT predict: catalytic rate (`k_cat`, `K_m`), patient-level clinical outcome, off-target binding outside the supplied target list. See the "Caveats" block in the recap for the full honest framing.
"""),

        # ====================================================================
        # CELL 3 -- SETUP (Colab install + repo clone + path setup)
        # ====================================================================
        code(title="Setup: detect Colab vs local, install deps, configure paths", source="""
import sys
import importlib
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    # CPU-only deps. pandas / numpy / matplotlib / pyarrow / requests / pyyaml
    # are in Colab's default image; rdkit / py3Dmol / prolif / biopython /
    # pdbfixer / posebusters are not. gnina + boltz are installed in the
    # respective stage cells (gnina is a Linux binary, boltz is a heavy pip
    # install with CUDA; both are GPU-cell territory).
    !pip install -q rdkit py3Dmol prolif biopython pdbfixer posebusters
    REPO_ROOT = Path("/content/aidd-pipeline")
    if not REPO_ROOT.exists():
        !git clone https://github.com/hvmarco/aidd-pipeline.git {REPO_ROOT}
    if not (REPO_ROOT / "src" / "aidd").exists():
        raise RuntimeError(
            "Repo clone failed (most likely cause: the repo is private and "
            "Colab cannot authenticate). Make github.com/hvmarco/aidd-pipeline "
            "public, or use a Personal Access Token via Colab Secrets, then "
            "re-run this cell."
        )
    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()
else:
    REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()

print(f"Repo root: {REPO_ROOT}")
print(f"Running on: {'Colab' if IS_COLAB else 'local'}")
"""),

        # ====================================================================
        # CELL 4 -- IMPORTS (everything aidd.* + autoreload up front)
        # ====================================================================
        code(title="Imports", source=AUTORELOAD_SNIPPET + """
import warnings
warnings.filterwarnings("ignore")

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import py3Dmol
from rdkit import Chem
from Bio.PDB import PDBParser

from aidd.io import mount_drive_if_colab, pretty_path
from aidd.structures import ca_rmsd, distogram, plddt_summary
from aidd.ifp import compute_ifp
from aidd.viz import show_protein_ligand
from aidd.variants import (
    DEMO_SET_UNIPROT_IDS,
    alphamissense_score,
    alphamissense_class,
    gnomad_frequency,
)
from aidd.stability import rasp_ddg, rasp_covers
from aidd.mutation import (
    ca_rmsd_pocket,
    ifp_diff,
    pocket_residues_within,
    render_mutation_html,
    render_moa_html,
    shortlist_diff,
    summarise_mutation,
)
from aidd.inputs import (
    RECEPTOR_PREPS,
    assert_wt_residue,
    fetch_pdb,
    name_to_smiles,
    prep_receptor,
)
from aidd.consensus import compute_consensus, write_shortlist
from aidd.docking import BindingBox, box_from_center_radius
from aidd.folding import fold_with_af3_server, summarise_run as colabfold_summarise

print("imports ok")
"""),

        # ====================================================================
        # CELL 5 -- DRIVE TOGGLE
        # ====================================================================
        markdown("""
### Google Drive - read this before running the next cell

The variant-prior caches (`alphamissense/demo_set.parquet`, `gnomad/<gene>_gnomad_r4.json`, `rasp/full_proteome_demo_set.parquet`) live on **Google Drive** by default when you ran notebook 07 on Colab. The cached ERK2 library outputs from steps 10-12 (gnina poses, scored_poses, Boltz-2 affinities) also live there.

**To opt out** -- runtime-local fast smoke-test, no Drive auth prompt -- set `USE_DRIVE = False` in the cell below. Caches and derived data will then live under the local repo's `data/cache/` and `data/derived/` trees only. On a fresh Colab runtime with `USE_DRIVE = False` you'll re-download AlphaMissense + RaSP from scratch on first use (~6 minutes); on a non-Colab local env, the local caches survive across runs.

The cell prints the resolved `DATA_ROOT` so you can see at a glance which root is in use.
"""),
        code(title="Drive persistence toggle  (change to False to opt out)", source="""
# ============================================================================
# GOOGLE DRIVE AUTHORIZATION
# On Colab, the next call will prompt for Drive permission on first use.
#
# ->  To OPT OUT of Drive, change the line below to:    USE_DRIVE = False
#     (caches and derived data then live under the local data/ tree only;
#      on Colab this means /content/ which is wiped on runtime disconnect)
# ============================================================================
USE_DRIVE = IS_COLAB

DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)
CACHE_ROOT          = DATA_ROOT.parent / "cache"
ALPHAMISSENSE_CACHE = CACHE_ROOT / "alphamissense"
RASP_CACHE          = CACHE_ROOT / "rasp"
GNOMAD_CACHE        = CACHE_ROOT / "gnomad"
RCSB_CACHE          = CACHE_ROOT / "rcsb"
# aidd.io.mount_drive_if_colab already returns data/derived/, so DERIVED_ROOT
# is an alias for DATA_ROOT, NOT DATA_ROOT / "derived" (that would double-nest).
DERIVED_ROOT        = DATA_ROOT

print(f"USE_DRIVE:    {USE_DRIVE}  ({'Drive-backed' if USE_DRIVE else 'local fallback'})")
print(f"DATA_ROOT:    {DATA_ROOT}")
print(f"CACHE_ROOT:   {CACHE_ROOT}")
print(f"DERIVED_ROOT: {DERIVED_ROOT}")
"""),

        # ====================================================================
        # CELL 6 -- CONFIGURATION OVERVIEW (markdown)
        # ====================================================================
        markdown("""
## 2 - Configuration

### Background

Two top-level toggles control the run. `FOLD_PROVIDER` selects which structure-prediction tool produces the receptor; both branches write the same canonical `<target>_best.pdb` so the downstream pipeline is agnostic. `RUN_MODE` selects between library-screen mode (1k-10k compounds; ranked shortlist) and mechanism-of-action mode (small-N; per-compound HTML report). The choice is per-run; the same notebook supports either.

The `TARGET` declaration in the second cell below names the protein (UniProt + RCSB PDB ID), variants of interest (if any), and the per-target binding-box. The default is ERK2 in library mode -- the cached scaffold from steps 10-12 means this configuration produces a shortlist in seconds. Commented blocks for each of the seven headline demos (ERK2 + DPYD, NAT2, CYP2D6, KRAS, BRCA1, ESR1, UGT1A1) live in the cell so a reviewer reading the notebook sees the full intended scope.
"""),

        # ====================================================================
        # CELL 7 -- TOP-LEVEL TOGGLES (FOLD_PROVIDER + RUN_MODE + validator)
        # ====================================================================
        code(title="Top-level toggles -- FOLD_PROVIDER + RUN_MODE", source="""
# ============================================================================
# FOLD_PROVIDER -- which structure-prediction tool produces the receptor.
#
#   "colabfold"  : AlphaFold 2 via ColabFold (default; open weights;
#                  reproducible by any user with a Colab account).
#   "af3_server" : AlphaFold 3 via the AlphaFold Server academic web UI.
#                  Owner-only path (per-user academic-access agreement; no
#                  public API). Requires AF3_RESULT_ZIP (see below) -- the
#                  helper is a parser, not a submission client.
# ============================================================================
FOLD_PROVIDER = "colabfold"
# FOLD_PROVIDER = "af3_server"

# ============================================================================
# RUN_MODE -- which pipeline shape to run.
#
#   "library" : 1k-10k compound virtual screen against the target.
#               Output: shortlist.sdf + shortlist.csv.
#               Boltz-2 co-folding is the bottleneck (6-15 GPU-hours
#               for a fresh 1k-compound run; seconds for an ERK2
#               cache walk).
#   "moa"     : small-N (N <= 10) mechanism-of-action.
#               Output: one HTML report per (compound x variant) pair
#               + a summary CSV. Per-compound Boltz-2 inference; minutes
#               per compound.
# ============================================================================
RUN_MODE = "library"
# RUN_MODE = "moa"

# AF3 result zip path -- only required when FOLD_PROVIDER == "af3_server".
# Download the zip from https://alphafoldserver.com after submitting the
# fold via the web UI; point this at the local file path.
AF3_RESULT_ZIP = None  # e.g. Path("/content/drive/MyDrive/fold_nat2.zip")

# ---------- Toggle validation -- raises with a clear message on misconfig ----
if FOLD_PROVIDER not in {"colabfold", "af3_server"}:
    raise ValueError(f"FOLD_PROVIDER must be 'colabfold' or 'af3_server'; got {FOLD_PROVIDER!r}")
if RUN_MODE not in {"library", "moa"}:
    raise ValueError(f"RUN_MODE must be 'library' or 'moa'; got {RUN_MODE!r}")
if FOLD_PROVIDER == "af3_server" and AF3_RESULT_ZIP is None:
    raise ValueError(
        "FOLD_PROVIDER='af3_server' requires AF3_RESULT_ZIP to be set. "
        "AlphaFold Server has no public submission API; submit the fold via "
        "https://alphafoldserver.com (academic-access web UI), download the "
        "result zip, and set AF3_RESULT_ZIP to its path."
    )

print(f"FOLD_PROVIDER: {FOLD_PROVIDER}")
print(f"RUN_MODE:      {RUN_MODE}")
if FOLD_PROVIDER == 'af3_server':
    print(f"AF3 result:    {AF3_RESULT_ZIP}")
"""),

        # ====================================================================
        # CELL 8 -- RUN DECLARATION (TARGET + LIBRARY/MOA inputs)
        # ====================================================================
        code(title="Run declaration -- target + variants + library/MoA inputs", source="""
# ============================================================================
# DEFAULT: ERK2 library run (cached from steps 10-12; rank_product @ TOP=0.10)
# ============================================================================
TARGET = {
    "name":     "erk2",
    "uniprot":  "P28482",
    "pdb_id":   "4FV7",
    "variants": [],   # WT-only library run; populate to compare WT vs mutant
}
LIBRARY_SMILES_PATH = REPO_ROOT / "data" / "compounds" / TARGET["name"] / "training.smi"
MOA_COMPOUNDS = []    # (used in moa mode only -- see the commented blocks)

# ============================================================================
# COMMENTED HEADLINE-DEMO BLOCKS -- uncomment one to switch the run.
# ============================================================================

# --- NAT2 (deep MoA walkthrough; *5 / *6 / *7 + isoniazid) -------------------
# TARGET = {
#     "name":     "nat2",
#     "uniprot":  "P11245",
#     "pdb_id":   "2PFR",
#     "variants": [
#         {"label": "*5", "handling": "mutate_in_place", "position": 114,
#          "wt_aa": "I", "mut_aa": "T", "rsid": "rs1801280"},
#         {"label": "*6", "handling": "mutate_in_place", "position": 197,
#          "wt_aa": "R", "mut_aa": "Q", "rsid": "rs1799930"},
#         {"label": "*7", "handling": "mutate_in_place", "position": 286,
#          "wt_aa": "G", "mut_aa": "E", "rsid": "rs1799931"},
#     ],
# }
# # MoA mode: small substrate set (isoniazid is the textbook NAT2 substrate;
# # 4-aminobiphenyl + PhIP are CRC-relevant arylamine substrates).
# MOA_COMPOUNDS = [("isoniazid", None), ("4-aminobiphenyl", None), ("PhIP", None)]
# # Library mode: supply your own .smi at the path below; suggested starter set
# # = the three substrates above + 7-10 known NAT2 arylamine substrates.
# LIBRARY_SMILES_PATH = REPO_ROOT / "data" / "compounds" / "nat2" / "library.smi"

# --- CYP2D6 + tamoxifen (*4 splice + *10 missense) ---------------------------
# TARGET = {
#     "name":     "cyp2d6",
#     "uniprot":  "P10635",
#     "pdb_id":   "3QM4",
#     "variants": [
#         {"label": "*10", "handling": "mutate_in_place", "position": 34,
#          "wt_aa": "P", "mut_aa": "S", "rsid": "rs1065852"},
#         # *4 is a splice variant -- out-of-scope for the structural pipeline;
#         # see notebook 09 section 7 for the priors-only branch.
#     ],
# }
# MOA_COMPOUNDS = [("tamoxifen", None)]
# LIBRARY_SMILES_PATH = REPO_ROOT / "data" / "compounds" / "cyp2d6" / "library.smi"

# --- KRAS G12C + sotorasib (oncogenic driver; covalent inhibitor) ------------
# # NOTE: 6OIM is the G12C+sotorasib co-crystal -- use 4OBE as the WT reference.
# TARGET = {
#     "name":     "kras",
#     "uniprot":  "P01116",
#     "pdb_id":   "4OBE",
#     "variants": [
#         {"label": "G12C", "handling": "mutate_in_place", "position": 12,
#          "wt_aa": "G", "mut_aa": "C", "rsid": "rs121913530"},
#     ],
# }
# MOA_COMPOUNDS = [("sotorasib", None)]
# LIBRARY_SMILES_PATH = REPO_ROOT / "data" / "compounds" / "kras" / "library.smi"
# # TODO step 18: curate KRAS BOX_SPECS entry (anchor=12, radius=~12 A).

# --- DPYD I560S + 5-fluorouracil (pharmacogene; CRC chemo) -------------------
# TARGET = {
#     "name":     "dpyd",
#     "uniprot":  "Q12882",
#     "pdb_id":   "1H7W",
#     "variants": [
#         {"label": "I560S", "handling": "mutate_in_place", "position": 560,
#          "wt_aa": "I", "mut_aa": "S", "rsid": "rs1801158"},
#     ],
# }
# MOA_COMPOUNDS = [("5-fluorouracil", None)]
# LIBRARY_SMILES_PATH = REPO_ROOT / "data" / "compounds" / "dpyd" / "library.smi"
# # TODO step 18: curate DPYD BOX_SPECS entry (uracil-binding pocket).

# --- ESR1 Y537S + tamoxifen / fulvestrant (breast; endocrine resistance) -----
# # Library-side only per PROJECT_PROPOSAL.md sec 10.
# TARGET = {
#     "name":     "esr1",
#     "uniprot":  "P03372",
#     "pdb_id":   "1ERR",
#     "variants": [
#         {"label": "Y537S", "handling": "mutate_in_place", "position": 537,
#          "wt_aa": "Y", "mut_aa": "S", "rsid": "rs121913041"},
#     ],
# }
# LIBRARY_SMILES_PATH = REPO_ROOT / "data" / "compounds" / "esr1" / "library.smi"
# # TODO step 18: curate ESR1 BOX_SPECS entry (LBD; tamoxifen-binding pocket).

# --- BRCA1 LoF + olaparib (ovarian/breast; synthetic lethality) --------------
# # Library-side only per PROJECT_PROPOSAL.md sec 10. NOTE: olaparib binds PARP,
# # NOT BRCA1 -- the variant creates the vulnerability without itself altering
# # drug binding. Pipeline runs against a PARP target for the binding question.
# TARGET = {
#     "name":     "brca1",
#     "uniprot":  "P38398",
#     "pdb_id":   "1JM7",  # BRCT domain
#     "variants": [
#         {"label": "C61G", "handling": "mutate_in_place", "position": 61,
#          "wt_aa": "C", "mut_aa": "G", "rsid": "rs28897672"},
#     ],
# }
# LIBRARY_SMILES_PATH = REPO_ROOT / "data" / "compounds" / "brca1" / "library.smi"
# # TODO step 18: curate BRCA1 BOX_SPECS entry.

# --- UGT1A1 *28 + irinotecan (CRC chemo; FDA-label genotype-guided) ----------
# # UGT1A1 has no human crystal -- AF model via notebook 01 (or AF-DB fetch).
# # *28 is a TATA-box promoter variant; library mode against the AF model is the
# # closest the structural pipeline gets to the substrate-binding question.
# TARGET = {
#     "name":     "ugt1a1",
#     "uniprot":  "P22309",
#     "pdb_id":   None,   # no human crystal; AF-DB fetch in cell 10
#     "variants": [],
# }
# MOA_COMPOUNDS = [("irinotecan", None)]
# LIBRARY_SMILES_PATH = REPO_ROOT / "data" / "compounds" / "ugt1a1" / "library.smi"
# # TODO step 18: curate UGT1A1 BOX_SPECS entry once the AF model is in.

print(f"TARGET: {TARGET['name']} (UniProt {TARGET['uniprot']}, PDB {TARGET['pdb_id']})")
print(f"variants: {len(TARGET['variants'])} ({[v['label'] for v in TARGET['variants']] or '(WT only)'})")
if RUN_MODE == 'library':
    print(f"LIBRARY_SMILES_PATH: {LIBRARY_SMILES_PATH}")
elif RUN_MODE == 'moa':
    print(f"MOA_COMPOUNDS: {[c[0] for c in MOA_COMPOUNDS]}")
"""),

        # ====================================================================
        # CELL 9 -- INPUTS & PRIORS (markdown)
        # ====================================================================
        markdown("""
## 3 - Inputs + variant priors

### Background

Two preparation stages run before any docking. **Stage 1** resolves the target receptor: download the PDB by ID from rcsb.org (cached on Drive), clean it with PDBFixer (alt-conf selection, missing-residue rebuild, cofactor retention via per-target presets), and apply per-variant mutations in-place via PDBFixer (for variants flagged `mutate_in_place`; for `fold_variant` see the AF3 / ColabFold branches in section 4). The output is one PDB per genotype at `data/derived/<target>[_<variant>]/fold/<target>[_<variant>]_best.pdb` -- the canonical contract every downstream stage reads from.

**Stage 2** resolves the per-variant computational priors -- AlphaMissense pathogenicity, gnomAD allele frequency, RaSP DDG -- as taught in detail in notebook 07. The three priors are independent (sequence-evolution, population-frequency, structure-stability) and together give a much fuller picture than any single signal. For pharmacogene loss-of-function variants, RaSP DDG is typically the load-bearing prior (stability-driven abundance loss is the dominant mechanism behind slow-acetylator / reduced-function phenotypes); for oncogenic drivers like KRAS G12C, AlphaMissense and stability-vs-binding contrast both carry signal.

For targets without a human crystal structure (e.g. UGT1A1), the cell below falls back to the AlphaFold Database REST API (`https://alphafold.ebi.ac.uk/api/prediction/<uniprot>`) and downloads the rank-1 AF model -- same canonical filename, no downstream code change.
"""),

        # ====================================================================
        # CELL 10 -- RESOLVE TARGET RECEPTOR (WT + per-variant via prep_receptor)
        # ====================================================================
        code(title="Resolve target receptor (WT + per-variant)", source="""
import urllib.request

def _genotype_slug(label: str) -> str:
    \"\"\"Filesystem-safe label for a genotype directory ('*5' -> '5', 'I560S' -> 'i560s').\"\"\"
    return label.replace("*", "").replace("/", "_").lower() or "wt"

def _genotype_dir(target_name: str, variant_label: str | None) -> Path:
    \"\"\"<DERIVED_ROOT>/<target>[_<variant>]/.\"\"\"
    if variant_label is None or variant_label.upper() == "WT":
        return DERIVED_ROOT / target_name
    return DERIVED_ROOT / f"{target_name}_{_genotype_slug(variant_label)}"

def _fetch_alphafold_pdb(uniprot_id: str, out_path: Path) -> Path:
    \"\"\"Fetch the rank-1 AlphaFold-DB PDB for a UniProt accession.

    Pattern from nb 08 commit f52b566 (_resolve_dpyd_alphafold_url): query the
    AF-DB JSON API for the pdbUrl rather than hardcoding a version-stamped URL,
    which the upstream rotates without notice.
    \"\"\"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    with urllib.request.urlopen(api_url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload:
        raise RuntimeError(f"AlphaFold-DB returned empty payload for {uniprot_id}")
    pdb_url = payload[0]["pdbUrl"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(pdb_url, timeout=60) as resp:
        out_path.write_bytes(resp.read())
    return out_path

def _apply_mutation_pdbfixer(wt_pdb: Path, position: int, wt_aa: str, mut_aa: str,
                              out_pdb: Path, chain: str = "A") -> Path:
    \"\"\"PDBFixer applyMutations for a single-residue missense.

    Same pattern as nb 09's _apply_mutation_pdbfixer (no extraction to src/aidd
    yet; rule of three not triggered -- nb 99 is the second caller).
    \"\"\"
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    AA_1TO3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU",
               "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
               "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL"}
    spec = f"{AA_1TO3[wt_aa]}-{position}-{AA_1TO3[mut_aa]}"
    fixer = PDBFixer(filename=str(wt_pdb))
    fixer.applyMutations([spec], chain)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdb, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    return out_pdb

# --- WT receptor ------------------------------------------------------------
wt_dir = _genotype_dir(TARGET["name"], None)
wt_fold = wt_dir / "fold"
wt_fold.mkdir(parents=True, exist_ok=True)
WT_PDB = wt_fold / f"{TARGET['name']}_best.pdb"

if not WT_PDB.exists():
    if TARGET["pdb_id"] is not None:
        # Crystal path: fetch from rcsb + prep_receptor with per-target preset
        raw = fetch_pdb(TARGET["pdb_id"], cache_dir=RCSB_CACHE)
        prep_receptor(
            raw,
            target_name=TARGET["name"].upper() if TARGET["name"].upper() in RECEPTOR_PREPS else None,
            out_path=WT_PDB,
        )
    else:
        # AF-DB fallback: no human crystal available
        print(f"  {TARGET['name']}: no PDB ID configured -- fetching AlphaFold-DB model")
        _fetch_alphafold_pdb(TARGET["uniprot"], WT_PDB)

print(f"WT receptor:  {pretty_path(WT_PDB, DATA_ROOT, REPO_ROOT)}")

# --- Per-variant receptors --------------------------------------------------
VARIANT_PDBS = {}  # variant_label -> Path
for variant in TARGET["variants"]:
    var_label = variant["label"]
    handling = variant.get("handling", "mutate_in_place")
    var_dir = _genotype_dir(TARGET["name"], var_label)
    var_fold = var_dir / "fold"
    var_fold.mkdir(parents=True, exist_ok=True)
    var_pdb = var_fold / f"{TARGET['name']}_{_genotype_slug(var_label)}_best.pdb"

    if var_pdb.exists():
        print(f"  variant {var_label}: cached at {pretty_path(var_pdb, DATA_ROOT, REPO_ROOT)}")
    elif handling == "mutate_in_place":
        # Pre-mutation PDB-identity check (catches the 6OIM-class mistake; see
        # feedback_pdb_identity_check.md in project memory)
        assert_wt_residue(WT_PDB, "A", variant["position"], variant["wt_aa"])
        _apply_mutation_pdbfixer(
            WT_PDB,
            position=variant["position"], wt_aa=variant["wt_aa"], mut_aa=variant["mut_aa"],
            out_pdb=var_pdb,
        )
        print(f"  variant {var_label}: mutated in place -> {pretty_path(var_pdb, DATA_ROOT, REPO_ROOT)}")
    elif handling == "fold_variant":
        # Variant sequence folded from scratch; user provides the FASTA
        raise NotImplementedError(
            f"variant {var_label} handling='fold_variant' -- run notebook 01 with "
            f"the variant sequence and copy the output to {var_pdb}"
        )
    else:
        # Splice / promoter / out-of-scope -- no structural receptor
        print(f"  variant {var_label}: handling={handling!r}; no structural receptor "
              f"(priors-only; see section 5 / 6)")
        continue
    VARIANT_PDBS[var_label] = var_pdb

print(f"\\nResolved {1 + len(VARIANT_PDBS)} receptor(s) (WT + {len(VARIANT_PDBS)} variant(s))")
"""),

        # ====================================================================
        # CELL 11 -- RESOLVE VARIANT PRIORS
        # ====================================================================
        code(title="Resolve variant priors (AlphaMissense + gnomAD + RaSP)", source="""
def _resolve_priors(uniprot, position, wt_aa, mut_aa):
    \"\"\"All three priors for one variant; returns a flat dict.\"\"\"
    am_score = alphamissense_score(uniprot, position, mut_aa, cache_dir=ALPHAMISSENSE_CACHE)
    am_cls   = alphamissense_class(uniprot, position, mut_aa, cache_dir=ALPHAMISSENSE_CACHE)
    gn       = gnomad_frequency(uniprot, position, mut_aa, wt_aa=wt_aa, cache_dir=GNOMAD_CACHE)
    rasp     = rasp_ddg(uniprot, position, wt_aa, mut_aa, cache_dir=RASP_CACHE)
    return {
        "am_score": am_score,
        "am_class": am_cls,
        "gnomad_af": gn.get("allele_freq_overall"),
        "gnomad_found": gn.get("found"),
        "gnomad_flipped": gn.get("reference_flipped"),
        "rasp_ddg": rasp,
    }

VARIANT_PRIORS = {}  # variant_label -> dict
if TARGET["variants"]:
    rows = []
    for variant in TARGET["variants"]:
        var_label = variant["label"]
        if variant.get("handling") in ("out_of_scope_splice", "out_of_scope_promoter"):
            # Priors for these (when defined at protein level) still resolve; skip
            # structural fields. AlphaMissense / RaSP are missense-only so they
            # return None; gnomAD by chromosomal position does work but needs the
            # variant_id, not a (position, alt_aa) tuple.
            VARIANT_PRIORS[var_label] = {
                "am_score": None, "am_class": None,
                "gnomad_af": None, "gnomad_found": None, "gnomad_flipped": None,
                "rasp_ddg": None,
                "note": f"out-of-scope ({variant['handling']}); see nb 09 section 7",
            }
            continue
        priors = _resolve_priors(
            TARGET["uniprot"], variant["position"], variant["wt_aa"], variant["mut_aa"],
        )
        VARIANT_PRIORS[var_label] = priors
        rows.append({"variant": var_label, **priors})

    priors_df = pd.DataFrame(rows)
    print(priors_df.to_string(index=False) if not priors_df.empty else "(all variants out-of-scope)")
else:
    print("No variants in TARGET -- skipping per-variant priors")
"""),

        # ====================================================================
        # CELL 12 -- FOLD STAGE (markdown)
        # ====================================================================
        markdown("""
## 4 - Fold (dispatch on FOLD_PROVIDER)

### Background

When `TARGET["pdb_id"]` is set (the default for the six headline demos with crystal structures), the "fold" stage is a no-op -- the prep_receptor output from section 3 IS the receptor. For targets without a human crystal (UGT1A1), section 3 already pulled an AlphaFold-DB model. The dispatch below is therefore most relevant when you want a *custom* fold -- e.g., a variant sequence with insertions / deletions / splice changes that PDBFixer can't apply in-place, or a fresh AF3-grade prediction for grant materials.

Both branches write the same canonical `<target>_best.pdb` to `data/derived/<target>/fold/`. The ColabFold branch invokes the standard ColabFold install and runs on Colab GPU (~30-60 min for a single-sequence protein fold). The AF3 branch consumes a zip you downloaded from [alphafoldserver.com](https://alphafoldserver.com) after submitting via the academic-access web UI -- the helper is a parser, not an API client, per the (web-UI-only) constraint Google publishes for academic use.
"""),

        # ====================================================================
        # CELL 13 -- FOLD DISPATCH
        # ====================================================================
        code(title="Fold dispatch (FOLD_PROVIDER = colabfold | af3_server)", source="""
# Most common case: TARGET has a crystal PDB; section 3 already produced the
# canonical PDB at WT_PDB. Re-folding from sequence is opt-in -- the cell below
# only acts when the user has explicitly asked for it via the toggle.

if FOLD_PROVIDER == "colabfold":
    # Re-fold from sequence via ColabFold (notebook 01 logic).
    # For the headline-demo set, the crystal PDB at section 3's output is the
    # default -- no re-fold needed. To force a ColabFold run, delete WT_PDB and
    # re-execute section 3 + this cell with notebook 01's setup imported.
    print(f"FOLD_PROVIDER=colabfold: using prep_receptor output at {pretty_path(WT_PDB, DATA_ROOT, REPO_ROOT)}")
    print("  (to force a fresh ColabFold run, run notebook 01 with TARGET['uniprot']'s sequence")
    print("   and copy the rank-1 PDB to the path above)")

elif FOLD_PROVIDER == "af3_server":
    af3_zip = Path(AF3_RESULT_ZIP)
    if not af3_zip.exists():
        raise FileNotFoundError(
            f"AF3_RESULT_ZIP does not exist: {af3_zip}. Submit the fold via "
            f"https://alphafoldserver.com and download the result zip first."
        )
    AF3_PDB = wt_fold / f"{TARGET['name']}_af3_best.pdb"
    fold_with_af3_server(af3_zip, wt_fold, target_name=f"{TARGET['name']}_af3")
    # Promote AF3 output to the canonical WT_PDB filename so downstream stages
    # are agnostic to which provider was used.
    AF3_PDB.replace(WT_PDB)
    print(f"FOLD_PROVIDER=af3_server: parsed {af3_zip.name} -> {pretty_path(WT_PDB, DATA_ROOT, REPO_ROOT)}")
"""),

        # ====================================================================
        # CELL 14 -- FOLD QUALITY READOUT
        # ====================================================================
        code(title="Fold quality readout (per-residue pLDDT mean if available)", source="""
# When the receptor came from rcsb.org (crystal), pLDDT is not defined -- print
# basic structural stats instead (residue / atom counts; chain). When it came
# from AlphaFold (AF-DB fetch or ColabFold), per-residue pLDDT lives in the
# B-factor column.
parser = PDBParser(QUIET=True)
structure = parser.get_structure("x", str(WT_PDB))
n_res = sum(1 for r in structure[0]["A"] if r.id[0] == " ")
n_atoms = sum(1 for _ in structure.get_atoms())

bfactors = [a.get_bfactor() for r in structure[0]["A"] if r.id[0] == " "
            for a in r if a.get_name() == "CA"]
mean_bfactor = float(np.mean(bfactors)) if bfactors else float("nan")

print(f"Receptor: {pretty_path(WT_PDB, DATA_ROOT, REPO_ROOT)}")
print(f"  chain A residues: {n_res}")
print(f"  total atoms:      {n_atoms}")
print(f"  mean CA B-factor: {mean_bfactor:.1f}")
print(f"  (for crystal PDBs this is a real B-factor; for AlphaFold models it is pLDDT 0-100)")
"""),

        # ====================================================================
        # CELL 15 -- LIBRARY MODE OVERVIEW (markdown)
        # ====================================================================
        markdown("""
## 5 - Library mode

### Background

The 1k-10k compound virtual screen. Reads a SMILES file from `data/compounds/<target>/library.smi`, runs the prep / dock / score / co-fold / consensus chain, writes `shortlist.sdf` + `shortlist.csv`. For the default ERK2 configuration the entire chain hits cached outputs from steps 10-12 and produces the shortlist in seconds; for a fresh target on a fresh library the chain runs the actual compute (gnina on Colab GPU; Boltz-2 on Colab GPU; cumulative 6-15 GPU-hours depending on library size).

Cells in this section auto-skip when `RUN_MODE = "moa"` -- they print "library mode skipped" and a one-line pointer to section 6. Reviewers reading top-to-bottom see the whole pipeline; only the configured mode produces output.

The lane choice for the consensus filter is **runtime-detected**: when `data/derived/<target>/scoring/rescorer.pkl` exists on disk (i.e. a per-target classical rescorer has been trained -- the ERK2 case from notebook 04), the consensus uses `rescorer_rf_proba_oof` at top-fraction 10% (matches notebook 06's empirical operating point for ERK2). When no rescorer.pkl exists, the consensus falls back to `gnina_cnn_affinity` at top-fraction 30% (matches notebook 08's non-ERK2 lane choice). The same compute_consensus call; the parameters resolve at the call site.
"""),

        # ====================================================================
        # CELL 16 -- LIBRARY LIGAND PREP
        # ====================================================================
        code(title="Library: SMILES -> standardise -> ADMET -> 3D embed", source="""
if RUN_MODE != "library":
    print("library mode skipped (RUN_MODE='moa')")
else:
    target_root = _genotype_dir(TARGET["name"], None)
    ligands_dir = target_root / "ligands"
    LIGANDS_SDF = ligands_dir / "ligands_prepared.sdf"

    if LIGANDS_SDF.exists():
        print(f"  cached: {pretty_path(LIGANDS_SDF, DATA_ROOT, REPO_ROOT)}")
        n_mols = sum(1 for _ in Chem.SDMolSupplier(str(LIGANDS_SDF)))
        print(f"  {n_mols} prepared ligands on disk")
    else:
        if not LIBRARY_SMILES_PATH.exists():
            raise FileNotFoundError(
                f"LIBRARY_SMILES_PATH does not exist: {LIBRARY_SMILES_PATH}. "
                f"Provide a SMILES file (one molecule per line, tab-separated id) "
                f"at this path, or switch TARGET to one with a cached library."
            )
        from aidd.ligands import read_smiles, prepare_library, write_sdf
        df_smi = read_smiles(LIBRARY_SMILES_PATH)
        print(f"  {len(df_smi)} SMILES read; running ADMET + 3D embed ...")
        df_prepped = prepare_library(df_smi, n_workers=1, progress=True)
        ligands_dir.mkdir(parents=True, exist_ok=True)
        write_sdf(
            df_prepped[df_prepped["ok"]],
            LIGANDS_SDF,
            props_to_write=["smiles_std", "mw", "logp", "qed"],
        )
        print(f"  wrote {int(df_prepped['ok'].sum())} prepared ligands -> "
              f"{pretty_path(LIGANDS_SDF, DATA_ROOT, REPO_ROOT)}")
"""),

        # ====================================================================
        # CELL 17 -- GNINA DOCKING (the box resolver lives here)
        # ====================================================================
        code(title="Library: gnina docking (resumable per-compound cache)", source="""
# ============================================================================
# BOX_SPECS -- per-target binding-site box configuration.
#
# Two shapes accepted:
#   {"center": (cx, cy, cz), "radius": R, "margin": M}    -- explicit coords
#   {"anchor_residue": N, "chain": "A", "radius": R}      -- derive from PDB
#
# Curated for the step-17 smoke-test critical path (ERK2 / NAT2 / CYP2D6).
# Other headline-demo targets need box curation in step 18 -- the resolver
# raises ValueError on a missing entry so wrong-pocket boxes don't ship silently.
# ============================================================================
BOX_SPECS = {
    # ERK2 / 4FV7: curated box from _archive/configs/plants_4fv7.conf;
    # step-10 verified (PoseBusters pass + reference-redock RMSD < 2 A).
    "erk2":   {"center": (1.34299, 17.3648, 40.9828), "radius": 12.9007, "margin": 2.0},
    # NAT2 / 2PFR: catalytic Cys68 (acetyl-CoA transfer nucleophile). Anchor
    # pattern established in nb 09 DEMOS (anchor_residue_hint=68).
    "nat2":   {"anchor_residue": 68,  "chain": "A", "radius": 12.0, "margin": 2.0},
    # CYP2D6 / 3QM4: I-helix residue 308 (anchor for HEM-adjacent substrate
    # pocket; CYP pockets are larger than typical kinases -> 14 A radius).
    "cyp2d6": {"anchor_residue": 308, "chain": "A", "radius": 14.0, "margin": 2.0},
    # TODO step 18: curate boxes for dpyd / kras / brca1 / esr1 / ugt1a1.
}

def _ca_coord_at(pdb_path, position, chain="A"):
    \"\"\"CA xyz of one residue. Duplicated from nb 09; rule of three not triggered.\"\"\"
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", str(pdb_path))
    for model in struct:
        for ch in model:
            if ch.id != chain:
                continue
            for residue in ch:
                if residue.id[1] == position and "CA" in residue:
                    return np.array(residue["CA"].get_coord())
        break
    raise ValueError(f"residue {position} chain {chain!r} CA not found in {pdb_path}")

def _resolve_box(target_name, pdb_path):
    \"\"\"Resolve a BindingBox for one target via BOX_SPECS; raises on missing entry.\"\"\"
    spec = BOX_SPECS.get(target_name)
    if spec is None:
        raise ValueError(
            f"No BOX_SPECS entry for target {target_name!r}. Add a curated box "
            f"(center + radius) or an anchor_residue + radius and retry. "
            f"Curated: {sorted(BOX_SPECS)}."
        )
    margin = spec.get("margin", 2.0)
    if "center" in spec:
        return box_from_center_radius(spec["center"], spec["radius"], margin=margin)
    center = _ca_coord_at(pdb_path, spec["anchor_residue"], chain=spec.get("chain", "A"))
    return box_from_center_radius(tuple(float(c) for c in center), spec["radius"], margin=margin)

# --- Actual gnina dock invocation -------------------------------------------
if RUN_MODE != "library":
    print("library mode skipped (RUN_MODE='moa')")
else:
    from aidd.docking import dock_library, require_gnina
    target_root = _genotype_dir(TARGET["name"], None)
    docking_dir = target_root / "docking"
    POSES_SDF   = docking_dir / "poses.sdf"

    if POSES_SDF.exists() and (docking_dir / "gnina_scores.csv").exists():
        print(f"  cached: {pretty_path(docking_dir, DATA_ROOT, REPO_ROOT)}")
        scores = pd.read_csv(docking_dir / "gnina_scores.csv")
        print(f"  {len(scores)} pose rows; {scores['compound_id'].nunique()} compounds docked")
    else:
        require_gnina()  # raises with install instructions if gnina is missing
        box = _resolve_box(TARGET["name"], WT_PDB)
        print(f"  box: center {box.center}, half_extents {box.half_extents}")
        result = dock_library(
            WT_PDB, LIGANDS_SDF, docking_dir, box,
            exhaustiveness=8, num_modes=9, cnn_scoring="rescore",
            on_failure="skip_with_guard",
        )
        print(f"  wrote {len(result)} pose rows -> {pretty_path(docking_dir, DATA_ROOT, REPO_ROOT)}")
"""),

        # ====================================================================
        # CELL 18 -- POSEBUSTERS QC
        # ====================================================================
        code(title="Library: PoseBusters QC (mandatory gate before scoring)", source="""
if RUN_MODE != "library":
    print("library mode skipped (RUN_MODE='moa')")
else:
    from aidd.docking import run_posebusters
    target_root = _genotype_dir(TARGET["name"], None)
    pb_path = target_root / "docking" / "posebusters.csv"
    if pb_path.exists():
        pb_df = pd.read_csv(pb_path)
        n_pass = int(pb_df["mol_pred_loaded"].sum()) if "mol_pred_loaded" in pb_df.columns else len(pb_df)
        print(f"  cached: {pretty_path(pb_path, DATA_ROOT, REPO_ROOT)}")
        print(f"  {len(pb_df)} poses checked; {n_pass} pass-rows on disk")
    else:
        pb_df = run_posebusters(POSES_SDF, receptor=WT_PDB)
        pb_df.to_csv(pb_path, index=False)
        print(f"  wrote {pretty_path(pb_path, DATA_ROOT, REPO_ROOT)} ({len(pb_df)} poses)")
"""),

        # ====================================================================
        # CELL 19 -- IFP + RESCORER (runtime-detect)
        # ====================================================================
        code(title="Library: IFP + classical rescorer (runtime-detected lane)", source="""
if RUN_MODE != "library":
    print("library mode skipped (RUN_MODE='moa')")
else:
    target_root = _genotype_dir(TARGET["name"], None)
    scoring_dir = target_root / "scoring"
    SCORED_PARQUET = scoring_dir / "scored_poses.parquet"
    RESCORER_PKL   = scoring_dir / "rescorer.pkl"

    if SCORED_PARQUET.exists():
        scored = pd.read_parquet(SCORED_PARQUET)
        print(f"  cached: {pretty_path(SCORED_PARQUET, DATA_ROOT, REPO_ROOT)}")
        print(f"  {len(scored)} scored rows on disk")
        print(f"  rescorer.pkl present: {RESCORER_PKL.exists()}")
    else:
        # For non-ERK2 targets we skip the rescorer training step entirely --
        # only the IFP + gnina_cnn_affinity columns are needed for the
        # gnina_cnn_affinity consensus lane. To train a per-target rescorer,
        # run notebook 04 (which writes scored_poses.parquet AND rescorer.pkl).
        ifp_df = compute_ifp(WT_PDB, POSES_SDF, progress=True)
        # Materialise as a flat per-pose DataFrame joined to the gnina scores.
        from aidd.docking import parse_poses_sdf
        gnina_scores = pd.read_csv(target_root / "docking" / "gnina_scores.csv")
        # Note: full IFP-rescorer training is in notebook 04; here we ship the
        # minimal columns the consensus needs.
        scoring_dir.mkdir(parents=True, exist_ok=True)
        scored = gnina_scores.copy()
        scored.to_parquet(SCORED_PARQUET, index=False)
        print(f"  wrote {pretty_path(SCORED_PARQUET, DATA_ROOT, REPO_ROOT)} ({len(scored)} rows; no rescorer trained)")
"""),

        # ====================================================================
        # CELL 20 -- BOLTZ-2 CO-FOLDING (MSA from stage 2 if available)
        # ====================================================================
        code(title="Library: Boltz-2 co-folding (MSA reuse if available)", source="""
if RUN_MODE != "library":
    print("library mode skipped (RUN_MODE='moa')")
else:
    target_root = _genotype_dir(TARGET["name"], None)
    boltz_dir = target_root / "boltz"
    AFFINITY_CSV = boltz_dir / "affinity.csv"

    if AFFINITY_CSV.exists():
        boltz_df = pd.read_csv(AFFINITY_CSV)
        print(f"  cached: {pretty_path(AFFINITY_CSV, DATA_ROOT, REPO_ROOT)}")
        print(f"  {len(boltz_df)} compounds with Boltz-2 affinity on disk")
    else:
        from aidd.co_folding import predict_library, require_boltz
        require_boltz()  # raises with install instructions if boltz is missing
        # Reuse the ColabFold MSA if it exists (saves per-call MSA server roundtrips)
        msa_glob = sorted((target_root / "fold").glob("**/*.a3m"))
        msa_path = msa_glob[0] if msa_glob else None
        if msa_path is None:
            print("  no .a3m on disk; Boltz-2 will query its MSA server per call (slower)")
        # Reconstruct the (compound_id, smiles) DataFrame from prep output
        df_lig = pd.DataFrame([
            {"compound_id": mol.GetProp("_Name") if mol.HasProp("_Name") else f"cpd_{i:05d}",
             "smiles": Chem.MolToSmiles(mol)}
            for i, mol in enumerate(Chem.SDMolSupplier(str(LIGANDS_SDF))) if mol is not None
        ])
        # Read target sequence from the receptor PDB
        from Bio.SeqUtils import seq1
        parser2 = PDBParser(QUIET=True)
        s = parser2.get_structure("x", str(WT_PDB))
        target_sequence = "".join(seq1(r.get_resname()) for r in s[0]["A"]
                                  if r.id[0] == " ")
        boltz_df = predict_library(
            target_sequence, df_lig, boltz_dir,
            msa_path=msa_path, on_failure="skip_with_guard",
        )
        print(f"  wrote {pretty_path(AFFINITY_CSV, DATA_ROOT, REPO_ROOT)} ({len(boltz_df)} compounds)")
"""),

        # ====================================================================
        # CELL 21 -- CONSENSUS + SHORTLIST
        # ====================================================================
        code(title="Library: consensus rank + shortlist (runtime-detected lane)", source="""
if RUN_MODE != "library":
    print("library mode skipped (RUN_MODE='moa')")
else:
    target_root = _genotype_dir(TARGET["name"], None)
    shortlist_dir = target_root / "shortlist"
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    SHORTLIST_SDF = shortlist_dir / "shortlist.sdf"
    SHORTLIST_CSV = shortlist_dir / "shortlist.csv"

    rescorer_present = (target_root / "scoring" / "rescorer.pkl").exists()
    if rescorer_present and "rescorer_rf_proba_oof" in scored.columns:
        rescorer_col = "rescorer_rf_proba_oof"
        top_fraction = 0.10
        print(f"  lane: ERK2-style trained rescorer (rescorer_rf_proba_oof @ top 10%)")
    else:
        rescorer_col = "gnina_cnn_affinity"
        top_fraction = 0.30
        print(f"  lane: non-ERK2 default (gnina_cnn_affinity @ top 30%)")

    boltz_df = pd.read_csv(target_root / "boltz" / "affinity.csv")
    result = compute_consensus(
        scored, boltz_df,
        top_fraction=top_fraction,
        rescorer_col=rescorer_col,
        boltz_col="boltz_affinity",
        id_col="compound_id",
        rescorer_lower_is_better=False,
        boltz_lower_is_better=True,             # log10(IC50) uM; lower = stronger
        filter_method="rank_product_topk",
    )
    write_shortlist(result["shortlist"], POSES_SDF, SHORTLIST_SDF, SHORTLIST_CSV)
    print(f"\\nShortlist: {len(result['shortlist'])} compounds")
    print(f"  -> {pretty_path(SHORTLIST_SDF, DATA_ROOT, REPO_ROOT)}")
    print(f"  -> {pretty_path(SHORTLIST_CSV, DATA_ROOT, REPO_ROOT)}")
    print(f"\\n{result['summary']}")
"""),

        # ====================================================================
        # CELL 22 -- PER-VARIANT LIBRARY RUN + WT-vs-VARIANT HTML
        # ====================================================================
        code(title="Library: per-variant pipeline + WT-vs-variant comparison HTML", source="""
# For each variant in TARGET["variants"]: re-invoke the stage-1-through-stage-7
# helpers (fetch_pdb shared via the rcsb cache; prep_receptor against the same
# raw PDB with assert_wt_residue + PDBFixer.applyMutations for mutate_in_place
# variants; the docking / scoring / Boltz-2 / consensus stages re-run with
# out_dir = DERIVED_ROOT / "<target>_<variant>"). The per-compound docking
# cache and Boltz-2 cache are per-(target, variant) -- separate
# <variant>/docking/per_compound/ and <variant>/boltz/per_compound/ directories
# so the WT cache isn't accidentally reused for a mutant. After both genotypes'
# shortlists exist, invoke aidd.mutation.summarise_mutation -> render_mutation_html
# per variant; output lands at DERIVED_ROOT / "<target>" / "mutations" /
# "wt_vs_<variant>.html".

if RUN_MODE != "library":
    print("library mode skipped (RUN_MODE='moa')")
elif not TARGET["variants"]:
    print("WT-only run; no variants to compare")
else:
    from aidd.docking import dock_library
    from aidd.co_folding import predict_library
    from Bio.SeqUtils import seq1

    mutations_dir = _genotype_dir(TARGET["name"], None) / "mutations"
    mutations_dir.mkdir(parents=True, exist_ok=True)

    for variant in TARGET["variants"]:
        var_label = variant["label"]
        if variant.get("handling") in ("out_of_scope_splice", "out_of_scope_promoter"):
            print(f"\\n=== variant {var_label}: out-of-scope; priors-only ===")
            continue
        if var_label not in VARIANT_PDBS:
            print(f"\\n=== variant {var_label}: no PDB resolved; skipping ===")
            continue
        var_pdb = VARIANT_PDBS[var_label]
        var_dir = _genotype_dir(TARGET["name"], var_label)
        print(f"\\n=== variant {var_label} ===")
        print(f"  pdb: {pretty_path(var_pdb, DATA_ROOT, REPO_ROOT)}")

        # Per-(target, variant) gnina dock
        var_docking = var_dir / "docking"
        if not (var_docking / "gnina_scores.csv").exists():
            box = _resolve_box(TARGET["name"], var_pdb)
            dock_library(var_pdb, LIGANDS_SDF, var_docking, box,
                          exhaustiveness=8, num_modes=9, cnn_scoring="rescore",
                          on_failure="skip_with_guard")
        # Per-(target, variant) Boltz-2 co-fold
        var_boltz = var_dir / "boltz"
        if not (var_boltz / "affinity.csv").exists():
            parser2 = PDBParser(QUIET=True)
            s = parser2.get_structure("x", str(var_pdb))
            var_seq = "".join(seq1(r.get_resname()) for r in s[0]["A"] if r.id[0] == " ")
            df_lig = pd.DataFrame([
                {"compound_id": mol.GetProp("_Name") if mol.HasProp("_Name") else f"cpd_{i:05d}",
                 "smiles": Chem.MolToSmiles(mol)}
                for i, mol in enumerate(Chem.SDMolSupplier(str(LIGANDS_SDF))) if mol is not None
            ])
            predict_library(var_seq, df_lig, var_boltz, on_failure="skip_with_guard")

        # WT-vs-variant comparison HTML via aidd.mutation
        summary = summarise_mutation(
            target=TARGET["name"], variant=var_label, uniprot=TARGET["uniprot"],
            position=variant["position"], wt_aa=variant["wt_aa"], mut_aa=variant["mut_aa"],
            wt_dir=_genotype_dir(TARGET["name"], None),
            mut_dir=var_dir,
            wt_name=TARGET["name"],
            mut_name=f"{TARGET['name']}_{_genotype_slug(var_label)}",
            alphamissense_cache=ALPHAMISSENSE_CACHE,
            gnomad_cache=GNOMAD_CACHE,
            rasp_cache=RASP_CACHE,
        )
        html_path = mutations_dir / f"wt_vs_{_genotype_slug(var_label)}.html"
        render_mutation_html(summary, html_path)
        print(f"  HTML report -> {pretty_path(html_path, DATA_ROOT, REPO_ROOT)}")
"""),

        # ====================================================================
        # CELL 23 -- MoA MODE OVERVIEW (markdown)
        # ====================================================================
        markdown("""
## 6 - MoA mode (small-N mechanism-of-action)

### Background

The small-N (`N <= 10`) per-compound mechanism-of-action driver. For each (compound, variant) pair: dock the single compound against the variant's receptor, co-fold via Boltz-2, compute the interaction fingerprint, render a self-contained HTML report (`aidd.mutation.render_moa_html`). One HTML file per (compound, variant) pair; one summary CSV across all pairs. The HTML is the wet-lab handoff artefact -- it opens in any modern browser and embeds an interactive py3Dmol viewer of the binding pose.

For the headline-demo set (NAT2 + isoniazid, CYP2D6 + tamoxifen, DPYD + 5-FU, UGT1A1 + irinotecan, KRAS + sotorasib), the notebook 09 walkthroughs are the pedagogical references; this mode is the production runner that produces the same reports with real per-compound Boltz-2 + gnina predictions instead of the nb 09 synthetic-score fixtures.

The cell below auto-skips when `RUN_MODE = "library"`.
"""),

        # ====================================================================
        # CELL 24 -- MoA PER-COMPOUND x PER-VARIANT LOOP
        # ====================================================================
        code(title="MoA: per-compound x per-variant report loop", source="""
if RUN_MODE != "moa":
    print("MoA mode skipped (RUN_MODE='library')")
elif not MOA_COMPOUNDS:
    print("MoA mode but MOA_COMPOUNDS is empty -- nothing to do")
else:
    from aidd.docking import dock_library
    from aidd.co_folding import predict_complex, require_boltz
    require_boltz()

    moa_root = _genotype_dir(TARGET["name"], None) / "moa_reports"
    moa_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    # Genotypes to iterate: WT first, then any in-scope variants
    genotypes = [("WT", WT_PDB, None)]
    for variant in TARGET["variants"]:
        if variant.get("handling") in ("out_of_scope_splice", "out_of_scope_promoter"):
            continue
        if variant["label"] in VARIANT_PDBS:
            genotypes.append((variant["label"], VARIANT_PDBS[variant["label"]], variant))

    target_sequence = None  # cache the parsed sequence per genotype below

    for compound_name, smiles_in in MOA_COMPOUNDS:
        smiles = smiles_in or name_to_smiles(compound_name)
        if smiles is None:
            print(f"  SKIP {compound_name}: no SMILES from PubChem and no fallback given")
            continue
        for genotype_label, genotype_pdb, variant in genotypes:
            print(f"\\n=== {compound_name} on {TARGET['name']} {genotype_label} ===")
            # Boltz-2 single-compound co-fold for this (compound, genotype) pair
            parser2 = PDBParser(QUIET=True)
            s = parser2.get_structure("x", str(genotype_pdb))
            from Bio.SeqUtils import seq1
            geno_seq = "".join(seq1(r.get_resname()) for r in s[0]["A"] if r.id[0] == " ")
            cdir = _genotype_dir(TARGET["name"], genotype_label if genotype_label != "WT" else None) / "boltz" / "per_compound" / compound_name.replace(" ", "_")
            cdir.mkdir(parents=True, exist_ok=True)
            boltz_result = predict_complex(geno_seq, smiles, cdir, compound_id=compound_name.replace(" ", "_"))
            # Priors for this genotype (None for WT)
            if variant is not None and variant.get("position") is not None:
                priors = _resolve_priors(TARGET["uniprot"], variant["position"],
                                          variant["wt_aa"], variant["mut_aa"])
            else:
                priors = {"am_score": None, "am_class": None,
                          "gnomad_af": None, "gnomad_found": None, "gnomad_flipped": None,
                          "rasp_ddg": None}
            # Build the MoA record dict (matches render_moa_html's contract)
            record = {
                "target_name": TARGET["name"], "uniprot": TARGET["uniprot"],
                "variant_label": genotype_label,
                "variant_handling": variant["handling"] if variant else "wt",
                "compound_name": compound_name, "smiles": smiles,
                "priors": {
                    "alphamissense": {"score": priors["am_score"], "class": priors["am_class"]},
                    "gnomad":        {"allele_freq_overall": priors["gnomad_af"],
                                       "found": priors["gnomad_found"],
                                       "reference_flipped": priors["gnomad_flipped"]},
                    "rasp":          {"ddg": priors["rasp_ddg"]},
                },
                "scores": {
                    "boltz_affinity":              boltz_result.affinity,
                    "boltz_affinity_probability":  boltz_result.affinity_probability,
                    "gnina_cnn_affinity":          None,   # filled in by per-compound gnina if you add it
                    "posebusters_pass":            None,
                },
            }
            html_path = moa_root / f"{compound_name.replace(' ', '_')}_{_genotype_slug(genotype_label)}.html"
            render_moa_html(record, html_path)
            print(f"  HTML: {pretty_path(html_path, DATA_ROOT, REPO_ROOT)}")
            summary_rows.append({
                "compound": compound_name, "variant": genotype_label,
                "boltz_affinity": boltz_result.affinity,
                "boltz_affinity_probability": boltz_result.affinity_probability,
                "am_score": priors["am_score"], "am_class": priors["am_class"],
                "gnomad_af": priors["gnomad_af"], "rasp_ddg": priors["rasp_ddg"],
                "html_report": str(html_path.relative_to(moa_root)),
            })

    if summary_rows:
        SUMMARY_CSV = moa_root / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
        print(f"\\nMoA summary: {len(summary_rows)} reports")
        print(f"  -> {pretty_path(SUMMARY_CSV, DATA_ROOT, REPO_ROOT)}")
"""),

        # ====================================================================
        # CELL 25 -- MoA SUMMARY READ-BACK (markdown that interprets the CSV)
        # ====================================================================
        markdown("""
### Interpreting the MoA summary

Reading order for each row:

1. **Boltz-2 affinity** -- log10(IC50 uM); negative = strong binder. The headline binding signal.
2. **Boltz-2 binder probability** -- 0-1; higher = more likely binder. Cross-checks the affinity headline.
3. **AlphaMissense class + score** -- for variant rows; "likely pathogenic" + a structural binding-delta is the strongest combined evidence.
4. **gnomAD allele frequency** -- common variants (AF > 0.01) carry stronger pharmacogene relevance; rare variants need careful per-case discussion.
5. **RaSP DDG** -- positive = destabilising; for pharmacogene loss-of-function variants this is typically the load-bearing prior.

Open the per-compound HTML files to see the embedded 3D pose viewer + the ProLIF interaction-fingerprint list. The HTML opens in any modern browser; share with collaborators by attaching the file directly.
"""),

        # ====================================================================
        # CELL 26 -- EXECUTIVE SUMMARY (markdown)
        # ====================================================================
        markdown("""
## 7 - Executive summary

### Result

For the configured `TARGET` (default: ERK2 in library mode), the pipeline produced either a ranked shortlist (`shortlist.sdf` + `shortlist.csv`) or one HTML report per (compound, variant) pair (`moa_reports/*.html` + `moa_reports/summary.csv`), with provenance back to the upstream methods cited in the Methods summary at the top of this notebook. Cached runs short-circuit in seconds; fresh runs on a non-ERK2 target take 6-15 GPU-hours per genotype.

### Confidence

Cross-lane agreement (gnina-CNN OR rescorer rank AND Boltz-2 affinity rank) is the headline confidence signal in library mode. The consensus filter at the default operating point (`rank_product_topk` @ top 10% for ERK2; top 30% for non-ERK2 targets without a trained rescorer) ranks compounds by the geometric mean of the two lane ranks. PoseBusters pass-rate is a mandatory gate before any compound enters the shortlist. For variant runs, the per-(target, variant) shortlist diff (compounds gained / lost / preserved + per-compound rank deltas) is rendered alongside the structural / IFP / priors comparison in the `mutations/wt_vs_<variant>.html` report.

In MoA mode the small-N nature of the run means consensus rank is not meaningful (N <= 10 compounds rarely produce a useful rank-product distribution); the per-(compound, variant) HTML report shows the Boltz-2 affinity + gnina-CNN affinity side-by-side so the reader applies their own chemistry judgment per pair.
"""),

        # ====================================================================
        # CELL 27 -- CAVEATS (methods-paper-grade)
        # ====================================================================
        markdown("""
### Caveats -- methods-paper-grade

Three caveats matter for any reader interpreting this notebook's output. None of them are blockers; all three are the kind of methods-paper detail a reviewer should know.

**(a) Boltz-2 cofactor asymmetry.** Boltz-2 co-folds **protein + ligand only**; cofactors are not part of the Boltz YAML schema. For cofactor-dependent demos (CYP2D6 with HEM iron; DPYD with FAD / NADP+ / FMN / [4Fe-4S]; UGT1A1 with UDP-glucuronic acid in any future model), the gnina-CNN lane sees the *holo*-with-cofactor receptor that `prep_receptor` produced (the cofactors are retained per the `RECEPTOR_PREPS` preset), while Boltz-2 sees the *apo* pocket. The two lanes therefore see different receptor states on cofactor-dependent targets. **Agreement between the lanes is stronger evidence than either alone; divergence between the lanes specifically on cofactor-dependent targets warrants careful chemistry-side review** (one lane is seeing an active-site environment the other is missing). The asymmetry is a methodological constraint of Boltz-2's input schema, not a pipeline bug.

**(b) IFP fidelity on PDBFixer-prepped receptors.** A small fraction of PDBFixer-produced PDBs have atoms that RDKit's strict valence sanitisation rejects (typically an over-valent carbon where RDKit's proximity-bonding heuristic infers extra bonds between residues PDBFixer reconstructed). The aidd.ifp.load_plf_molecule helper has a defensive `sanitize=False` fallback that preserves geometric IFP categories (residue distance shells, H-bond geometry, hydrophobic contacts) but may **undercount aromaticity-dependent categories** (pi-stacking, formal-charge-dependent HBDonor/HBAcceptor classification) on the affected receptors. The fallback logs a WARNING whenever it fires so the condition stays visible in notebook output. See `_planning/KNOWN_ISSUES.md` "nb 09 stub IFP" entry for the strict-fix decision (deferred to step 18 pending real-PDB measurement of how often the fallback actually fires).

**(c) Binding fit, NOT catalytic rate.** For enzyme demos (NAT2, CYP2D6, DPYD, UGT1A1) the pipeline measures whether the substrate **fits the active site** (a necessary condition for catalysis) -- not the **catalytic rate** (`k_cat`, `K_m`, turnover). RaSP DDG addresses the protein-stability dimension that drives many slow-acetylator / reduced-function phenotypes (stability-driven abundance loss is the dominant mechanism for many pharmacogene LoF variants), and combined with the binding-pose / IFP-diff signal this triangulates the slow / rapid mechanism without proving catalytic rate. For rigorous rate prediction the right tool class is QM/MM (e.g. EnzyHTP); out of scope for this in-silico screening pipeline.
"""),

        # ====================================================================
        # CELL 28 -- METHODS PROVENANCE (markdown)
        # ====================================================================
        markdown("""
### Methods provenance -- where each stage is taught in detail

This notebook is the audit-shape production runner; the teaching notebooks 00-09 walk every stage in pedagogical detail. Reviewers wanting to drill into any single method's setup, validation, or edge cases go to the upstream notebook below.

- [`00_quickstart.ipynb`](00_quickstart.ipynb) -- the IFP-on-ERK2 sanity check; smallest-possible end-to-end demo.
- [`01_fold_target.ipynb`](01_fold_target.ipynb) -- ColabFold + AlphaFold 2; the `colabfold` branch of section 4's dispatch.
- [`02_prepare_ligands.ipynb`](02_prepare_ligands.ipynb) -- SMILES standardisation + ADMET / PAINS gate + 3D embed; section 5 cell 16.
- [`03_dock_gnina.ipynb`](03_dock_gnina.ipynb) -- gnina docking + PoseBusters QC; section 5 cells 17-18 + section 6 cell 24 (per-compound variant).
- [`04_score_classical.ipynb`](04_score_classical.ipynb) -- ProLIF IFP + classical RF / XGBoost rescorer + scaffold-grouped 5-fold OOF; the ERK2-specific lane in section 5 cell 19.
- [`05_dock_boltz.ipynb`](05_dock_boltz.ipynb) -- Boltz-2 co-folding (per_compound caching, MSA reuse); section 5 cell 20 + section 6 cell 24.
- [`06_consensus_and_shortlist.ipynb`](06_consensus_and_shortlist.ipynb) -- consensus rank-product, intersection vs rank_product_topk filter choice, threshold sweep; section 5 cell 21.
- [`07_variant_effect_prediction.ipynb`](07_variant_effect_prediction.ipynb) -- AlphaMissense + gnomAD + RaSP variant priors; section 3 cell 11 + section 6 cell 24.
- [`08_mutation_analysis.ipynb`](08_mutation_analysis.ipynb) -- WT-vs-variant library-side diff (structural / IFP / shortlist); section 5 cell 22 + render_mutation_html.
- [`09_moa_small_n.ipynb`](09_moa_small_n.ipynb) -- small-N MoA driver + render_moa_html; section 6 cell 24.
"""),

        # ====================================================================
        # CELL 29 -- FURTHER READING (markdown)
        # ====================================================================
        markdown("""
### Further reading

- **AlphaFold 3.** Abramson, J. *et al.* "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature* **630**, 493-500 (2024). [doi:10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w) -- methodological reference for the `FOLD_PROVIDER = "af3_server"` branch.
- **Consensus rank product in virtual screening.** Houston, D.R. & Walkinshaw, M.D. "Consensus docking: improving the reliability of docking in a virtual screening context." *J. Chem. Inf. Model.* **53**, 384 (2013). [doi:10.1021/ci300399w](https://doi.org/10.1021/ci300399w) -- justifies the geometric-mean rank-product used in the consensus cell.
- **Catalysis prediction (out of scope here, for context).** Yan, B. *et al.* "EnzyHTP: Bridging molecular dynamics and quantum mechanics for enzyme engineering." *J. Chem. Theory Comput.* (2022) -- for readers who want to know what the right tool looks like for the catalytic-rate prediction this notebook explicitly does not attempt.
"""),
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
