"""Builder for 08_mutation_analysis.ipynb.

Source of truth for the library-side WT-vs-mutant comparison notebook. Cells
appear below in narrative order. Never edit the .ipynb directly -- see
``CLAUDE.md`` section *Notebook workflow*.

Regenerate:
    python notebooks/_build_08_mutation_analysis.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import AUTORELOAD_SNIPPET, code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "08_mutation_analysis.ipynb"


def build() -> None:
    nb = notebook(
        markdown("""
# 08 - Mutation analysis (library-side WT-vs-variant comparison)

**aidd-pipeline - Notebook 8 of the screening workflow**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hvmarco/aidd-pipeline/blob/main/notebooks/08_mutation_analysis.ipynb)

This notebook takes two complete pipeline runs - one against a wild-type protein, one against a mutant variant of the same protein - and produces a side-by-side comparison that a wet-lab chemist or oncologist can read in five minutes:

- **Variant context.** The three per-variant priors from notebook 07 (AlphaMissense pathogenicity, gnomAD allele frequency, RaSP ΔΔG stability) anchor the mutation in a clinical / population / structural reading.
- **Structural diff.** Pocket-restricted Cα-RMSD plus a full-protein Cα-RMSD plus a side-by-side 3-D viewer. Tells you whether the binding pocket has actually reshaped or whether the mutation is geometrically silent.
- **Interaction-fingerprint (IFP) diff.** Which protein-ligand interactions are gained / lost / preserved between WT and mutant pocket. The chemistry-level readout of "did the binding mode change."
- **Shortlist diff.** Which compounds survive the consensus filter in both genotypes, which only in one. Rank-movement deltas. The wet-lab-handoff readout: "in WT we'd prioritise these N compounds; in the mutant the priorities reshuffle to these M, of which X are new candidates."
- **One self-contained HTML report per (WT, variant) pair** at `data/derived/<target>/mutations/wt_vs_<variant>.html`. Embedded interactive py3Dmol viewer; opens in any modern browser; doubles as the artefact you forward to a wet-lab collaborator or paste into a clinical-collaboration meeting.

The DPYD I560S walkthrough is the deep pedagogical centrepiece (dihydropyrimidine dehydrogenase, 5-fluorouracil metabolism, colorectal-cancer pharmacogenomics). Six brief follow-up demos (KRAS G12C, ESR1 Y537S, BRCA1 LoF C61G, CYP2D6, NAT2, UGT1A1\\*28) confirm the same template applies; full WT-vs-mutant compute for those lands at notebook 99's production runner.

## Learning objectives

After running this notebook you will be able to:

- Explain in plain language why a **WT-vs-variant diff** is more robust than any absolute affinity score: systematic over- or under-estimation by Boltz-2 / gnina largely cancels when you subtract.
- Read a **pocket-restricted Cα-RMSD** + **distogram diff** + **side-by-side 3-D viewer** and place them against the variant's biological context.
- Read an **IFP diff** as a chemist would: which residue-interaction pairs are gained / lost / preserved between the WT and the variant pocket.
- Read a **shortlist diff** as a wet-lab chemist receiving the WT-vs-variant comparison: which compounds drop out, which survive, which newly appear, and by how many ranks.
- Recognise the four mechanistic archetypes the headline-demo set covers (pharmacogene LoF, oncogenic GoF driver, ligand-pocket GoF, synthetic-lethality LoF) and know which signals matter most for each.
- Generate one HTML report per (WT, variant) pair with embedded interactive 3D, suitable for forwarding to a wet-lab collaborator.

## Audience

You'll get value if you are:

- A clinician or wet-lab biologist who wants the one-page WT-vs-variant readout before commissioning experimental work.
- A medicinal chemist asked "does our compound library still bind this patient's variant?" - the shortlist diff is the operative answer.
- A pharmacogenomics researcher triaging which variants justify deeper structural follow-up.
- A reviewer / grant auditor: every diff surface has a citation, every method is target-agnostic, and the methodology limits are explicit.

## Prerequisites

- The conda environment `aidd` (`conda env create -f environment.yml`), or running on Google Colab (the setup cell installs everything).
- **Notebook 07** completed at least once for the variants of interest (so AlphaMissense + gnomAD + RaSP caches are warm under `data/cache/`). The DPYD I560S walkthrough uses these caches.
- For the **DPYD walkthrough**: the notebook can run end-to-end against **synthetic stub trees** generated in section 3 (step-15 mode; `USE_STUB_TREES = True`). Real DPYD WT + I560S pipeline runs are nb 99 production territory.
- For the **six follow-up demos**: only the nb 07 priors are exercised; full WT-vs-mutant pipeline trees are not required at step-15 closure.

## Runtime

- **CPU-only.** No GPU dependency anywhere. Free Colab CPU runtime, free Colab T4, Windows / macOS local - all work identically.
- **First run on a fresh Colab T4: 5 - 10 minutes.** Dominated by the one-time AlphaFold model download (DPYD wild-type structure from AlphaFold-DB, ~1 MB) plus the IFP computation across the 11 stub ligand poses.
- **Cached re-runs: under 60 seconds.** The stub trees, the variant-prior caches, and the structural files all live on Drive (Colab) or in the repo's `data/cache/` + `data/derived/` (local).
"""),

        markdown("""
## Key terms used in this notebook

| Term | Meaning in one line |
|---|---|
| **Wild-type (WT)** | The reference protein sequence and structure that the pipeline was originally run against. |
| **Variant / mutant** | A protein with one (or more) amino-acid substitutions relative to WT - encoded as `<wt_aa><position><alt_aa>` (for example, `I560S` = isoleucine at position 560 changed to serine). |
| **Missense variant** | A single-base DNA change that swaps one amino acid for another. The structural pipeline can handle these directly. |
| **Splice variant** | A change to mRNA splicing that produces a truncated or different protein (DPYD\\*2A is the canonical example). Out of scope for missense-only protein-level tools - flagged honestly when encountered. |
| **Cα-RMSD (alpha-carbon RMSD)** | Root-mean-square deviation of Cα-atom positions between two superposed structures, in Å. Measures backbone geometric agreement. |
| **Pocket-restricted Cα-RMSD** | Cα-RMSD computed only over residues within a defined shell of the binding pocket. More sensitive to local pocket reshape than whole-protein RMSD. |
| **Distogram** | Cα-Cα distance matrix; useful as a 2-D image diff for spotting which residues move relative to each other under the mutation. |
| **Interaction fingerprint (IFP)** | A binary record of which protein-ligand contacts a pose makes, indexed by (residue, interaction type). Computed via ProLIF. |
| **IFP diff** | The per-(residue, interaction-type) set difference between WT and mutant IFP. Three categories: gained (in mutant only), lost (in WT only), preserved (in both). |
| **Shortlist** | The compounds that survive the consensus filter (notebook 06). One shortlist per genotype; diffing them tells you the genotype's effect on prioritisation. |
| **Consensus filter (non-ERK2)** | For pharmacogenes and other non-ERK2 targets, there is no labelled training set to fit a per-target rescorer on, so the consensus uses Boltz-2 affinity and gnina CNN-affinity as the two independent lanes (instead of the rescorer + Boltz-2 pair used for ERK2 in notebook 06). |
| **AlphaMissense / gnomAD / RaSP ΔΔG** | The three per-variant computational priors from notebook 07. Loaded as added columns alongside the structural / IFP / shortlist diff. |
| **DPYD\\*2A** | The canonical clinical DPYD pharmacogene allele (`IVS14+1G>A`, splice). NOT a missense variant; AlphaMissense and RaSP cannot score it directly. We use DPYD I560S (\\*13), a real pathogenic missense, as the structural probe instead. |
| **Embedded py3Dmol** | A self-contained HTML+JavaScript chunk that renders an interactive protein-ligand 3D viewer in any modern browser. The HTML report uses these for the side-by-side WT/mutant views. |
| **PRISM** | The PRISM file format (YAML header + whitespace-separated data block) used by upstream RaSP to ship per-protein saturated-scan predictions. Relevant only if you re-run the operator-step probe for the full-proteome RaSP cache. |
"""),

        markdown("""
## Why WT-vs-variant diffs - the chemistry reasoning

In oncology, **amino-acid variants in or near the active sites of enzymes** drive much of the clinical landscape. Two mechanisms dominate:

- **Resistance mutations.** A drug works initially, the tumour evolves a single amino-acid change that reshapes the binding pocket, and the drug loses potency. EGFR T790M (osimertinib resistance), BRAF V600E (vemurafenib activation), ESR1 Y537S (tamoxifen resistance), ABL T315I - these define the prescribing landscape for second- and third-line therapies.
- **Pharmacogenomic loss-of-function.** Germline variants in drug-metabolising enzymes (DPYD, UGT1A1, NAT2, TPMT, CYP2D6) change how a patient metabolises standard chemotherapy. DPYD\\*2A homozygotes given full-dose 5-fluorouracil experience life-threatening toxicity; the pharmacogene structural change is the mechanism.

Any cancer-drug-discovery pipeline - especially one shaped for common solid tumours and precision oncology - has to answer "*what happens when I run this variant against the same drugs and substrates as the wild-type?*"

### The four levels of investigation (per `_planning/PROJECT_PROPOSAL.md` §8)

| Level | Question | How this notebook answers it |
|---|---|---|
| 1. Mutant structure | What does the mutant look like? | Read the mutant PDB from `data/derived/<target>_<variant>/fold/`. |
| 2. Structural difference | How is the mutant fold different from WT? | Pocket-restricted Cα-RMSD + distogram diff + side-by-side viewer in §6. |
| 3. Stability change (ΔΔG) | Is the mutant more or less stable? | RaSP ΔΔG from notebook 07's helper, surfaced in §5. |
| 4. Effect on drug binding | Does the candidate library still bind the mutant? | IFP diff (§7) and shortlist diff (§8) against the same compounds run through both genotypes. |
| 5. Atom-level dynamics | Did the mutation change a hinge motion or allosteric loop? | **Out of scope.** Requires molecular-dynamics simulation; see the Recap for pointers. |

### Why diff signals beat absolute affinities

For a single (compound, target) pair, both Boltz-2 affinity and gnina CNN-affinity carry **systematic** biases (they over- or under-estimate certain chemotypes, certain pocket sizes, certain target families). On a WT-vs-mutant comparison those biases largely **cancel** - the same compound through the same pipeline against two genotypes of the same protein. The deltas are therefore more trustworthy than the absolute numbers. This is the central methodological reason variants are the right scientific frame for this pipeline, not generic SBVS.

### Honesty about what we measure

The pipeline measures **whether the substrate fits the active site** (a necessary condition for catalysis), not the **catalytic rate** (k_cat, K_m, turnover). For the pharmacogene demos (DPYD, NAT2, CYP2D6, UGT1A1), the clinically meaningful question often involves catalysis, not just binding. The pipeline captures one mechanism contributing to slow / poor metabolism (binding-site fit + protein-stability change via RaSP ΔΔG), not the full catalytic-rate story. QM/MM and EnzyHTP-style methods are the right tools for catalysis rate; both are out of scope here. The Recap carries the explicit caveat language.
"""),

        markdown("""
## 1 - Setup

### What this section does

Detect Colab vs local, install dependencies that are not in Colab's default image (rdkit, py3Dmol, prolif, biopython, pyyaml), set up the import path for `src/aidd/`, and mount Google Drive on Colab so the (one-time) AlphaFold model download and the AlphaMissense / RaSP / gnomAD caches from notebook 07 survive runtime restarts.
"""),

        code(title="Setup: detect Colab vs local, configure paths", source="""
import sys
import importlib
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    # CPU-only deps. pandas / numpy / matplotlib / pyarrow / requests / pyyaml
    # are in Colab's default image; rdkit / py3Dmol / prolif / biopython are not.
    !pip install -q rdkit py3Dmol prolif biopython
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

        code(title="Imports", source=AUTORELOAD_SNIPPET + """
import warnings
warnings.filterwarnings("ignore")

import io
import json
import math
import re
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBIO, PDBParser, Select, Superimposer

from aidd.io import mount_drive_if_colab, pretty_path
from aidd.structures import ca_coordinates, ca_rmsd, distogram, read_pdb
from aidd.ifp import compute_ifp
from aidd.viz import (
    show_protein_ligand,
    show_binding_site,
    show_structure_colored_by_plddt,
)
from aidd.consensus import compute_consensus
from aidd.variants import (
    DEMO_SET_UNIPROT_IDS,
    alphamissense_score,
    alphamissense_class,
    gnomad_frequency,
)
from aidd.stability import rasp_ddg, rasp_covers, FULL_PROTEOME_PARQUET
from aidd.mutation import (
    ca_rmsd_pocket,
    ifp_diff,
    pocket_residues_within,
    render_mutation_html,
    shortlist_diff,
    summarise_mutation,
)

print("imports ok")
"""),

        markdown("""
### Google Drive - read this before running the next cell

The variant-prior caches built by notebook 07 (`alphamissense/demo_set.parquet`, `gnomad/<gene>_gnomad_r4.json`, `rasp/full_proteome_demo_set.parquet` once the operator probe lands) live on **Google Drive** by default when you ran notebook 07 on Colab. The same Drive tree also hosts this notebook's stub-tree fixtures under `data/derived/dpyd_wt/` and `data/derived/dpyd_i560s/`.

**To opt out**, set `USE_DRIVE = False` in the cell *before* running it. The notebook will then look for caches and derived data under the local repo's `data/cache/` + `data/derived/` trees - which works if you copied them down from Drive, or if you ran every upstream notebook locally.
"""),

        code(title="Drive persistence toggle  (change to False to opt out)", source="""
# ════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE AUTHORIZATION
# On Colab, the next call will prompt for Drive permission on first use.
#
# →  To OPT OUT of Drive, change the line below to:    USE_DRIVE = False
#    (caches and derived data then live under the local data/ tree only)
# ════════════════════════════════════════════════════════════════════════
USE_DRIVE = IS_COLAB

DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)
CACHE_ROOT = DATA_ROOT.parent / "cache"
ALPHAMISSENSE_CACHE = CACHE_ROOT / "alphamissense"
RASP_CACHE          = CACHE_ROOT / "rasp"
GNOMAD_CACHE        = CACHE_ROOT / "gnomad"
# aidd.io.mount_drive_if_colab already returns data/derived/ (per its docstring),
# so DERIVED_ROOT is an alias for DATA_ROOT, NOT DATA_ROOT / "derived" (that
# double-nested on the first Colab run and produced data/derived/derived/...).
DERIVED_ROOT        = DATA_ROOT

print(f"USE_DRIVE: {USE_DRIVE}  "
      f"({'Drive-backed' if USE_DRIVE else 'local /content (or local repo)'})")
print(f"Cache root:   {CACHE_ROOT}")
print(f"Derived root: {DERIVED_ROOT}")
"""),

        markdown("""
## 2 - Inputs and pre-cache verification

### What this section does

Defines the WT and variant target identifiers for the DPYD walkthrough, then verifies that the upstream pipeline outputs (fold, docking, scoring, boltz) are present under `data/derived/<target>_wt/` and `data/derived/<target>_<variant>/`. If anything is missing, the assertion lists the missing paths so you know exactly what is needed.

**For step 15** the WT and variant trees for DPYD are **synthetic stub fixtures** generated by the next cell (section 3). The architecture is the same as for real production runs - first this cell verifies presence, then the rest of the notebook reads from those paths - so the comparison logic exercises the same code paths in step 15 and at nb 99 production time. When real DPYD runs land at nb 99, the stub-generator toggle in section 3 flips to `False` and this cell asserts against the real outputs.
"""),

        code(title="Inputs: WT and variant target identifiers + pre-cache verification", source="""
# DPYD I560S deep walkthrough - see section 5 for the *2A vs I560S rationale.
TARGET           = "dpyd"
VARIANT          = "I560S"
UNIPROT          = "Q12882"
POSITION         = 560
WT_AA            = "I"
MUT_AA           = "S"

WT_DIR  = DERIVED_ROOT / f"{TARGET}_wt"
MUT_DIR = DERIVED_ROOT / f"{TARGET}_{VARIANT.lower()}"

# Output directory for the HTML report (per PROJECT_PROPOSAL.md §8).
MUTATIONS_DIR = DERIVED_ROOT / TARGET / "mutations"

# The four upstream artefacts the comparison logic reads. Per-genotype paths.
REQUIRED_ARTEFACTS = {
    "fold":     "fold/{name}_best.pdb",
    "docking":  "docking/poses.sdf",
    "scoring":  "scoring/scored_poses.parquet",
    "boltz":    "boltz/affinity.csv",
}


def _expected_paths(genotype_dir: Path, genotype_name: str) -> dict[str, Path]:
    return {
        stage: genotype_dir / template.format(name=genotype_name)
        for stage, template in REQUIRED_ARTEFACTS.items()
    }


def verify_trees(*, build_if_missing: bool) -> dict[str, dict[str, Path]]:
    \"\"\"Return resolved paths per (genotype, stage). Raise listing the gaps if any are missing.\"\"\"
    expected = {
        "wt":  _expected_paths(WT_DIR,  f"{TARGET}_wt"),
        VARIANT.lower(): _expected_paths(MUT_DIR, f"{TARGET}_{VARIANT.lower()}"),
    }
    missing: list[str] = []
    for genotype, paths in expected.items():
        for stage, path in paths.items():
            if not path.exists():
                missing.append(f"{genotype}/{stage}: {path}")
    if missing and not build_if_missing:
        raise FileNotFoundError(
            "Upstream pipeline artefacts not found for "
            f"{TARGET} {{wt, {VARIANT}}}:\\n  - " + "\\n  - ".join(missing) +
            "\\n\\nRun notebooks 01 -> 05 for each genotype, or enable the "
            "step-15 stub fixture in section 3 by leaving USE_STUB_TREES = True."
        )
    return expected


# At this stage we are still in step-15 (stub-fixture) mode; section 3 will
# populate the missing paths. We therefore call verify_trees with
# build_if_missing=True so this cell does not raise here. The real assertion
# fires post-section-3.
_ = verify_trees(build_if_missing=True)
print(f"WT  tree expected at: {WT_DIR}")
print(f"Mut tree expected at: {MUT_DIR}")
print(f"HTML reports go to:   {MUTATIONS_DIR}")
"""),

        markdown("""
## 3 - DPYD stub trees (step-15 fixture)

### Why this section exists

Step 15's mandate is **methodology verification, not novel empirical findings**. Running real DPYD WT and DPYD I560S pipelines end-to-end (fold + dock + score + Boltz-2 + consensus, for each genotype) is GPU compute that belongs to notebook 99's production runner. To exercise every diff surface in this notebook without that compute, we ship a **synthetic stub fixture** for DPYD: real protein structures (AlphaFold WT model + a coarse I→S surgery for the mutant), real ligand SMILES for the 5-fluorouracil family and unrelated drug-like negative controls, and synthetic Boltz-2 / gnina-CNN scores designed to populate every cell of the diff tables.

**What the stub fixture is honest about:**

- The protein structures: WT comes from AlphaFold-DB (real); the mutant is the WT model with residue 560's sidechain replaced by a minimal serine surrogate (small, coarse - real PDBFixer-based modelling is what nb 99 will use).
- The ligands: SMILES for 5-fluorouracil and 5 closely-related fluoropyrimidines (floxuridine, capecitabine, 5,6-dihydrouracil, tegafur, gimeracil) are the **real molecules** from PubChem. The 5 negative-control compounds are also real drugs (aspirin, ibuprofen, caffeine, diphenhydramine, propranolol) but chemically unrelated to fluoropyrimidines - they're stand-ins for "library noise", not predictions about real binding behaviour.
- The poses: 3-D conformers generated by RDKit and translated to sit near the WT residue 560 Cα. They are not docked poses; they are placement stubs whose only purpose is to exercise the IFP-computation pathway.
- The Boltz-2 affinity / gnina CNN-affinity values: **synthetic and designed-to-exercise every shortlist-diff branch**. The substrate plus three close analogs (floxuridine, capecitabine) survive in both genotypes' shortlists. One analog (gimeracil) sits inside the WT top-K and drops below the threshold in the mutant — this is the `wt_only` branch. One negative-control compound (diphenhydramine) starts at the bottom in WT and promotes into the mutant top-K — this is the `mut_only` branch. The rank-movement artefact is the verification that all three branches (both / wt_only / mut_only) fire, not a clinical prediction. All scores are explicit per-compound constants in the cell so the stub is deterministic across kernel restarts.

When real DPYD WT and I560S production runs land at notebook 99, flip the toggle below to `False` and section 2's verification will assert against those real outputs instead.

The stub-generator cell is **idempotent**: when the stub trees already exist on disk, it skips the rebuild.
"""),

        code(title="Build DPYD stub trees (step-15 fixture; idempotent)", source="""
# ════════════════════════════════════════════════════════════════════════
# STEP-15 STUB FIXTURE TOGGLE
# Set USE_STUB_TREES = False when real DPYD WT + I560S production runs
# exist under DERIVED_ROOT / {dpyd_wt, dpyd_i560s} / ...
# ════════════════════════════════════════════════════════════════════════
USE_STUB_TREES = True

ALPHAFOLD_DB_API = "https://alphafold.ebi.ac.uk/api/prediction/Q12882"


def _resolve_dpyd_alphafold_url() -> str:
    \"\"\"Look up the current AlphaFold-DB PDB URL for UniProt Q12882 (DPYD).

    AlphaFold-DB rolls its model version periodically (v4 in 2022, v6 by mid-2025);
    hitting the JSON API once returns the current pdbUrl regardless of version, so
    a hardcoded version-stamped URL does not re-break the stub fixture every time
    the upstream bumps. The API is unauthenticated and CORS-friendly.

    The resolver assumes the F1 canonical isoform is the first entry in the
    returned list. Verified for Q12882 (2 entries today, primary is F1); a
    future agent extending the pattern to other UniProt accessions in step 16
    or step 17 should re-verify per accession or filter explicitly on the
    ``isoform`` field rather than blindly indexing ``payload[0]``.
    \"\"\"
    with urllib.request.urlopen(ALPHAFOLD_DB_API, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Unexpected AlphaFold-DB API response: {payload!r}")
    pdb_url = payload[0].get("pdbUrl")
    if not pdb_url:
        raise RuntimeError(
            f"No pdbUrl in AlphaFold-DB primary entry for Q12882: {payload[0]!r}"
        )
    return pdb_url

# Real fluoropyrimidine ligands. SMILES + PubChem CIDs from pubchem.ncbi.nlm.nih.gov.
DPYD_LIGANDS = [
    # name, SMILES, PubChem CID, family
    ("5_fluorouracil",     "O=C1NC(=O)C(F)=CN1",                                       "3385",  "substrate"),
    ("floxuridine",        "OCC1OC(CC1O)N1C=C(F)C(=O)NC1=O",                           "5790",  "analog"),
    ("capecitabine",       "CCCCCOC(=O)NC1=NC(=O)N(C=C1F)C1OC(C)C(O)C1O",              "60953", "analog"),
    ("dihydrouracil",      "O=C1NCCC(=O)N1",                                           "649",   "analog"),
    ("tegafur",            "O=C1NC(=O)C(F)=CN1C1CCCO1",                                "5386",  "analog"),
    ("gimeracil",          "OC1=NC=C(Cl)C(=O)C1",                                      "65628", "analog"),
    # Unrelated drug-like negative controls (real drugs, chemically unrelated to fluoropyrimidines).
    ("aspirin",            "CC(=O)Oc1ccccc1C(=O)O",                                    "2244",  "decoy"),
    ("ibuprofen",          "CC(C)Cc1ccc(C(C)C(=O)O)cc1",                               "3672",  "decoy"),
    ("caffeine",           "Cn1c(=O)c2c(ncn2C)n(C)c1=O",                               "2519",  "decoy"),
    ("diphenhydramine",    "CN(C)CCOC(c1ccccc1)c1ccccc1",                              "3100",  "decoy"),
    ("propranolol",        "CC(C)NCC(O)COc1cccc2ccccc12",                              "4946",  "decoy"),
]


def _download_dpyd_wt_pdb(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_url = _resolve_dpyd_alphafold_url()
    print(f"  downloading {pdb_url}")
    urllib.request.urlretrieve(pdb_url, out_path)


class _IsoleucineSidechainToSerineSelect(Select):
    \"\"\"Biopython Select that drops I560's ethyl branch (CG1/CG2/CD1) and OG-stub at CB.

    Coarse step-15-only single-residue surgery; nb 99 uses PDBFixer with proper
    bond-length / rotamer placement.
    \"\"\"
    def __init__(self, position: int, chain: str = "A"):
        self.position = position
        self.chain = chain
    def accept_residue(self, residue):
        return True
    def accept_atom(self, atom):
        residue = atom.get_parent()
        if residue.id[1] == self.position and residue.get_parent().id == self.chain:
            return atom.get_name() in {"N", "CA", "C", "O", "CB"}
        return True


def _mutate_i560s_inplace(src_pdb: Path, dst_pdb: Path) -> None:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("dpyd_wt", str(src_pdb))
    # Rename residue 560 from ILE to SER in the in-memory structure.
    for model in struct:
        for chain in model:
            if chain.id != "A":
                continue
            for residue in chain:
                if residue.id[1] == POSITION:
                    residue.resname = "SER"
        break
    dst_pdb.parent.mkdir(parents=True, exist_ok=True)
    io_ = PDBIO()
    io_.set_structure(struct)
    io_.save(str(dst_pdb), _IsoleucineSidechainToSerineSelect(POSITION))


def _embed_and_translate(smiles: str, anchor_xyz: np.ndarray, seed: int) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # MMFF parameter gap on uncommon atoms; positions are still embedded.
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    coords = np.array([[c.x, c.y, c.z] for c in coords]) if hasattr(coords[0], "x") else coords
    coords -= coords.mean(axis=0)
    coords += anchor_xyz
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, coords[i].tolist())
    return mol


def _write_pose_sdf(out_sdf: Path, mol_list: list[tuple[str, Chem.Mol]]) -> None:
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_sdf))
    try:
        for compound_id, mol in mol_list:
            mol = Chem.Mol(mol)
            mol.SetProp("_Name", compound_id)
            writer.write(mol)
    finally:
        writer.close()


def _resnum_ca_coord(pdb_path: Path, position: int, chain: str = "A") -> np.ndarray:
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
    raise ValueError(f"residue {position} chain {chain} CA not found in {pdb_path}")


def _build_genotype_stub(genotype_dir: Path, genotype_name: str, perturb: bool) -> None:
    genotype_dir.mkdir(parents=True, exist_ok=True)
    pdb_out = genotype_dir / "fold" / f"{genotype_name}_best.pdb"
    if not pdb_out.exists():
        if genotype_name.endswith("_wt"):
            _download_dpyd_wt_pdb(pdb_out)
        else:
            wt_pdb = WT_DIR / "fold" / f"{TARGET}_wt_best.pdb"
            if not wt_pdb.exists():
                _download_dpyd_wt_pdb(wt_pdb)
            _mutate_i560s_inplace(wt_pdb, pdb_out)

    poses_out = genotype_dir / "docking" / "poses.sdf"
    if not poses_out.exists():
        anchor = _resnum_ca_coord(pdb_out, POSITION)
        # Place each ligand ~6 Å from residue 560's Cα in slightly different directions
        # so IFPs across compounds are distinguishable.
        offsets = np.array([
            [ 6.0,  0.0,  0.0], [ 5.0,  3.0,  2.0], [ 4.0, -3.0,  3.0],
            [-5.0,  2.0,  3.0], [ 6.0,  0.0, -3.0], [ 3.0,  5.0, -2.0],
            [-3.0,  5.0,  3.0], [-4.0, -4.0,  2.0], [ 2.0, -5.0, -3.0],
            [-2.0,  6.0,  1.0], [ 5.0, -2.0,  4.0],
        ])
        mols: list[tuple[str, Chem.Mol]] = []
        for i, (cid, smi, _pubchem, _fam) in enumerate(DPYD_LIGANDS):
            jitter = np.array([0.4, -0.3, 0.5]) if perturb else np.zeros(3)
            mol = _embed_and_translate(smi, anchor + offsets[i] + jitter, seed=42 + i)
            mols.append((cid, mol))
        _write_pose_sdf(poses_out, mols)

    # Explicit per-compound stub scores. Designed so the shortlist diff at
    # TOP=0.30 (K=4 of 11) produces 3 + 1 + 1 across the both/wt_only/mut_only
    # branches:
    #   WT  top-4 by rank product: 5-FU, floxuridine, gimeracil, capecitabine
    #   Mut top-4 by rank product: 5-FU, floxuridine, diphenhydramine, capecitabine
    #   → both = {5-FU, floxuridine, capecitabine}
    #     wt_only  = {gimeracil}        (analog that drops in mutant)
    #     mut_only = {diphenhydramine}  (decoy that promotes in mutant)
    # Sign conventions: gnina_cnn_affinity higher = stronger;
    # boltz_affinity (log10(IC50) µM) lower = stronger.
    GNINA_WT = {
        "5_fluorouracil":    9.50,
        "floxuridine":       8.30,
        "capecitabine":      8.10,
        "gimeracil":         8.00,
        "dihydrouracil":     7.90,
        "tegafur":           7.70,
        "aspirin":           3.50,
        "ibuprofen":         3.35,
        "caffeine":          3.20,
        "diphenhydramine":   3.05,
        "propranolol":       2.90,
    }
    BOLTZ_WT = {
        "5_fluorouracil":   -2.80,
        "gimeracil":        -2.50,
        "floxuridine":      -2.45,
        "capecitabine":     -2.40,
        "dihydrouracil":    -2.35,
        "tegafur":          -2.30,
        "aspirin":          +0.50,
        "ibuprofen":        +0.60,
        "caffeine":         +0.70,
        "diphenhydramine":  +0.80,
        "propranolol":      +0.90,
    }
    GNINA_MUT_DELTA = {"gimeracil": -3.00, "diphenhydramine": +5.00}
    BOLTZ_MUT_DELTA = {"gimeracil": +2.00, "diphenhydramine": -3.50}

    scored_out = genotype_dir / "scoring" / "scored_poses.parquet"
    if not scored_out.exists():
        rows = []
        for cid, smi, pubchem, fam in DPYD_LIGANDS:
            gnina_val = GNINA_WT[cid] + (GNINA_MUT_DELTA.get(cid, 0.0) if perturb else 0.0)
            rows.append({
                "compound_id":          cid,
                "smiles":               smi,
                "pubchem_cid":          pubchem,
                "family":               fam,
                "gnina_cnn_affinity":   gnina_val,
                "gnina_affinity":       gnina_val * -0.7,  # raw Vina-like; lower = stronger
            })
        df = pd.DataFrame(rows)
        scored_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(scored_out, index=False)

    boltz_out = genotype_dir / "boltz" / "affinity.csv"
    if not boltz_out.exists():
        rows = []
        for cid, smi, pubchem, fam in DPYD_LIGANDS:
            boltz_val = BOLTZ_WT[cid] + (BOLTZ_MUT_DELTA.get(cid, 0.0) if perturb else 0.0)
            rows.append({
                "compound_id":                cid,
                "boltz_affinity":             boltz_val,
                "boltz_affinity_probability": float(np.clip(0.5 - boltz_val / 4, 0.05, 0.95)),
            })
        df = pd.DataFrame(rows)
        boltz_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(boltz_out, index=False)


if USE_STUB_TREES:
    print("Building DPYD stub trees (idempotent; will skip files that exist)...")
    _build_genotype_stub(WT_DIR,  f"{TARGET}_wt",                  perturb=False)
    _build_genotype_stub(MUT_DIR, f"{TARGET}_{VARIANT.lower()}",    perturb=True)
    print("Stub trees ready.")
else:
    print("USE_STUB_TREES = False -- expecting real DPYD WT + I560S trees on disk.")

# Re-verify; this time we want a hard error if anything is still missing.
verified = verify_trees(build_if_missing=False)
for genotype, paths in verified.items():
    print(f"\\n{genotype}:")
    for stage, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {stage:>10s}: {str(pretty_path(path)):<70s}  ({size_kb:.1f} KB)")
"""),

        markdown("""
## 4 - Comparison helpers (loaded from `src/aidd/mutation.py`)

### What this section does

Six small helper functions used by every comparison cell below. They live in [`src/aidd/mutation.py`](../src/aidd/mutation.py) - extracted from this notebook in step 16 commit 2/4 to give notebook 09 (small-N MoA driver) and notebook 99 (production runner) `RUN_MODE = "moa"` a single source of truth across all three callers. The Q1 reuse-mechanism decision (extract vs verbatim-copy vs builder-side source-of-truth) was made in favour of extract. Imported at the top of this notebook in §1; the cells below print a confirmation that the imports are live.

- `pocket_residues_within(pdb, anchor_residue, radius_a)` - returns the set of residue numbers whose any-heavy-atom Cα is within `radius_a` Å of the anchor residue's Cα. Used to define the "pocket shell" for the pocket-restricted Cα-RMSD.
- `ca_rmsd_pocket(pdb_wt, pdb_mut, pocket_residues)` - Cα-RMSD restricted to the pocket-residue set; uses Biopython's Superimposer on matched-by-resnum Cα atoms.
- `ifp_diff(ifp_wt_df, ifp_mut_df)` - returns `{gained, lost, preserved}` lists of `(residue, interaction_type)` pairs. Aggregates over poses: an interaction is "on" if any pose makes it.
- `shortlist_diff(shortlist_wt, shortlist_mut)` - classifies compounds as `both / wt_only / mut_only`; reports `rank_delta = mut_rank - wt_rank` for compounds in both.
- `summarise_mutation(...)` - top-level aggregator: pulls the three diffs plus the three nb 07 priors into one dict ready for HTML rendering.
- `render_mutation_html(summary, out_path)` - writes the summary dict as a self-contained HTML file with an embedded py3Dmol viewer. Output target per `PROJECT_PROPOSAL.md` §8: `data/derived/<target>/mutations/wt_vs_<variant>.html`.

A small naming caveat for the consensus call below: `compute_consensus`'s `rescorer_col` parameter is a historical artefact from notebook 06's ERK2-rescorer design. Semantically it accepts any "second-lane score column"; for DPYD and the other non-ERK2 demos, we point it at `gnina_cnn_affinity`. A future step-17 refactor may rename it to `lane2_col` or `companion_col` when notebook 99 lands and all four consensus-using notebooks (06, 08, 09, 99) get touched in one architectural pass.
"""),

        code(title="Helpers: geometry + IFP + shortlist diffs (loaded from aidd.mutation)", source="""
# pocket_residues_within / ca_rmsd_pocket / ifp_diff / shortlist_diff live in
# src/aidd/mutation.py (extracted from this notebook in step 16 commit 2/4 so
# notebooks 09 and 99 RUN_MODE="moa" share one source of truth). Imported in
# section 1; this cell prints the same confirmation string the inline-defs
# cell used, so the saved Colab output from step 15 remains accurate after
# the move.
print("geometry + IFP + shortlist helpers defined")
"""),

        code(title="Helpers: summary aggregator + HTML renderer (loaded from aidd.mutation)", source="""
# render_mutation_html / summarise_mutation also live in src/aidd/mutation.py.
# Imported in section 1; this cell prints the legacy confirmation string so
# the saved Colab output from step 15 remains accurate after the move.
print("summary aggregator + HTML renderer defined")
"""),

        markdown("""
## 5 - DPYD walkthrough: variant context (notebook 07 priors)

### Background

**Dihydropyrimidine dehydrogenase (DPD)** is the rate-limiting enzyme in **pyrimidine catabolism** - it reduces 5-fluorouracil (5-FU, the backbone chemotherapy for colorectal cancer) and uracil at the C5-C6 double bond. Patients with reduced DPD activity accumulate 5-FU and experience severe, sometimes life-threatening, toxicity at standard chemotherapy doses. The clinical mandate is well-established: **CPIC** publishes formal dose-reduction guidelines genotype-by-genotype, and the **FDA label** for 5-fluorouracil names DPD-deficiency variants.

The **DPYD\\*2A allele** (`IVS14+1G>A`, intron-14 splice donor variant) is the canonical clinical example: about 1.5% allele frequency in non-Finnish Europeans, much rarer in East Asians, and a strong recommendation in CPIC dosing guidance. However, DPYD\\*2A is a **splice variant**, not a missense - the disrupted splicing produces a truncated protein with the catalytic domain (or large parts of it) missing. AlphaMissense and RaSP are **missense-only** protein-level tools. Neither has a meaningful answer for a splice variant.

For this notebook's deep walkthrough we therefore use **DPYD I560S** (the \\*13 allele) as the **structural probe**: a real, pathogenic, well-characterised DPYD missense variant with documented reduced enzyme activity. Same gene, same pharmacology, but a substitution AlphaMissense and RaSP **can** score. The methodological point - "AlphaMissense and RaSP are protein-level missense-only tools; splice variants need different methods" - is itself part of what this notebook teaches. DPYD\\*2A is mentioned in the recap as future-work scope (modelling the truncated protein via notebook 01's mutant-sequence fold path).

### Cell intro

Look up the three nb 07 priors for I560S and print them as a single context table. If RaSP returns `None`, it is because the operator-step probe for the full-proteome cache has not been run yet (see `_planning/PROJECT_PROPOSAL.md` § 0 row for the AlphaFold-RaSP swap). AlphaMissense + gnomAD work regardless.
"""),

        code(title="DPYD I560S priors (AlphaMissense + gnomAD + RaSP)", source="""
am_prob  = alphamissense_score(UNIPROT, POSITION, MUT_AA, cache_dir=ALPHAMISSENSE_CACHE)
am_label = alphamissense_class(UNIPROT, POSITION, MUT_AA, cache_dir=ALPHAMISSENSE_CACHE)
gn       = gnomad_frequency(UNIPROT, POSITION, MUT_AA, wt_aa=WT_AA, cache_dir=GNOMAD_CACHE)
ddg      = rasp_ddg(UNIPROT, POSITION, WT_AA, MUT_AA, cache_dir=RASP_CACHE)
rasp_covered = rasp_covers(UNIPROT, cache_dir=RASP_CACHE)

priors_table = pd.DataFrame([
    {"prior": "AlphaMissense pathogenicity",
     "value": am_prob,
     "context": f"class: {am_label}"},
    {"prior": "gnomAD allele frequency (overall)",
     "value": gn["allele_freq_overall"] if gn["found"] else 0.0,
     "context": f"found: {gn['found']}, flip: {gn['reference_flipped']}, "
                f"hgvsp: {gn['hgvsp']}"},
    {"prior": "RaSP ΔΔG (kcal/mol; positive = destabilising)",
     "value": ddg,
     "context": (f"protein covered: {rasp_covered}"
                 if ddg is None else "positive = destabilising")},
])
print(f"DPYD I560S priors (UniProt {UNIPROT}, position {POSITION}, I->S)")
priors_table
"""),

        markdown("""
### Interpretation

The three priors carry independent information. For DPYD I560S we **expect**:

- **AlphaMissense pathogenic** (probability > 0.5, class `likely_pathogenic`). I560S is a buried hydrophobic-to-polar substitution in the FAD-binding region; evolutionary conservation flags it.
- **gnomAD rare** (allele frequency near zero). I560S (the \\*13 allele) is a low-frequency LoF allele; carriers are seen but not common in healthy populations.
- **RaSP destabilising** (ΔΔG > 0). Replacing the buried I560 ethyl branch with a smaller, polar serine sidechain removes packing volume in the protein core - the canonical destabilising-by-misfolding mechanism. **The numerical RaSP value will be `None` until the full-proteome cache lands** (the operator-step probe; see commit `ef701d0`'s description). AlphaMissense + gnomAD alone already point at the loss-of-function-by-misfolding hypothesis; RaSP closes the structural case once available.

Read the three together: a variant that AlphaMissense calls pathogenic, RaSP calls destabilising, and gnomAD shows is rare is the textbook **loss-of-function-by-misfolding** signature - the variant doesn't fold correctly, so the enzyme isn't there, so the substrate accumulates. This is the structural mechanism behind DPYD\\*13's clinical reduced-DPD-activity phenotype.
"""),

        markdown("""
## 6 - DPYD walkthrough: structural diff

### Background

Two RMSD numbers tell complementary stories. **Full-protein Cα-RMSD** is dominated by the parts of the protein that *didn't* change - if a single residue's sidechain is the only difference, the full-protein number is essentially zero (the backbone barely moves). It is reassuring as a sanity check but uninformative about the binding site.

**Pocket-restricted Cα-RMSD** measures Cα displacement only over residues whose Cα lies within a chosen shell of the mutated residue (default here: 8 Å). This is much more sensitive to local geometric change. A polar→hydrophobic switch in a buried position with bulky neighbours can push surrounding backbone by 0.3-0.8 Å pocket-RMSD without moving the full-protein RMSD at all. That deflection is the structural mechanism the IFP diff in the next section will pick up.

The 8 Å cutoff is a generic, target-agnostic default. A literature-curated per-target pocket-residue list (e.g. residues that contact the substrate in a co-crystal) would be more chemistry-faithful per target but adds curation burden and breaks the "one consistent definition across the notebook" pedagogy. Use the geometric default for triage; swap to a curated list if the variant is in a target with a well-defined published pocket.

### Cell intro

Compute the pocket-restricted Cα-RMSD and the full-protein Cα-RMSD for DPYD WT vs I560S; print both. Then render the two backbones overlaid in one py3Dmol view (WT in gold, mutant in cyan) centred on the mutated residue.
"""),

        code(title="DPYD structural diff: pocket-restricted + full Cα-RMSD + overlaid 3-D viewer", source="""
WT_PDB  = WT_DIR  / "fold" / f"{TARGET}_wt_best.pdb"
MUT_PDB = MUT_DIR / "fold" / f"{TARGET}_{VARIANT.lower()}_best.pdb"

pocket_set = pocket_residues_within(WT_PDB, POSITION, radius_a=8.0)
print(f"Pocket shell (radius 8 Å around residue {POSITION}): "
      f"{len(pocket_set)} residues = {sorted(pocket_set)}")

pocket_metric = ca_rmsd_pocket(WT_PDB, MUT_PDB, pocket_set)
print(f"\\nPocket-restricted Cα-RMSD: {pocket_metric['rmsd_pocket_A']:.3f} Å "
      f"(n_matched = {pocket_metric['n_matched']})")

# Full-protein Cα-RMSD via Superimposer on all matched-by-resnum Cα atoms.
parser = PDBParser(QUIET=True)
def _all_ca(path: Path):
    struct = parser.get_structure("x", str(path))
    for model in struct:
        for ch in model:
            if ch.id != "A":
                continue
            return {r.id[1]: r["CA"] for r in ch if "CA" in r}
    return {}
wt_all  = _all_ca(WT_PDB)
mut_all = _all_ca(MUT_PDB)
common = sorted(set(wt_all) & set(mut_all))
sup_full = Superimposer()
sup_full.set_atoms([wt_all[r] for r in common], [mut_all[r] for r in common])
rmsd_full = float(sup_full.rms)
print(f"Full-protein Cα-RMSD:      {rmsd_full:.3f} Å (n_common = {len(common)})")

# Overlaid py3Dmol viewer (WT gold, mutant cyan), centred on the mutated residue.
view = py3Dmol.view(width=720, height=480)
view.addModel(WT_PDB.read_text(),  format="pdb")
view.setStyle({"model": 0}, {"cartoon": {"color": "gold",      "opacity": 0.85}})
view.addModel(MUT_PDB.read_text(), format="pdb")
view.setStyle({"model": 1}, {"cartoon": {"color": "lightblue", "opacity": 0.85}})
view.setStyle({"model": 0, "resi": str(POSITION)}, {"stick": {"colorscheme": "yellowCarbon"}})
view.setStyle({"model": 1, "resi": str(POSITION)}, {"stick": {"colorscheme": "cyanCarbon"}})
view.zoomTo({"resi": str(POSITION)})
view.show()
"""),

        markdown("""
### Interpretation

Two patterns to look for:

- **Full-protein Cα-RMSD near zero** (under 0.2 Å) with **pocket Cα-RMSD modestly positive** (0.1 - 0.8 Å) is the **localised pocket reshape** signature. Most of the protein is unchanged; the variant pushes a small neighbourhood around the mutated residue. This is the most common pattern for single-residue substitutions and is exactly what should show for the I560S stub fixture (the mutant differs from WT by one sidechain only).
- **Both numbers larger** (1+ Å) would suggest the fold is differently arranged - either a real allosteric rearrangement (rare for single missense at this scale) or, in stub-fixture mode, an artefact of the coarse sidechain surgery. In production runs with PDBFixer-based modelling the localised-reshape pattern is the norm.

Note on the stub fixture: the I→S surgery used here drops the I560 ethyl-branch atoms without adding an explicit OG, so the pocket-restricted RMSD will read as essentially zero in stub mode (no backbone movement). In a real DPYD WT-vs-I560S production run, the pocket-restricted RMSD should pick up the buried-to-polar swap with a small but real deflection. The diff-surface helpers exercise the same code path in either case.
"""),

        markdown("""
## 7 - DPYD walkthrough: interaction-fingerprint (IFP) diff

### Background

An **interaction fingerprint** is a binary record of which protein-ligand contacts a pose makes, indexed by (residue, interaction type). ProLIF computes one IFP per pose; aggregating over all poses for a given genotype gives the set of interactions any pose makes against that genotype's binding site. Diffing the WT and mutant sets gives the per-(residue, interaction-type) **gained / lost / preserved** classification.

Why this matters for a clinical reading: the IFP diff is the **chemistry-level translation** of the structural change. The structural RMSD tells you *that* the pocket reshaped; the IFP diff tells you *which interactions* the reshape changed - and that maps directly onto the medicinal chemist's question of whether the same chemotype can still bind.

For DPYD I560S, the I→S swap removes the buried isoleucine ethyl-branch hydrophobic surface. Any pose that was making a Hydrophobic interaction to I560's CG1/CG2/CD1 in the WT pocket loses that interaction in the mutant. Whether a new H-bond (S560 OG as donor or acceptor) appears in the mutant depends on pose geometry; the diff exposes which case held in the stub poses.

### Cell intro

Compute IFPs for both genotypes' pose SDFs using the geometry-only stub poses defined in section 3, then call `ifp_diff` to surface gained / lost / preserved per (residue, interaction-type).
"""),

        code(title="DPYD IFP diff: gained / lost / preserved per (residue, interaction)", source="""
WT_POSES  = WT_DIR  / "docking" / "poses.sdf"
MUT_POSES = MUT_DIR / "docking" / "poses.sdf"

print(f"Computing WT  IFP from {WT_POSES.name}...")
ifp_wt  = compute_ifp(WT_PDB,  WT_POSES,  progress=False)
print(f"Computing Mut IFP from {MUT_POSES.name}...")
ifp_mut = compute_ifp(MUT_PDB, MUT_POSES, progress=False)

ifp_d = ifp_diff(ifp_wt, ifp_mut)
print(f"\\nIFP set sizes: WT = {ifp_d['n_wt']}, Mut = {ifp_d['n_mut']}")
print(f"Preserved (WT and Mut both):       {len(ifp_d['preserved'])}")
print(f"Gained    (Mut only, new in mut):  {len(ifp_d['gained'])}")
print(f"Lost      (WT only, lost in mut):  {len(ifp_d['lost'])}")

def _render_interaction_list(label: str, pairs: list) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(columns=["residue", "interaction_type", "status"])
    return pd.DataFrame([
        {"residue": r, "interaction_type": it, "status": label} for r, it in pairs
    ])

ifp_table = pd.concat([
    _render_interaction_list("preserved", ifp_d["preserved"][:10]),
    _render_interaction_list("gained",    ifp_d["gained"][:10]),
    _render_interaction_list("lost",      ifp_d["lost"][:10]),
], ignore_index=True)
ifp_table
"""),

        markdown("""
### Interpretation

Three readings to take from the diff:

- **Lost interactions naming the mutated residue.** Any `(residue 560, Hydrophobic)` or similar entry that appears in `lost` is the direct chemical translation of the I→S sidechain change - that interaction can't exist in the mutant because the atoms that made it are gone.
- **Gained interactions naming the mutated residue.** `(residue 560, HBAcceptor)` or `HBDonor` appearing in `gained` would indicate the new serine OG (when present in production runs) is reaching far enough into the pocket to make a hydrogen bond the WT couldn't. In stub-fixture mode the surgery removes I560 atoms without adding an OG, so this branch typically stays empty - real PDBFixer-based mutants would populate it when geometry allows.
- **Preserved interactions far from the mutated residue.** These are the binding contacts the variant *doesn't* affect - usually the bulk of the binding evidence. A short `preserved` list with a large `lost` list would be the worrying pattern (most contacts gone); a long `preserved` list with a small `lost` and `gained` set is the more common localised-impact pattern.

For a wet-lab reading: the `lost` list is the answer to "what bonds does the variant break that the WT compound was relying on?" If the chemist designed a hinge-binding inhibitor whose key hydrogen bond is on the lost list, that compound's potency against the variant is in doubt and a structural redesign is in order.
"""),

        markdown("""
## 8 - DPYD walkthrough: shortlist diff (with non-ERK2 consensus rule note)

### Background

The **shortlist** is what the pipeline hands to a wet-lab chemist: the small ordered set of compounds the consensus filter survived. Diffing the WT and mutant shortlists is the **wet-lab-handoff readout** of the variant's effect on prioritisation. Three categories:

- **In both shortlists** - the same compounds the chemist would have ordered against WT also survive against the mutant. The genotype doesn't reshape priorities. Rank delta tells you whether ordering shifts within the shared set.
- **WT-only** - compounds that survived against WT but didn't make the mutant shortlist. These are the compounds whose potency is most at risk from the variant.
- **Mutant-only** - compounds that didn't survive against WT but do survive against the mutant. These are the new candidates the variant *opens up* - useful when designing a variant-selective inhibitor, less useful when looking for a single drug that works against both genotypes.

**Non-ERK2 consensus rule** (per `_planning/PROJECT_PROPOSAL.md` Q4 decision for step 15). Notebook 06 trained a per-target rescorer on ERK2 activity labels for the original ERK2 walkthrough. For DPYD and the other six headline-demo targets, **no labelled compound set exists** to train a per-target rescorer on, so the rescorer column from notebook 04's pipeline is not available. The consensus rule therefore switches to **Boltz-2 affinity + gnina CNN-affinity** as the two independent lanes. Both have meaningful directional information for binding strength (Boltz-2 from co-folding; gnina-CNN from a separately-trained ML scoring model), and the two-lane consensus pattern is preserved without retraining. The `compute_consensus()` parameter `rescorer_col` accepts any column name; we point it at `gnina_cnn_affinity` and pass `rescorer_lower_is_better=False`. (The parameter name `rescorer_col` is a historical artefact from nb 06's ERK2-rescorer design; semantically it's just "the second lane that complements Boltz-2". A step-17 refactor may rename it when nb 99 lands.)

### Cell intro

Build WT and mutant shortlists by running `compute_consensus` on each genotype's scored_poses + boltz tables (with the non-ERK2 lane choice). Diff the two shortlists. Print classification table with rank deltas for compounds in both.
"""),

        code(title="DPYD shortlist diff: build per-genotype shortlists + diff", source="""
def _build_shortlist(genotype_dir: Path, top_fraction: float = 0.30) -> pd.DataFrame:
    scored = pd.read_parquet(genotype_dir / "scoring" / "scored_poses.parquet")
    boltz  = pd.read_csv(   genotype_dir / "boltz"   / "affinity.csv")
    result = compute_consensus(
        scored, boltz,
        top_fraction=top_fraction,
        rescorer_col="gnina_cnn_affinity",
        boltz_col="boltz_affinity",
        id_col="compound_id",
        rescorer_lower_is_better=False,
        boltz_lower_is_better=True,
        filter_method="rank_product_topk",
    )
    return result["shortlist"]

shortlist_wt  = _build_shortlist(WT_DIR)
shortlist_mut = _build_shortlist(MUT_DIR)
print(f"WT shortlist:  {len(shortlist_wt)} compounds")
print(f"Mut shortlist: {len(shortlist_mut)} compounds")

sl_d = shortlist_diff(shortlist_wt, shortlist_mut)
print(f"\\nClassification:")
print(f"  both:       {sl_d['n_both']}")
print(f"  WT only:    {sl_d['n_wt_only']}")
print(f"  Mut only:   {sl_d['n_mut_only']}")

print("\\nBoth (with rank deltas; rank_delta > 0 means dropped in mutant):")
print(sl_d["both"].to_string(index=False) if sl_d["n_both"] else "(none)")
print("\\nWT-only:")
print(sl_d["wt_only"].to_string(index=False) if sl_d["n_wt_only"] else "(none)")
print("\\nMutant-only:")
print(sl_d["mut_only"].to_string(index=False) if sl_d["n_mut_only"] else "(none)")
"""),

        markdown("""
### Interpretation

Three reading patterns:

- **Mostly the same compounds, small rank deltas** - the variant doesn't change priorities. A drug developed for WT should still bind the variant; clinical implication is "no dose adjustment needed on structural grounds." This is the common case for variants far from the binding pocket.
- **Compounds drop out of the mutant shortlist** - those compounds' potency is at risk against carriers of the variant. For pharmacogenes this is the clinically actionable signal (a compound that loses binding against a slow-acetylator NAT2 variant won't be turned over normally). For oncology drug-resistance work this signals which drugs the resistance mutation defeats.
- **New compounds appear in the mutant shortlist** - candidates for variant-selective therapy. Common in oncology resistance (KRAS G12C-selective inhibitors like sotorasib that bind the cysteine specifically introduced by the mutation).

For the DPYD I560S stub fixture: by design, the synthetic scores were perturbed so one analog (gimeracil) drops out in mutant and one decoy (diphenhydramine) promotes - the diff-surface verification, not a clinical claim. In a real DPYD WT-vs-I560S production run we would expect 5-FU and its close analogs to survive in **both** shortlists (the binding-site fit is robust for these tight pyrimidine-shaped substrates), with the chemistry-relevant question being whether **rank deltas** within the preserved set tilt toward one genotype - a softer "preferential prioritisation" rather than a hard "drops out."
"""),

        markdown("""
## 9 - DPYD walkthrough: HTML report

### Background

The HTML report is the **wet-lab handoff artefact**: one self-contained file per (WT, variant) pair at `data/derived/<target>/mutations/wt_vs_<variant>.html`. It includes the variant priors, the structural diff, the IFP diff, the shortlist diff, and an embedded interactive py3Dmol viewer of WT and mutant overlaid. Open it in any modern browser; forward it to a wet-lab collaborator; paste it into a clinical-collaboration meeting.

**Embedded py3Dmol vs static screenshots.** This v1 uses embedded py3Dmol so the reader can rotate / zoom / hide chains interactively. File size lands at 1 - 5 MB for a single (WT, variant) report; the embedded JavaScript runs in every browser released in the last decade. A static-screenshot export (smaller, no JS, slides-ready) is a useful future addition - tracked in the Recap as out-of-scope future work.

### Cell intro

Assemble the summary dict via the inline aggregator helper, render an embedded py3Dmol viewer of the WT-vs-mutant overlay, write the self-contained HTML, and preview it inline.
"""),

        code(title="DPYD HTML report assembly", source="""
# Render the WT-vs-mutant overlay viewer as embeddable HTML.
viewer = py3Dmol.view(width=720, height=480)
viewer.addModel(WT_PDB.read_text(),  format="pdb")
viewer.setStyle({"model": 0}, {"cartoon": {"color": "gold",      "opacity": 0.85}})
viewer.addModel(MUT_PDB.read_text(), format="pdb")
viewer.setStyle({"model": 1}, {"cartoon": {"color": "lightblue", "opacity": 0.85}})
viewer.setStyle({"model": 0, "resi": str(POSITION)}, {"stick": {"colorscheme": "yellowCarbon"}})
viewer.setStyle({"model": 1, "resi": str(POSITION)}, {"stick": {"colorscheme": "cyanCarbon"}})
viewer.zoomTo({"resi": str(POSITION)})

# py3Dmol's _make_html() returns the standalone HTML+JS chunk we can embed.
viewer_html = viewer._make_html()

summary = summarise_mutation(
    TARGET, VARIANT, UNIPROT, POSITION, WT_AA, MUT_AA,
    wt_dir=WT_DIR, mut_dir=MUT_DIR,
    wt_name=f"{TARGET}_wt",
    mut_name=f"{TARGET}_{VARIANT.lower()}",
    pocket_radius_a=8.0,
    alphamissense_cache=ALPHAMISSENSE_CACHE,
    gnomad_cache=GNOMAD_CACHE,
    rasp_cache=RASP_CACHE,
    viewer_html=viewer_html,
)

MUTATIONS_DIR.mkdir(parents=True, exist_ok=True)
html_path = MUTATIONS_DIR / f"wt_vs_{VARIANT.lower()}.html"
render_mutation_html(summary, html_path, embed_3d=True)
size_kb = html_path.stat().st_size / 1024
print(f"Wrote {pretty_path(html_path)}  ({size_kb:.1f} KB)")

# Inline preview (uncomment if you want to view here in the notebook):
# from IPython.display import IFrame
# IFrame(str(html_path.relative_to(REPO_ROOT)), width=900, height=600)
"""),

        markdown("""
### Interpretation

The HTML report is the artefact you forward to a collaborator. File-size pragmatics:

- **Around 1 - 2 MB per report** when py3Dmol is embedded (the file carries the WT + mutant PDB text and the embedded JavaScript viewer code). Opens in any modern browser; needs no server.
- **Around 100 - 300 KB without the 3D viewer** (`embed_3d=False`) - text + tables only. Useful for grant-figure-style static appendices.
- The browser-JavaScript dependency is intrinsic to interactive 3D in HTML; a script-free alternative would be PNG screenshots of the viewer captured at notebook-run time. Worth adding as a `viewer_as_image=True` toggle when grant materials need static figures.
"""),

        markdown("""
## 10 - Six follow-up template demos

### Background

The DPYD walkthrough above demonstrates the comparison template end-to-end. The same template applies to the other six headline demos (per `PROJECT_PROPOSAL.md` § 10): KRAS G12C, ESR1 Y537S, BRCA1 LoF C61G, CYP2D6\\*4 P34S, NAT2\\*5 I114T, UGT1A1\\*28.

For step-15 closure we exercise only the **template-coverage** dimension: for each demo, we resolve the nb 07 priors (cheap; uses only the already-warm caches) and confirm that the structural / IFP / shortlist diff branches would fire if pre-cached genotype trees were present. Real WT-vs-mutant pipeline runs for each demo are nb 99 production territory - that's where the full library compute lands.

One of the six demos is intentionally **out of scope** for the missense-shaped tools: **UGT1A1\\*28** is a TATA-box-promoter insertion (`*28 = (TA)7TAA` vs WT `(TA)6TAA`), not a missense variant. AlphaMissense and RaSP correctly return `None` for it. The template's purpose for that demo is to demonstrate the **honest out-of-scope branch**: clinicians and researchers seeing this section should know that promoter-variant analysis needs different methodology (transcriptional regulation modelling, mRNA expression analysis), not a missense-shaped tool returning a fabricated number.

### Cell intro

Loop over the six demos. For each, call the three nb 07 priors and print the result; check whether pre-cached genotype trees exist; print one short status line per demo. The UGT1A1\\*28 demo prints an explicit "out-of-scope for missense-shaped tools" note.
"""),

        code(title="Six follow-up demos: resolve priors + report tree status", source="""
FOLLOWUP_DEMOS = [
    {"target": "kras",   "variant": "G12C",    "uniprot": "P01116", "position": 12,
     "wt_aa": "G", "mut_aa": "C",
     "notes": "Oncogenic driver, covalent-cysteine site; sotorasib binds the variant cysteine."},
    {"target": "esr1",   "variant": "Y537S",   "uniprot": "P03372", "position": 537,
     "wt_aa": "Y", "mut_aa": "S",
     "notes": "Ligand-binding-domain GoF; tamoxifen / fulvestrant resistance in breast cancer."},
    {"target": "brca1",  "variant": "C61G",    "uniprot": "P38398", "position": 61,
     "wt_aa": "C", "mut_aa": "G",
     "notes": "RING-domain LoF (synthetic lethality with olaparib via PARP)."},
    {"target": "cyp2d6", "variant": "P34S",    "uniprot": "P10635", "position": 34,
     "wt_aa": "P", "mut_aa": "S",
     "notes": "Common reduced-function allele; tamoxifen activation pharmacogene."},
    {"target": "nat2",   "variant": "I114T",   "uniprot": "P11245", "position": 114,
     "wt_aa": "I", "mut_aa": "T",
     "notes": "Slow-acetylator marker (the *5 allele); carcinogen-detoxification PGx."},
    {"target": "ugt1a1", "variant": "TA7",     "uniprot": "P22309", "position": 0,
     "wt_aa": "-", "mut_aa": "-",
     "notes": "TATA-box-promoter insertion, NOT a missense; intentionally out-of-scope here."},
]


def _resolve_priors(demo: dict) -> dict:
    if demo["variant"] == "TA7":
        return {
            "am_prob": None, "am_class": None,
            "gn":     {"found": False, "allele_freq_overall": None,
                       "reference_flipped": False, "hgvsp": None},
            "ddg":    None, "rasp_covered": False,
            "out_of_scope_reason": (
                "UGT1A1*28 is a TATA-box-promoter insertion ((TA)7TAA vs WT (TA)6TAA), "
                "not a missense variant. AlphaMissense and RaSP are missense-only "
                "protein-level tools and return None. Promoter-variant analysis needs "
                "different methodology (transcriptional regulation modelling, mRNA "
                "expression analysis); not in scope for this notebook."
            ),
        }
    am_prob  = alphamissense_score(demo["uniprot"], demo["position"], demo["mut_aa"],
                                   cache_dir=ALPHAMISSENSE_CACHE)
    am_class = alphamissense_class(demo["uniprot"], demo["position"], demo["mut_aa"],
                                   cache_dir=ALPHAMISSENSE_CACHE)
    gn       = gnomad_frequency(demo["uniprot"], demo["position"], demo["mut_aa"],
                                wt_aa=demo["wt_aa"], cache_dir=GNOMAD_CACHE)
    ddg      = rasp_ddg(demo["uniprot"], demo["position"], demo["wt_aa"], demo["mut_aa"],
                        cache_dir=RASP_CACHE)
    rasp_cov = rasp_covers(demo["uniprot"], cache_dir=RASP_CACHE)
    return {
        "am_prob": am_prob, "am_class": am_class,
        "gn": gn, "ddg": ddg, "rasp_covered": rasp_cov,
        "out_of_scope_reason": None,
    }


followup_rows = []
print("=" * 88)
print(f"{'target':>8s}  {'variant':>8s}  {'AM_prob':>8s}  {'AM_class':>20s}  "
      f"{'gnomAD_AF':>10s}  {'RaSP_ddG':>10s}  trees")
print("-" * 88)
for demo in FOLLOWUP_DEMOS:
    priors = _resolve_priors(demo)
    wt_dir  = DERIVED_ROOT / f"{demo['target']}_wt"
    mut_dir = DERIVED_ROOT / f"{demo['target']}_{demo['variant'].lower()}"
    trees_ready = wt_dir.exists() and mut_dir.exists()
    tree_status = "cached -> ready" if trees_ready else "not cached (nb 99 runs)"

    am_prob_str  = f"{priors['am_prob']:.3f}" if isinstance(priors['am_prob'], (int, float)) else "-"
    am_class_str = str(priors['am_class'] or "-")
    af_str       = (f"{priors['gn']['allele_freq_overall']:.5f}"
                    if priors['gn']['found'] else "-")
    ddg_str      = f"{priors['ddg']:.3f}" if isinstance(priors['ddg'], (int, float)) else "-"
    print(f"{demo['target']:>8s}  {demo['variant']:>8s}  {am_prob_str:>8s}  {am_class_str:>20s}  "
          f"{af_str:>10s}  {ddg_str:>10s}  {tree_status}")
    if priors["out_of_scope_reason"]:
        print(f"    note: {priors['out_of_scope_reason']}")
    followup_rows.append({"demo": demo, "priors": priors, "trees_ready": trees_ready})
print("=" * 88)
"""),

        markdown("""
### Interpretation

For the missense demos (KRAS G12C, ESR1 Y537S, BRCA1 C61G, CYP2D6 P34S, NAT2 I114T) the expected pattern is the same as the DPYD I560S example above: AlphaMissense + gnomAD resolve cleanly; RaSP ΔΔG resolves once the full-proteome cache lands via the operator-step probe. Mechanistically the five span four distinct archetypes:

- **KRAS G12C** is the textbook **oncogenic-driver-not-stability** case: AlphaMissense pathogenic, gnomAD near-zero germline (it's a somatic driver), RaSP near zero (G12C does not destabilise the GTPase fold; the activation mechanism is altered nucleotide binding). Sotorasib's covalent-cysteine engagement is the structural angle.
- **ESR1 Y537S** is **ligand-pocket GoF / endocrine resistance**: AlphaMissense pathogenic, gnomAD rare, RaSP near zero or small (the variant stabilises the agonist-bound conformation by removing the H-bond that pinned the apo state). Tamoxifen resistance is the clinical readout.
- **BRCA1 C61G** is **DDR LoF**: pathogenic + rare + destabilising (a buried zinc-coordinating cysteine substitution wrecks the RING domain). Synthetic-lethality with olaparib is the clinical mechanism - olaparib binds PARP, not BRCA1; the variant creates the vulnerability.
- **CYP2D6 P34S** is **reduced-function PGx (Phase I oxidation)**: common in some populations, AlphaMissense ambiguous-to-pathogenic, RaSP modestly destabilising. Tamoxifen-activation impairment is the breast-cancer connection.
- **NAT2 I114T (\\*5)** is the **slow-acetylator PGx (Phase II acetylation)**: common, AlphaMissense ambiguous-to-benign (a known AlphaMissense weakness on functional-but-common variants), RaSP positive (destabilising). The combined three-prior reading correctly identifies the stability mechanism.
- **UGT1A1\\*28** is the **out-of-scope promoter variant**: priors are intentionally `None`. The template's job here is to teach the limit, not to fabricate a number.

For all six, the **structural / IFP / shortlist diffs** will run automatically when the pre-cached WT + variant trees land - same code path as the DPYD walkthrough, no new logic. The Recap points to nb 99's library mode as the production-runner home for those trees.
"""),

        markdown("""
## 11 - Step-15 DPYD WT-vs-I560S SUMMARY block

### Cell intro

Boxed summary mirroring nb 06's OPERATING-POINT SUMMARY and nb 07's CALIBRATION SUMMARY conventions: cohort header, per-diff-surface row, six-follow-up status row, file paths. The block is the copy-pasteable artefact for the step-15 closure commit message.
"""),

        code(title="DPYD WT-vs-I560S SUMMARY block (copy-pasteable for step-15 closure)", source="""
print("=" * 78)
print("STEP-15 DPYD WT-vs-I560S SUMMARY")
print("=" * 78)
print(f"Mode: {'STUB FIXTURE (USE_STUB_TREES=True)' if USE_STUB_TREES else 'REAL DPYD RUNS'}")
print(f"Target: DPYD (UniProt {UNIPROT})   Variant: I560S (*13 missense; *2A splice in sidebar)")
print(f"WT  tree:  {WT_DIR}")
print(f"Mut tree:  {MUT_DIR}")
print()
print("Per-variant priors (from notebook 07 helpers):")
print(f"  AlphaMissense: prob={am_prob}, class={am_label}")
print(f"  gnomAD:        af={gn['allele_freq_overall']}, found={gn['found']}, "
      f"flipped={gn['reference_flipped']}")
print(f"  RaSP ddG:      {ddg}   (covered={rasp_covered})")
print()
print("Structural diff:")
print(f"  Pocket-restricted Cα-RMSD ({summary['structural']['radius_a']:.0f} Å shell, "
      f"{summary['structural']['n_matched']} residues): "
      f"{summary['structural']['rmsd_pocket_A']:.3f} Å")
print(f"  Full-protein  Cα-RMSD: {summary['structural']['rmsd_full_A']:.3f} Å")
print()
print("IFP diff:")
print(f"  gained:    {len(ifp_d['gained']):>3d}")
print(f"  lost:      {len(ifp_d['lost']):>3d}")
print(f"  preserved: {len(ifp_d['preserved']):>3d}")
print()
print("Shortlist diff (Boltz-2 + gnina-CNN consensus; rank_product_topk @ TOP=0.30):")
print(f"  both:      {sl_d['n_both']:>3d}")
print(f"  WT only:   {sl_d['n_wt_only']:>3d}")
print(f"  Mut only:  {sl_d['n_mut_only']:>3d}")
print()
print(f"HTML report: {html_path}  ({html_path.stat().st_size / 1024:.1f} KB)")
print()
print("Six follow-up demos (priors-only template coverage):")
for row in followup_rows:
    demo = row["demo"]; priors = row["priors"]
    if priors["out_of_scope_reason"]:
        status = "OUT OF SCOPE (promoter variant)"
    else:
        ok = (priors["am_prob"] is not None) and priors["gn"]["found"]
        status = "priors resolved" if ok else "priors partial"
    print(f"  {demo['target']:>8s} {demo['variant']:>8s}: {status}")
print("=" * 78)
"""),

        markdown("""
## Recap

### Biomedical takeaway

For any human missense variant in one of the seven demo-set genes (DPYD, NAT2, CYP2D6, UGT1A1, KRAS, BRCA1, ESR1), this notebook now produces a **WT-vs-variant comparison report** that combines three independent computational priors (AlphaMissense pathogenicity, gnomAD allele frequency, RaSP ΔΔG stability) with three structural / chemical diff surfaces (pocket-restricted Cα-RMSD, IFP diff, shortlist diff). The DPYD I560S walkthrough exercised every surface end-to-end against a synthetic stub fixture; the six follow-up demos confirmed the same template applies across the four mechanistic archetypes (pharmacogene LoF, oncogenic driver, ligand-pocket GoF, DDR synthetic-lethality LoF) plus the honest out-of-scope branch for non-missense variants. The HTML report is the wet-lab-handoff artefact: one self-contained file per (WT, variant) pair, embedded interactive 3D, suitable for forwarding to a wet-lab collaborator or pasting into a clinical-collaboration meeting.

### Technical takeaway

The notebook contains **five inline comparison helpers** (`pocket_residues_within`, `ca_rmsd_pocket`, `ifp_diff`, `shortlist_diff`, `summarise_mutation` + the HTML renderer) defined in section 4. Step 16 (notebook 09) is when the reuse-mechanism decision opens (verbatim copy-paste into nb 09, extract to a shared `src/aidd/mutation.py` module, or builder-side source-of-truth verbatim-inline mechanism); step 15's mandate is to write the helpers cleanly and let their shape settle before that decision lands.

The consensus rule **switches** for non-ERK2 targets: instead of the (rescorer, Boltz-2) lanes that nb 06 trained on ERK2 labels, nb 08 uses (Boltz-2, gnina-CNN-affinity) as the two independent lanes via `compute_consensus(rescorer_col="gnina_cnn_affinity", rescorer_lower_is_better=False, ...)`. The parameter-name `rescorer_col` is a historical artefact from nb 06's design; a step-17 refactor when nb 99 lands may rename to `lane2_col` or `companion_col`.

The **stub fixture** in section 3 is step-15 scope only: when real DPYD WT and I560S production runs land at nb 99, flip `USE_STUB_TREES = False` and section 2's verification will assert against those real outputs.

### What's next in the pipeline

- **`09_moa_small_n.ipynb`** (step 16) - small-N mechanism-of-action driver: same diff surfaces at per-compound × per-variant scope, with one HTML report per (compound, variant) pair. Reuses the helpers from section 4 of this notebook (verbatim or via a shared `src/aidd/mutation.py` module - decided at step 16).
- **`99_screen_library.ipynb`** (step 17) - production runner with `RUN_MODE = library | moa` toggle. In `library` mode with `mutations=[...]` it runs the full WT + variant pipeline for each variant and invokes nb 08's logic to produce one HTML report per variant. This is where the real DPYD WT-vs-I560S production runs land - and where the six follow-up demos' actual WT-vs-mutant trees get computed.
- **Out of scope for v1**: atom-level dynamics (MD simulation), catalysis-rate prediction (QM/MM, EnzyHTP-style methods), promoter-variant analysis (transcriptional regulation modelling), splice-variant fold generation (would extend nb 01 to handle truncated proteins). Mentioned in the further-reading pointers below.

### Further reading

- **Variant priors design (combining three signals).** Cheng, J. *et al.* "Accurate proteome-wide missense variant effect prediction with AlphaMissense." *Science* **381**, eadg7492 (2023). [doi:10.1126/science.adg7492](https://doi.org/10.1126/science.adg7492); Blaabjerg, L.M. *et al.* "Rapid protein stability prediction using deep learning representations." *eLife* **12**, e82593 (2023). [doi:10.7554/eLife.82593](https://doi.org/10.7554/eLife.82593).
- **Consensus virtual screening on uncorrelated lanes.** Wang, R. & Wang, S. "How does consensus scoring work for virtual library screening?" *J. Chem. Inf. Comput. Sci.* **41**, 1422 (2001). [doi:10.1021/ci010025x](https://doi.org/10.1021/ci010025x); Houston, D.R. & Walkinshaw, M.D. "Consensus docking: improving the reliability of docking in a virtual screening context." *J. Chem. Inf. Model.* **53**, 384 (2013). [doi:10.1021/ci300399w](https://doi.org/10.1021/ci300399w).
- **DPYD clinical pharmacogenomics (the deep-walkthrough background).** Amstutz, U. *et al.* "Clinical Pharmacogenetics Implementation Consortium (CPIC) Guideline for Dihydropyrimidine Dehydrogenase Genotype and Fluoropyrimidine Dosing: 2017 Update." *Clin. Pharmacol. Ther.* **103**, 210 (2018). [doi:10.1002/cpt.911](https://doi.org/10.1002/cpt.911).
- **Catalysis-rate prediction (out of scope here, for context).** Yan, B. *et al.* "EnzyHTP: A high-throughput computational platform for enzyme modeling." *J. Chem. Inf. Model.* (2022) - for readers wanting to know what the right tool looks like for k_cat / K_m prediction this notebook explicitly does not attempt.
"""),
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
