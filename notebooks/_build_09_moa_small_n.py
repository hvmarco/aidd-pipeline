"""Builder for 09_moa_small_n.ipynb.

Source of truth for the small-N mechanism-of-action driver. Per-compound
counterpart to notebook 08's library-side WT-vs-variant comparison.

Cells appear below in narrative order. Never edit the .ipynb directly --
see ``CLAUDE.md`` section *Notebook workflow*.

Regenerate:
    python notebooks/_build_09_moa_small_n.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import AUTORELOAD_SNIPPET, code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "09_moa_small_n.ipynb"


def build() -> None:
    nb = notebook(
        markdown("""
# 09 - Small-N mechanism-of-action (MoA) driver

**aidd-pipeline - Notebook 9 of the screening workflow**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hvmarco/aidd-pipeline/blob/main/notebooks/09_moa_small_n.ipynb)

This notebook is the **per-compound** counterpart to notebook 08's library-side WT-vs-variant comparison. Where notebook 08 takes a target + a candidate library + a variant and emits a ranked shortlist diff, this notebook takes a target + a handful of compounds (N <= 10) + optional variant list and emits **one HTML report per (compound x variant) pair** plus a **summary CSV**.

The use case is single-compound clinical questions:

- **"Does this drug bind this protein?"** Mechanism-of-action investigation when the MoA is unknown.
- **"Is binding altered by this inherited variant?"** Pharmacogenomic question (NAT2 slow acetylators, CYP2D6 poor metabolisers, DPYD\\*2A, UGT1A1\\*28, KRAS G12C).

Five demos cover Natallia's actual research focus - colorectal-cancer pharmacogenomics + cancer driver mutations - and span the full handling matrix:

1. **NAT2 \\*5 / \\*6 / \\*7 + isoniazid** *(deep walkthrough)* - missense, `mutate_in_place`. Slow-acetylator phenotype is the textbook NAT2 story; the structural argument generalises to aromatic-amine carcinogens for the CRC risk angle.
2. **CYP2D6 \\*4 / \\*10 + tamoxifen** - one splice (`*4`, `out_of_scope_splice`) plus one missense (`*10` P34S, `mutate_in_place`). Pair demonstrates both handling branches within one target.
3. **DPYD\\*2A + 5-fluorouracil** - splice donor variant; `out_of_scope_splice`. Library-side deep walkthrough for DPYD (using I560S as the structural probe) lives in notebook 08.
4. **Irinotecan + UGT1A1\\*28** - TATA-box promoter variant; `out_of_scope_promoter`. FDA-label genotype-guided dose.
5. **Sotorasib + KRAS G12C** - oncogenic driver, covalent inhibitor at the active site; `mutate_in_place`.

## What this notebook does and does NOT predict

For the enzyme demos (NAT2, CYP2D6, DPYD, UGT1A1 - four of the five), the pipeline measures **whether the substrate fits the active site**, not the **catalytic rate** (k_cat, K_m, turnover). For variant-vs-WT comparisons on enzymes, the questions the pipeline can answer are: *"Does the substrate still fit in the variant active site, and how does binding-site geometry / IFP differ from WT? Is the variant predicted to destabilise the fold (RaSP DDG)? Is it evolutionarily flagged (AlphaMissense)? Is it common in the population (gnomAD)?"* The pipeline **cannot** answer: *"How much slower is catalysis in the variant?"* and definitely **cannot** answer: *"Will this patient have toxicity from drug X?"*

**Notebook 07's RaSP DDG meaningfully strengthens the slow/rapid story for enzyme variants** - for many slow-acetylator and reduced-function alleles, protein-stability change (and consequent reduced cellular abundance) is the dominant mechanism behind the kinetic phenotype, and DDG captures it directly. Combined with the binding-pose / IFP-diff signal from notebooks 08 and 09 and the AlphaMissense pathogenicity prior, the structural mechanism story is much more complete than binding-pose alone. Still circumstantial for catalytic rate - k_cat prediction needs QM/MM (out of scope; see Recap for pointers).

This caveat appears in three places in this notebook (per `_planning/MECHANISM_OF_ACTION_SCOPE.md`): here in the title cell, once around the enzyme demos, and at the bottom of every HTML report.

## Learning objectives

After running this notebook you will be able to:

- Explain in plain language why a **per-compound HTML report** is the canonical wet-lab handoff artefact for small-N MoA / pharmacogenomic questions, and how it differs from the library-side ranked shortlist (notebook 08).
- Read a **per-(compound x variant) report** as a clinician would: variant context (AlphaMissense + gnomAD + RaSP), single-compound binding pose, IFP diff against WT (when applicable), bottom-line caveat.
- Distinguish the four `variant_handling` branches: `wt` (baseline), `mutate_in_place` (missense via PDBFixer applyMutations on WT crystal), `out_of_scope_splice` (splice variant; structural pipeline cannot model), `out_of_scope_promoter` (promoter variant; no protein-sequence change).
- Generate a summary CSV across multiple (compound x variant) pairs that a wet-lab collaborator can read in 60 seconds.
- Recognise that for enzyme demos the pipeline measures binding fit + protein stability, not catalytic rate.

## Audience

You'll get value if you are:

- A clinician or wet-lab biologist who wants the per-compound binding-site readout for a single drug and a known patient variant.
- A pharmacogenomics researcher triaging which compounds work in which genotypes for a pre-clinical study design.
- A medicinal chemist asked "does compound X still bind in the *Y* variant?" - the (compound x variant) HTML report is the operative answer.
- A reviewer / grant auditor: every report carries provenance back to notebook 07 (variant priors) + notebook 08 (comparison helpers via `aidd.mutation`); methodology limits are explicit in three places.

## Prerequisites

- The conda environment `aidd` (`conda env create -f environment.yml`), or running on Google Colab (the setup cell installs everything including `pdbfixer`).
- **Notebook 07** completed at least once for the variants of interest (AlphaMissense + gnomAD + RaSP caches warm under `data/cache/`).
- For the **deep walkthrough**: this notebook can run end-to-end against **synthetic stub fixtures** generated in section 3 (step-16 mode; `USE_STUB_TREES = True`). Real per-(compound x variant) Boltz-2 + gnina runs land at notebook 99's production runner under `RUN_MODE = "moa"` (step 17).
- For the **four brief follow-ups**: only the nb 07 priors are exercised end-to-end at step-16 closure for the out-of-scope branches (DPYD\\*2A, UGT1A1\\*28, CYP2D6\\*4); the missense follow-ups (CYP2D6 \\*10 P34S, KRAS G12C) run the structural + IFP diff path via the same stub-fixture mechanism as NAT2.

## Runtime

- **CPU-only at step-16 closure.** No GPU dependency in stub mode. Free Colab CPU runtime, free Colab T4, Windows / macOS local - all work identically.
- **First run on a fresh Colab T4: 5 - 15 minutes.** Dominated by the one-time PDB downloads (NAT2 2PFR, CYP2D6 3QM4, KRAS 6OIM, ~0.5 MB each) plus PDBFixer mutations for the 6 missense variants (NAT2 \\*5/\\*6/\\*7, CYP2D6 \\*10, KRAS G12C) plus IFP computation across 9 (variant, compound) pose pairs.
- **Cached re-runs: under 60 seconds.** The stub trees, the variant-prior caches, and the structural files all live on Drive (Colab) or in the repo's `data/cache/` + `data/derived/` (local).
- **Production runs at nb 99 (step 17): real Boltz-2 single-prediction per (compound, variant) pair needs Colab GPU.** Order ~2 minutes per (compound, variant) on an A100; the 9 small-N cases here would be ~20 minutes of GPU time, not the multi-hour library scale.
"""),

        markdown("""
## Key terms used in this notebook

| Term | Meaning in one line |
|---|---|
| **Mechanism of action (MoA)** | The biochemical mechanism by which a drug exerts its effect; in this notebook, narrowly the question "does drug X bind protein Y, and how does inherited variation alter that binding?" |
| **Small-N** | A handful of compounds (N <= 10) rather than a library (1k-10k). The right scope for clinical "this patient, this drug, this variant" questions. |
| **Per-(compound x variant) report** | One HTML report per pair of (compound, variant). The canonical wet-lab handover artefact from this notebook. |
| **Pharmacogenomic variant** | An inherited DNA change that affects how a patient responds to a drug. Categories include drug-metabolism enzyme variants (DPYD, NAT2, CYP2D6, UGT1A1) and drug-target variants (KRAS G12C and similar). |
| **Slow acetylator** | A patient genotype of NAT2 (the *5, *6, *7 alleles in our walkthrough) where N-acetylation of aromatic amines is slow, producing a metabolic phenotype with both pharmacological and CRC-risk consequences. |
| **NAT2 \\*5 / \\*6 / \\*7** | The three most common clinical slow-acetylator alleles. \\*5 = I114T (rs1801280), \\*6 = R197Q (rs1799930), \\*7 = G286E (rs1799931). UniProt P11245. |
| **CYP2D6 \\*4 / \\*10** | Two common reduced-function alleles. \\*4 (rs3892097) is a splice variant; \\*10 (P34S, rs1065852) is a missense. UniProt P10635. |
| **DPYD\\*2A** | The canonical clinical DPYD pharmacogene allele (`IVS14+1G>A`, splice donor variant). Causes severe 5-FU toxicity. UniProt Q12882. |
| **UGT1A1\\*28** | TATA-box promoter variant (rs8175347; `(TA)7TAA` repeat instead of `(TA)6TAA`). FDA label adjusts irinotecan dose by genotype. UniProt P22309. |
| **KRAS G12C** | Single missense at codon 12; oncogenic gain-of-function driver in metastatic CRC and NSCLC. Sotorasib binds covalently at the G12C cysteine in the GDP-bound state. UniProt P01116. |
| **`variant_handling`** | A label declared per variant that selects which branch of this notebook's pipeline applies: `wt` (no variant change, baseline), `mutate_in_place` (PDBFixer applyMutations on WT crystal), `fold_variant` (notebook 01 fold of the variant sequence; not exercised in step 16), `out_of_scope_splice` (splice variant; protein-structural pipeline cannot model directly), `out_of_scope_promoter` (promoter variant; no protein-sequence change). |
| **`name_to_smiles`** | Helper in `src/aidd/inputs.py` that resolves a compound name (brand / generic / synonym) to its canonical SMILES via PubChem PUG REST. Warns when stereo features are present. |
| **`fetch_pdb` / `prep_receptor`** | Helpers in `src/aidd/inputs.py` for receptor preparation: download a PDB by ID and clean it (waters, alt-confs, cofactor whitelist, missing residues / atoms). `RECEPTOR_PREPS` dict ships sensible cofactor defaults for the five demo targets. |
| **Embedded py3Dmol** | A self-contained HTML+JavaScript chunk that renders an interactive protein-ligand 3D viewer in any modern browser. Each HTML report embeds one for the (compound x variant) pose. |
| **Catalysis vs binding** | The pipeline measures binding-site fit, not catalytic rate. For enzyme variants, RaSP DDG captures the protein-stability dimension; reaction-rate prediction needs QM/MM (out of scope). |
"""),

        markdown("""
## Why small-N differs from library screening

The library-screening flow (notebooks 00-06) ranks 1k-10k compounds against one target and emits the top ~5%. Consensus rank across two scoring lanes (Boltz-2 affinity + a second lane: gnina CNN-affinity for non-ERK2 demos) is the operative summary; the shortlist diff (notebook 08) compares two genotypes' shortlists.

At N <= 10 the consensus rank is **meaningless** - ranking 5 compounds tells you nothing useful, and `shortlist_diff` produces trivial output (`both = {all 5}`, `wt_only = {}`, `mut_only = {}`). The clinically interesting comparison shifts to **per-compound**: how does the binding pose for this one drug change between WT and variant? Which interactions are gained / lost? Are the nb 07 priors flagging the variant as pathogenic / common / destabilising?

The output therefore shifts too:

| Surface | Notebook 08 (library) | Notebook 09 (small-N MoA, this notebook) |
|---|---|---|
| Input | target + 1k-10k SMILES + variant | target + N <= 10 SMILES + variant(s) |
| Output | ranked shortlist diff (`shortlist.sdf` + `shortlist.csv` + HTML report per variant) | one HTML report per (compound x variant) pair + summary CSV |
| Consensus filter | rank_product_topk over many compounds | skipped (meaningless at N <= 10) |
| Variant priors (nb 07) | added columns on the shortlist | priors table inside each HTML report |
| Comparison helpers | inline in nb 08 (was; extracted in step 16 commit 2/4) | imported from `aidd.mutation` |
"""),

        markdown("""
## 1 - Setup

### What this section does

Detect Colab vs local, install dependencies that are not in Colab's default image (rdkit, py3Dmol, prolif, biopython, pyyaml, pdbfixer), set up the import path for `src/aidd/`, and mount Google Drive on Colab so the PDB cache (`data/cache/rcsb/`) and the AlphaMissense / RaSP / gnomAD caches from notebook 07 survive runtime restarts.
"""),

        code(title="Setup: detect Colab vs local, configure paths", source="""
import sys
import importlib
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    # CPU-only deps. pandas / numpy / matplotlib / pyarrow / requests / pyyaml
    # are in Colab's default image; rdkit / py3Dmol / prolif / biopython /
    # pdbfixer are not.
    !pip install -q rdkit py3Dmol prolif biopython pdbfixer
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

import json
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
    render_moa_html,
)
from aidd.inputs import (
    RECEPTOR_PREPS,
    fetch_pdb,
    name_to_smiles,
    prep_receptor,
)

print("imports ok")
"""),

        markdown("""
### Google Drive - read this before running the next cell

The variant-prior caches built by notebook 07 (`alphamissense/demo_set.parquet`, `gnomad/<gene>_gnomad_r4.json`, `rasp/full_proteome_demo_set.parquet` once the operator probe lands) live on **Google Drive** by default when you ran notebook 07 on Colab. This notebook's PDB cache (`data/cache/rcsb/`) and the stub trees under `data/derived/<target>_<variant>/` also live there.

**To opt out**, set `USE_DRIVE = False` in the cell *before* running it. The notebook will then look for caches and derived data under the local repo's `data/cache/` + `data/derived/` trees - which works if you copied them down from Drive, or if you ran every upstream notebook locally.
"""),

        code(title="Drive persistence toggle  (change to False to opt out)", source="""
# ============================================================================
# GOOGLE DRIVE AUTHORIZATION
# On Colab, the next call will prompt for Drive permission on first use.
#
# ->  To OPT OUT of Drive, change the line below to:    USE_DRIVE = False
#     (caches and derived data then live under the local data/ tree only)
# ============================================================================
USE_DRIVE = IS_COLAB

DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)
CACHE_ROOT          = DATA_ROOT.parent / "cache"
ALPHAMISSENSE_CACHE = CACHE_ROOT / "alphamissense"
RASP_CACHE          = CACHE_ROOT / "rasp"
GNOMAD_CACHE        = CACHE_ROOT / "gnomad"
RCSB_CACHE          = CACHE_ROOT / "rcsb"
# aidd.io.mount_drive_if_colab already returns data/derived/ (per its docstring),
# so DERIVED_ROOT is an alias for DATA_ROOT, NOT DATA_ROOT / "derived" (that
# double-nested on the first Colab run for nb 08 and produced data/derived/derived/...).
DERIVED_ROOT        = DATA_ROOT
MOA_REPORTS_ROOT    = DERIVED_ROOT / "moa_reports"
MOA_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

print(f"USE_DRIVE: {USE_DRIVE}  "
      f"({'Drive-backed' if USE_DRIVE else 'local /content (or local repo)'})")
print(f"Cache root:        {CACHE_ROOT}")
print(f"Derived root:      {DERIVED_ROOT}")
print(f"MoA reports root:  {MOA_REPORTS_ROOT}")
"""),

        markdown("""
## 2 - Demo declarations

### What this section does

Defines the five demos as a single `DEMOS` list - one dict per (target, compound) pair, with a `variants` list inside each. The `variant_handling` key on each variant selects the pipeline branch: `wt` for the WT baseline (no variant change), `mutate_in_place` for missense variants (PDBFixer applyMutations on WT crystal), `out_of_scope_splice` for splice variants, `out_of_scope_promoter` for promoter variants.

The compound SMILES are looked up live from PubChem via `name_to_smiles` in section 5 / 6, falling back to the hardcoded value in this cell if PubChem is unreachable. Pre-hardcoding the SMILES alongside the name keeps the notebook reproducible across PubChem outages and lets you run the demo offline if you've cached the PDB downloads.
"""),

        code(title="DEMOS = the five (target, variant set, compound) declarations", source="""
DEMOS = [
    # ---------------------------------------------------------------- 1: deep walkthrough
    {
        "target_name":         "NAT2",
        "uniprot":             "P11245",
        "pdb_id":              "2PFR",
        "compound":            {"name": "isoniazid",
                                "smiles_fallback": "C1=CN=CC=C1C(=O)NN"},
        "anchor_residue_hint": 68,   # NAT2 catalytic cysteine; pocket reference for WT report
        "variants": [
            {"label": "WT",  "handling": "wt",
             "position": None, "wt_aa": None, "mut_aa": None, "rsid": None},
            {"label": "*5",  "handling": "mutate_in_place",
             "position": 114,  "wt_aa": "I",  "mut_aa": "T", "rsid": "rs1801280"},
            {"label": "*6",  "handling": "mutate_in_place",
             "position": 197,  "wt_aa": "R",  "mut_aa": "Q", "rsid": "rs1799930"},
            {"label": "*7",  "handling": "mutate_in_place",
             "position": 286,  "wt_aa": "G",  "mut_aa": "E", "rsid": "rs1799931"},
        ],
        "is_deep_walkthrough": True,
    },

    # ---------------------------------------------------------------- 2: CYP2D6 + tamoxifen
    {
        "target_name":         "CYP2D6",
        "uniprot":             "P10635",
        "pdb_id":              "3QM4",
        "compound":            {"name": "tamoxifen",
                                "smiles_fallback":
                                "CC/C(=C(\\\\C1=CC=CC=C1)/C2=CC=C(C=C2)OCCN(C)C)/C3=CC=CC=C3"},
        "anchor_residue_hint": 308,    # CYP2D6 active-site I-helix anchor
        "variants": [
            {"label": "*4",  "handling": "out_of_scope_splice",
             "position": None, "wt_aa": None, "mut_aa": None, "rsid": "rs3892097",
             "out_of_scope_reason":
                 ("CYP2D6*4 is a splice-acceptor variant (rs3892097, 1846G>A) "
                  "that produces a non-functional protein via aberrant splicing. "
                  "AlphaMissense and RaSP are missense-only protein-level tools "
                  "and cannot score it directly; gnomAD lists it by chromosomal "
                  "position. The structural pipeline cannot model the truncated "
                  "protein.")},
            {"label": "*10", "handling": "mutate_in_place",
             "position": 34,   "wt_aa": "P", "mut_aa": "S", "rsid": "rs1065852"},
        ],
    },

    # ---------------------------------------------------------------- 3: DPYD + 5-FU
    {
        "target_name":         "DPYD",
        "uniprot":             "Q12882",
        "pdb_id":              "1H7W",
        "compound":            {"name": "5-fluorouracil",
                                "smiles_fallback": "C1=C(C(=O)NC(=O)N1)F"},
        "anchor_residue_hint": 560,
        "variants": [
            {"label": "*2A", "handling": "out_of_scope_splice",
             "position": None, "wt_aa": None, "mut_aa": None, "rsid": "rs3918290",
             "out_of_scope_reason":
                 ("DPYD*2A is a splice-donor variant (IVS14+1G>A, rs3918290) "
                  "that produces a truncated protein with the catalytic domain "
                  "missing. AlphaMissense and RaSP are missense-only and cannot "
                  "score it; gnomAD lists it by chromosomal position. The "
                  "library-side deep walkthrough in notebook 08 uses DPYD I560S "
                  "(*13, missense) as the structural probe for the same gene.")},
        ],
    },

    # ---------------------------------------------------------------- 4: UGT1A1 + irinotecan
    {
        "target_name":         "UGT1A1",
        "uniprot":             "P22309",
        "pdb_id":              None,        # no human crystal; AF model path
        "compound":            {"name": "irinotecan",
                                "smiles_fallback":
                                "CCC1=C2CN3C(=CC4=C(C3=O)COC(=O)C4(CC)O)C2=NC5=C1C=C(C=C5)"
                                "OC(=O)N6CCC(CC6)N7CCCCC7"},
        "anchor_residue_hint": None,
        "variants": [
            {"label": "*28", "handling": "out_of_scope_promoter",
             "position": None, "wt_aa": None, "mut_aa": None, "rsid": "rs8175347",
             "out_of_scope_reason":
                 ("UGT1A1*28 is a TATA-box promoter variant ((TA)7TAA instead of "
                  "(TA)6TAA, rs8175347). It does NOT change the protein sequence "
                  "- the reduced enzyme activity is driven by reduced "
                  "transcription, not by altered protein structure. AlphaMissense "
                  "and RaSP are not relevant; gnomAD lists it by chromosomal "
                  "position. UGT1A1 also has no human crystal structure, so even "
                  "for missense variants in this gene the pipeline would need an "
                  "AlphaFold model via notebook 01.")},
        ],
    },

    # ---------------------------------------------------------------- 5: KRAS + sotorasib
    {
        "target_name":         "KRAS",
        "uniprot":             "P01116",
        # WT K-RAS + GDP. Do NOT use 6OIM here -- 6OIM is the famous sotorasib+G12C
        # co-crystal, i.e. the VARIANT complex (residue 12 in 6OIM is already CYS),
        # so PDBFixer's applyMutations(\"GLY-12-CYS\") raises a residue-name mismatch.
        # 4OBE is the canonical WT reference; G12C is produced from it via the same
        # applyMutations call the other missense demos use.
        "pdb_id":              "4OBE",
        "compound":            {"name": "sotorasib",
                                "smiles_fallback":
                                "C[C@H]1CN(CCN1C2=NC(=O)N(C3=NC(=C(C=C32)F)C4=C(C=CC=C4F)O)"
                                "C5=C(C=CN=C5C(C)C)C)C(=O)C=C"},
        "anchor_residue_hint": 12,
        "variants": [
            {"label": "G12C", "handling": "mutate_in_place",
             "position": 12,  "wt_aa": "G", "mut_aa": "C", "rsid": "rs121913530"},
        ],
    },
]

print(f"{len(DEMOS)} demos declared:")
for d in DEMOS:
    n_var = len(d["variants"])
    star = " (deep walkthrough)" if d.get("is_deep_walkthrough") else ""
    print(f"  - {d['target_name']:<8s} + {d['compound']['name']:<20s}: {n_var} variant(s){star}")
"""),

        markdown("""
## 3 - Stub fixtures (step-16 mode)

### Why this section exists

Step 16's mandate is **methodology verification, not novel empirical findings** - same as step 15 was for notebook 08. Running real per-(compound, variant) Boltz-2 + gnina predictions for the nine demo cases is GPU compute that belongs to notebook 99's production runner under `RUN_MODE = "moa"`. To exercise every diff surface in this notebook (priors lookup, PDBFixer mutation, IFP computation, structural diff, HTML rendering) without that GPU compute, we ship **synthetic stub scores** alongside real protein-prep + real RDKit pose embedding.

**What the stub fixture is honest about:**

- The protein structures: real crystal PDBs from RCSB (downloaded via `fetch_pdb`); real prep via `prep_receptor` with the target-specific cofactor preset from `RECEPTOR_PREPS`; real variants applied via PDBFixer's `applyMutations` (the production approach nb 99 will use).
- The ligands: real SMILES (looked up via `name_to_smiles` from PubChem, falling back to the hardcoded SMILES in the DEMOS cell on network failure). RDKit ETKDG embedding + MMFF minimisation; the conformer is translated to sit near the variant residue's Cα (or the anchor residue for WT records).
- The Boltz-2 affinity / gnina CNN-affinity values: **synthetic, designed to exercise the report rendering**. We do not predict real binding here. The nb 99 production runner (step 17) flips `USE_STUB_TREES = False` and computes real predictions.
- The variant priors: **fully real** - looked up from notebook 07's caches (AlphaMissense + gnomAD; RaSP where covered). No stubs in this section.

The stub-generator cell is **idempotent**: when a per-(target, variant) tree already exists on disk, it skips the rebuild. Re-running the notebook is cheap.
"""),

        code(title="Build (target, variant, compound) stub fixtures (idempotent)", source="""
# ============================================================================
# STEP-16 STUB FIXTURE TOGGLE
# Set USE_STUB_TREES = False when real per-(compound, variant) predictions
# exist under DERIVED_ROOT / <target>_<variant> / ... (step 17, nb 99).
# ============================================================================
USE_STUB_TREES = True


def _resolve_smiles(compound_decl: dict) -> str:
    \"\"\"Try PubChem first; fall back to the hardcoded SMILES on failure.\"\"\"
    try:
        smiles = name_to_smiles(compound_decl["name"])
        if smiles:
            return smiles
    except Exception as exc:
        print(f"  name_to_smiles({compound_decl['name']!r}) failed: {exc}; "
              f"falling back to hardcoded SMILES")
    return compound_decl["smiles_fallback"]


def _embed_ligand_at(smiles: str, anchor_xyz: np.ndarray, seed: int) -> Chem.Mol:
    \"\"\"RDKit ETKDG embed + MMFF minimise; translate centroid to anchor_xyz.\"\"\"
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
        pass  # MMFF parameter gap on uncommon atoms; positions still embedded.
    conf = mol.GetConformer()
    coords = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z]
                       for i in range(mol.GetNumAtoms())])
    coords -= coords.mean(axis=0)
    coords += anchor_xyz
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, coords[i].tolist())
    return mol


def _ca_coord_at(pdb_path: Path, position: int, chain: str = "A") -> np.ndarray:
    \"\"\"Return the Calpha xyz of a given residue number.\"\"\"
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


def _apply_mutation_pdbfixer(wt_pdb: Path, position: int, wt_aa: str, mut_aa: str,
                              out_pdb: Path, chain: str = "A") -> Path:
    \"\"\"PDBFixer applyMutations: single-residue missense.

    Used in step 16 stub mode for missense variants where the WT crystal
    captures the binding-site geometry well and the mutation is local
    (NAT2 *5/*6/*7, CYP2D6 *10, KRAS G12C). For variants where the
    mutation might cause global rearrangement, use the fold_variant branch
    (notebook 01 fold of the variant sequence) instead.
    \"\"\"
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    AA_1TO3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
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


def _genotype_dir(target_name: str, variant_label: str) -> Path:
    safe = variant_label.replace("*", "").replace("/", "_").lower() or "wt"
    return DERIVED_ROOT / f"{target_name.lower()}_{safe}"


# Synthetic per-genotype score deltas. WT values are the baselines; each missense
# variant moves both numbers in the direction the slow-acetylator / reduced-function
# story predicts (slightly weaker for the substrate, much weaker for tamoxifen
# in CYP2D6 *10). KRAS G12C is the opposite: sotorasib is DESIGNED to bind the
# variant covalently, so its predicted affinity is much stronger for G12C than WT.
SYNTHETIC_SCORES: dict[tuple[str, str], dict] = {
    # (target, variant_label) -> {boltz_affinity, boltz_affinity_probability,
    #                              gnina_cnn_affinity, posebusters_pass}
    ("NAT2",   "WT"):    {"boltz_affinity": -1.85, "boltz_affinity_probability": 0.66, "gnina_cnn_affinity": 5.20, "posebusters_pass": True},
    ("NAT2",   "*5"):    {"boltz_affinity": -1.55, "boltz_affinity_probability": 0.54, "gnina_cnn_affinity": 4.85, "posebusters_pass": True},
    ("NAT2",   "*6"):    {"boltz_affinity": -1.60, "boltz_affinity_probability": 0.55, "gnina_cnn_affinity": 4.90, "posebusters_pass": True},
    ("NAT2",   "*7"):    {"boltz_affinity": -1.50, "boltz_affinity_probability": 0.52, "gnina_cnn_affinity": 4.75, "posebusters_pass": True},
    ("CYP2D6", "WT"):    {"boltz_affinity": -2.10, "boltz_affinity_probability": 0.75, "gnina_cnn_affinity": 6.40, "posebusters_pass": True},
    ("CYP2D6", "*10"):   {"boltz_affinity": -1.40, "boltz_affinity_probability": 0.48, "gnina_cnn_affinity": 5.10, "posebusters_pass": True},
    ("KRAS",   "WT"):    {"boltz_affinity": -0.50, "boltz_affinity_probability": 0.32, "gnina_cnn_affinity": 3.20, "posebusters_pass": True},
    ("KRAS",   "G12C"):  {"boltz_affinity": -3.20, "boltz_affinity_probability": 0.88, "gnina_cnn_affinity": 7.50, "posebusters_pass": True},
}


def _build_genotype_tree(demo: dict, variant: dict, smiles: str) -> dict:
    \"\"\"Build per-(target, variant) stub tree; return paths + scores.

    Idempotent: skips re-prep / re-mutate / re-embed when files exist on disk.
    Returns a dict with: pdb_path, pose_sdf_path, scores.
    \"\"\"
    target = demo["target_name"]
    pdb_id = demo["pdb_id"]
    var_label = variant["label"]
    handling = variant["handling"]

    genotype_dir = _genotype_dir(target, var_label)
    fold_dir     = genotype_dir / "fold"
    docking_dir  = genotype_dir / "docking"
    fold_dir.mkdir(parents=True, exist_ok=True)
    docking_dir.mkdir(parents=True, exist_ok=True)

    pdb_path  = fold_dir   / f"{target.lower()}_{var_label.replace('*', '').lower() or 'wt'}_best.pdb"
    pose_path = docking_dir / "poses.sdf"

    # --- Stage 1: PDB ---
    if not pdb_path.exists():
        raw_pdb = fetch_pdb(pdb_id, cache_dir=RCSB_CACHE)
        wt_prepped = raw_pdb.with_suffix(".prepped.pdb")
        prep_receptor(raw_pdb, target_name=target, out_path=wt_prepped)
        if handling == "wt":
            wt_prepped.replace(pdb_path)
        elif handling == "mutate_in_place":
            _apply_mutation_pdbfixer(
                wt_prepped,
                position=variant["position"],
                wt_aa=variant["wt_aa"],
                mut_aa=variant["mut_aa"],
                out_pdb=pdb_path,
            )
        else:
            raise ValueError(f"unsupported handling {handling!r} for structural stub")

    # --- Stage 2: pose SDF ---
    if not pose_path.exists():
        anchor_pos = variant["position"] if variant["position"] is not None \
            else demo["anchor_residue_hint"]
        if anchor_pos is None:
            raise ValueError(
                f"{target} {var_label}: no anchor residue (variant position) "
                "and no anchor_residue_hint in DEMOS"
            )
        anchor_xyz = _ca_coord_at(pdb_path, anchor_pos)
        # Place ~6 A from the anchor Calpha; small offset so the ligand sits
        # in the pocket rather than overlapping the sidechain.
        offset = np.array([6.0, 0.0, 0.0])
        mol = _embed_ligand_at(smiles, anchor_xyz + offset, seed=42)
        mol.SetProp("_Name", demo["compound"]["name"])
        writer = Chem.SDWriter(str(pose_path))
        try:
            writer.write(mol)
        finally:
            writer.close()

    scores = SYNTHETIC_SCORES.get((target, var_label), {})
    return {"pdb_path": pdb_path, "pose_sdf_path": pose_path, "scores": scores}


# Build the stub trees for every (target, variant) pair that has a structural
# pipeline branch (wt + mutate_in_place). Out-of-scope variants don't need a
# structural tree.
if USE_STUB_TREES:
    print("Building stub trees (idempotent; will skip files that exist)...")
    SMILES_BY_DEMO: dict[str, str] = {}
    FIXTURES: dict[tuple[str, str], dict] = {}
    for demo in DEMOS:
        target = demo["target_name"]
        smiles = _resolve_smiles(demo["compound"])
        SMILES_BY_DEMO[target] = smiles
        for variant in demo["variants"]:
            if variant["handling"] in ("wt", "mutate_in_place"):
                print(f"  {target:<8s} {variant['label']:<6s}: building...")
                FIXTURES[(target, variant["label"])] = _build_genotype_tree(
                    demo, variant, smiles,
                )
            else:
                print(f"  {target:<8s} {variant['label']:<6s}: out-of-scope branch (no structural fixture)")
    print(f"\\n{len(FIXTURES)} structural fixtures + {len(SMILES_BY_DEMO)} resolved SMILES ready.")
else:
    print("USE_STUB_TREES = False -- expecting real per-(target, variant) trees on disk (step 17 territory).")
    SMILES_BY_DEMO = {}
    FIXTURES = {}
"""),

        markdown("""
## 4 - Helpers (loaded from `src/aidd/mutation.py` and `src/aidd/inputs.py`)

The structural / IFP comparison helpers used below are imported in §1:

- `pocket_residues_within(pdb, anchor_residue, radius_a)` (from `aidd.mutation`)
- `ca_rmsd_pocket(pdb_wt, pdb_mut, pocket_residues)` (from `aidd.mutation`)
- `ifp_diff(ifp_wt_df, ifp_mut_df)` (from `aidd.mutation`)
- `render_moa_html(record, out_path)` (from `aidd.mutation`) - the per-(compound x variant) HTML renderer

The fourth nb-08 helper `shortlist_diff` is NOT used here: at N <= 10 the shortlist diff is meaningless. The orchestrator `summarise_mutation` is also not used; nb 09 builds its own per-(compound, variant) record dict via the cells below, since the rendering layout differs from nb 08's (no shortlist diff, optional out-of-scope branch).

The new input helpers from `src/aidd/inputs.py` are also imported in §1:

- `fetch_pdb(pdb_id)` - rcsb.org download with on-disk caching.
- `prep_receptor(pdb_path, target_name=...)` - cleanup with cofactor presets via `RECEPTOR_PREPS`.
- `name_to_smiles(name)` - PubChem REST lookup.
"""),

        code(title="Confirm helpers loaded", source="""
import inspect
helpers_loaded = [
    ("pocket_residues_within", pocket_residues_within),
    ("ca_rmsd_pocket",         ca_rmsd_pocket),
    ("ifp_diff",               ifp_diff),
    ("render_moa_html",        render_moa_html),
    ("fetch_pdb",              fetch_pdb),
    ("prep_receptor",          prep_receptor),
    ("name_to_smiles",         name_to_smiles),
]
for name, fn in helpers_loaded:
    sig = inspect.signature(fn)
    print(f"  {name:<28s} {sig}")
print("\\nhelpers loaded from aidd.mutation + aidd.inputs")
"""),

        markdown("""
## 5 - Deep walkthrough: NAT2 \\*5 / \\*6 / \\*7 + isoniazid

### Background

**N-acetyltransferase 2 (NAT2)** is the Phase II conjugation enzyme that acetylates aromatic amines - both the tuberculosis drug **isoniazid** and a much wider class of **dietary aromatic amines** (PhIP, 4-aminobiphenyl, heterocyclic amines from cooked meat) that are recognised CRC carcinogens. The slow-acetylator phenotype - reduced NAT2 activity - means less efficient detoxification of these carcinogens; that mechanism is the structural argument behind a published **higher CRC risk in slow acetylators**.

The slow-acetylator phenotype is encoded by three common alleles in non-Asian populations:

- **NAT2\\*5** carries the missense variant **I114T** (`rs1801280`, c.341T>C). Position 114 is in the substrate-binding region.
- **NAT2\\*6** carries the missense variant **R197Q** (`rs1799930`, c.590G>A). Position 197 is in a loop adjacent to the active site.
- **NAT2\\*7** carries the missense variant **G286E** (`rs1799931`, c.857G>A). Position 286 is further from the active site but in a structurally constrained region.

(There are also \\*5B / \\*5C / \\*6B / \\*7B sub-alleles with additional silent or non-coding co-occurring SNPs; the *defining* missense for each \\*N class is what we model here.)

**Isoniazid** is the textbook NAT2 substrate - the molecule on which the slow-acetylator phenotype was originally characterised in the 1950s pharmacogenetics literature. It is the right deep-walkthrough compound for the same reason: the full clinical and pharmacological literature on NAT2 phenotype-genotype travels with it. The CRC carcinogen-detoxification argument generalises directly: if the active site fits isoniazid more poorly in the slow-acetylator variants, the same structural argument applies to the aromatic-amine carcinogens.

### Catalysis-vs-binding reminder (specific to enzymes)

For NAT2 specifically, the slow-acetylator phenotype's clinical signal is **k_cat / V_max**, not just K_d. The pipeline measures whether isoniazid still fits the variant active site (a necessary condition) and how the binding-site geometry / IFP differs from WT; it does NOT predict the acetylation rate. Notebook 07's RaSP DDG captures the protein-stability mechanism (slow acetylators often have reduced enzyme abundance due to misfolding); the binding-fit signal here captures the substrate-recognition mechanism. Together they triangulate the slow-acetylator story without claiming to predict catalytic rate.

### What this section produces

Four per-(compound x variant) HTML reports + four rows in the summary CSV:

- `nat2_wt_isoniazid.html` - baseline; no diff section
- `nat2_5_isoniazid.html`  - WT-vs-\\*5 structural + IFP diff
- `nat2_6_isoniazid.html`  - WT-vs-\\*6 structural + IFP diff
- `nat2_7_isoniazid.html`  - WT-vs-\\*7 structural + IFP diff

The diff sections are computed against the WT PDB built in §3. RaSP DDG numbers depend on the nb 07 cache state - the demo-set cache covers NAT2 missenses, so the numbers should be real (not None) for \\*5 / \\*6 / \\*7.
"""),

        code(title="NAT2 deep walkthrough: priors + structural diff + IFP + HTML reports", source="""
nat2_demo = next(d for d in DEMOS if d["target_name"] == "NAT2")
nat2_smiles = SMILES_BY_DEMO["NAT2"]
nat2_compound = nat2_demo["compound"]["name"]

# Reference WT structures + IFP (computed once for diff use across variants).
wt_fixture = FIXTURES[("NAT2", "WT")]
wt_pdb     = wt_fixture["pdb_path"]
wt_poses   = wt_fixture["pose_sdf_path"]
print(f"WT NAT2 PDB:   {pretty_path(wt_pdb, DERIVED_ROOT)}")
print(f"WT poses SDF:  {pretty_path(wt_poses, DERIVED_ROOT)}")

print(f"\\nComputing WT IFP for {nat2_compound}...")
ifp_wt = compute_ifp(wt_pdb, wt_poses, progress=False)
print(f"  WT IFP shape: {ifp_wt.shape if ifp_wt is not None and not ifp_wt.empty else '(empty)'}")


def _viewer_for_pose(pdb_path: Path, poses_sdf: Path, anchor_pos: int | None) -> str:
    \"\"\"Embedded py3Dmol HTML: receptor cartoon + ligand sticks + variant residue highlighted.\"\"\"
    view = py3Dmol.view(width=640, height=420)
    view.addModel(pdb_path.read_text(), format="pdb")
    view.setStyle({}, {"cartoon": {"color": "spectrum", "opacity": 0.85}})
    # Read first ligand from SDF; render as sticks.
    suppl = Chem.SDMolSupplier(str(poses_sdf), removeHs=False, sanitize=False)
    for mol in suppl:
        if mol is None:
            continue
        lig_pdb = Chem.MolToPDBBlock(mol)
        view.addModel(lig_pdb, format="pdb")
        view.setStyle({"model": -1}, {"stick": {"colorscheme": "yellowCarbon"}})
        break
    if anchor_pos is not None:
        view.setStyle({"resi": str(anchor_pos)},
                      {"stick": {"colorscheme": "cyanCarbon"}, "cartoon": {"color": "cyan"}})
        view.zoomTo({"resi": str(anchor_pos)})
    else:
        view.zoomTo()
    return view._make_html()


def _build_record(target: str, demo: dict, variant: dict, compound_name: str, smiles: str,
                  *, fixture: dict | None = None,
                  ifp_wt_for_diff: "pd.DataFrame | None" = None,
                  wt_pdb_for_diff: Path | None = None) -> dict:
    \"\"\"Assemble the per-(compound x variant) record dict for render_moa_html.\"\"\"
    handling = variant["handling"]
    label    = variant["label"]
    uniprot  = demo["uniprot"]

    # Priors (always; nb 07 cache lookups).
    am_score = None; am_label = None; gn = None; ddg = None; rasp_cov = None
    if variant.get("position") is not None and variant.get("mut_aa") is not None:
        am_score = alphamissense_score(uniprot, variant["position"], variant["mut_aa"],
                                       cache_dir=ALPHAMISSENSE_CACHE)
        am_label = alphamissense_class(uniprot, variant["position"], variant["mut_aa"],
                                       cache_dir=ALPHAMISSENSE_CACHE)
        gn       = gnomad_frequency(uniprot, variant["position"], variant["mut_aa"],
                                    wt_aa=variant["wt_aa"], cache_dir=GNOMAD_CACHE)
        ddg      = rasp_ddg(uniprot, variant["position"], variant["wt_aa"], variant["mut_aa"],
                            cache_dir=RASP_CACHE)
        rasp_cov = rasp_covers(uniprot, cache_dir=RASP_CACHE)
    elif handling.startswith("out_of_scope_"):
        # For splice / promoter variants, AlphaMissense / RaSP return None; gnomAD
        # can still resolve by rsid when available (the gnomad_frequency helper
        # accepts rsid lookups when position/alt_aa are None).
        am_score = None
        am_label = None
        gn = {"allele_freq_overall": None, "found": False, "reference_flipped": None,
              "hgvsp": None, "rsid": variant.get("rsid")}
        ddg = None
        rasp_cov = None

    priors = {
        "alphamissense": {"score": am_score, "class": am_label},
        "gnomad":        gn or {},
        "rasp":          {"ddg": ddg, "covered": rasp_cov},
    }

    record: dict = {
        "target_name":      target,
        "uniprot":          uniprot,
        "variant_label":    label,
        "variant_handling": handling,
        "compound_name":    compound_name,
        "smiles":           smiles,
        "priors":           priors,
    }

    if handling.startswith("out_of_scope_"):
        record["out_of_scope_reason"] = variant.get("out_of_scope_reason", "")
        return record

    # Structural / IFP branches (wt + mutate_in_place).
    assert fixture is not None, "fixture required for in-scope variant"
    variant_pdb   = fixture["pdb_path"]
    variant_poses = fixture["pose_sdf_path"]
    record["scores"] = fixture["scores"]
    record["viewer_html"] = _viewer_for_pose(
        variant_pdb, variant_poses,
        anchor_pos=variant.get("position") or demo.get("anchor_residue_hint"),
    )

    ifp_var = compute_ifp(variant_pdb, variant_poses, progress=False)
    # ProLIF column structure: (ligand_resid, protein_resid, interaction_type).
    if ifp_var is not None and not ifp_var.empty:
        any_pose = ifp_var.any(axis=0)
        record["ifp_per_residue"] = [
            (str(prot_resid), str(itype))
            for (_lig, prot_resid, itype), present in any_pose.items()
            if bool(present)
        ]
    else:
        record["ifp_per_residue"] = []

    if handling == "mutate_in_place" and ifp_wt_for_diff is not None and wt_pdb_for_diff is not None:
        # WT-vs-variant structural diff + IFP diff.
        anchor = variant["position"]
        pocket_set = pocket_residues_within(wt_pdb_for_diff, anchor, radius_a=8.0)
        pocket_metric = ca_rmsd_pocket(wt_pdb_for_diff, variant_pdb, pocket_set)

        parser = PDBParser(QUIET=True)
        def _all_ca(path: Path) -> dict:
            struct = parser.get_structure("x", str(path))
            for model in struct:
                for ch in model:
                    if ch.id != "A":
                        continue
                    return {r.id[1]: r["CA"] for r in ch if "CA" in r}
            return {}
        wt_all = _all_ca(wt_pdb_for_diff)
        mut_all = _all_ca(variant_pdb)
        common = sorted(set(wt_all) & set(mut_all))
        sup = Superimposer()
        sup.set_atoms([wt_all[r] for r in common], [mut_all[r] for r in common])
        rmsd_full = float(sup.rms)

        record["structural"] = {
            "rmsd_pocket_A": pocket_metric["rmsd_pocket_A"],
            "rmsd_full_A":   rmsd_full,
            "n_matched":     pocket_metric["n_matched"],
            "radius_a":      8.0,
        }
        record["ifp_diff"] = ifp_diff(ifp_wt_for_diff, ifp_var)

    return record


# Build records for all 4 NAT2 variants (WT + 3 missense).
nat2_records: list[dict] = []
for variant in nat2_demo["variants"]:
    label = variant["label"]
    if variant["handling"] == "wt":
        fixture = FIXTURES[("NAT2", "WT")]
        record = _build_record("NAT2", nat2_demo, variant, nat2_compound, nat2_smiles,
                               fixture=fixture)
    else:
        fixture = FIXTURES[("NAT2", label)]
        record = _build_record("NAT2", nat2_demo, variant, nat2_compound, nat2_smiles,
                               fixture=fixture,
                               ifp_wt_for_diff=ifp_wt, wt_pdb_for_diff=wt_pdb)
    # Render HTML.
    safe_label = label.replace("*", "").lower() or "wt"
    html_path = MOA_REPORTS_ROOT / "nat2" / f"nat2_{safe_label}_isoniazid.html"
    render_moa_html(record, html_path)
    record["html_path"] = html_path
    nat2_records.append(record)
    am_score_str = "n/a" if record["priors"]["alphamissense"]["score"] is None else f"{record['priors']['alphamissense']['score']:.3f}"
    ddg_str = "n/a" if record["priors"]["rasp"]["ddg"] is None else f"{record['priors']['rasp']['ddg']:+.2f}"
    rmsd_str = "n/a" if "structural" not in record else f"{record['structural']['rmsd_pocket_A']:.3f}"
    n_gained = len(record.get("ifp_diff", {}).get("gained", [])) if "ifp_diff" in record else 0
    n_lost   = len(record.get("ifp_diff", {}).get("lost",   [])) if "ifp_diff" in record else 0
    print(f"  NAT2 {label:<4s}: AM={am_score_str:<6s} RaSP={ddg_str:<6s} "
          f"pocket-RMSD={rmsd_str:<6s} ifp(gain/lost)={n_gained}/{n_lost} -> "
          f"{pretty_path(html_path, DERIVED_ROOT)}")

print(f"\\nNAT2 walkthrough: {len(nat2_records)} HTML reports written to "
      f"{pretty_path(MOA_REPORTS_ROOT / 'nat2', DERIVED_ROOT)}")
"""),

        markdown("""
### Interpreting the NAT2 walkthrough

Three readings to take from the 4 reports:

- **AlphaMissense pathogenicity (priors table, row 5).** I114T, R197Q, G286E should all flag with probability >= 0.5 ("likely_pathogenic"). The evolutionary signal is independent of structure; if even AlphaMissense thinks the variant matters, that's a real prior.
- **gnomAD allele frequency (priors table, row 6).** Allele frequency varies by ancestry: \\*5 is ~28% in Europeans but rarer in East Asians; \\*6 ~30% in Europeans, \\*7 ~3% in Europeans. Ancestry stratification matters clinically; the report shows the overall frequency, but gnomAD per-population breakdowns are available via `gnomad_frequency`'s full return dict.
- **IFP diff (section 5 of the report).** Interactions naming the mutated residue (`THR114` gained, `ILE114` lost in \\*5) are the direct chemistry translation of the sidechain change. Interactions on other residues that change indicate longer-range pocket reshape - more interesting and more dependent on the structural model.

Combined reading: a variant that AlphaMissense calls pathogenic, gnomAD shows is common in the population at risk, and the IFP diff shows reshapes the isoniazid binding pocket is the textbook **slow-acetylator-by-substrate-binding-disruption** signature. Combined with RaSP DDG (stability), the structural mechanism is well-triangulated. Catalytic rate prediction stays a wet-lab question.
"""),

        markdown("""
## 6 - Brief follow-up demos

The same template applies to four more clinical cases - one per follow-up demo, brief instead of deep. Each produces an HTML report + a summary CSV row. The two missense follow-ups (CYP2D6 \\*10, KRAS G12C) exercise the structural + IFP diff path; the three out-of-scope follow-ups (CYP2D6 \\*4, DPYD\\*2A, UGT1A1\\*28) exercise the priors-only branch with the `out_of_scope_reason` text rendered in section 2 of the HTML.

The four-demo set was chosen to span the full handling matrix: two splice variants (CYP2D6\\*4, DPYD\\*2A), one promoter variant (UGT1A1\\*28), and the canonical oncogenic-driver missense (KRAS G12C). Together with the deep walkthrough (NAT2 \\*5/\\*6/\\*7) this exercises every branch of the pipeline.
"""),

        markdown("""
### 6.1 - CYP2D6 \\*4 / \\*10 + tamoxifen

**CYP2D6** is the Phase I oxidative enzyme that converts tamoxifen to its active metabolite **endoxifen** (4-hydroxy-N-desmethyl-tamoxifen, ~100x more potent than tamoxifen itself). Poor metabolisers (PMs) under-activate tamoxifen and have well-documented worse breast-cancer outcomes on tamoxifen-based endocrine therapy. CPIC publishes formal dose-adjustment guidance by CYP2D6 phenotype.

Two common reduced-function alleles:

- **CYP2D6\\*4** is a splice-acceptor variant (`rs3892097`, 1846G>A) that produces a non-functional protein. AlphaMissense and RaSP are missense-only; gnomAD lists it by position. The structural pipeline cannot model the splice consequence. *Branch:* `out_of_scope_splice`.
- **CYP2D6\\*10** is a missense variant (**P34S**, `rs1065852`, c.100C>T). The proline-to-serine swap is well outside the active-site I-helix, in the N-terminal membrane-anchor region; the variant nonetheless reduces enzyme activity by destabilising the protein. *Branch:* `mutate_in_place`.

**Catalysis-vs-binding reminder.** Same caveat as NAT2: the pipeline measures binding fit; the actual oxidation rate (P450 turnover) is a catalysis question. RaSP DDG captures the stability mechanism that drives \\*10's reduced-function phenotype.
"""),

        code(title="CYP2D6 *4 + *10 + tamoxifen: priors + (selective) structural", source="""
cyp_demo = next(d for d in DEMOS if d["target_name"] == "CYP2D6")
cyp_smiles = SMILES_BY_DEMO["CYP2D6"]
cyp_compound = cyp_demo["compound"]["name"]

cyp_wt_fixture = FIXTURES[("CYP2D6", "WT")]
ifp_wt_cyp = compute_ifp(cyp_wt_fixture["pdb_path"], cyp_wt_fixture["pose_sdf_path"], progress=False)

cyp_records: list[dict] = []
for variant in cyp_demo["variants"]:
    label = variant["label"]
    if variant["handling"] == "mutate_in_place":
        fixture = FIXTURES[("CYP2D6", label)]
        record = _build_record("CYP2D6", cyp_demo, variant, cyp_compound, cyp_smiles,
                               fixture=fixture,
                               ifp_wt_for_diff=ifp_wt_cyp,
                               wt_pdb_for_diff=cyp_wt_fixture["pdb_path"])
    else:  # out_of_scope_splice
        record = _build_record("CYP2D6", cyp_demo, variant, cyp_compound, cyp_smiles)
    safe_label = label.replace("*", "").lower() or "wt"
    html_path = MOA_REPORTS_ROOT / "cyp2d6" / f"cyp2d6_{safe_label}_tamoxifen.html"
    render_moa_html(record, html_path)
    record["html_path"] = html_path
    cyp_records.append(record)
    am_score = record["priors"]["alphamissense"]["score"]
    am_str = "n/a" if am_score is None else f"{am_score:.3f}"
    print(f"  CYP2D6 {label:<4s} ({variant['handling']:<20s}): AM={am_str:<6s} -> "
          f"{pretty_path(html_path, DERIVED_ROOT)}")
"""),

        markdown("""
### 6.2 - DPYD\\*2A + 5-fluorouracil

**Dihydropyrimidine dehydrogenase (DPYD)** is the rate-limiting Phase I enzyme for 5-fluorouracil (5-FU) catabolism. **DPYD\\*2A** (`IVS14+1G>A`, `rs3918290`) is a splice-donor variant: the disrupted splicing produces a truncated protein with the catalytic domain missing. CPIC and the FDA label both recommend dose reduction for carriers.

For the library-side **deep** walkthrough of DPYD, see notebook 08 - it uses **DPYD I560S** (the \\*13 allele) as the structural probe, because \\*13 is a real, pathogenic, well-characterised missense that AlphaMissense and RaSP **can** score. \\*2A here is the brief follow-up for the *canonical clinical allele*, with the splice limitation surfaced honestly. *Branch:* `out_of_scope_splice`.
"""),

        code(title="DPYD *2A + 5-fluorouracil: priors-only (out_of_scope_splice)", source="""
dpyd_demo = next(d for d in DEMOS if d["target_name"] == "DPYD")
dpyd_smiles = SMILES_BY_DEMO["DPYD"]
dpyd_compound = dpyd_demo["compound"]["name"]

dpyd_records: list[dict] = []
for variant in dpyd_demo["variants"]:
    record = _build_record("DPYD", dpyd_demo, variant, dpyd_compound, dpyd_smiles)
    safe_label = variant["label"].replace("*", "").lower()
    html_path = MOA_REPORTS_ROOT / "dpyd" / f"dpyd_{safe_label}_5fu.html"
    render_moa_html(record, html_path)
    record["html_path"] = html_path
    dpyd_records.append(record)
    print(f"  DPYD {variant['label']:<4s} (out_of_scope_splice): priors-only -> "
          f"{pretty_path(html_path, DERIVED_ROOT)}")
"""),

        markdown("""
### 6.3 - Irinotecan + UGT1A1\\*28

**UDP-glucuronosyltransferase 1A1 (UGT1A1)** is the Phase II enzyme that glucuronidates the active metabolite of irinotecan (SN-38) for biliary excretion. **UGT1A1\\*28** (`(TA)7TAA` instead of `(TA)6TAA` in the TATA box, `rs8175347`) is a **promoter** variant - it does NOT change the protein sequence; the reduced enzyme activity comes from reduced *transcription*, not from altered protein structure. AlphaMissense and RaSP are not relevant; gnomAD lists it by chromosomal position. UGT1A1 also has no human crystal structure, so even for missense variants in this gene the pipeline would need an AlphaFold model via notebook 01.

The FDA label for irinotecan adjusts the starting dose by UGT1A1 genotype. CPIC publishes formal dose-reduction guidance for \\*28 homozygotes. *Branch:* `out_of_scope_promoter`.
"""),

        code(title="Irinotecan + UGT1A1 *28: priors-only (out_of_scope_promoter)", source="""
ugt_demo = next(d for d in DEMOS if d["target_name"] == "UGT1A1")
ugt_smiles = SMILES_BY_DEMO["UGT1A1"]
ugt_compound = ugt_demo["compound"]["name"]

ugt_records: list[dict] = []
for variant in ugt_demo["variants"]:
    record = _build_record("UGT1A1", ugt_demo, variant, ugt_compound, ugt_smiles)
    safe_label = variant["label"].replace("*", "").lower()
    html_path = MOA_REPORTS_ROOT / "ugt1a1" / f"ugt1a1_{safe_label}_irinotecan.html"
    render_moa_html(record, html_path)
    record["html_path"] = html_path
    ugt_records.append(record)
    print(f"  UGT1A1 {variant['label']:<4s} (out_of_scope_promoter): priors-only -> "
          f"{pretty_path(html_path, DERIVED_ROOT)}")
"""),

        markdown("""
### 6.4 - Sotorasib + KRAS G12C

**KRAS** is the small GTPase whose oncogenic driver mutations (G12C, G12D, G12V, G13D, ...) drive metastatic colorectal cancer, non-small-cell lung cancer, and pancreatic cancer. **Sotorasib** (AMG 510, marketed as Lumakras) is the first FDA-approved KRAS G12C-selective covalent inhibitor: its acrylamide warhead forms an irreversible bond with the variant-introduced cysteine at codon 12 in the GDP-bound state. The drug is *specifically designed* to bind the variant, not WT - so the variant report should show *stronger* predicted binding for G12C than for WT, the opposite direction of the enzyme demos.

*Branch:* `mutate_in_place`. The KRAS WT vs G12C diff exercises the same structural + IFP machinery as the NAT2 walkthrough; what's new is the clinical interpretation - here the "gained" interactions naming residue 12 (`CYS12 Hydrophobic` or `CYS12 HBDonor` if the warhead has formed in the pose) are the *desired* signal, not the worrying signal.
"""),

        code(title="Sotorasib + KRAS G12C: priors + structural + IFP (mutate_in_place)", source="""
kras_demo = next(d for d in DEMOS if d["target_name"] == "KRAS")
kras_smiles = SMILES_BY_DEMO["KRAS"]
kras_compound = kras_demo["compound"]["name"]

kras_wt_fixture = FIXTURES[("KRAS", "WT")]
ifp_wt_kras = compute_ifp(kras_wt_fixture["pdb_path"], kras_wt_fixture["pose_sdf_path"], progress=False)

kras_records: list[dict] = []
for variant in kras_demo["variants"]:
    label = variant["label"]
    fixture = FIXTURES[("KRAS", label)]
    record = _build_record("KRAS", kras_demo, variant, kras_compound, kras_smiles,
                           fixture=fixture,
                           ifp_wt_for_diff=ifp_wt_kras,
                           wt_pdb_for_diff=kras_wt_fixture["pdb_path"])
    safe_label = label.replace("*", "").lower() or "wt"
    html_path = MOA_REPORTS_ROOT / "kras" / f"kras_{safe_label}_sotorasib.html"
    render_moa_html(record, html_path)
    record["html_path"] = html_path
    kras_records.append(record)
    am_score = record["priors"]["alphamissense"]["score"]
    am_str = "n/a" if am_score is None else f"{am_score:.3f}"
    rmsd_str = "n/a" if "structural" not in record else f"{record['structural']['rmsd_pocket_A']:.3f}"
    n_gained = len(record.get("ifp_diff", {}).get("gained", [])) if "ifp_diff" in record else 0
    n_lost   = len(record.get("ifp_diff", {}).get("lost",   [])) if "ifp_diff" in record else 0
    print(f"  KRAS {label:<5s}: AM={am_str:<6s} pocket-RMSD={rmsd_str:<6s} "
          f"ifp(gain/lost)={n_gained}/{n_lost} -> "
          f"{pretty_path(html_path, DERIVED_ROOT)}")
"""),

        markdown("""
## 7 - Summary CSV (one row per compound x variant)

### What this section does

Concatenate all 9 per-(compound x variant) records (4 NAT2 + 2 CYP2D6 + 1 DPYD + 1 UGT1A1 + 1 KRAS) into a single DataFrame; emit `summary.csv` at `data/derived/moa_reports/summary.csv`. The column set covers compound identity, variant identity, scores (when in-scope), nb 07 priors, structural / IFP diff summary numbers, and the path to the per-(compound x variant) HTML report.
"""),

        code(title="Emit summary CSV across all demos", source="""
def _record_to_summary_row(record: dict) -> dict:
    scores = record.get("scores") or {}
    priors = record.get("priors") or {}
    am  = priors.get("alphamissense", {}) or {}
    gn  = priors.get("gnomad",        {}) or {}
    rp  = priors.get("rasp",          {}) or {}
    structural = record.get("structural") or {}
    ifp_diff_rec = record.get("ifp_diff") or {}
    row = {
        "target_name":             record["target_name"],
        "uniprot":                 record["uniprot"],
        "variant_label":           record["variant_label"],
        "variant_handling":        record["variant_handling"],
        "compound_name":           record["compound_name"],
        "smiles":                  record["smiles"],
        "boltz_affinity":          scores.get("boltz_affinity"),
        "boltz_affinity_probability": scores.get("boltz_affinity_probability"),
        "gnina_cnn_affinity":      scores.get("gnina_cnn_affinity"),
        "posebusters_pass":        scores.get("posebusters_pass"),
        "alphamissense_score":     am.get("score"),
        "alphamissense_class":     am.get("class"),
        "gnomad_allele_freq_overall":  gn.get("allele_freq_overall"),
        "gnomad_reference_flipped":    gn.get("reference_flipped"),
        "gnomad_rsid":             gn.get("rsid"),
        "rasp_ddg":                rp.get("ddg"),
        "rasp_covered":            rp.get("covered"),
        "rmsd_pocket_A":           structural.get("rmsd_pocket_A"),
        "rmsd_full_A":             structural.get("rmsd_full_A"),
        "n_pocket_matched":        structural.get("n_matched"),
        "ifp_gained_n":            len(ifp_diff_rec.get("gained", [])) if ifp_diff_rec else None,
        "ifp_lost_n":              len(ifp_diff_rec.get("lost",   [])) if ifp_diff_rec else None,
        "ifp_preserved_n":         len(ifp_diff_rec.get("preserved", [])) if ifp_diff_rec else None,
        "html_path":               str(record.get("html_path", "")),
    }
    return row


all_records = nat2_records + cyp_records + dpyd_records + ugt_records + kras_records
summary_df = pd.DataFrame([_record_to_summary_row(r) for r in all_records])
summary_csv = MOA_REPORTS_ROOT / "summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Summary CSV written to {pretty_path(summary_csv, DERIVED_ROOT)} "
      f"({len(summary_df)} rows x {len(summary_df.columns)} cols)")
print("\\nSample columns:", list(summary_df.columns)[:10], "...")
summary_df[["target_name", "variant_label", "variant_handling", "compound_name",
            "alphamissense_score", "rasp_ddg", "rmsd_pocket_A",
            "ifp_gained_n", "ifp_lost_n"]]
"""),

        markdown("""
## Summary

### What landed in this notebook

- 9 per-(compound x variant) HTML reports under `data/derived/moa_reports/<target>/<target>_<variant>_<compound>.html`.
- 1 combined summary CSV at `data/derived/moa_reports/summary.csv`.
- 6 PDB stub trees (NAT2 WT + 3 variants, CYP2D6 WT + \\*10, KRAS WT + G12C) for the structural-pipeline branches.
- 0 GPU calls. Real Boltz-2 / gnina single-compound predictions are nb 99 production territory (step 17).

### Copy-pasteable summary block for the step-16 closing commit
"""),

        code(title="Emit copy-pasteable SUMMARY block", source="""
print(\"\"\"
======================================================================
NB-09 STEP-16 SUMMARY BLOCK (copy-paste into step-closing commit body)
======================================================================
\"\"\")

print(f"5 demos, 9 (compound x variant) reports, 1 summary CSV.")
print()
print("Deep walkthrough -- NAT2 *5/*6/*7 + isoniazid (UniProt P11245):")
for record in nat2_records:
    label = record["variant_label"]
    am = record["priors"]["alphamissense"]["score"]
    ddg = record["priors"]["rasp"]["ddg"]
    rmsd = record.get("structural", {}).get("rmsd_pocket_A")
    n_g = len(record.get("ifp_diff", {}).get("gained", [])) if "ifp_diff" in record else 0
    n_l = len(record.get("ifp_diff", {}).get("lost",   [])) if "ifp_diff" in record else 0
    am_s   = "n/a"  if am   is None else f"{am:.3f}"
    ddg_s  = "n/a"  if ddg  is None else f"{ddg:+.2f}"
    rmsd_s = "n/a"  if rmsd is None else f"{rmsd:.3f} A"
    print(f"  {label:<4s}: AM={am_s:<6s} RaSP_ddG={ddg_s:<7s} pocket-RMSD={rmsd_s:<10s} ifp_gained/lost={n_g}/{n_l}")

print()
print("Brief follow-ups:")
for record_list, target in [(cyp_records, "CYP2D6 + tamoxifen"),
                             (dpyd_records, "DPYD + 5-FU"),
                             (ugt_records, "UGT1A1 + irinotecan"),
                             (kras_records, "KRAS + sotorasib")]:
    for record in record_list:
        label = record["variant_label"]
        handling = record["variant_handling"]
        am = record["priors"]["alphamissense"]["score"]
        ddg = record["priors"]["rasp"]["ddg"]
        rmsd = record.get("structural", {}).get("rmsd_pocket_A")
        n_g = len(record.get("ifp_diff", {}).get("gained", [])) if "ifp_diff" in record else 0
        n_l = len(record.get("ifp_diff", {}).get("lost",   [])) if "ifp_diff" in record else 0
        am_s   = "n/a"  if am   is None else f"{am:.3f}"
        ddg_s  = "n/a"  if ddg  is None else f"{ddg:+.2f}"
        rmsd_s = "n/a"  if rmsd is None else f"{rmsd:.3f} A"
        oos = " (out_of_scope)" if handling.startswith("out_of_scope_") else ""
        print(f"  {target:<24s} {label:<5s}{oos}: AM={am_s:<6s} RaSP_ddG={ddg_s:<7s} pocket-RMSD={rmsd_s:<10s} ifp_g/l={n_g}/{n_l}")

print()
print(f"Total: {len(all_records)} reports, {len(summary_df)} summary CSV rows.")
print(f"HTML reports root: {pretty_path(MOA_REPORTS_ROOT, DERIVED_ROOT)}")
print(f"Summary CSV:       {pretty_path(summary_csv, DERIVED_ROOT)}")
print("======================================================================")
"""),

        markdown("""
## Recap

### Biomedical takeaway

The per-(compound x variant) HTML report is the canonical wet-lab handover artefact for single-compound clinical questions: "does drug X bind protein Y" (when MoA is unknown) and "is binding altered by inherited variation Z" (pharmacogenomic question). The five demos in this notebook span Natallia's research focus - colorectal-cancer pharmacogenomics (DPYD, NAT2, CYP2D6, UGT1A1) with a breast-cancer cross-over (CYP2D6 + tamoxifen) and one oncogenic-driver demo (KRAS G12C). The slow-acetylator demo (NAT2 \\*5/\\*6/\\*7) deep-walks the methodology: each variant is missense, AlphaMissense flags pathogenic, RaSP captures the protein-stability mechanism, the IFP diff captures the substrate-binding mechanism, and the combined signal triangulates the clinical phenotype without claiming to predict catalytic rate.

For the enzyme demos specifically, the pipeline measures binding fit + protein stability, NOT catalytic rate. RaSP DDG addresses one mechanism (stability-driven reduced enzyme abundance) that drives many slow-acetylator and reduced-function phenotypes; reaction-rate prediction (k_cat, K_m) needs QM/MM-class methods and is out of scope. The bottom-line caveat at the foot of each HTML report ("Computational hypothesis-generation only. Wet-lab validation required.") is the right framing for every report this notebook produces.

### Technical takeaway

This notebook closes step 16 of the project plan (see `_planning/PROJECT_PROPOSAL.md` §7 and `_planning/MECHANISM_OF_ACTION_SCOPE.md`). Architectural lessons:

- **Per-(compound x variant) is the right unit** at N <= 10; consensus rank is meaningless and `shortlist_diff` produces trivial output. The HTML report layout for nb 09 (`render_moa_html`) is a sibling renderer to nb 08's `render_mutation_html` - different sections, same input-helper conventions.
- **The `variant_handling` enum** (`wt | mutate_in_place | fold_variant | out_of_scope_splice | out_of_scope_promoter`) keeps the methodology limits explicit in the input declarations rather than hidden in code paths. Future agents extending the demo set know which branch they're invoking.
- **The five demo targets** (NAT2, CYP2D6, DPYD, UGT1A1, KRAS) populate the `RECEPTOR_PREPS` cofactor-preset dict in `src/aidd/inputs.py`. Extending the dict for a new target is one line plus a comment about which cofactors are structurally required.
- **PDBFixer `applyMutations` is the production missense path.** Step 16 stub mode uses this same call; nb 99 (step 17) reuses the same logic and just flips `USE_STUB_TREES = False` to compute real Boltz-2 + gnina scores per (compound, variant) pair.

### Further reading

- Yang Z. et al. *Genetic variation in NAT2 metabolism and its effect on colorectal cancer susceptibility: a meta-analysis.* DOI: 10.1186/s12876-022-02418-3 - NAT2 slow-acetylator alleles and CRC risk meta-analysis.
- Cheng J. et al. *Accurate proteome-wide missense variant effect prediction with AlphaMissense.* Science 2023. DOI: 10.1126/science.adg7492 - the AlphaMissense pathogenicity model used in section 5's priors.
- Blaabjerg L.M. et al. *Rapid protein stability prediction using deep learning representations.* eLife 2023. DOI: 10.7554/eLife.82593 - RaSP, the protein-stability predictor used in section 5's priors.
- Bouysset C. & Fiorucci S. *ProLIF: a library to encode molecular interactions as fingerprints.* J. Cheminform. 2021. DOI: 10.1186/s13321-021-00548-6 - the ProLIF interaction-fingerprint library used for the IFP diff sections.
"""),
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
