"""Builder for 04_score_classical.ipynb.

Source of truth for the classical-rescorer notebook. Cells appear below in
narrative order. Never edit the .ipynb directly — see ``CLAUDE.md`` §
*Notebook workflow*.

Regenerate:
    python notebooks/_build_04_score_classical.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import AUTORELOAD_SNIPPET, code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "04_score_classical.ipynb"


def build() -> None:
    nb = notebook(
        markdown("""
# 04 — Classical rescorer: train a per-target ML model on gnina + ProLIF features

**aidd-pipeline · Notebook 4 of the screening workflow**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hvmarco/aidd-pipeline/blob/main/notebooks/04_score_classical.ipynb)

> **Upstream reference:** This notebook adapts the workflow of `_archive/Week_3_Tuesday_Challenge_{1,2}.ipynb` (Leiden / ULLA AI-in-Drug-Discovery 2025), modernised to use **gnina** features instead of PLANTS. Same shape, different docker.

The previous notebook (`03_dock_gnina`) gave us 3-D poses + raw gnina scores for every compound. Raw docking scores are a sensible *first* ranking, but they leave information on the table:

- They don't know which **specific interactions** the ligand makes (a kinase-inhibitor pose is much more interesting when it engages the canonical hinge hydrogen bond).
- They don't learn from the **target's own activity history** (compounds that have been measured in a binding assay).

This notebook trains a small machine-learning model that **rescores** the gnina poses using:

1. The gnina numeric outputs (`affinity`, `cnn_score`, `cnn_affinity`, `cnn_vs`).
2. A **ProLIF interaction fingerprint** of every pose — one feature per (residue, interaction type).
3. The ERK2 **activity labels** (`Active = 0/1`) from a published screening campaign.

We train two models — **Random Forest** (interpretable, gives feature importances) and **XGBoost** (usually the strongest classifier on tabular IFP data) — and compare both to the raw gnina-CNN-affinity baseline on a held-out test fold.

## Learning objectives

After running this notebook you will be able to:

- Explain in plain language what a **rescorer** is and why it improves a virtual-screening pipeline.
- Combine docking scores with **interaction-fingerprint features** in a single feature table.
- Handle **class imbalance** in screening data with SMOTE oversampling and class weighting.
- Read a **ROC curve** and an **enrichment factor at 1% (EF1%)** — the two metrics medicinal-chemistry teams actually look at.
- Inspect which protein residues / interactions the model learned to use, via Random Forest **feature importances**.
- Save a trained rescorer alongside a per-compound scored table for downstream consensus ranking.

## Audience

- A clinician / biologist who wants to understand how the pipeline turns docking output into a *learned* ranking.
- A data / ML person new to drug discovery — the imbalanced-classifier patterns here mirror what you'd do on any rare-event problem.
- A student writing a thesis chapter on machine-learning-assisted virtual screening.

## Prerequisites

- The repo is checked out locally with the conda `aidd` environment created (`conda env create -f environment.yml`), **or** you're running on Google Colab (the setup cell installs everything).
- `data/labels/erk2_training.tsv` ships with the repo (ERK2 actives / inactives from the Leiden course).
- `data/compounds/erk2/training.smi` ships with the repo (numeric IDs that match the labels).
- The reference receptor structure (`data/structures/erk2_4fv7.pdb`) ships with the repo. The AlphaFold model from notebook `01` is used if present (and aligned to the crystal frame, same pattern as notebook `03`).

## Runtime

- **First run, cold cache:** ~60–90 min on a Colab T4 GPU. Dominated by docking ~700 compounds with gnina (the labelled subset). The cache is written to Google Drive so this only happens once per target.
- **Re-runs with cache:** ~2–5 min on any CPU (the rescorer itself trains in seconds; the bulk of the time is IFP computation + feature engineering).
- **Local Windows / macOS:** if the docking cache already exists on Drive (from a previous Colab run), the rescorer training step works on Windows / macOS. The docking step itself is Linux-only and runs on Colab.

> ⚠ **Why a T4 GPU is recommended even though the rescorer is CPU-friendly.** The notebook's *first* run needs to dock ~700 compounds with gnina (Linux + CUDA). Subsequent re-runs read the cached docks from Drive and run on plain CPU — locally on Windows is fine. Open on T4 the first time, anywhere thereafter.
"""),

        markdown("""
## Key terms used in this notebook

| Term | Meaning in one line |
|---|---|
| **Rescorer** | A second, learned scoring function applied *on top* of docking. Improves ranking by using docking outputs as features, not as the final answer. |
| **Active / Inactive** | The binary label from a binding or activity assay — `1` if the compound is a confirmed binder, `0` if not. |
| **Class imbalance** | When one class is much rarer than the other. In screening data, < 1% of compounds are usually actives. |
| **SMOTE** | *Synthetic Minority Oversampling Technique* — generates synthetic positive examples to balance the training set. Implemented in [imbalanced-learn](https://imbalanced-learn.org/). |
| **Class weighting** | An alternative to SMOTE: tells the classifier that errors on the minority class count more. Both can be combined. |
| **ROC curve** | A plot of *true positive rate* vs *false positive rate* at every classification threshold. Diagonal = random; top-left corner = perfect. |
| **ROC-AUC** | Area under the ROC curve, 0–1. 0.5 = random; 1.0 = perfect. Common across ML; in drug discovery, > 0.7 is useful, > 0.8 is good. |
| **Enrichment factor (EF)** | How many times more actives sit at the top of a ranking than would by chance. EF1% is the standard early-recognition metric in virtual screening. |
| **Random Forest (RF)** | An ensemble of decision trees. Robust on small data, gives interpretable feature importances. |
| **XGBoost** | A gradient-boosted decision-tree ensemble. Usually the strongest classifier on tabular IFP-style features. |
| **Feature importance** | A per-feature score telling you how much each input column contributed to the model's decisions. |
| **Held-out fold** | A subset of the data not seen during training, used to estimate how well the model generalises. We use a single stratified 80/20 split with `seed=42`. |
| **Stratified split** | A train/test split that preserves the active/inactive ratio in both halves. Essential for imbalanced data. |
| **Pose selection** | Choosing one pose per compound from the multiple poses gnina returns. Default here: gnina's own top-1. |
"""),

        markdown("""
## Why a rescorer — the biological reasoning

Raw docking scores answer one question: *can this compound geometrically fit in the pocket?* A rescorer trained on real activity labels adds two more:

1. **Does it make the right interactions?** Two compounds can score identically on raw gnina affinity but make very different interaction patterns. The rescorer learns from labelled data that — for *this target* — a hinge hydrogen bond plus hydrophobic packing in a specific subpocket correlates with measured activity, while contacts in the wrong subpocket do not. ProLIF interaction fingerprints supply that information per pose.

2. **What does the assay's own pharmacology say?** Different targets reward different chemistry. ATP-competitive kinase inhibitors look chemically different from allosteric inhibitors, and within ATP-competitive inhibitors, different sub-pockets accept different functional groups. A rescorer trained on the target's labelled history captures this without us having to encode it by hand.

The trade-off: the rescorer is only as good as the training labels. On ERK2 we have a few hundred actives across a few tens of thousands of inactives — small for ML, large for medicinal chemistry. The result will be useful as one input into a consensus ranker (notebook `06`), not as the final word.

What this notebook does **not** do:

- It does not retrain a new docking scoring function. We treat gnina's outputs as *features*, full stop.
- It does not use 2-D chemical descriptors (Morgan fingerprints, MACCS keys). Those are useful but they encode *the molecule alone*, not *how it sits in this pocket* — a different (and complementary) modelling story. We keep this notebook focused on structure-based features.
- It does not handle external validation across targets. Generalisation between kinases is its own research project (TeachOpenCADD T015 has a good treatment).
"""),

        markdown("""
## 1. Setup

### What this section does

Detect Colab vs local, clone the repo on Colab, install **gnina** (Linux-native — Colab downloads its static binary; local Windows / macOS users hit the gnina step only if the docked cache isn't on Drive yet — in which case open this notebook on Colab once to populate the cache, then come back). The rescorer itself trains on CPU, so re-runs after the cache is built work locally.

### About the gnina install (read once, then forget)

The install logic is identical to notebook `03_dock_gnina`. We pin to **gnina v1.3.2** (CUDA-linked from v1.3 onward — needs a GPU runtime on Colab). To update: bump `GNINA_VERSION` + `GNINA_ASSET` below, re-run on a fresh runtime, commit on success.
"""),

        code(title="Setup: clone repo, install gnina, verify GPU + binary loads", source="""
import sys
import importlib
import shutil
import subprocess
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

# gnina pin — same as notebook 03. Bump in lockstep.
GNINA_VERSION = "v1.3.2"
GNINA_ASSET   = "gnina.1.3.2"
LAST_VERIFIED = "2026-05-11"


def _gnina_works() -> bool:
    if not shutil.which("gnina"):
        return False
    try:
        r = subprocess.run(["gnina", "--version"], capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


if IS_COLAB:
    # Repo clone. Must be public for unauthenticated clone from Colab.
    REPO_ROOT = Path("/content/aidd-pipeline")
    if not REPO_ROOT.exists():
        !git clone https://github.com/hvmarco/aidd-pipeline.git {REPO_ROOT}
    if not (REPO_ROOT / "src" / "aidd").exists():
        raise RuntimeError(
            "Repo clone failed (likely cause: the repo is private and Colab cannot "
            "authenticate). Make github.com/hvmarco/aidd-pipeline public, or use a "
            "Personal Access Token via Colab Secrets, then re-run this cell."
        )

    # gnina is only needed when the docked cache hasn't been built yet for this
    # labelled subset. We probe in the next cell; for now, install if absent and
    # a GPU is attached (skip silently on CPU runtimes — the cached path will
    # surface a clearer message if the cache is also missing).
    if not _gnina_works():
        gpu_ok = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
        if gpu_ok:
            print(f"Downloading gnina {GNINA_VERSION} ({GNINA_ASSET}, verified {LAST_VERIFIED})…")
            url = f"https://github.com/gnina/gnina/releases/download/{GNINA_VERSION}/{GNINA_ASSET}"
            !wget -q -O /usr/local/bin/gnina {url}
            !chmod +x /usr/local/bin/gnina
            probe = subprocess.run(["gnina", "--version"], capture_output=True, text=True)
            if probe.returncode != 0:
                raise RuntimeError(
                    f"gnina installed but `gnina --version` exited {probe.returncode}. "
                    f"stderr:\\n{probe.stderr}\\n"
                )

    # ML stack + structural-biology deps (not in Colab's default image).
    !pip install -q posebusters rdkit datamol "prolif>=2.0" py3Dmol biopython \
        scikit-learn xgboost imbalanced-learn

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from rdkit import Chem

from aidd.io import mount_drive_if_colab, pretty_path
from aidd.ligands import pick_labeled_subset, prepare_library, write_sdf
from aidd.ifp import compute_ifp, to_wide_features
from aidd.docking import (
    BindingBox, box_from_center_radius,
    dock_library, gnina_available,
)
from aidd.scoring import (
    select_top_pose, align_ifp_to_top_pose, build_feature_table,
    enrichment_factor, evaluate_scores, roc_points,
    make_rescorer, feature_importances,
)
from aidd.structures import superpose_by_resnum

sns.set_theme(style="whitegrid")
print("imports ok")
"""),

        markdown("""
### ⚠ Google Drive authorization — read this before running the next cell

The docking cache for this notebook is **hundreds of megabytes** and takes 60–90 min of GPU time to build. We default to writing it on Google Drive so it survives Colab runtime restarts (idle timeout, browser close, disconnect). The first time you run the next cell on Colab, you will see a Drive permission dialog — click through to allow.

**If you do not want to authorize Google Drive**, change the line `USE_DRIVE = IS_COLAB` in the next cell to `USE_DRIVE = False` *before* running it. The notebook will still run end-to-end; caches go to `/content/aidd-pipeline/data/derived/` on the Colab session disk and **disappear when the runtime ends** — you'll have to redo the docking on every reconnect.

If you accidentally dismiss the Drive dialog (clicking outside it, closing the tab), the cell will crash. Re-run it and either authorize, or change the line to `False` first.
"""),

        code(title="Drive persistence toggle  (change to False to opt out)", source="""
# ════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE AUTHORIZATION
# On Colab, the next call will prompt for Drive permission on first use.
#
# →  To OPT OUT of Drive, change the line below to:    USE_DRIVE = False
#    (caches then live on /content/ and disappear when the runtime ends)
# ════════════════════════════════════════════════════════════════════════
USE_DRIVE = IS_COLAB   # ←── change to False if you do NOT want Drive

DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)
print(f"USE_DRIVE: {USE_DRIVE}  "
      f"({'Drive-backed (persistent)' if USE_DRIVE else 'local /content (ephemeral, lost on runtime kill)'})")
print(f"DATA_ROOT: {DATA_ROOT}")
"""),

        markdown("""
## 2. Inputs — the labelled ERK2 subset

### Background

The ERK2 labels file (`data/labels/erk2_training.tsv`) holds 47,203 compounds with binary `Active` annotations from the Leiden / ULLA course. The class imbalance is severe — about **231 actives among 46,972 inactives (~0.5%)** — which is typical of real screening data and is where most rescorers struggle.

We need to be deliberate about *which* compounds we dock. Two extreme strategies:

1. **Dock all 47k compounds.** Most rigorous training set possible, but ~24 GPU-hours on a T4. Overkill for a kinase rescorer where actives saturate well before 47k.
2. **Dock a small handful.** Fast, but with so few actives the ROC-AUC becomes noisy and the model can't learn IFP patterns reliably.

The pragmatic middle ground is to **keep every known active and sample inactives at a fixed multiplier**. With 231 actives at a 2× inactive multiplier, we dock 693 compounds total — under 90 min on a T4 — and retain enough actives for a stable held-out fold (≈ 46 actives per 20 % test split).

`aidd.ligands.pick_labeled_subset` builds this subset deterministically (seeded). The output is one row per compound with the SMILES, the numeric ID, and the activity label.
"""),

        code(title="Build the labelled subset (~693 compounds)", source="""
TARGET = "erk2"

INPUT_SMI = REPO_ROOT / "data" / "compounds" / TARGET / "training.smi"
LABELS    = REPO_ROOT / "data" / "labels" / "erk2_training.tsv"

# Output paths follow the stage-hierarchy convention: per-stage dirs at
# data/derived/<target>/<stage>/, with named sub-runs under docking/. Notebook
# 03's 10-compound smoke-test goes to data/derived/erk2/docking/ directly;
# the larger labelled-subset run for the rescorer lives under
# data/derived/erk2/docking/labeled_subset/. They coexist without colliding.
SCORING_DIR  = DATA_ROOT / TARGET / "scoring"
DOCK_DIR     = DATA_ROOT / TARGET / "docking" / "labeled_subset"
SCORING_DIR.mkdir(parents=True, exist_ok=True)
DOCK_DIR.mkdir(parents=True, exist_ok=True)

LABELED_SMI         = SCORING_DIR / "labeled_subset.smi"
LIGANDS_PREPARED    = SCORING_DIR / "ligands_labeled_prepared.sdf"
POSES_SDF           = DOCK_DIR / "poses.sdf"
GNINA_SCORES_CSV    = DOCK_DIR / "gnina_scores.csv"
TRAINING_IFP        = SCORING_DIR / "training_ifp.csv.gz"
RESCORER_PKL        = SCORING_DIR / "rescorer.pkl"
SCORED_POSES        = SCORING_DIR / "scored_poses.parquet"

# pick_labeled_subset is deterministic given the seed; we cache its output as
# a tiny .smi so downstream stages have a stable input. Re-runs short-circuit.
INACTIVE_MULTIPLIER = 2
CAP                 = 2000   # ceiling on total subset size; rarely hit on ERK2
SEED                = 42

subset_df = pick_labeled_subset(
    INPUT_SMI, LABELS,
    inactive_multiplier=INACTIVE_MULTIPLIER,
    cap=CAP,
    seed=SEED,
)
print(f"Subset: {len(subset_df):,} compounds  "
      f"({int(subset_df['Active'].sum())} actives, "
      f"{int((subset_df['Active']==0).sum())} inactives)")

# Cache the subset as a .smi for the prep stage. Columns: smiles, name.
subset_df[["smiles", "name"]].to_csv(LABELED_SMI, sep="\\t", header=False, index=False)
print(f"Wrote {pretty_path(LABELED_SMI, DATA_ROOT, REPO_ROOT)}")

# Keep the labels in memory for later — same DataFrame, just rename.
labels = subset_df[["name", "Active"]].rename(columns={"name": "compound_id"})
labels["compound_id"] = labels["compound_id"].astype(str)
labels.head()
"""),

        markdown("""
### Why this sampling matters

The 2× inactive multiplier is a deliberate compromise between **statistical stability** (more inactives = more reliable EF1%) and **wall time** (each compound costs ~7 s on a T4). At 1:2 active:inactive in training data, the model still sees enough signal from the negatives; pushing to 1:5 or 1:10 marginally tightens the AUC but doubles the docking budget.

The `seed=42` makes the inactive sampling reproducible — anyone re-running this notebook gets the exact same 462 inactives, so the rescorer is comparable across sessions and reviewers.
"""),

        markdown("""
## 3. Stage A — prepare the ligands

### Background

We re-use the standardisation + 3-D embed pipeline from `02_prepare_ligands` against this labelled subset. Idempotent: skips if the prepared SDF already exists at `data/derived/erk2/scoring/ligands_labeled_prepared.sdf`. About **1–2 minutes for 693 compounds** on Colab Linux with 4 workers.
"""),

        code(title="Prepare ligands (standardise → properties → PAINS → 3-D embed)", source="""
if LIGANDS_PREPARED.exists():
    n_cached = sum(1 for _ in Chem.SDMolSupplier(str(LIGANDS_PREPARED)))
    print(f"Using cached prepared library: {n_cached:,} compounds at "
          f"{pretty_path(LIGANDS_PREPARED, DATA_ROOT, REPO_ROOT)}")
else:
    n_workers = 4 if IS_COLAB else 1  # Windows Jupyter prefers serial
    print(f"Preparing {len(subset_df):,} compounds (n_workers={n_workers})…")
    prepared = prepare_library(subset_df, n_workers=n_workers, progress=True)

    passing = prepared[
        prepared["ok"]
        & prepared["lipinski_pass"]
        & (prepared["pains"] == "")
    ].copy()
    n_written = write_sdf(
        passing, LIGANDS_PREPARED,
        props_to_write=["smiles_std", "mw", "logp", "qed", "Active"],
    )
    print(f"Standardise + 3-D embed: {prepared['ok'].sum():,}/{len(prepared):,} succeeded")
    print(f"After Lipinski + PAINS gates: {len(passing):,} compounds")
    print(f"Wrote {n_written:,} → {pretty_path(LIGANDS_PREPARED, DATA_ROOT, REPO_ROOT)}")
"""),

        markdown("""
### Interpreting the gate output

The Lipinski + PAINS gate typically retains 70–90 % of the labelled subset. **Some actives will be filtered out** — this is the well-known tension between "drug-like" filters and "true binders" (a few clinically successful drugs fail Lipinski). We accept that loss here because the goal is a *generalisable rescorer*, not maximum recall on this particular labelled set; compounds that won't ever be drugs aren't useful training signal even if active in the assay. Notebook `02` has a deeper discussion of this trade-off.
"""),

        markdown("""
## 4. Stage B — dock the labelled subset

### Background

This is the long step. Each compound runs through gnina at exhaustiveness 8, producing up to 9 poses per compound. Wall time: ~5 s per compound on a Colab T4 GPU, so 693 compounds ≈ **60–90 minutes**.

The receptor + binding-site geometry are reused from notebook `03`:
- Receptor: the AlphaFold model from notebook `01` aligned to the 4FV7 crystal frame when available, else the crystal directly.
- Binding-site box: derived from the 4FV7 crystal coordinates (`_archive/configs/plants_4fv7.conf`).

**Idempotent.** If `poses.sdf` + `gnina_scores.csv` already exist on Drive from a previous run, gnina is skipped and we read the cached outputs. To force a redock, delete those two files (or pass `overwrite=True`).

### When the cache is missing and you're not on Colab GPU

`dock_library` raises a clear instruction. The fix is one of:
- Switch the runtime to T4 GPU (Colab) and re-run this cell.
- On a local Linux machine with CUDA, install gnina (`conda install -c bioconda gnina`) and re-run.

The training cells below will work locally on Windows / macOS *as long as the cache exists on Drive*.
"""),

        code(title="Dock the labelled subset (cached after first run)", source="""
CRYSTAL_RECEPTOR = REPO_ROOT / "data" / "structures" / "erk2_4fv7.pdb"
AF_RECEPTOR = DATA_ROOT / TARGET / "fold" / f"{TARGET}_best.pdb"

if AF_RECEPTOR.exists():
    ALIGNED_AF = AF_RECEPTOR.with_name(f"{TARGET}_best_aligned_to_4fv7.pdb")
    if not ALIGNED_AF.exists():
        info = superpose_by_resnum(AF_RECEPTOR, CRYSTAL_RECEPTOR, ALIGNED_AF)
        print(f"Aligned AF model onto 4FV7 crystal: RMSD {info['rmsd']:.2f} Å "
              f"over {info['n_matched_residues']} Cα atoms")
    RECEPTOR = ALIGNED_AF
    RECEPTOR_KIND = "AlphaFold (notebook 01), aligned to 4FV7 crystal frame"
else:
    print("AF model from notebook 01 not found — using the 4FV7 crystal directly.")
    RECEPTOR = CRYSTAL_RECEPTOR
    RECEPTOR_KIND = "4FV7 crystal (AF model unavailable)"

# Binding-site geometry — 4FV7 frame, from _archive/configs/plants_4fv7.conf.
BINDING_SITE_CENTER = (1.34299, 17.3648, 40.9828)
BINDING_SITE_RADIUS = 12.9007
BOX = box_from_center_radius(BINDING_SITE_CENTER, BINDING_SITE_RADIUS, margin=2.0)

EXHAUSTIVENESS = 8
NUM_MODES = 9

print(f"Receptor:   {pretty_path(RECEPTOR, DATA_ROOT, REPO_ROOT)}")
print(f"            ({RECEPTOR_KIND})")
print(f"Output:     {pretty_path(DOCK_DIR, DATA_ROOT, REPO_ROOT)}/")
print()

if POSES_SDF.exists() and GNINA_SCORES_CSV.exists():
    gnina_scores = pd.read_csv(GNINA_SCORES_CSV)
    print(f"Cached docking found: {len(gnina_scores):,} poses across "
          f"{gnina_scores['compound_id'].nunique():,} compounds — skipping gnina run.")
elif not gnina_available():
    raise RuntimeError(
        "No cached docking output and gnina is not available on this machine. "
        "Open this notebook on a Colab T4 GPU once to populate the docking cache "
        "on Google Drive; subsequent re-runs of this notebook (even on Windows / "
        "macOS) will use the cache. See the markdown above this cell."
    )
else:
    gnina_scores = dock_library(
        receptor=RECEPTOR,
        ligands_sdf=LIGANDS_PREPARED,
        out_dir=DOCK_DIR,
        box=BOX,
        exhaustiveness=EXHAUSTIVENESS,
        num_modes=NUM_MODES,
        cnn_scoring="rescore",
        seed=42,
    )
    print(f"Docked {gnina_scores['compound_id'].nunique():,} compounds "
          f"→ {len(gnina_scores):,} poses")

gnina_scores["compound_id"] = gnina_scores["compound_id"].astype(str)
gnina_scores.head()
"""),

        markdown("""
### How to read the score table

One row per pose; ~9 poses per compound. The columns are the same as in notebook `03`: `affinity` (Vina kcal/mol, lower better), `cnn_score` (0–1, higher better), `cnn_affinity` (pK_d, higher better), `cnn_vs` (combined, higher better). A healthy labelled set will show CNN-affinity peaking around 5–7 with the strongest actives ≥ 8.

We don't filter on PoseBusters here even though notebook `03` did. The reason: a compound whose top-1 pose fails PoseBusters might still have *useful features* for ranking (the IFP can capture the spatial pattern even if the geometry has small flaws), and we don't want to silently drop labelled compounds — that biases the rescorer. The PoseBusters flag remains *available* on each pose for downstream filtering of *individual screening hits*.
"""),

        markdown("""
## 5. Stage C — compute interaction fingerprints (IFPs)

### Background

ProLIF turns each docked pose into a binary vector: one feature per (protein residue, interaction type) combination, with `1` indicating "this interaction was detected" and `0` otherwise. For an ERK2 ATP-pocket-sized binding box you typically get 80–150 features.

We compute IFPs in **wide format** (one row per pose, columns = features) so they can be concatenated to the gnina numeric columns. Cached as `.csv.gz` to disk so re-runs are instant.

Wall time: ~5–15 minutes on CPU for 700 compounds × 9 poses.
"""),

        code(title="Compute IFPs on all docked poses", source="""
if TRAINING_IFP.exists():
    ifp_wide = pd.read_csv(TRAINING_IFP, index_col=0)
    print(f"Cached IFP: {ifp_wide.shape[0]:,} rows × {ifp_wide.shape[1]} features "
          f"at {pretty_path(TRAINING_IFP, DATA_ROOT, REPO_ROOT)}")
else:
    print("Computing ProLIF interaction fingerprints on docked poses…")
    ifp_df = compute_ifp(RECEPTOR, POSES_SDF, progress=True)
    ifp_wide = to_wide_features(ifp_df)
    ifp_wide.to_csv(TRAINING_IFP, compression="gzip")
    print(f"Wrote {ifp_wide.shape[0]:,} × {ifp_wide.shape[1]} → {pretty_path(TRAINING_IFP, DATA_ROOT, REPO_ROOT)}")

ifp_wide.head()
"""),

        markdown("""
### What the IFP columns mean

Each column is a `RES{number}.{chain}_{interaction}` label, e.g. `LYS54.A_HBAcceptor` = "the ligand accepted a hydrogen bond from the side chain of lysine-54 in chain A". The values are 0 or 1. Most columns are sparse — a typical pose makes 5–20 interactions, not 100 — which is fine for tree-based classifiers (they handle sparsity natively).

If you scroll the column names, you should see the **ATP-pocket residues** as the most populated columns: `LYS54`, `MET108` (gatekeeper), `LEU107` (hinge), `ASP106`, `LYS114`. Residues elsewhere on the protein surface will rarely be touched and contribute little.
"""),

        markdown("""
## 6. Stage D — pose selection + feature merge + label merge

### Background

Each compound has up to 9 poses. To train one model with one prediction per compound, we have to **collapse poses to compounds**. The classical approaches:

- **Top-1 by gnina rank.** Trust gnina's own pose ranking; take its first pose for every compound. Simplest, deterministic, what most rescorer pipelines do. ← default.
- **Top-1 by CNN affinity.** Take whichever pose has the highest CNN affinity per compound. Slightly stronger biased-by-the-scorer-we're-using-to-rerank signal.
- **Boltzmann-weighted aggregate.** Combine all poses with weights from their scores. More principled, more complex; the literature shows ~3–5 % AUC improvement on hard targets.

For a teaching notebook the gnina-top-1 path is honest about its assumptions and gives results comparable to published gnina-rescoring papers. The other options are commented out below for anyone who wants to experiment.
"""),

        code(title="Pose-select + merge IFP + attach labels → training table", source="""
top_pose = select_top_pose(gnina_scores, by="gnina_top1")
aligned_ifp = align_ifp_to_top_pose(ifp_wide, top_pose)
features = build_feature_table(top_pose, aligned_ifp)
features["compound_id"] = features["compound_id"].astype(str)

# Attach labels — inner join drops the compounds that lost their label during
# the prep gate (e.g. a Lipinski failure removed them upstream). We track the
# n_lost number so the merge isn't silent.
n_before = len(features)
table = features.merge(labels, on="compound_id", how="inner")
n_after = len(table)
print(f"Feature table: {n_after:,} compounds × {table.shape[1] - 2:,} features "
      f"(+ compound_id, Active). Lost {n_before - n_after} to label / prep mismatch.")
print(f"Active: {int(table['Active'].sum())}    Inactive: {int((table['Active']==0).sum())}")
table.head()
"""),

        markdown("""
### What "lost to label / prep mismatch" means

`n_lost` should usually be 0 — every compound in the labelled subset survives prep + docking. Non-zero values indicate compounds that failed standardisation, failed Lipinski / PAINS, or failed to dock — for any of those, gnina has no pose to score and we can't include them in training. If `n_lost` is large (> 5 % of the subset), check the prep + dock cells above for an upstream problem.
"""),

        markdown("""
## 7. Stage E — train Random Forest and XGBoost rescorers

### Background

We train two classifiers on the same training fold and compare them honestly on the same held-out fold:

- **Random Forest** — the interpretable baseline. Gives feature importances (Section 10 below). `class_weight='balanced'` compensates for imbalance; we additionally apply SMOTE on the training fold for a fairer comparison with XGBoost.
- **XGBoost** — usually the strongest classifier on tabular IFP data, at the cost of interpretability. Configured to use a histogram tree method and a binary-classification log-loss.

We also compute the **raw gnina baseline**: use `cnn_affinity` of the top pose directly as the ranking score, no learning involved. This is what the rescorer has to beat. The brief's "done signal" is **rescorer AUC − baseline AUC > 0.02 on the held-out fold**.

### About the split — two passes, two AUCs

We evaluate on **two splits** because each tells a different — and honest — story:

1. **Random stratified split (80/20, `seed=42`).** Preserves active/inactive ratio in both halves; gives ~46 actives in the held-out fold. Simple and stable. **But it leaks chemistry**: when an active and a chemically similar inactive land on opposite sides of the split, the model effectively "memorises" the scaffold and the AUC is optimistic.

2. **Scaffold-grouped split (`GroupShuffleSplit`, `seed=42`).** Compounds are grouped by their **Murcko scaffold** (the molecular backbone, stripped of side-chains) and the split is taken at the *scaffold* level — every molecule sharing a scaffold goes to the same side. This is the gold-standard QSAR evaluation: it estimates how the rescorer generalises to **new chemotypes**, not just new compounds of a known chemotype.

On this ERK2 subset the scaffold counts are surprisingly favourable: 646 distinct Murcko scaffolds across 693 compounds (94 % singletons), giving 38–57 actives in the held-out fold across random seeds. Scaffold-split AUC is **the headline number to trust**; the gap between the random AUC and the scaffold AUC is the "scaffold-leakage tax" — quantify it so you know how much of any apparent rescorer gain is real generalisation versus chemotype memorisation.

A 5-fold CV would tighten error bars further but takes 5× wall time. For the teaching path we report single splits and explain the trade-off in plain language.
"""),

        code(title="Compute Murcko scaffolds + define both splits", source="""
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split, GroupShuffleSplit

feature_cols = [c for c in table.columns if c not in ("compound_id", "Active")]
X = table[feature_cols].copy()
y = table["Active"].astype(int).to_numpy()

# Recover the SMILES for each compound from the labelled subset to compute
# scaffolds. (subset_df is built deterministically in Stage 2.)
smiles_map = dict(zip(subset_df["name"].astype(str), subset_df["smiles"]))
def _murcko(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

scaffolds = pd.Series(
    [_murcko(smiles_map.get(cid, "")) for cid in table["compound_id"]],
    index=table.index,
    name="scaffold",
)
print(f"Scaffold counts on the labelled subset:")
print(f"  distinct scaffolds       : {scaffolds.nunique():,}")
print(f"  singletons (one compound): {int((scaffolds.value_counts() == 1).sum()):,}")
print(f"  largest scaffold size    : {int(scaffolds.value_counts().max())}")
print()

# Split A — random stratified
rand_train_idx, rand_test_idx = train_test_split(
    table.index, test_size=0.20, random_state=42, stratify=y,
)

# Split B — scaffold-grouped (every molecule sharing a Murcko scaffold goes
# to the same side; ratio is approximate because group-shuffle can't perfectly
# stratify when groups are sticky).
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
scaff_train_idx, scaff_test_idx = next(gss.split(table.index, y, scaffolds))
scaff_train_idx = table.index[scaff_train_idx]
scaff_test_idx  = table.index[scaff_test_idx]

print(f"Random-stratified split   "
      f"test n = {len(rand_test_idx):3d}, "
      f"actives in test = {int(y[rand_test_idx].sum()):3d}")
print(f"Scaffold-grouped split    "
      f"test n = {len(scaff_test_idx):3d}, "
      f"actives in test = {int(y[scaff_test_idx].sum()):3d}")
"""),

        code(title="Train RF + XGBoost on each split, evaluate both", source="""
def _train_and_eval(train_idx, test_idx, label):
    X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
    y_tr, y_te = y[X.index.get_indexer(train_idx)], y[X.index.get_indexer(test_idx)]

    rf = make_rescorer("rf", smote=True, seed=42, n_estimators=300)
    rf.fit(X_tr, y_tr)
    rf_p = rf.predict_proba(X_te)[:, 1]

    xgb = make_rescorer("xgb", smote=True, seed=42, n_estimators=300)
    xgb.fit(X_tr, y_tr)
    xgb_p = xgb.predict_proba(X_te)[:, 1]

    base_p = X_te["cnn_affinity"].to_numpy()

    return {
        "label":    label,
        "test_idx": test_idx,
        "y_test":   y_te,
        "rf":       rf,  "rf_proba":  rf_p,  "rf_eval":  evaluate_scores(y_te, rf_p),
        "xgb":      xgb, "xgb_proba": xgb_p, "xgb_eval": evaluate_scores(y_te, xgb_p),
        "base_proba": base_p, "base_eval": evaluate_scores(y_te, base_p),
    }

run_random   = _train_and_eval(rand_train_idx,  rand_test_idx,  "random stratified")
run_scaffold = _train_and_eval(scaff_train_idx, scaff_test_idx, "scaffold grouped")

# Headline numbers — scaffold AUC is the one to trust.
results = pd.DataFrame([
    {"split": run_random["label"],   "model": "gnina CNN_affinity",  **run_random["base_eval"]},
    {"split": run_random["label"],   "model": "Random Forest",       **run_random["rf_eval"]},
    {"split": run_random["label"],   "model": "XGBoost",             **run_random["xgb_eval"]},
    {"split": run_scaffold["label"], "model": "gnina CNN_affinity",  **run_scaffold["base_eval"]},
    {"split": run_scaffold["label"], "model": "Random Forest",       **run_scaffold["rf_eval"]},
    {"split": run_scaffold["label"], "model": "XGBoost",             **run_scaffold["xgb_eval"]},
])
print("Held-out fold summary (scaffold AUC is the honest generalisation estimate):")
results
"""),

        markdown("""
## 8. Stage F — Out-of-fold predictions for downstream consumption

### Background

Stage E gave us an honest *aggregate* metric on a single 80/20 held-out fold. That's the right format for the section's "did the rescorer beat the baseline?" question. But the downstream consensus notebook (`06_consensus_and_shortlist`) needs something different: an honest **per-compound** prediction — i.e., for every compound in the labelled subset, the rescorer's predicted probability *as if that compound had not been in training*. A single 80/20 split only gives honest predictions for 20 % of compounds; the other 80 % were in the training set, so any prediction the model makes on them is a memorisation artefact.

The solution is **out-of-fold (OOF) cross-validation**:

1. Split the data into N folds (scaffold-grouped, same group key as Stage E).
2. For each fold, train a fresh model on the other N−1 folds and predict on the held-out fold.
3. Stitch the predictions back together — every compound now has a prediction from a model that did *not* see it during training.

We use 5 folds, grouped by Murcko scaffold (same `scaffolds` Series as Stage E). The methodology is identical to Stage E's `_train_and_eval` — only the bookkeeping differs (5 folds covering every compound vs. one 80/20 split).

### Why both OOF and trained-on-all are written to disk

The notebook saves *both* prediction types to `scored_poses.parquet`:

- **`rescorer_rf_proba_oof`** / **`rescorer_xgb_proba_oof`** — honest, suitable for ranking and held-out evaluation. **This is what notebook 06's consensus shortlist consumes.**
- **`rescorer_rf_proba`** / **`rescorer_xgb_proba`** — predictions of the model trained on *all* labelled compounds. Useful for chemistry inspection (e.g. "show me how the global model would score these never-labelled compounds when we deploy the pipeline on a new library"), but **not** for computing AUC against the labelled training set — that gives the meaningless 1.000 ceiling.

This split was added after a downstream notebook caught the leak; see `_planning/KNOWN_ISSUES.md` for the original bug spec.
"""),

        code(title="5-fold scaffold-grouped OOF predictions (RF + XGBoost)", source="""
from sklearn.model_selection import GroupKFold

N_OOF_FOLDS = 5
gkf = GroupKFold(n_splits=N_OOF_FOLDS)

oof_proba_rf  = np.full(len(X), np.nan)
oof_proba_xgb = np.full(len(X), np.nan)

for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=scaffolds), start=1):
    rf = make_rescorer("rf", smote=True, seed=42, n_estimators=300)
    rf.fit(X.iloc[tr_idx], y[tr_idx])
    oof_proba_rf[te_idx] = rf.predict_proba(X.iloc[te_idx])[:, 1]

    xgb = make_rescorer("xgb", smote=True, seed=42, n_estimators=300)
    xgb.fit(X.iloc[tr_idx], y[tr_idx])
    oof_proba_xgb[te_idx] = xgb.predict_proba(X.iloc[te_idx])[:, 1]

    rf_fold_auc  = evaluate_scores(y[te_idx], oof_proba_rf[te_idx])["auc"]
    xgb_fold_auc = evaluate_scores(y[te_idx], oof_proba_xgb[te_idx])["auc"]
    print(f"Fold {fold_i}: n_test={len(te_idx):3d}, "
          f"actives={int(y[te_idx].sum()):2d}, "
          f"RF AUC={rf_fold_auc:.3f}, XGB AUC={xgb_fold_auc:.3f}")

oof_rf_eval  = evaluate_scores(y, oof_proba_rf)
oof_xgb_eval = evaluate_scores(y, oof_proba_xgb)
print()
print(f"OOF (5-fold scaffold-grouped) — RF:  "
      f"AUC={oof_rf_eval['auc']:.3f}, EF1%={oof_rf_eval['ef1']:.2f}, EF5%={oof_rf_eval['ef5']:.2f}")
print(f"OOF (5-fold scaffold-grouped) — XGB: "
      f"AUC={oof_xgb_eval['auc']:.3f}, EF1%={oof_xgb_eval['ef1']:.2f}, EF5%={oof_xgb_eval['ef5']:.2f}")
"""),

        markdown("""
### How to read the OOF result

Every compound now has one prediction from a model that did not see it during training. The aggregate AUC across all 413 compounds is the rescorer's honest **per-compound generalisation estimate** — methodologically more rigorous than any single 80/20 split, because it averages over five disjoint scaffold partitions covering every compound, not one optimistic realisation.

The two numbers tell different stories:

- **Stage E (single 80/20 scaffold split)** — one realisation of the rescorer's performance on one held-out fold. On ERK2 this landed at scaffold AUC ≈ 0.66, but that single number sits inside a wide distribution.
- **This cell (5-fold scaffold OOF over all 413)** — the aggregate across five folds covering every compound. On ERK2 this comes in lower (RF ≈ 0.58, XGB ≈ 0.58), with substantial **per-fold variance** (the per-fold AUCs span roughly 0.40 to 0.69 — one fold lands below random, another sits near 0.70). Expect the aggregate to be **0.05–0.10 below** any optimistic single-split realisation as a rule of thumb.

The OOF aggregate is the number to anchor on when reporting the rescorer's expected performance on a new chemotype; it is what `notebook 06` consumes for the consensus shortlist. The per-fold spread is the **scientifically informative second number** — it tells you the rescorer generalises *unevenly* across chemotypes (which is precisely what scaffold splits exist to expose). Report both; don't average the spread away.

If the OOF AUC comes in materially lower (< 0.55) or higher (> 0.75), that's a finding to investigate before relying on it downstream. Pathological causes to consider:

- One fold dominates the actives — `n_actives` per fold in the print above should be roughly balanced (within ±10 % of the global active count).
- A scaffold collision between fold training sets (shouldn't happen with `GroupKFold`, but worth eyeballing).
- A subtle drift between this cell's feature pipeline and Stage E's — both use `X` and `y` exactly as built in Stage D, so this should not occur.
"""),

        markdown("""
## 9. Stage G — ROC curves on both splits

### Background

The whole point of a rescorer is to **beat the raw docker** at ranking compounds. The baseline takes gnina's own `cnn_affinity` for the top pose and treats it as a ranking score directly — no learning. We've already computed it on each split in Stage E (under `gnina CNN_affinity` rows). Now we overlay the ROC curves so the comparison is visual on both splits side-by-side.
"""),

        code(title="ROC curves — random vs scaffold split, baseline vs RF vs XGBoost", source="""
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=110)
for ax, run in zip(axes, [run_random, run_scaffold]):
    for label, scores, eval_dict, colour in [
        ("gnina CNN_affinity (baseline)", run["base_proba"], run["base_eval"], "tab:gray"),
        ("Random Forest",                 run["rf_proba"],   run["rf_eval"],   "tab:blue"),
        ("XGBoost",                       run["xgb_proba"],  run["xgb_eval"],  "tab:orange"),
    ]:
        fpr, tpr = roc_points(run["y_test"], scores)
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f"{label}  AUC = {eval_dict['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{run['label'].title()} split (n = {len(run['y_test'])})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
fig.suptitle("ERK2 rescorer vs raw gnina — random (optimistic) vs scaffold (honest)")
fig.tight_layout()
plt.show()
"""),

        markdown("""
### How to read the result

The "done signal" for this notebook is:

> **Rescorer scaffold-split AUC − baseline scaffold-split AUC > 0.02.**

We anchor on the **scaffold-split** number, not the random one. The random AUC is an upper bound (chemotype memorisation makes it look generous); the scaffold AUC reflects what the rescorer will do on a *new* chemotype that wasn't in training. Reviewers and funders look at the scaffold AUC for the same reason.

A 0.02 margin is small in absolute terms but large in terms of *what it buys you*: at fixed throughput (you can dose 100 compounds), a 0.02 AUC improvement typically means **2–5 extra true actives** in your hit list, depending on the prevalence curve. On a 700-compound screen with ~40–50 held-out actives, a 0.05 absolute AUC improvement is what XGBoost typically delivers over raw gnina on a well-curated kinase target.

**The random–scaffold AUC gap** itself is diagnostic. A large gap (> 0.10) means the model is mostly memorising chemotypes; a small gap (< 0.04) means it has learned generalisable interaction patterns. The IFP-driven features tend to compress this gap (interactions are scaffold-agnostic in a way that 2-D fingerprints are not), which is one structural reason to prefer them for rescoring.

EF1% is the more medicinal-chemistry-relevant number: it tells you how concentrated the top 1 % of your ranking is in real actives. Baseline EF1% on raw docking is usually 3–8; a good rescorer pushes this to 10–25. EF1% near 1 is random; below 1 means the rescorer is *worse than random* at the top of the list — investigate before trusting it.

If neither rescorer beats the baseline on the scaffold split:
- Look at how many actives ended up in the scaffold-grouped held-out fold (`run_scaffold["y_test"].sum()`). Below 30 is statistically shaky.
- Inspect the feature-importance plot below — if the top features are all gnina columns and no IFP columns appear, ProLIF isn't seeing useful interactions and the model has nothing to add over raw docking.
- Try `select_top_pose(by="cnn_affinity")` in Section 6 — sometimes the gnina pose-rank-1 isn't the *informationally* best pose.
"""),

        markdown("""
## 10. Stage H — what did the Random Forest learn?

### Background

Random Forest's `feature_importances_` ranks each input column by how much it contributed to the trees' splits. Tree-based feature importances are a *correlational* signal — high importance ≠ causal — but on imbalanced screening data they reliably reveal which residue–interaction pairs the model leans on.

Two things you want to see:

1. A mix of **gnina numeric columns** and **IFP residue interactions** in the top 15. Pure IFP-dominated importances mean the model ignored the docker (suspicious for a structure-based feature set); pure gnina-only importances mean it ignored the structural pattern and is doing little more than the baseline.
2. **Known catalytic / hinge residues** of the target appearing — for ERK2, `LYS54`, `MET108`, `LEU107`, `ASP167`. If the model is learning on residues nowhere near the ATP pocket, something's wrong upstream.
"""),

        code(title="Random Forest feature importances (top 20)", source="""
# We read importances from the scaffold-split RF since it's the model trained
# in a way that resists chemotype-memorisation; its importances are slightly
# more honest about *generalisable* signal.
fi = feature_importances(run_scaffold["rf"], feature_cols, top_n=20)
print("Top-20 features by Random Forest importance (scaffold-split model):")
print(fi.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 6), dpi=110)
ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="steelblue")
ax.set_xlabel("Mean decrease in Gini impurity")
ax.set_title("Random Forest — top-20 features (scaffold-split model)")
plt.tight_layout()
plt.show()
"""),

        markdown("""
## 11. Stage I — save the trained rescorer and the scored-poses table

### Background

The contract with downstream notebook `06_consensus_and_shortlist`:

- **`rescorer.pkl`** — the trained RF model. We save the Random Forest trained on **all** data (no held-out fold) so the model that ships downstream is the strongest one we can produce; the AUCs from the held-out folds above are the *performance estimate* for this final model, not the model itself. If a future run shows XGBoost consistently winning on the scaffold split, swap the saved model in one line.
- **`scored_poses.parquet`** — one row per compound, with **four prediction columns** + features + label + fold-membership flags:

| Column | Source | Use it for | Never use it for |
|---|---|---|---|
| `rescorer_rf_proba_oof`  | 5-fold scaffold OOF (Stage F) | **Ranking, consensus shortlist, held-out evaluation** | — |
| `rescorer_xgb_proba_oof` | 5-fold scaffold OOF (Stage F) | Comparator to the RF OOF; honest like above | — |
| `rescorer_rf_proba`      | RF trained on **all** 413 compounds | Chemistry inspection (e.g. score brand-new compounds at deployment time) | Held-out AUC against the labelled subset — gives a meaningless 1.000 |
| `rescorer_xgb_proba`     | XGBoost trained on **all** 413 compounds | Same as above | Same as above |

The honest OOF columns are the **default consumed by `aidd.consensus.compute_consensus`** (its `rescorer_col` parameter defaults to `rescorer_rf_proba_oof`). The leaked trained-on-all columns are preserved deliberately for two reasons: (1) the chemistry-inspection use case is real — at deployment time on a fresh library, the strongest available model is the one trained on all known data — and (2) they're cheap to keep and removing them would silently break any historical notebook that referenced them.

The fold-membership flags (`in_random_test_fold`, `in_scaffold_test_fold`) record which Stage-E split each compound landed in. These remain useful for ad-hoc evaluations against a specific 80/20 split, but they're orthogonal to the OOF columns above — the OOF predictions cover every compound, not just the test slice.

The notebook is reproducible: re-running it produces bit-identical artefacts (same seeds, same data, same caches).
"""),

        code(title="Train final rescorer on all data, save rescorer.pkl + scored_poses.parquet", source="""
# Final model: trained on every labelled compound so notebook 06 ships the
# strongest available rescorer. Use the same hyperparameters as in the
# evaluation runs above.
final_rf = make_rescorer("rf", smote=True, seed=42, n_estimators=300)
final_rf.fit(X, y)
all_proba_rf  = final_rf.predict_proba(X)[:, 1]

# XGBoost trained on all data too — saved as a side artefact for comparison.
final_xgb = make_rescorer("xgb", smote=True, seed=42, n_estimators=300)
final_xgb.fit(X, y)
all_proba_xgb = final_xgb.predict_proba(X)[:, 1]

scored = table[["compound_id", "Active"]].copy()
# Honest, OOF — what downstream consumers (notebook 06) rank on.
scored["rescorer_rf_proba_oof"]  = oof_proba_rf
scored["rescorer_xgb_proba_oof"] = oof_proba_xgb
# Leaked, trained-on-all — chemistry inspection only; see the table above.
scored["rescorer_rf_proba"]      = all_proba_rf
scored["rescorer_xgb_proba"]     = all_proba_xgb
scored["gnina_cnn_affinity"]     = X["cnn_affinity"].to_numpy()
scored["gnina_affinity"]         = X["affinity"].to_numpy()
scored["in_random_test_fold"]    = scored.index.isin(run_random["test_idx"])
scored["in_scaffold_test_fold"]  = scored.index.isin(run_scaffold["test_idx"])

scored.to_parquet(SCORED_POSES, index=False)
print(f"Wrote scored table → {pretty_path(SCORED_POSES, DATA_ROOT, REPO_ROOT)}  ({len(scored):,} rows)")

joblib.dump(
    {
        "model": final_rf,
        "feature_cols": feature_cols,
        "kind": "rf",
        "seed": SEED,
        "trained_on": "erk2_labeled_subset_v1",
        "scaffold_split_auc": run_scaffold["rf_eval"]["auc"],
        "random_split_auc":   run_random["rf_eval"]["auc"],
        "oof_scaffold_auc":   oof_rf_eval["auc"],
        "oof_scaffold_ef1":   oof_rf_eval["ef1"],
    },
    RESCORER_PKL,
)
print(f"Wrote model       → {pretty_path(RESCORER_PKL, DATA_ROOT, REPO_ROOT)}")
"""),

        markdown("""
## Recap

### Biomedical takeaway

We turned a labelled ERK2 compound set into a target-specific classifier that ranks docked candidates better than raw gnina alone — for this target. The rescorer learned which **residue–interaction patterns in the ATP pocket** correlate with measured activity. That learning is the bridge between *generic structure-based docking* and *target-aware ranking*: gnina knows kinases in general; the rescorer knows *this kinase, with its known active series*.

In a real triage, the rescorer's output is one of two orthogonal signals (the other being Boltz-2 co-folded affinity from notebook `05`). The final shortlist (notebook `06`) takes compounds that rank well on **both** signals — defensible consensus, not single-tool guesswork.

### Technical takeaway

`src/aidd/scoring.py` exposes the rescorer toolbox as small composable functions: `select_top_pose`, `align_ifp_to_top_pose`, `build_feature_table`, `make_rescorer`, `enrichment_factor`, `evaluate_scores`. The notebook orchestrates them around three caches on Drive (prepared SDF, docked poses, IFP table) so the expensive stages run **once per target**; subsequent re-runs are seconds, not hours.

Two design decisions worth carrying to other targets:

- **Two-split evaluation (random + scaffold).** Random AUC is the optimistic upper bound; scaffold AUC is the honest generalisation estimate; the gap is diagnostic. Don't report one without the other.
- **Class-imbalance pattern** (`class_weight='balanced'` + SMOTE in an `imblearn.Pipeline`). The workhorse for any rare-event screening problem; the same pattern you'd reach for in tox prediction, adverse-event modelling, or any binary classification where one class is < 5 %.

### What's next in the pipeline

- **`05_dock_boltz.ipynb`** — Boltz-2 co-folded affinity on the same labelled subset. Different physics (co-folded structure + ML affinity head) and a useful orthogonal signal.
- **`06_consensus_and_shortlist.ipynb`** — join `scored_poses.parquet` (this notebook) with Boltz-2 outputs and emit `shortlist.sdf` for wet-lab follow-up.

### Further reading

- McNutt et al., *J. Cheminform.* (2021), **13**, 43 — *GNINA 1.0: molecular docking with deep learning.* [doi:10.1186/s13321-021-00522-2](https://doi.org/10.1186/s13321-021-00522-2)
- Bouysset & Fiorucci, *J. Cheminform.* (2021), **13**, 72 — *ProLIF: a library to encode molecular interactions as fingerprints.* [doi:10.1186/s13321-021-00548-6](https://doi.org/10.1186/s13321-021-00548-6)
- Chen & Guestrin, *KDD* (2016) — *XGBoost: A Scalable Tree Boosting System.* [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
- Chawla et al., *J. Artif. Intell. Res.* (2002), **16**, 321 — *SMOTE: Synthetic Minority Over-sampling Technique.* [doi:10.1613/jair.953](https://doi.org/10.1613/jair.953)
- Volkamer Lab **TeachOpenCADD T015 / T010** — ML-based protein–ligand ranking. [projects.volkamerlab.org/teachopencadd](https://projects.volkamerlab.org/teachopencadd/)
"""),
        accelerator="GPU",
        gpu_type="T4",
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
