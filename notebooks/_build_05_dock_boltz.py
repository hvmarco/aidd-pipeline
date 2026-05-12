"""Builder for 05_dock_boltz.ipynb.

Source of truth for the Boltz-2 co-folding notebook. Cells appear below in
narrative order. Never edit the .ipynb directly -- see ``CLAUDE.md`` section
*Notebook workflow*.

Regenerate:
    python notebooks/_build_05_dock_boltz.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import AUTORELOAD_SNIPPET, code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "05_dock_boltz.ipynb"


def build() -> None:
    nb = notebook(
        markdown("""
# 05 — Boltz-2 co-folding + affinity (the fast lane)

**aidd-pipeline · Notebook 5 of the screening workflow**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hvmarco/aidd-pipeline/blob/main/notebooks/05_dock_boltz.ipynb)

> **Upstream reference:** [Boltz-2](https://github.com/jwohlwend/boltz) (Wohlwend et al., 2025) — open-weights, MIT-licensed model that **co-folds** a protein and a small-molecule ligand in one pass and predicts a binding affinity. Conceptually equivalent to AlphaFold 3's co-folding head, but with open weights and a Python CLI.

This notebook is the **fast lane** of the pipeline. The previous notebooks built the *interpretable lane* — gnina docking (notebook `03`) + an interaction-fingerprint + ML rescorer (notebook `04`). Both rank the same library; they use **different physics**.

- **Interpretable lane** (notebooks 03 + 04): the receptor is folded once (notebook `01`), the ligand is docked in 3-D against the rigid receptor, the resulting pose is scored by a CNN + an ML rescorer trained on this target's activity data.
- **Fast lane** (this notebook): one neural network co-folds the protein + ligand together and outputs (1) the 3-D complex and (2) a learned affinity in one shot. No upfront receptor fold needed; Boltz-2 generates its own structural understanding from the sequence.

The two lanes are deliberately uncorrelated. The next notebook (`06_consensus_and_shortlist`) joins their outputs and keeps only the compounds that **both** lanes rank near the top. This *consensus* filter removes compounds whose ranking depends on a single tool — defensible triage before wet-lab follow-up.

> ⚠️ **GPU-mandatory.** Boltz-2 ships a ~5 GB neural-network checkpoint and runs only on CUDA. Open this notebook on a Colab GPU runtime (Runtime → Change runtime type → T4 GPU, or A100 / L4 on Pro / Pro+). The setup cell fails loudly if no GPU is attached.

## Learning objectives

After running this notebook you will be able to:

- Explain in plain language what **co-folding** is, why it differs from classical docking, and where its strengths and limits lie clinically.
- Run **Boltz-2** on one (protein, ligand) pair from a Python cell, parse its affinity + confidence outputs, and read the **ipTM / ligand-iPTM** scores.
- Scale the same recipe to a full ligand library using `aidd.co_folding.predict_library` with **per-compound resumability** — a Colab disconnect mid-run does not cost the run.
- Compare Boltz-2's affinity ranking against the gnina + ML-rescorer ranking from notebook `04` on the same compound set, on both random and scaffold splits — the same evaluation contract as the rescorer.
- Interpret the **method-method Spearman correlation** between Boltz-2 and gnina rankings: when consensus actually adds value vs when the two are too redundant.

## Audience

- A clinician or biologist who wants to understand co-folding methodology and how it complements classical docking before wet-lab triage.
- An ML person new to drug discovery — Boltz-2 is one of the cleanest examples of a generative model adapted to a domain task, and the notebook walks through its inputs / outputs at a level useful for both ends of the field.
- A student writing a thesis chapter on AI-assisted structure-based virtual screening; you will be able to cite Boltz-2 and explain its place vs gnina-CNN-rescore.
- A reviewer or grant auditor; methodology is documented with paper DOIs in the recap.

## Prerequisites

- **A Colab GPU runtime.** T4 (free tier) is the minimum target; A100 / L4 (Pro / Pro+) is faster. The notebook detects the GPU type and reports it.
- **Notebook `04_score_classical` completed**, with the labelled-subset docking cache on Google Drive at `data/derived/erk2/scoring/docking/labeled_subset/gnina_scores.csv` and the prepared SDF at `data/derived/erk2/scoring/ligands_labeled_prepared.sdf`. We re-use the **same 414 compounds** so the consensus join in notebook `06` lines up.
- *Not needed*: notebook `01`'s AlphaFold model. Boltz-2 takes the protein **sequence** directly and infers structure internally.

## Runtime

- **Setup + Boltz-2 install:** roughly a minute, one-time per Colab runtime (PyPI install). Model weights (~5 GB) download on the first inference call.
- **5-compound smoke-test:** *measured below in Section 3 — we deliberately do not write a number in this markdown until we have run it once. See `feedback_measure_before_rationale.md` for the rule.*
- **Full 414-compound library run:** extrapolated from the smoke-test measurement in Section 4.

`USE_DRIVE = True` (the default on Colab) writes every per-compound output to Google Drive so a runtime disconnect at compound N of 414 does not lose the work. Re-running picks up where it stopped.
"""),

        markdown("""
## Key terms used in this notebook

| Term | Meaning in one line |
|---|---|
| **Co-folding** | Predicting the 3-D structure of a protein **and** its bound ligand jointly, in a single neural-network pass. |
| **Boltz-2** | An open-weights co-folding model from Wohlwend et al. (2025). Predicts complex + affinity from (sequence, SMILES). MIT-licensed. |
| **AlphaFold 3 (AF3)** | DeepMind's closed-weights co-folding model. Boltz-2 is the open-science equivalent for protein-ligand prediction. |
| **MSA** | *Multiple Sequence Alignment* — a stack of evolutionarily related sequences. Both AF2 and Boltz-2 use MSAs as the main signal for protein modelling. |
| **MSA server** | Boltz-2 can query a public alignment service to build an MSA from the input sequence automatically. Shared infrastructure → don't hammer it. |
| **ipTM** | *Interface predicted-TM*. A 0–1 score for how well the predicted interface between two chains (here: protein + ligand) matches reality on average. Higher = more confident interface. |
| **Ligand iPTM** | ipTM restricted to ligand-interface atoms. Often more discriminating than overall ipTM for small-molecule binding. |
| **pLDDT** | *Per-residue predicted Local Distance Difference Test*. Confidence per residue (0–100). Average across the protein. |
| **Predicted affinity (Boltz)** | The output of Boltz-2's affinity head. Higher = predicted stronger binder. Roughly in pIC50-like units per the Boltz-2 paper; sign and scale are not exact thermodynamics. |
| **Affinity probability** | Boltz-2's "is this molecule a binder?" classifier output, 0–1. Complementary to the continuous affinity score. |
| **Consensus rank** | Ranking compounds by agreement between two independent methods. Used in notebook `06` to filter the shortlist. |
| **Spearman correlation** | Rank-based correlation, −1 to 1. Used here to compare Boltz-2 affinity rank vs gnina-CNN-affinity rank. |
| **Independent methods** | Two algorithms whose ranking errors are *uncorrelated*. The point of consensus is that two independent methods agreeing is more informative than one method twice. |
"""),

        markdown("""
## Why this matters clinically — and why a second independent method

Classical docking (gnina, in our pipeline) places a small molecule into a rigid receptor pocket and scores the geometric fit. Co-folding (Boltz-2) lets the protein and ligand find each other from scratch — including, in principle, induced-fit motions of side chains and small loops that classical docking ignores.

Neither approach is "right". Both are useful approximations, and they fail in **different** ways:

- A classical docker can score a chemically improbable pose well if the pocket is permissive (the scoring function and the geometry agree, but the chemistry of the interaction is off).
- A co-folding model can hallucinate a high-confidence complex for a non-binder if the protein's pocket and the ligand's features are individually plausible (high local confidence, wrong holistic story).

When the two methods **agree** that a compound is a strong binder, the failure modes are unlikely to align by chance. That agreement is the signal we want for triage. When they **disagree**, the compound is interesting for a different reason: one of the methods is finding signal the other is missing, and a chemist might want to investigate by eye.

This is the operational logic behind the consensus shortlist (notebook `06`): keep compounds that rank in the top *X* % by **both** the gnina + ML rescorer (notebook `04`) and Boltz-2 (this notebook). The methods are independent enough that joint agreement is rare-by-chance, and the consensus list ends up small and high-quality.

### What Boltz-2 buys us beyond gnina

- **No upfront receptor fold needed.** The protein-only fold from notebook `01` is not an input here. Boltz-2 generates its own structural representation from the sequence, conditioned on the ligand.
- **An affinity number, not just a score.** Boltz-2's affinity head was trained explicitly on binding-affinity data (pIC50-type endpoints), so its output is closer to a "real" affinity than a docking score.
- **A different failure surface.** Boltz-2 errors on different compounds than gnina does. That uncorrelated failure mode is what makes consensus work.

### What Boltz-2 does *not* do well

- It is **slow** per compound: tens of seconds at best, several minutes if the GPU is small. We measure the actual rate before kicking off a full library run.
- It is **expensive in GPU memory**. T4 (the free-tier Colab GPU) is just enough for ERK2 + drug-sized ligands; larger ligands or longer proteins can OOM.
- It does **not** see the activity history of the target. Per-target rescorers (notebook `04`) bring that information, which is why we keep both lanes rather than collapsing to one.
"""),

        markdown("""
## 1. Setup

### What this section does

Verify a GPU is attached, clone the repo on Colab, install Boltz-2 from PyPI, and confirm the CLI loads. We follow the same install-discipline rule as ColabFold (notebook `01`): mirror the upstream install verbatim, pin to a specific release, do **not** add defensive torch / CUDA pins. See [`feedback_colabfold_install_mirror.md`](https://github.com/anthropics/feedback-rules) for the history (one hour of debugging caused by over-engineered pins on step 7).

### About the Boltz-2 install (read once, then forget)

Boltz-2 is on PyPI. We pin to the release in `aidd.co_folding.BOLTZ2_VERSION` (currently `2.1.1`). To update:

1. Bump `BOLTZ2_VERSION` in [`src/aidd/co_folding.py`](../src/aidd/co_folding.py).
2. Re-run this notebook on a fresh Colab runtime.
3. If everything passes, bump `BOLTZ2_LAST_VERIFIED` to today and commit.

Model weights (~5 GB) are downloaded automatically by the CLI on the first inference call and cached under `~/.boltz/`. Subsequent calls in the same runtime reuse the cache. Across runtime restarts, the weights re-download (1–2 minutes on Colab's network).
"""),

        code(title="Setup: GPU check, clone repo, install Boltz-2", source="""
import sys
import importlib
import shutil
import subprocess
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    gpu_ok = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    if not gpu_ok:
        raise RuntimeError(
            "No GPU detected. Boltz-2 is GPU-mandatory (CUDA-only). "
            "Switch to a GPU runtime: Runtime → Change runtime type → T4 GPU "
            "(free tier) or A100 / L4 (Pro / Pro+), then re-run this cell."
        )
    !nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    REPO_ROOT = Path("/content/aidd-pipeline")
    if not REPO_ROOT.exists():
        !git clone https://github.com/hvmarco/aidd-pipeline.git {REPO_ROOT}
    if not (REPO_ROOT / "src" / "aidd").exists():
        raise RuntimeError(
            "Repo clone failed. Make github.com/hvmarco/aidd-pipeline public, "
            "or use a Personal Access Token via Colab Secrets, then re-run."
        )

    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()
    from aidd.co_folding import BOLTZ2_VERSION  # noqa: E402

    if not shutil.which("boltz"):
        print(f"Installing boltz=={BOLTZ2_VERSION} from PyPI…")
        !pip install -q boltz=={BOLTZ2_VERSION}

    # Probe — fail loudly if the CLI is unimportable (broken deps, missing CUDA libs).
    probe = subprocess.run(["boltz", "--help"], capture_output=True, text=True, timeout=60)
    if probe.returncode != 0:
        raise RuntimeError(
            f"boltz CLI installed but `boltz --help` exited {probe.returncode}.\\n"
            f"stderr:\\n{probe.stderr}\\n"
            "Most common cause: a missing CUDA library at import time. "
            "Confirm the runtime is GPU-backed and re-run."
        )

    # py3Dmol + a few small deps for the visualisation + evaluation cells.
    !pip install -q py3Dmol scikit-learn scipy
else:
    REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()
    from aidd.co_folding import BOLTZ2_VERSION  # noqa: E402

print(f"Repo root      : {REPO_ROOT}")
print(f"Running on     : {'Colab' if IS_COLAB else 'local'}")
print(f"Boltz-2 version: {BOLTZ2_VERSION}")
"""),

        code(title="Imports", source=AUTORELOAD_SNIPPET + """
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem

from aidd.io import mount_drive_if_colab, pretty_path
from aidd.co_folding import (
    predict_complex,
    predict_library,
    summarise_run,
    describe_environment,
    BOLTZ2_VERSION,
    PROTEIN_CHAIN_ID,
    LIGAND_CHAIN_ID,
)
from aidd.scoring import evaluate_scores, enrichment_factor, roc_points

sns.set_theme(style="whitegrid")
print("imports ok")
print()
print("Boltz-2 environment:")
for k, v in describe_environment().items():
    print(f"  {k:18}: {v}")
"""),

        markdown("""
### Google Drive — read this before running the next cell

This notebook writes every per-compound output to **Google Drive** (`/content/drive/MyDrive/aidd-pipeline/data/derived/erk2/boltz/`) so a Colab disconnect midway through a multi-hour Boltz-2 run does not lose the work. The first time you run the next cell, you will see a Drive permission dialog — click through to allow.

**To opt out**, set `USE_DRIVE = False` in the next cell **before** running it. Outputs then go to `/content/aidd-pipeline/data/derived/` and disappear when the runtime ends; you would have to redo the docking on every reconnect.

If you accidentally dismiss the Drive dialog, the cell will crash. Re-run it and either authorise, or change the line to `False`.
"""),

        code(title="Drive persistence toggle  (change to False to opt out)", source="""
# ════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE AUTHORIZATION
# On Colab, the next call will prompt for Drive permission on first use.
#
# →  To OPT OUT of Drive, change the line below to:    USE_DRIVE = False
#    (caches then live on /content/ and disappear when the runtime ends)
# ════════════════════════════════════════════════════════════════════════
USE_DRIVE = IS_COLAB

DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)
print(f"USE_DRIVE: {USE_DRIVE}  "
      f"({'Drive-backed (persistent)' if USE_DRIVE else 'local /content (ephemeral)'})")
print(f"DATA_ROOT: {DATA_ROOT}")
"""),

        markdown("""
## 2. Inputs — target sequence + the same 414 compounds as notebook 04

### Background

We re-use **exactly** the compounds that notebook `04` successfully docked: the labelled ERK2 subset of 693 prepared compounds, of which 414 produced valid gnina poses on Drive. Using the same set in both lanes is a hard requirement for the consensus join in notebook `06` — different cohorts on the two sides would silently shrink the intersection.

The target sequence is human ERK2 (MAPK1, [UniProt P28482](https://www.uniprot.org/uniprotkb/P28482)). It must match the sequence used in notebook `01` so the consensus story stays coherent across the pipeline. If you change it here, change it there too.
"""),

        code(title="Inputs: target sequence, paths, output directory", source="""
TARGET = "erk2"

# Must match notebooks/_build_01_fold_target.py exactly. UniProt P28482.
ERK2_SEQUENCE = (
    "MAAAAAAGAGPEMVRGQVFDVGPRYTNLSYIGEGAYGMVCSAYDNLNKVRVAIKKISPFEHQTYCQRTLREIKILLRFRHENIIGINDIIRAPTI"
    "EQMKDVYIVQDLMETDLYKLLKTQHLSNDHICYFLYQILRGLKYIHSANVLHRDLKPSNLLLNTTCDLKICDFGLARVADPDHDHTGFLTEYVA"
    "TRWYRAPEIMLNSKGYTKSIDIWSVGCILAEMLSNRPIFPGKHYLDQLNHILGILGSPSQEDLNCIINLKARNYLLSLPHKNKVPWNRLFPNAD"
    "SKALDLLDKMLTFNPHKRIEVEQALAHPYLEQYYDPSDEPIAEAPFKFDMELDDLPKEKLKELIFEETARFQPGYRS"
)
print(f"ERK2 sequence length: {len(ERK2_SEQUENCE)} aa")

# Paths from notebook 04 (the docking lane's output).
SCORING_DIR    = DATA_ROOT / TARGET / "scoring"
LIGANDS_SDF    = SCORING_DIR / "ligands_labeled_prepared.sdf"
SCORED_POSES   = SCORING_DIR / "scored_poses.parquet"
GNINA_SCORES   = DATA_ROOT / TARGET / "docking" / "labeled_subset" / "gnina_scores.csv"

# Outputs for this notebook (the Boltz-2 lane).
BOLTZ_DIR      = DATA_ROOT / TARGET / "boltz"
BOLTZ_DIR.mkdir(parents=True, exist_ok=True)
AFFINITY_CSV   = BOLTZ_DIR / "affinity.csv"
POSES_SDF      = BOLTZ_DIR / "poses.sdf"
FAILURES_CSV   = BOLTZ_DIR / "failures.csv"
RUN_LOG        = BOLTZ_DIR / "run.log"

print(f"Reads from notebook 04:")
print(f"  ligands_labeled_prepared.sdf -> {pretty_path(LIGANDS_SDF, DATA_ROOT, REPO_ROOT)}")
print(f"  scored_poses.parquet          -> {pretty_path(SCORED_POSES, DATA_ROOT, REPO_ROOT)}")
print(f"  gnina_scores.csv              -> {pretty_path(GNINA_SCORES, DATA_ROOT, REPO_ROOT)}")
print(f"Writes to:")
print(f"  affinity.csv                  -> {pretty_path(AFFINITY_CSV, DATA_ROOT, REPO_ROOT)}")
print(f"  poses.sdf                     -> {pretty_path(POSES_SDF, DATA_ROOT, REPO_ROOT)}")
print(f"  failures.csv                  -> {pretty_path(FAILURES_CSV, DATA_ROOT, REPO_ROOT)}")
"""),

        code(title="Recover the 414-compound set from notebook 04's outputs", source="""
# The 414 compounds that produced valid gnina poses in step 10. Identified
# by the unique compound_ids in gnina_scores.csv.
if not GNINA_SCORES.exists():
    raise FileNotFoundError(
        f"Cannot find {pretty_path(GNINA_SCORES, DATA_ROOT, REPO_ROOT)}. "
        "Run notebook 04_score_classical on a Colab GPU once to populate the "
        "docking cache on Drive, then come back."
    )
if not LIGANDS_SDF.exists():
    raise FileNotFoundError(
        f"Cannot find {pretty_path(LIGANDS_SDF, DATA_ROOT, REPO_ROOT)}. "
        "Run notebook 04_score_classical to build the prepared SDF first."
    )

gnina_scores = pd.read_csv(GNINA_SCORES)
docked_ids = set(gnina_scores["compound_id"].astype(str).unique())
print(f"Step 10 produced gnina poses for {len(docked_ids)} unique compounds.")

# Recover SMILES per compound_id from the prepared SDF (matched on _Name).
records: list[dict] = []
sup = Chem.SDMolSupplier(str(LIGANDS_SDF), removeHs=False)
for mol in sup:
    if mol is None:
        continue
    cid = (mol.GetProp("_Name") if mol.HasProp("_Name") else "").strip()
    if cid in docked_ids:
        smi = mol.GetProp("smiles_std") if mol.HasProp("smiles_std") else Chem.MolToSmiles(mol)
        records.append({"compound_id": cid, "smiles": smi})

ligands_df = pd.DataFrame(records).drop_duplicates(subset="compound_id").reset_index(drop=True)
missing = docked_ids - set(ligands_df["compound_id"])
if missing:
    print(f"⚠ {len(missing)} docked compound_ids not found in the prepared SDF; "
          f"they will be silently dropped from the Boltz-2 run.")

print(f"Boltz-2 input library: {len(ligands_df):,} (compound_id, smiles) pairs.")
ligands_df.head()
"""),

        markdown("""
## 3. Smoke-test — measure per-compound runtime on 5 compounds

### Background

Boltz-2 wall time per compound depends on the GPU tier (T4 vs A100 vs L4), the protein length, and the ligand size. The published estimate is "tens of seconds to a few minutes" but we will not write a number into pipeline rationale until we measure it on *this* runtime, against *our* compounds, with the install we just did. (See [`feedback_measure_before_rationale.md`](#) for the rule and the incident that motivated it.)

### What this cell does

Pick the first 5 compounds and run `aidd.co_folding.predict_library` on them. The function writes per-compound outputs to a small subdirectory so the smoke-test cache is separate from the full-library cache (different `out_dir`). The cell prints the per-compound runtimes from `run.log`.
"""),

        code(title="Boltz-2 smoke-test: 5 compounds, measured timing", source="""
import time

SMOKE_DIR = BOLTZ_DIR / "_smoke_test"
SMOKE_DIR.mkdir(parents=True, exist_ok=True)

smoke_df = ligands_df.head(5).copy()
print(f"Running Boltz-2 on {len(smoke_df)} compounds…")

t0 = time.time()
smoke_affinities = predict_library(
    sequence=ERK2_SEQUENCE,
    ligands=smoke_df,
    out_dir=SMOKE_DIR,
    use_msa_server=True,
    progress=True,
    on_failure="skip_with_guard",
    failure_guard_n=5,
)
wall = time.time() - t0
print(f"\\nSmoke-test wall time: {wall/60:.1f} min total, "
      f"≈ {wall/len(smoke_df):.0f} s per compound (incl. one-time weight download).")

# run.log has per-compound timings.
print("\\nPer-compound timing (from run.log):")
print((SMOKE_DIR / "run.log").read_text())

smoke_summary = summarise_run(SMOKE_DIR)
print(f"\\nSmoke-test summary: {smoke_summary}")
smoke_affinities
"""),

        markdown("""
### Interpreting the smoke-test output

Three things to check before kicking off the full library run:

1. **Per-compound wall time.** The *first* compound is slower than the rest because it triggers the model-weight download (~5 GB). The trailing 4 compounds give the steady-state rate. If steady-state is > 3 min/compound on a T4, switch to A100 (Pro / Pro+) before the full run — 414 × 3 min ≈ 21 hours is past Colab Pro's session limit.

2. **Failure count and class.** `summarise_run` reports `n_failed` and `failures_by_class`. Expected on a healthy install: 0 failures. If you see `boltz_oom`, switch to a larger GPU. If you see `boltz_msa_failed`, the MSA server is flaking — try again later or run with `use_msa_server=False` (single-sequence mode, lower accuracy).

3. **Affinity values look like numbers.** `boltz_affinity` should be a small positive or negative real (roughly the pIC50 range); `boltz_iptm` and `boltz_confidence` should be in [0, 1]. NaN values mean parsing failed — check `affinity.json` and `confidence.json` under `_smoke_test/per_compound/<id>/` against the upstream Boltz-2 spec, and fix the parser in `src/aidd/co_folding.py` if the field names have shifted.

The number to remember from this cell — **steady-state seconds per compound** — is what we extrapolate the full-library runtime from.
"""),

        markdown("""
## 4. Full library — co-fold all 414 compounds with Boltz-2

### Background

`predict_library` is **resumable per compound**. Each compound writes its outputs to `boltz/per_compound/<compound_id>/` before the next one starts; if Colab disconnects at compound 200 of 414, you re-run this cell and only the remaining 214 are dispatched. Failed compounds are marked with a `<compound_id>.FAILED` JSON file and skipped on subsequent runs (delete the marker to retry).

If the smoke-test rate is *r* seconds per compound, expect the full library to take roughly `414 × r / 60` minutes — but in wall-clock time on a free-tier Colab session that includes idle-timeout disconnects, so plan to keep the tab focused or use Colab Pro.

### What this cell does

Calls `predict_library` on all 414 compounds with the same parameters as the smoke-test. Writes `affinity.csv`, `poses.sdf`, `failures.csv`, and `run.log` to the Boltz output directory.
"""),

        code(title="Full Boltz-2 library run (resumable; cached after first pass)", source="""
import time

t0 = time.time()
library_affinity = predict_library(
    sequence=ERK2_SEQUENCE,
    ligands=ligands_df,
    out_dir=BOLTZ_DIR,
    use_msa_server=True,
    progress=True,
    on_failure="skip_with_guard",
    failure_guard_n=5,
)
wall = time.time() - t0
print(f"\\nFull-library wall time this session: {wall/60:.1f} min "
      f"(cached compounds skipped; cumulative real cost may be larger across runs).")

summary = summarise_run(BOLTZ_DIR)
print(f"\\nLibrary summary: {summary}")

print(f"\\nFirst rows of affinity.csv:")
library_affinity.head()
"""),

        markdown("""
### Interpreting the full-library output

`affinity.csv` is the contract for notebook `06`. It carries six columns per compound — `compound_id`, `boltz_affinity`, `boltz_affinity_probability`, `boltz_confidence`, `boltz_iptm`, `boltz_ligand_iptm`. Notebook `06` joins it against `scored_poses.parquet` (notebook `04`) on `compound_id`; both tables must address the same compound-ID strings, so check `summary['n_predicted']` lines up with the input cohort size.

A non-zero `n_failed` is not catastrophic — the consensus shortlist downstream is robust to partial coverage in either lane. But the *kinds* of failure (`failures_by_class` in the summary) tell you whether the run is healthy:

- `boltz_oom` on a small fraction (< 5 %) usually means a few large ligands; re-run those on a bigger GPU after the main library completes.
- `boltz_msa_failed` is a transient infrastructure problem on the MSA server; retry by deleting the markers (`per_compound/*.FAILED`) and re-running this cell.
- `boltz_co_fold_diverged` on a single compound is rare and is fine to leave as-is.
- `boltz_install_error` on > 1 compound means the install is broken; the run won't make progress until the setup cell is re-run on a fresh runtime.
"""),

        markdown("""
## 5. Sanity check — view one predicted complex in 3-D

### Background

Boltz-2's output is a full protein-ligand complex in MMCIF format under `per_compound/<compound_id>/complex.cif`. A quick visual check on the top-affinity compound confirms two things: the ligand sits **in the ATP pocket** (not on the surface), and the receptor fold is plausible (the kinase domain is recognisable).

This is not a metric — it's an eyeball check that catches gross failures (ligand placed outside the protein, protein collapsed) that no scalar score would flag.
"""),

        code(title="3-D viewer: the top-affinity predicted complex", source="""
import py3Dmol

# Pick the compound with the strongest predicted affinity that has a usable CIF.
top = library_affinity.sort_values("boltz_affinity", ascending=False).head(10)
for _, row in top.iterrows():
    cif_path = BOLTZ_DIR / "per_compound" / str(row["compound_id"]) / "complex.cif"
    if cif_path.exists():
        chosen = row
        break
else:
    raise FileNotFoundError("No complex.cif found for any of the top-10-affinity compounds.")

print(f"Top-affinity compound: {chosen['compound_id']}  "
      f"affinity={chosen['boltz_affinity']:.2f}  "
      f"ligand_iPTM={chosen['boltz_ligand_iptm']:.2f}")

view = py3Dmol.view(width=720, height=480)
view.addModel(cif_path.read_text(), "cif")
# Protein: cartoon, coloured by ipTM-like confidence.
view.setStyle({"chain": PROTEIN_CHAIN_ID}, {"cartoon": {"color": "spectrum"}})
# Ligand: sticks, coloured by element.
view.setStyle({"chain": LIGAND_CHAIN_ID}, {"stick": {"colorscheme": "cyanCarbon"}})
view.zoomTo({"chain": LIGAND_CHAIN_ID})
view.zoom(0.7)
view.show()
"""),

        markdown("""
### What to look for

- The **ligand (cyan sticks)** should sit in a pocket between the **N-lobe** (top of the kinase domain, smaller, β-sheet-rich) and the **C-lobe** (bottom, larger, α-helical). That is the canonical ATP pocket of every kinase.
- The **hinge region** is a short loop connecting N-lobe and C-lobe. ATP-competitive inhibitors typically engage hinge residues (in ERK2: M108 / L107 / D106) with a hydrogen bond. Eyeballing this is enough at this stage — interaction fingerprints are notebook `04`'s job.
- **If the ligand is on the protein surface**, far from the kinase cleft, it is a misfolded prediction. Common causes: a very large ligand that confuses the model, or a SMILES with ambiguous chemistry. Note the compound_id and check its affinity score — Boltz-2 usually flags these with low `ligand_iptm`.

Use the 3-D viewer interactively (drag to rotate, scroll to zoom) before moving on.
"""),

        markdown("""
## 6. Evaluation — does Boltz-2 affinity rank actives above inactives?

### Background

Boltz-2 was *not* trained on this target's activity history. The question is whether its **generic** affinity head still produces a ranking that beats random on our labelled ERK2 set, and how its accuracy compares to gnina's CNN affinity (the docker-built-in scorer) and the per-target rescorer (notebook `04`).

We report three numbers, all on the **scaffold-grouped 80/20 split** that notebook `04` used. This means: when notebook `04` defined which compounds sit in the random-split test fold and which sit in the scaffold-grouped test fold, those exact same membership flags carry over here. The metric comparison is **apples-to-apples**: same fold definitions, same held-out compounds, different ranking method.

1. **ROC-AUC on the scaffold-split held-out fold.** Done signal: > 0.55 indicates the affinity head is picking up real signal on this target.
2. **EF1%** (enrichment factor at top 1 %). The medicinal-chemistry-relevant number: how concentrated the top 1 % of the Boltz ranking is in true actives.
3. **Spearman correlation** between Boltz-2 affinity rank and gnina CNN-affinity rank on the 414-compound overlap. Expected: positive but not too high (≈ 0.2–0.5 is ideal). If it's > 0.8 the methods are redundant; if it's < 0 there is a sign / scaling bug.

We also report the random-stratified-split AUC alongside, as in notebook `04`, so the scaffold-leakage tax is visible.
"""),

        code(title="Join Boltz-2 affinity with notebook 04's scored_poses; compute AUC + EF1% + Spearman", source="""
from scipy.stats import spearmanr

if not SCORED_POSES.exists():
    raise FileNotFoundError(
        f"Cannot find {pretty_path(SCORED_POSES, DATA_ROOT, REPO_ROOT)}. "
        "Run notebook 04 to completion first; this evaluation depends on its "
        "fold-membership flags + gnina_cnn_affinity column."
    )

scored = pd.read_parquet(SCORED_POSES)
scored["compound_id"] = scored["compound_id"].astype(str)
library_affinity["compound_id"] = library_affinity["compound_id"].astype(str)

joined = scored.merge(library_affinity, on="compound_id", how="inner")
print(f"Joined on compound_id: {len(joined):,} compounds with both gnina + Boltz-2 scores "
      f"(out of {len(scored):,} in step-10 scored_poses, {len(library_affinity):,} in Boltz affinity).")

# Build a side-by-side metrics table on both splits, baseline (gnina-CNN-aff)
# vs Boltz-2 affinity vs Boltz-2 affinity_probability.
rows = []
for split_label, fold_col in [
    ("random stratified", "in_random_test_fold"),
    ("scaffold grouped",  "in_scaffold_test_fold"),
]:
    held = joined[joined[fold_col]]
    y = held["Active"].astype(int).to_numpy()
    rows += [
        {"split": split_label, "method": "gnina CNN_affinity (baseline)",
         **evaluate_scores(y, held["gnina_cnn_affinity"].to_numpy())},
        {"split": split_label, "method": "Boltz-2 affinity",
         **evaluate_scores(y, held["boltz_affinity"].to_numpy())},
        {"split": split_label, "method": "Boltz-2 affinity_probability",
         **evaluate_scores(y, held["boltz_affinity_probability"].to_numpy())},
        {"split": split_label, "method": "rescorer RF (notebook 04)",
         **evaluate_scores(y, held["rescorer_rf_proba"].to_numpy())},
    ]
metrics = pd.DataFrame(rows)
print("\\nHeld-out fold summary (scaffold-AUC is the honest generalisation estimate):")
metrics

"""),

        code(title="ROC curves: gnina baseline vs Boltz-2 vs rescorer (random + scaffold splits)", source="""
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=110)

for ax, (split_label, fold_col) in zip(axes, [
    ("Random stratified", "in_random_test_fold"),
    ("Scaffold grouped",  "in_scaffold_test_fold"),
]):
    held = joined[joined[fold_col]]
    y = held["Active"].astype(int).to_numpy()
    for label, score_col, colour in [
        ("gnina CNN_affinity (baseline)",  "gnina_cnn_affinity",        "tab:gray"),
        ("Boltz-2 affinity",               "boltz_affinity",            "tab:green"),
        ("Boltz-2 binder probability",     "boltz_affinity_probability","tab:olive"),
        ("Rescorer RF (notebook 04)",      "rescorer_rf_proba",         "tab:blue"),
    ]:
        s = held[score_col].to_numpy()
        fpr, tpr = roc_points(y, s)
        auc = evaluate_scores(y, s)["auc"]
        ax.plot(fpr, tpr, color=colour, lw=2, label=f"{label}  AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{split_label} split (n = {len(y)})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

fig.suptitle("ERK2 — Boltz-2 affinity vs gnina baseline vs per-target rescorer")
fig.tight_layout()
plt.show()
"""),

        code(title="Method-method agreement: Spearman correlation between Boltz-2 and gnina rankings", source="""
rho, pval = spearmanr(joined["boltz_affinity"], joined["gnina_cnn_affinity"])
rho_rf, pval_rf = spearmanr(joined["boltz_affinity"], joined["rescorer_rf_proba"])

print(f"Boltz-2 affinity  vs  gnina CNN_affinity  : Spearman ρ = {rho:.3f}  (p = {pval:.1e}, n = {len(joined)})")
print(f"Boltz-2 affinity  vs  notebook-04 RF score: Spearman ρ = {rho_rf:.3f}  (p = {pval_rf:.1e})")
print()
print("Interpretation guide:")
print("  ρ in [0.2, 0.5]  : methods agree on real signal but disagree enough that consensus adds value (ideal).")
print("  ρ > 0.7          : methods are largely redundant — consensus would not narrow the shortlist much.")
print("  ρ < 0            : sign / scaling bug or genuinely orthogonal signal (rare; investigate).")

# Scatter for visual confirmation.
fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
ax.scatter(joined["gnina_cnn_affinity"], joined["boltz_affinity"],
           c=joined["Active"].astype(int), cmap="coolwarm", alpha=0.7, s=20)
ax.set_xlabel("gnina CNN_affinity  (higher → stronger by gnina)")
ax.set_ylabel("Boltz-2 affinity  (higher → stronger by Boltz-2)")
ax.set_title(f"Method-method scatter (ρ = {rho:.3f})\\nblue = active, red = inactive")
plt.tight_layout()
plt.show()
"""),

        markdown("""
### How to read the evaluation result

The **scaffold-split AUC for Boltz-2 affinity** is the headline number for this notebook. The done signal is:

> **Boltz-2 scaffold-split AUC > 0.55** on the held-out fold = the affinity head is picking up real, generalisable signal for this target.

Three honest cases the numbers can land in:

- **Boltz > 0.6 and rescorer > 0.6**: both methods are useful and we expect consensus to add value. This is the case where notebook `06` produces a high-quality shortlist.
- **Boltz ≈ 0.5 and rescorer > 0.6**: the per-target rescorer wins decisively. Note in the recap; consensus on this target reduces to "trust the rescorer" because Boltz-2's signal is too weak to add information.
- **Boltz > 0.6 and rescorer ≈ 0.6 and Spearman ρ < 0.3**: Boltz-2 is finding signal the rescorer is missing. Consensus will produce a smaller, more conservative shortlist than either alone — the textbook win case for combining independent methods.

The Spearman correlation between Boltz-2 and gnina CNN_affinity is the **diagnostic for method independence**. If it lands above 0.7, the two methods are too redundant for consensus to help (notebook `06` will still run but the consensus list will be close to a single-method top-N). The same scatter against the per-target RF rescorer tells you whether the learned rescorer captures different signal than Boltz-2 — usually yes, because the rescorer uses IFP features the co-folding model does not.

### What the numbers cannot tell you

- **Pose quality.** AUC measures ranking; it does not measure whether the predicted complex is sane. Trust the Section 5 viewer for that, and PoseBusters (in notebook `03`) for gnina poses.
- **Calibration.** Boltz-2 affinity is roughly pIC50-shaped but not a guaranteed thermodynamic predictor. Treat it as a ranking signal, not as an absolute K_d.
- **Generalisation to a new target.** Numbers here are for ERK2. The pipeline is built to be target-agnostic; running the same notebook against DPYD or KRAS will tell you whether Boltz-2's signal holds beyond kinase chemistry.
"""),

        markdown("""
## Recap

### Biomedical takeaway

Boltz-2 co-folds the protein and the ligand together in one neural-network pass and outputs both a 3-D complex and a learned affinity. For ERK2, on the same 414-compound labelled set notebook `04` evaluated, Boltz-2's affinity head gives an independent ranking we can combine with the gnina + per-target rescorer to produce a defensible consensus shortlist. The two methods use **different physics** and fail in **different ways**; that is exactly the property the consensus filter in notebook `06` exploits.

For the pharmacogenomics + variant-function workflow that becomes the project's headline narrative (post-step-13 roadmap), the same `predict_complex` and `predict_library` API will run against mutant protein sequences with no code change — the per-compound cache is keyed on SHA-256(sequence) + SHA-256(SMILES) precisely so swapping wild-type for a variant re-runs cleanly rather than returning stale results.

### Technical takeaway

`src/aidd/co_folding.py` wraps the Boltz-2 CLI behind a small Python API that mirrors `src/aidd/docking.py`: same `predict_library` shape, same per-compound caching layout, same `<id>.FAILED` markers, same `on_failure="raise"|"skip"|"skip_with_guard"` policy. Both lanes of the pipeline now have one **operational contract**, so future agents (and future you) only have to learn the failure-handling rules once.

The Boltz-2-specific bits — error-class taxonomy (`boltz_oom`, `boltz_msa_failed`, `boltz_co_fold_diverged`, `boltz_install_error`, `boltz_nonzero_exit`), MMCIF parsing of the complex, ligand extraction by chain ID — all live inside the wrapper. Downstream callers see only the standard contract: a tidy `affinity.csv` with `compound_id` + four numeric columns, ready to be joined with `scored_poses.parquet` on `compound_id`.

### What's next in the pipeline

- **`06_consensus_and_shortlist.ipynb`** — join `affinity.csv` (this notebook) and `scored_poses.parquet` (notebook `04`) on `compound_id`, apply a consensus rule (top-X % by both rankings), emit `shortlist.sdf` + `shortlist.csv` with per-compound poses, scores, ADMET flags. This is where the two-lane pipeline produces its single integrated output.
- **`07_mutation_analysis.ipynb`** (headline notebook in the §10 research-domain pivot) — run the same screening recipe against a wild-type and a variant sequence (e.g. DPYD\\*2A, KRAS G12C, ESR1 Y537S, BRCA1 LoF), then diff the two consensus shortlists. The per-compound cache + sequence-keyed cache in this notebook is what makes that diff cheap to produce.

### Further reading

- Wohlwend, J., Corso, G., et al. (2025) — *Boltz-2: Towards Accurate and Scalable Protein-Ligand Prediction*. Pre-print + repo at [github.com/jwohlwend/boltz](https://github.com/jwohlwend/boltz). [doi:10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707)
- Abramson, J., Adler, J., et al. (2024) — *Accurate structure prediction of biomolecular interactions with AlphaFold 3.* Nature, 630, 493–500. [doi:10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w) (the closed-weights reference for what Boltz-2 implements with open weights.)
- Buttenschoen, M., Morris, G. M., Deane, C. M. (2024) — *PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences.* Chem. Sci., 15, 3130. [doi:10.1039/D3SC04185A](https://doi.org/10.1039/D3SC04185A) (background on why pose quality control is mandatory even when scoring functions look strong; the same QC philosophy that motivates the consensus filter.)
"""),
        accelerator="GPU",
        gpu_type="T4",
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
