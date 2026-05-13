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
- **3-compound smoke-test:** *measured below in Section 5 — we deliberately do not write a number in this markdown until we have run it once. See `feedback_measure_before_rationale.md` for the rule.*
- **Full 414-compound library run:** extrapolated from the smoke-test measurement in Section 6, using a pre-cached MSA from Section 3 so per-compound time is dominated by inference rather than MSA-server queries.

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

### About the Boltz-2 install — mirror upstream, restart the kernel once

The install command is taken from [sokrypton's Boltz1.ipynb](https://github.com/sokrypton/ColabFold/blob/main/Boltz1.ipynb) — the community-vetted Colab pattern for Boltz on T4 / A100 / L4 runtimes:

```
pip install -q --no-warn-conflicts boltz
```

We add a pinned version (`==2.2.1`) so the cached outputs of a screen are reproducible against a specific Boltz release. Two intentional differences from Boltz's own README (`pip install boltz[cuda] -U`):

- **No `[cuda]` extra.** Colab's image already ships a CUDA-pinned torch wheel; the `[cuda]` extra would force a redundant torch reinstall that adds version-skew risk (we hit exactly this on attempts 1–2 of this notebook). Bare `boltz` lets pip reuse Colab's torch.
- **`--no-warn-conflicts`.** Suppresses the cosmetic "pip's dependency resolver does not currently take into account…" warnings about Colab's unrelated packages wanting `numpy>=2`. Those warnings have always been benign (none of the conflicting packages are imported by our notebook) but they're scary to read mid-install.

To bump the pin: edit `BOLTZ2_VERSION` in [`src/aidd/co_folding.py`](../src/aidd/co_folding.py), bump `BOLTZ_VERSION_PIN` in the setup cell to match, delete `/content/_aidd_boltz_setup_done`, re-run on a fresh Colab runtime, and on success bump `BOLTZ2_LAST_VERIFIED` to today and commit.

#### Per-package audit — what comes from where

| Package | Boltz 2.2.1 pin (from [pyproject.toml](https://github.com/jwohlwend/boltz/blob/main/pyproject.toml)) | Notebook needs it? | Source |
|---|---|---|---|
| `torch` | `>=2.2` | yes (Boltz inference) | **Colab pre-installed** (CUDA wheel) |
| `numpy` | `>=1.26,<2.0` | yes | Boltz transitive (downgrades Colab default) |
| `scipy` | `==1.13.1` | yes (`scipy.stats.spearmanr` in evaluation cell) | Boltz transitive |
| `scikit-learn` | `==1.6.1` | yes (`aidd.scoring.evaluate_scores` uses it) | Boltz transitive |
| `pandas` | `>=2.2.2` | yes | Boltz transitive |
| `rdkit` | `>=2024.3.2` | yes (`Chem.SDMolSupplier`) | Boltz transitive |
| `biopython` | `==1.84` | no (not used in this notebook) | Boltz transitive |
| `cuequivariance-torch` | not a Boltz dep | yes on Ampere+ GPUs (A100/H100) | **Explicit install** |
| `cuequivariance-ops-torch-cu12` | not a Boltz dep | yes on Ampere+ GPUs — the actual CUDA-12 kernel binaries | **Explicit install** |
| `colabfold` | not a Boltz dep | yes (Section 3 MSA generator, only on cache miss) | **Colab pre-installed via setup cell's transitive deps** |
| `py3Dmol` | not a Boltz dep | yes (Section 7 sanity-check viewer) | **Explicit install** |
| `matplotlib` | not a Boltz dep | yes (ROC + scatter plots) | **Explicit install** |
| `seaborn` | not a Boltz dep | yes (theme; one `sns.set_theme()` call) | **Explicit install** |

### Why cuEquivariance is an explicit install

Boltz-2's README states: "On recent NVIDIA GPUs, Boltz leverages the acceleration provided by NVIDIA cuEquivariance kernels." On Ampere+ GPUs (A100, H100) Boltz late-imports `cuequivariance_torch` inside its triangular-multiplicative-update layer when `use_kernels=True`, which is the **default** on those architectures. The dependency is **not** in Boltz's `pyproject.toml` — they expect users with A100/H100 hardware to install it explicitly. There is no graceful fallback path: missing the package crashes the warm-up cell with `ModuleNotFoundError: No module named 'cuequivariance_torch'`.

Two PyPI packages needed:
- **`cuequivariance-torch`** — core Python package (hyphens-vs-underscores: the `cuequivariance_torch` import name maps to this distribution).
- **`cuequivariance-ops-torch-cu12`** — the compiled CUDA-12 kernel binaries. This is where the actual GPU acceleration lives. If Colab moves to CUDA 13, bump the suffix to `-cu13`; the core package has no CUDA-version suffix.

On T4 (Turing) Boltz uses a different code path that does not invoke these kernels; the install is harmless on T4 (the packages just sit unused). So we install unconditionally regardless of GPU tier — keeps the setup cell simple.

That's the principle: install Boltz with `[cuda]`, **do not** layer `numpy / scipy / scikit-learn / rdkit / pandas` defensively (Boltz pins them at exact versions), but **do** explicitly install `py3Dmol / matplotlib / seaborn` because they are notebook-only plotting deps not in Boltz's tree. The two failed Colab attempts on this notebook both stemmed from violating the first half of this principle.

#### Why the kernel restart

Boltz-2's `numpy>=1.26,<2.0` and `scipy==1.13.1` and `scikit-learn==1.6.1` pins **downgrade** Colab's pre-installed versions (Colab's defaults drift forward; Boltz pins backward to its tested set). The downgrade happens after the running Python kernel has already imported the newer numpy/scipy/sklearn, leaving the kernel in a half-bumped state where any later `import sklearn` fails with cryptic ABI errors (the symptom Natallia hit on attempt 2: `No module named 'numpy.char'` inside scipy's array-API shim).

The fix is to kill the Python kernel right after the install. The runtime stays warm; only Python restarts. A `/content/_aidd_boltz_setup_done` sentinel file makes the second pass of the cell skip the install and finish with the GPU + CLI probe.

#### Flow when you run this cell

1. **Run the setup cell once.** It installs `boltz==2.2.1 cuequivariance-torch cuequivariance-ops-torch-cu12 py3Dmol matplotlib seaborn` with `--no-warn-conflicts`, drops the sentinel, and kills the Python kernel.
2. **Wait ~10 s for Colab to reconnect.** The runtime stays warm; the Python process restarts.
3. **Re-run the setup cell.** The sentinel is present; the cell skips the install and finishes with the GPU + CLI probe + version summary.

#### Expected pass-1 output (first run, before kernel restart)

```
Tesla T4, 15360 MiB                        # or "NVIDIA A100-SXM4-40GB, 40960 MiB" on Pro+
Cloning into '/content/aidd-pipeline'...
...
Receiving objects: 100% (...), done.
Installing boltz==2.2.1 + cuequivariance + py3Dmol + matplotlib + seaborn… (~3-5 min on Colab)

Install complete. Restarting the Python kernel so the newly
installed numpy / scipy / scikit-learn versions take effect.

→  Once the kernel comes back up, RE-RUN THIS CELL to finish setup.
   (Subsequent runs are fast: the sentinel file skips the install.)
```

Then the kernel dies (Colab UI shows "Your session crashed" or "Runtime disconnected" briefly).

#### Expected pass-2 output (after kernel restart, re-run the cell)

```
Tesla T4, 15360 MiB
                                            # no clone print — directory exists
                                            # no install print — sentinel exists
Repo root      : /content/aidd-pipeline
Running on     : Colab
Boltz-2 version: 2.2.1
```

If you see `ModuleNotFoundError` on `matplotlib` or `seaborn` after pass 2, Colab's defaults have changed — paste the traceback in chat and we'll widen the install line. (Low risk; both packages have been Colab defaults for years.)

Model weights (~GB-scale) and the CCD ligand-chemistry dataset are downloaded by the CLI on the **first** `boltz predict` call, not at install time. The next section (Section 2) explicitly warms that download up so the runtime measurements in Section 5 are clean.
"""),

        code(title="Setup: GPU check, clone repo, install Boltz-2 (kernel restarts once)", source="""
import sys
import os
import importlib
import shutil
import subprocess
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

# Must match BOLTZ2_VERSION in src/aidd/co_folding.py. Hardcoded here so the
# install can happen before we can import the module (the module needs the
# install to have happened first -- chicken and egg). Keep in sync.
BOLTZ_VERSION_PIN = "2.2.1"

# Sentinel file on /content. Persists for the lifetime of the runtime but
# does not survive a hard-disconnect; that is the right scope -- the install
# IS valid for the lifetime of the runtime.
SETUP_SENTINEL = Path("/content/_aidd_boltz_setup_done")

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

    if not SETUP_SENTINEL.exists():
        # Mirror sokrypton's Boltz Colab pattern (no [cuda] extra,
        # --no-warn-conflicts), plus cuequivariance-* for the Ampere+
        # kernel path Boltz uses on A100/H100 (see the markdown audit
        # table above for the full rationale). Single line for shell
        # compatibility -- backslash continuation in Jupyter `!` cells
        # is fragile, so we keep it flat. Boltz brings numpy / scipy /
        # scikit-learn / rdkit / pandas at exact version pins; DO NOT
        # add them to this install line.
        print(f"Installing boltz=={BOLTZ_VERSION_PIN} + cuequivariance + py3Dmol + matplotlib + seaborn… "
              f"(~3-5 min on Colab)")
        !pip install -q --no-warn-conflicts "boltz=={BOLTZ_VERSION_PIN}" cuequivariance-torch cuequivariance-ops-torch-cu12 py3Dmol matplotlib seaborn
        SETUP_SENTINEL.touch()
        print()
        print("Install complete. Restarting the Python kernel so the newly")
        print("installed numpy / scipy / scikit-learn versions take effect.")
        print()
        print("→  Once the kernel comes back up, RE-RUN THIS CELL to finish setup.")
        print("   (Subsequent runs are fast: the sentinel file skips the install.)")
        os.kill(os.getpid(), 9)

    # Post-install path: the install has happened, the kernel has restarted,
    # and we are running this cell for the second time. Finish the setup.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()
    from aidd.co_folding import BOLTZ2_VERSION  # noqa: E402

    if BOLTZ2_VERSION != BOLTZ_VERSION_PIN:
        print(f"⚠  BOLTZ2_VERSION in aidd.co_folding is {BOLTZ2_VERSION!r}, but the "
              f"install above pinned {BOLTZ_VERSION_PIN!r}. Bump one to match the "
              f"other and re-run on a fresh runtime.")

    probe = subprocess.run(["boltz", "--help"], capture_output=True, text=True, timeout=60)
    if probe.returncode != 0:
        raise RuntimeError(
            f"boltz CLI installed but `boltz --help` exited {probe.returncode}.\\n"
            f"stderr:\\n{probe.stderr}\\n"
            "Most common cause: a missing CUDA library at import time. "
            "Confirm the runtime is GPU-backed and re-run."
        )
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
## 2. Warm up Boltz-2 (downloads weights + CCD on first run)

### Background

The Boltz-2 CLI fetches its model weights (~GB-scale) and the [CCD](https://www.wwpdb.org/data/ccd) ligand-chemistry dataset the **first** time `boltz predict` runs in a session. The download is not part of the install — it happens at first inference. Two consequences if we don't surface it explicitly:

- The first compound of any Colab run will be **much slower than steady-state**, and the per-compound runtime in Section 5's smoke-test would be skewed by the weight download.
- If the download fails (network blip, mirror down), it would fail mid-way through Section 5 rather than at a visible setup step.

So we warm Boltz up here with a tiny throwaway prediction — a 5-residue peptide + a small ligand — that triggers the weight download in **single-sequence mode** (no MSA server query, just to keep this step fast and deterministic). On a healthy runtime this takes a few minutes the first time and a couple of seconds on every subsequent Colab session. Output goes to `/content/_boltz_warmup/`, never to Drive.

This pattern follows the same logic as our resumable-docking design: surface expensive one-time setup as its own visible step so the user sees it complete before kicking off the long loop. The Boltz community's own example notebooks (e.g. AtharvaTilewale's) use a comparable warm-up cell.
"""),

        code(title="Warm up Boltz-2: tiny prediction triggers weight + CCD download", source="""
import time
from pathlib import Path

# Tiny inputs: a 5-residue peptide and the simplest aromatic SMILES.
# Goal is end-to-end protocol coverage, NOT a meaningful biological result.
WARMUP_SEQUENCE = "ACDEF"        # 5 amino acids — alanine, cysteine, aspartate, glutamate, phenylalanine
WARMUP_SMILES   = "c1ccccc1"     # benzene
WARMUP_DIR      = Path("/content/_boltz_warmup")

if WARMUP_DIR.exists() and (WARMUP_DIR / "metadata.json").exists():
    print(f"Warm-up cache present at {WARMUP_DIR}; skipping re-run.")
else:
    print("Warming up Boltz-2 (downloads model weights + CCD on the first call)…")
    t0 = time.time()
    result = predict_complex(
        sequence=WARMUP_SEQUENCE,
        smiles=WARMUP_SMILES,
        output_dir=WARMUP_DIR,
        compound_id="warmup",
        use_msa_server=False,  # single-sequence mode -- faster, no MSA-server traffic
        cache=False,
    )
    wall = time.time() - t0
    print(f"\\nWarm-up complete in {wall/60:.1f} min ({wall:.0f} s).")
    print(f"  affinity            : {result.affinity:.3f}")
    print(f"  affinity_probability: {result.affinity_probability:.3f}")
    print(f"  confidence          : {result.confidence:.3f}")
    print(f"  iptm                : {result.iptm:.3f}")
    print(f"  ligand_iptm         : {result.ligand_iptm:.3f}")
"""),

        markdown("""
### Interpreting the warm-up output

Two things to confirm before continuing:

1. **The cell completed.** Weight + CCD downloads can take a few minutes on a fresh runtime; on subsequent runtimes the cache is present and the cell is near-instant. If the cell hangs or fails mid-download, check Colab's network indicator and try again on a fresh runtime.
2. **The numeric fields parse.** `affinity`, `confidence`, `iptm`, `ligand_iptm` should all be real numbers, not `NaN`. NaN values mean the affinity / confidence JSON schema parser is misaligned with Boltz-2's actual output for this version — fix in [`src/aidd/co_folding.py`](../src/aidd/co_folding.py) (`_parse_boltz_outputs`) before continuing.

The numeric values themselves are **meaningless** here — a 5-residue peptide + benzene is not a real binding system. This is purely a protocol-coverage check.
"""),

        markdown("""
## 3. Locate (or generate) the protein MSA

### Background

Boltz-2 needs a **multiple-sequence alignment** (MSA) of the protein. Without one, it runs in *single-sequence mode* — much less accurate, basically a sequence-conditioned random fold. With one, it uses the co-evolution signal in the MSA the same way AF2 does (Boltz-2 is built on the same modelling principles).

By default Boltz queries `api.colabfold.com` for a fresh MSA on **every** CLI invocation. For a library of N compounds against the **same** protein, that's N redundant queries against shared community infrastructure — wasteful, slow (each query dominates per-compound wall time), and rate-limit-prone. Smoke-test attempt #3 measured ~6.5 min per compound on a T4 in this mode for ERK2; a full 414-compound run would have been ~45 GPU-hours.

Boltz-2 accepts a **pre-computed MSA** via the `msa:` field on the protein chain in the input YAML (and our wrapper threads it through `predict_complex` / `predict_library`). Same MSA, reused for every per-compound call — Boltz only spends time on actual inference, not on hammering the MSA server.

### What this cell does

Probes three locations in priority order. The first hit wins. Non-canonical hits are **copied to the canonical location** so future runs short-circuit at the first probe:

1. **`data/derived/<target>/fold/msa/<target>.a3m`** — the canonical location. ✅ Hit here means "we've located the MSA before; just use it".
2. **`data/derived/<target>/fold/*.a3m`** — notebook 01's default output. ColabFold's `colabfold_batch` writes `<input_stem>.a3m` directly in the output directory, not under a `msa/` subdir. If notebook 01 has run for this target, the file lives here.
3. **`data/derived/<target>/boltz/per_compound/*/boltz_in_*/out/boltz_results_*/msa/*_unpaired_tmp_env/bfd.mgnify30.metaeuk30.smag30.a3m`** — Boltz's own intermediate MSA from any prior per-compound run on this protein. Reusing this avoids generating a new MSA when one already exists somewhere on Drive. We prefer the deeper BFD/MGnify/MetaEuk/SMAG variant over the UniRef-only one.

If none of the three find anything, the genuine fallback installs **ColabFold** (mirroring notebook 01's pinned commit) and runs `colabfold_batch --msa-only` to generate a fresh MSA. The fresh MSA lands in the canonical `fold/msa/<target>.a3m` location.

> ⚠ The ColabFold-install branch of the fallback is **best-effort**: it does not include a kernel restart, so if ColabFold's deps downgrade numpy / scipy / sklearn (the same risk Section 1's setup cell handles via a kernel restart), subsequent cells may fail with cryptic ABI errors. In practice this branch is rarely needed — the probes above usually find an existing MSA. If you hit this branch and subsequent cells break, run the install in Section 1's setup cell and re-run from cell 1.

The MSA file is a few hundred KB to a few MB, lives on Drive, survives across Colab sessions.
"""),

        code(title="Locate (or generate) the ERK2 protein MSA", source="""
import shutil

TARGET = "erk2"
FOLD_DIR         = DATA_ROOT / TARGET / "fold"
MSA_DIR          = FOLD_DIR / "msa"
TARGET_BOLTZ_DIR = DATA_ROOT / TARGET / "boltz"
CANONICAL_MSA    = MSA_DIR / f"{TARGET}.a3m"

# Must match notebooks/_build_01_fold_target.py exactly. UniProt P28482.
ERK2_SEQUENCE = (
    "MAAAAAAGAGPEMVRGQVFDVGPRYTNLSYIGEGAYGMVCSAYDNLNKVRVAIKKISPFEHQTYCQRTLREIKILLRFRHENIIGINDIIRAPTI"
    "EQMKDVYIVQDLMETDLYKLLKTQHLSNDHICYFLYQILRGLKYIHSANVLHRDLKPSNLLLNTTCDLKICDFGLARVADPDHDHTGFLTEYVA"
    "TRWYRAPEIMLNSKGYTKSIDIWSVGCILAEMLSNRPIFPGKHYLDQLNHILGILGSPSQEDLNCIINLKARNYLLSLPHKNKVPWNRLFPNAD"
    "SKALDLLDKMLTFNPHKRIEVEQALAHPYLEQYYDPSDEPIAEAPFKFDMELDDLPKEKLKELIFEETARFQPGYRS"
)

# Probe priority — first hit wins. Non-canonical hits are copied to
# CANONICAL_MSA so future runs short-circuit at probe #1.
found, origin = None, None

if CANONICAL_MSA.exists():
    found, origin = CANONICAL_MSA, "canonical (fold/msa/)"
elif FOLD_DIR.exists():
    fold_root = sorted(FOLD_DIR.glob("*.a3m"))
    if fold_root:
        found, origin = fold_root[0], "notebook-01 colabfold_batch default (fold/)"

if found is None and TARGET_BOLTZ_DIR.exists():
    boltz_msas = sorted(TARGET_BOLTZ_DIR.glob(
        "**/boltz_in_*/out/boltz_results_*/msa/*_unpaired_tmp_env/bfd.mgnify30.metaeuk30.smag30.a3m"
    ))
    if boltz_msas:
        found, origin = boltz_msas[0], "Boltz internal MSA from a prior per-compound run"

if found is not None:
    if found.resolve() != CANONICAL_MSA.resolve():
        MSA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(found, CANONICAL_MSA)
        print(f"Found MSA at non-canonical location:")
        print(f"  source: {pretty_path(found, DATA_ROOT, REPO_ROOT)}")
        print(f"  origin: {origin}")
        print(f"  copied to canonical: {pretty_path(CANONICAL_MSA, DATA_ROOT, REPO_ROOT)}")
    else:
        print(f"Reusing MSA at canonical location:")
        print(f"  {pretty_path(CANONICAL_MSA, DATA_ROOT, REPO_ROOT)}")
    MSA_PATH = CANONICAL_MSA.resolve()
else:
    print(f"No .a3m found at canonical (fold/msa/), notebook-01 default (fold/),")
    print(f"or any boltz/per_compound/.../boltz_in_*/.../msa/ tempdir. Generating fresh…")
    if not shutil.which("colabfold_batch"):
        # ColabFold install pinned to notebook 01's verified commit. No kernel
        # restart afterward -- best-effort path (see markdown note above).
        COLABFOLD_COMMIT = "de5ab5f795ed95c70a7a9b6a9dc6bb5625016142"  # v1.6.1
        print(f"  Installing ColabFold {COLABFOLD_COMMIT[:7]} (~3-5 min, one-time)…")
        !pip install -q --no-warn-conflicts \\
            "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold@{COLABFOLD_COMMIT}"
    MSA_DIR.mkdir(parents=True, exist_ok=True)
    fasta_path = MSA_DIR / f"{TARGET}.fasta"
    fasta_path.write_text(f">{TARGET}\\n{ERK2_SEQUENCE}\\n")
    !colabfold_batch "{fasta_path}" "{MSA_DIR}" --msa-only
    msa_candidates = sorted(MSA_DIR.glob("*.a3m"))
    if not msa_candidates:
        raise RuntimeError(
            f"colabfold_batch --msa-only produced no .a3m files under {MSA_DIR}. "
            "Check the cell output above for the actual error; common causes are "
            "MMseqs2 server overload (retry in a few minutes) and a network blip."
        )
    generated = msa_candidates[0]
    if generated.resolve() != CANONICAL_MSA.resolve():
        shutil.move(str(generated), CANONICAL_MSA)
    MSA_PATH = CANONICAL_MSA.resolve()
    print(f"Generated MSA: {pretty_path(MSA_PATH, DATA_ROOT, REPO_ROOT)}")

size_kb = MSA_PATH.stat().st_size / 1024
size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
print(f"\\nMSA_PATH = {pretty_path(MSA_PATH, DATA_ROOT, REPO_ROOT)}  ({size_str})")
"""),

        markdown("""
### Interpreting the cell output

Three branches you might see:

- **"Reusing MSA at canonical location"** — happiest path. The .a3m is already at `fold/msa/<target>.a3m`, no work to do.
- **"Found MSA at non-canonical location … copied to canonical"** — we located the MSA in a fallback spot (notebook 01's `fold/` or a Boltz tempdir) and copied it to the canonical path. Future runs hit the first branch.
- **"Generating fresh …"** — no MSA found anywhere on Drive. Installs ColabFold (if needed) and runs `colabfold_batch --msa-only`. ~5–10 min one-time. The generated MSA lands at the canonical location.

`MSA_PATH` is the variable Sections 5 and 6 below use; it must be set after this cell runs.
"""),

        markdown("""
## 4. Inputs — target sequence + the same 414 compounds as notebook 04

### Background

We re-use **exactly** the compounds that notebook `04` successfully docked: the labelled ERK2 subset of 693 prepared compounds, of which 414 produced valid gnina poses on Drive. Using the same set in both lanes is a hard requirement for the consensus join in notebook `06` — different cohorts on the two sides would silently shrink the intersection.

The target sequence is human ERK2 (MAPK1, [UniProt P28482](https://www.uniprot.org/uniprotkb/P28482)). It must match the sequence used in notebook `01` so the consensus story stays coherent across the pipeline. If you change it here, change it there too.
"""),

        code(title="Inputs: paths, output directory (TARGET + ERK2_SEQUENCE come from Section 3)", source="""
# TARGET and ERK2_SEQUENCE are defined in Section 3 (the MSA-locate cell);
# we use the same constants here so the sequence string never drifts
# between cells.
print(f"Target: {TARGET}")
print(f"ERK2 sequence length: {len(ERK2_SEQUENCE)} aa")
print(f"MSA file:             {pretty_path(MSA_PATH, DATA_ROOT, REPO_ROOT)}")

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
## 5. Smoke-test — measure per-compound runtime on 3 fresh compounds with MSA caching

### Background

Boltz-2 wall time per compound depends on the GPU tier (T4 vs A100 vs L4), the protein length, the ligand size, and — critically — whether the MSA is queried fresh from the server (~5 min per call on T4 for ERK2) or pre-cached on disk (skipped, so per-compound time drops to inference-only). We will not write a number into pipeline rationale until we measure it on *this* runtime, against *our* compounds, with the install we just did. (See [`feedback_measure_before_rationale.md`](#) for the rule and the incident that motivated it.)

### What this cell does

Pick **three compounds the previous (MSA-server-mode) attempt never saw** — indices 5, 6, 7 of `ligands_df` — and run `predict_library(msa_path=MSA_PATH, ...)`. All three use the same on-disk MSA from Section 3 (no MSA-server calls), so the measured per-compound runtime is the clean "inference-only" number we extrapolate the full-library cost from. The function writes per-compound outputs to `boltz/_smoke_test/`, separate from the full-library cache. Per-compound runtimes are printed from `run.log` at the end, plus the actual compound IDs so re-runs are reproducible.
"""),

        code(title="Boltz-2 smoke-test: 3 fresh compounds with cached MSA", source="""
import time

SMOKE_DIR = BOLTZ_DIR / "_smoke_test"
SMOKE_DIR.mkdir(parents=True, exist_ok=True)

# Three compounds the previous (MSA-server-mode) smoke-test never reached.
# The actual compound_ids are printed for reproducibility before the run starts.
smoke_df = ligands_df.iloc[5:8].copy().reset_index(drop=True)
print(f"Smoke-test compounds (indices 5-7 of ligands_df):")
for _, row in smoke_df.iterrows():
    print(f"  compound_id={row['compound_id']}    smiles={row['smiles']}")
print()
print(f"Using MSA from: {pretty_path(MSA_PATH, DATA_ROOT, REPO_ROOT)}")
print(f"Running Boltz-2 on {len(smoke_df)} compounds with MSA caching enabled…")
print()

t0 = time.time()
smoke_affinities = predict_library(
    sequence=ERK2_SEQUENCE,
    ligands=smoke_df,
    out_dir=SMOKE_DIR,
    msa_path=MSA_PATH,
    progress=True,
    on_failure="skip_with_guard",
    failure_guard_n=5,
)
wall = time.time() - t0
n_fresh = int(smoke_affinities['boltz_affinity'].notna().sum())
print(f"\\nSmoke-test wall time: {wall/60:.1f} min total, "
      f"{n_fresh} successful predictions; "
      f"≈ {wall/max(n_fresh, 1):.0f} s per fresh compound.")

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
## 6. Full library — co-fold all 414 compounds with Boltz-2

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
    msa_path=MSA_PATH,
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

- `boltz_input_invalid`: Boltz's internal RDKit pipeline rejected the SMILES (kekulization, valence, fragment-chooser). The CLI prints `Failed to process … Skipping. Error: …` and exits 0; the wrapper catches the pattern and marks the compound. A small fraction (< 5 %) is expected even on cleanly-prepared libraries — Boltz's `LARGEST_FRAGMENT_CHOOSER` is stricter than the standardisation in notebook `02`. If > 10 %, pre-filter the input SDF through the same chooser in notebook `02` before re-running.
- `boltz_oom` on a small fraction (< 5 %) usually means a few large ligands; re-run those on a bigger GPU after the main library completes.
- `boltz_msa_failed` is a transient infrastructure problem on the MSA server; retry by deleting the markers (`per_compound/*.FAILED`) and re-running this cell.
- `boltz_co_fold_diverged` on a single compound is rare and is fine to leave as-is.
- `boltz_install_error` on > 1 compound means the install is broken; the run won't make progress until the setup cell is re-run on a fresh runtime.
"""),

        markdown("""
## 7. Sanity check — view one predicted complex in 3-D

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
## 8. Evaluation — does Boltz-2 affinity rank actives above inactives?

### Background

Boltz-2 was *not* trained on this target's activity history. The question is whether its **generic** affinity head still produces a ranking that beats random on our labelled ERK2 set, and how its accuracy compares to gnina's CNN affinity (the docker-built-in scorer) and the per-target rescorer (notebook `04`).

We report three numbers, all on the **scaffold-grouped 80/20 split** that notebook `04` used. This means: when notebook `04` defined which compounds sit in the random-split test fold and which sit in the scaffold-grouped test fold, those exact same membership flags carry over here. The metric comparison is **apples-to-apples**: same fold definitions, same held-out compounds, different ranking method.

1. **ROC-AUC on the scaffold-split held-out fold.** Done signal: > 0.55 indicates the affinity head is picking up real signal on this target.
2. **EF1%** (enrichment factor at top 1 %). The medicinal-chemistry-relevant number: how concentrated the top 1 % of the Boltz ranking is in true actives.
3. **Spearman correlation** between Boltz-2 affinity rank and gnina CNN-affinity rank on the 412-compound overlap. Expected: positive in the ≈ 0.2–0.5 range is ideal — methods agree on real signal but disagree enough that consensus adds value. ρ > 0.7 means the methods are largely redundant.

We also report the random-stratified-split AUC alongside, as in notebook `04`, so the scaffold-leakage tax is visible.

### Critical: Boltz-2's affinity sign convention

Boltz-2 writes `affinity_pred_value` as **log10(IC50) µM** units — **lower means stronger binder**. This is the opposite of `gnina_cnn_affinity` / `rescorer_rf_proba` / EF/AUC conventions, which all assume **higher = better**.

If we feed raw `boltz_affinity` straight into `evaluate_scores`, the actives sort to the *bottom* of the ranking and AUC comes back as `1 − real_AUC` (looks broken — e.g. scaffold AUC = 0.35 instead of 0.65). The metrics cell below precomputes

```python
joined["boltz_affinity_signed"] = -joined["boltz_affinity"]
```

once, immediately after the join, and uses the signed column for every downstream metric. The raw `boltz_affinity` value in `affinity.csv` stays as Boltz wrote it (so we don't fight upstream's tooling); the negation happens at consumption time. **Any future notebook that joins against `affinity.csv` for rank-based metrics must do the same.** The principle and the incident that motivated it are documented in the project memory's `feedback_boltz_affinity_sign.md`.
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

# Boltz-2's affinity_pred_value is log10(IC50) µM (lower = stronger binder).
# Negate once here so every downstream rank-based metric uses the standard
# "higher = better" convention without having to remember to flip. See the
# markdown above and feedback_boltz_affinity_sign.md.
joined["boltz_affinity_signed"] = -joined["boltz_affinity"]

# Build a side-by-side metrics table on both splits, baseline (gnina-CNN-aff)
# vs Boltz-2 affinity (signed) vs Boltz-2 affinity_probability vs rescorer.
#
# NOTE on the rescorer column: scored_poses.parquet currently records
# final_rf.predict_proba(X) from a model trained on ALL labelled compounds,
# including those in the test fold. When we filter to test-fold here and
# compute AUC, we are testing the model on its own training data -- the
# resulting AUC = 1.000 is a data leak, NOT a real performance number. The
# honest scaffold AUC for the rescorer (from notebook 04's own held-out
# evaluation) is 0.66. See _planning/KNOWN_ISSUES.md for the fix scope.
# Until notebook 04 saves train-fold-only OOF predictions, treat the
# rescorer row of this table as decorative -- do not compare Boltz-2 against
# 1.000 in any reporting context.
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
        {"split": split_label, "method": "Boltz-2 affinity (sign-corrected)",
         **evaluate_scores(y, held["boltz_affinity_signed"].to_numpy())},
        {"split": split_label, "method": "Boltz-2 affinity_probability",
         **evaluate_scores(y, held["boltz_affinity_probability"].to_numpy())},
        {"split": split_label, "method": "rescorer RF (LEAKED — see comment)",
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
        ("Boltz-2 affinity (sign-corrected)", "boltz_affinity_signed",  "tab:green"),
        ("Boltz-2 binder probability",     "boltz_affinity_probability","tab:olive"),
        ("Rescorer RF (LEAKED)",           "rescorer_rf_proba",         "tab:blue"),
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
rho, pval = spearmanr(joined["boltz_affinity_signed"], joined["gnina_cnn_affinity"])
rho_rf, pval_rf = spearmanr(joined["boltz_affinity_signed"], joined["rescorer_rf_proba"])

print(f"Boltz-2 (sign-corrected)  vs  gnina CNN_affinity  : Spearman ρ = {rho:+.3f}  (p = {pval:.1e}, n = {len(joined)})")
print(f"Boltz-2 (sign-corrected)  vs  rescorer RF (LEAKED): Spearman ρ = {rho_rf:+.3f}  (p = {pval_rf:.1e})")
print()
print("Interpretation guide (Boltz-2 has been sign-corrected so positive correlation means agreement):")
print("  ρ in [0.2, 0.5]  : methods agree on real signal but disagree enough that consensus adds value (ideal).")
print("  ρ > 0.7          : methods are largely redundant — consensus would not narrow the shortlist much.")
print("  ρ < 0            : disagreement after sign-correction — investigate either the sign convention or the data join.")

# Scatter for visual confirmation. y-axis is the sign-corrected affinity
# (negated boltz_affinity) so both axes share the same "higher = stronger"
# convention -- positive correlation visible as a diagonal.
fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
ax.scatter(joined["gnina_cnn_affinity"], joined["boltz_affinity_signed"],
           c=joined["Active"].astype(int), cmap="coolwarm", alpha=0.7, s=20)
ax.set_xlabel("gnina CNN_affinity  (higher → stronger by gnina)")
ax.set_ylabel("Boltz-2 affinity, sign-corrected  (higher → stronger by Boltz-2)")
ax.set_title(f"Method-method scatter (ρ = {rho:+.3f})\\nblue = active, red = inactive")
plt.tight_layout()
plt.show()
"""),

        markdown("""
### How to read the evaluation result

The **scaffold-split AUC for Boltz-2 affinity (sign-corrected)** is the headline number for this notebook. The done signal is:

> **Boltz-2 scaffold-split AUC > 0.55** on the held-out fold = the affinity head is picking up real, generalisable signal for this target.

Three honest cases the numbers can land in:

- **Boltz > 0.6 and rescorer > 0.6**: both methods are useful and we expect consensus to add value. This is the case where notebook `06` produces a high-quality shortlist.
- **Boltz ≈ 0.5 and rescorer > 0.6**: the per-target rescorer wins decisively. Note in the recap; consensus on this target reduces to "trust the rescorer" because Boltz-2's signal is too weak to add information.
- **Boltz > 0.6 and rescorer ≈ 0.6 and Spearman ρ < 0.3**: Boltz-2 is finding signal the rescorer is missing. Consensus will produce a smaller, more conservative shortlist than either alone — the textbook win case for combining independent methods.

The Spearman correlation between sign-corrected Boltz-2 affinity and gnina CNN_affinity is the **diagnostic for method independence**. If it lands above 0.7, the two methods are too redundant for consensus to help (notebook `06` will still run but the consensus list will be close to a single-method top-N).

### About the rescorer's leaked AUC

`scored_poses.parquet` from notebook `04` records `rescorer_rf_proba` as the prediction of a Random Forest trained on the *full* labelled subset (including the test-fold compounds). When this notebook filters to the test fold and computes AUC on that column, the model is being tested on its own training data — the resulting AUC = 1.000 is a **data leak**, not a real performance number. The honest scaffold AUC for the rescorer (from notebook `04`'s own held-out evaluation) is **0.66**.

**Do not compare Boltz-2 against the 1.000 number** in any reporting context. The rescorer row of the metrics table above is decorative until notebook `04` is fixed to save train-fold-only OOF predictions in a separate column. See `_planning/KNOWN_ISSUES.md` for the fix scope.

### What the numbers cannot tell you

- **Pose quality.** AUC measures ranking; it does not measure whether the predicted complex is sane. Trust the Section 7 viewer for that, and PoseBusters (in notebook `03`) for gnina poses.
- **Calibration.** Boltz-2 affinity is in log10(IC50) µM units but is not a guaranteed thermodynamic predictor. Treat it as a ranking signal, not as an absolute IC50.
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
