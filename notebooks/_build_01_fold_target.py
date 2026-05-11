"""Builder for 01_fold_target.ipynb.

Source of truth for the ColabFold / AlphaFold2 structure-prediction notebook.
Cells appear below in narrative order. Never edit the .ipynb directly — see
``CLAUDE.md`` § *Notebook workflow*.

Regenerate:
    python notebooks/_build_01_fold_target.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import AUTORELOAD_SNIPPET, code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "01_fold_target.ipynb"


def build() -> None:
    # accelerator + gpu_type cause Colab to open this notebook with T4 GPU
    # pre-selected — mirrors ColabFold's AlphaFold2.ipynb metadata so reviewers /
    # users don't have to remember to switch the runtime before running cell 1.
    nb = notebook(
        markdown("""
# 01 — Fold a target protein with ColabFold / AlphaFold2

**aidd-pipeline · Notebook 1 of the screening workflow**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hvmarco/aidd-pipeline/blob/main/notebooks/01_fold_target.ipynb)

> **Upstream reference:** This notebook is an adapted, pinned version of [`ColabFold/AlphaFold2.ipynb`](https://github.com/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb) (a.k.a. the official ColabFold AlphaFold2 notebook). For the canonical, always-current ColabFold notebook open the link directly: [Open in Colab ↗](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb). Our version differs in three ways: (1) the install is pinned to a specific ColabFold commit for reproducibility, (2) we wrap the call to `colabfold_batch` in a thinner orchestration so the output is picked up by the rest of *this* pipeline, (3) the markdown is rewritten for our pedagogical conventions.

When you do not have an experimental 3-D structure of your target — and even when you do, but it lacks a part of the sequence, contains a mutation you care about, or is a homolog rather than your actual construct — you predict the structure from the amino-acid sequence. The state of the art for that since 2021 is **AlphaFold2 (AF2)**, run via the community wrapper **ColabFold**.

This notebook takes a protein sequence as input, runs ColabFold on a Colab GPU, picks the best of five predicted models, scores the prediction's confidence, and writes the structure to `data/derived/<target>/fold/`. The output PDB is what the docking notebooks downstream consume.

> ⚠️ **This notebook must run on Google Colab with a GPU.** ColabFold's inference is a deep-learning model with billions of parameters; it is impractical on CPU and the install on Windows / macOS without CUDA is painful. On Colab a typical 360-residue protein takes 30–45 min from "Run All" to a saved PDB. We will not try to run it locally.

## Learning objectives

After running this notebook you will be able to:

- Explain what AlphaFold2 / ColabFold does and what its outputs mean.
- Read the **pLDDT** confidence score and recognise the four AlphaFold confidence bands (very high / confident / low / disordered).
- Read the **pTM** score and know what it tells you about overall fold quality.
- Run ColabFold on Colab from a Jupyter cell and parse the output.
- Use `src/aidd/structures.py` to extract pLDDT, compute a distogram, and compare two structures by RMSD.

## Audience

- A biologist / clinician who has a sequence (or a UniProt ID) for the target and needs a 3-D model to dock candidates against.
- A data / ML person who has read about AlphaFold but never used it; this notebook is a hands-on first run.
- A student writing a thesis chapter that includes structure prediction.

## Prerequisites

- A Google account, signed into Colab. The free tier works but a Pro subscription is more reliable for the runtime (no idle disconnects, longer max session).
- A target sequence in single-letter amino-acid format. We provide ERK2 (UniProt **P28482**) as the default example; replace with your sequence when you wire in a real target.
- Optionally, an experimental or previously-predicted PDB at `data/structures/<target>_ref.pdb` to compare against (we use the archived `erk2_af.pdb` for the demo).

## Runtime

- Install ColabFold on Colab: **5–10 min** (one-time per Colab runtime).
- MSA search (remote MMseqs2): **2–5 min**.
- AlphaFold2 inference (5 models): **15–30 min** on a T4, **5–10 min** on an A100.
- **Total: ~30–45 min** for a typical 300–400 residue protein on free Colab.
"""),

        markdown("""
## Key terms used in this notebook

| Term | Meaning in one line |
|---|---|
| **AlphaFold (AF)** | DeepMind's protein-structure prediction neural network. AF2 (2021) is open-source and the de-facto standard. AF3 (2024) extends to complexes but has closed weights. |
| **ColabFold** | A community wrapper around AF2 that swaps in fast MMseqs2 for the slow MSA step and runs free on Colab. The de-facto way to use AF2 if you do not have a local cluster. |
| **MSA** | *Multiple Sequence Alignment* — a stack of evolutionarily related protein sequences. AF2 reads co-evolution patterns in the MSA to infer 3-D contacts. |
| **MMseqs2** | A fast sequence-search engine; ColabFold builds MSAs with it instead of the original AF database tools. |
| **pLDDT** | *predicted Local Distance Difference Test* — AF2's per-residue confidence score, 0–100. Higher = more confident. Stored in the PDB B-factor column by convention. |
| **pTM** | *predicted TM-score* — AF2's overall fold-quality score, 0–1. > 0.7 is a "good" fold; > 0.8 is excellent. |
| **TM-score** | *Template-Modelling score* — a global measure of structural similarity between two protein folds; > 0.5 means same fold. |
| **RMSD** | *Root-Mean-Square Deviation* — average Cα–Cα distance between two superposed structures, in Å. < 2 Å = very similar; > 5 Å = different. |
| **Distogram** | The matrix of pairwise Cα–Cα distances; visualises the fold's contact pattern. |
| **B-factor** | A PDB column originally meant for X-ray thermal motion; AF / ColabFold repurpose it to store pLDDT. |
"""),

        markdown("""
## Why this matters clinically

Roughly **a third of proteins in the human proteome had no experimental structure** at the time of AlphaFold's public release (2021). For cancer targets specifically, this is still a problem: many mutated oncoproteins, splice variants, or domain combinations have no PDB entry — but designing a small-molecule inhibitor against them requires a 3-D model of the pocket you intend to fill.

AlphaFold2 changed the game. For globular proteins of normal size, AF2 predictions are accurate enough to plan a docking campaign against them. The community uses AF models as starting points for:

- Targeting mutant forms of well-known proteins (a known structure of wild-type + an AF model of the mutant).
- Predicting structures of human kinases / GPCRs / nuclear receptors not yet crystallised in the conformation of interest.
- Pre-screening libraries against orphan targets before investing in protein-X-ray work.

For *our* pipeline, the AF model is what gets handed to the docking step. The pLDDT scores tell us *which parts* of the model to trust — high-pLDDT residues in the binding pocket are fine to dock against; low-pLDDT loops far from the pocket are usually safe to ignore.

What AlphaFold2 does **not** do: predict conformational dynamics, protein–ligand co-folding (AF3 / Boltz-2 do this), post-translational modifications, alternative conformations, or membrane insertion. Treat the AF model as one snapshot, not the truth.
"""),

        markdown("""
## 1. Setup

### About this install (read once, then forget)

This cell **mirrors the install pattern of ColabFold's official notebook**, [`AlphaFold2.ipynb`](https://github.com/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb). The only difference: we pin ColabFold to a specific commit (`de5ab5f`, v1.6.1, verified working on a fresh Colab T4 runtime on **2026-05-11**) so the install is reproducible six months from now.

**Why we trust Colab's default JAX rather than pinning it ourselves:** ColabFold's notebook deliberately uses Colab's default JAX on GPU runtimes (it only pins JAX on TPU). We tested this directly on **2026-05-11**: their notebook ran a full AlphaFold2 prediction end-to-end with no version pinning, no `dm-haiku` forcing, and no TF reinstall — proving Colab's current default JAX is compatible with the bundled AlphaFold code. Earlier we tried pinning JAX 0.4.26 + `dm-haiku` 0.0.12 + TF force-reinstall as a defensive layer; that turned out to *cause* problems (PJRT plugin mismatches, haiku symbol skew) rather than solve them. Less is more here.

**To update the ColabFold pin** (recommended every ~3 months, or whenever a Colab runtime change breaks this cell):
1. Visit [`AlphaFold2.ipynb`](https://github.com/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb) at the current ColabFold master.
2. Run their install cell on a fresh Colab runtime; confirm a small test prediction completes.
3. Find the latest commit hash on [ColabFold's commit history](https://github.com/sokrypton/ColabFold/commits/main); update `COLABFOLD_COMMIT` below.
4. Re-run this notebook end-to-end; commit on success. The git log becomes the methodological provenance.

**If a future Colab JAX upgrade breaks this cell** (i.e. ColabFold's vanilla install starts failing because Colab's default JAX advances past what AlphaFold supports), the historical defensive pins are preserved in git: see commit `08d4806` for the pinning approach we used before this simplification.

### What the cell does

On Colab, in order:

1. **Installs ColabFold + the bundled AlphaFold** at the pinned commit (`alphafold-minus-jax` extras — we don't touch JAX, ColabFold's deps include the matching `dm-haiku`).
2. **Adds `tpu-info`** so ColabFold's CPU/GPU/TPU selector takes its tpu-info code path and avoids a stale `import tensorflow as tf` fallback that can fail on cudnn ABI mismatches.
3. **Creates the symlinks** ColabFold's own notebook does (their convention; harmless on JupyterLab, mandatory in some ColabFold internal paths).
4. **Removes specific broken TF Lite `.so` files** (their official "hack to fix TF crash"). Cheap and safe.
5. **Clones the repo** and aborts loudly if the clone failed.
6. **Invalidates Python's import cache** so freshly-cloned `aidd.*` modules are findable.
7. **Verifies JAX sees the GPU** in a subprocess (the env `colabfold_batch` will actually use).

Total wall time on a fresh Colab runtime: ~3–5 min.
"""),

        code(title="Setup: install ColabFold + clone repo (~3–5 min)", source="""
import sys
import importlib
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

# ColabFold install pin — update via the procedure described in the markdown above.
COLABFOLD_COMMIT = "de5ab5f795ed95c70a7a9b6a9dc6bb5625016142"   # ColabFold v1.6.1
LAST_VERIFIED    = "2026-05-11"

if not IS_COLAB:
    print(
        "This notebook is meant to run on Google Colab (it needs a GPU for AlphaFold2).\\n"
        "If you only want to test the parsing/visualisation cells locally, place a\\n"
        "previously-generated ColabFold output directory at\\n"
        "data/derived/erk2/fold/ and skip §3 (the actual run)."
    )
    REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()
else:
    print(f"Installing ColabFold {COLABFOLD_COMMIT[:7]} (verified {LAST_VERIFIED})…")
    # 1. ColabFold + the bundled AlphaFold (minus JAX). We deliberately do NOT pin
    #    JAX / dm-haiku / TF here — ColabFold's official notebook uses Colab's
    #    default versions on GPU and verified-working as of LAST_VERIFIED.
    !pip install -q --no-warn-conflicts \\
        "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold@{COLABFOLD_COMMIT}" \\
        tpu-info

    # 2. ColabFold's own symlinks (their hack from AlphaFold2.ipynb).
    !ln -sf /usr/local/lib/python3.*/dist-packages/colabfold colabfold
    !ln -sf /usr/local/lib/python3.*/dist-packages/alphafold alphafold

    # 3. ColabFold's official "TF crash fix" — remove specific broken .so files.
    !rm -f /usr/local/lib/python3.*/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so \\
           /usr/local/lib/python3.*/dist-packages/tensorflow/lite/python/*/*.so 2>/dev/null

    # 4. Repo. Must be public for unauthenticated clone from Colab.
    REPO_ROOT = Path("/content/aidd-pipeline")
    if not REPO_ROOT.exists():
        !git clone https://github.com/hvmarco/aidd-pipeline.git {REPO_ROOT}
    if not (REPO_ROOT / "src" / "aidd").exists():
        raise RuntimeError(
            "Repo clone failed (likely cause: the repo is private and Colab cannot "
            "authenticate). Make github.com/hvmarco/aidd-pipeline public, or use a "
            "Personal Access Token via Colab Secrets, then re-run this cell."
        )
    sys.path.insert(0, str(REPO_ROOT / "src"))
    # 5. Re-scan sys.path so freshly-cloned modules are findable.
    importlib.invalidate_caches()

    # 6. Verify JAX sees the GPU in a subprocess (the env colabfold_batch will use).
    !python -c "import jax; print('JAX', jax.__version__, '— devices:', jax.devices())"

print()
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

from aidd.structures import extract_plddt, plddt_summary, distogram, ca_rmsd
from aidd.folding import parse_colabfold_ranking, best_model_path, summarise_run
from aidd.viz import show_structure_colored_by_plddt
from aidd.io import mount_drive_if_colab

# Set up the data/derived/ root. On Colab this mounts Google Drive (one
# OAuth prompt on first call per runtime, then silent) and resolves to a
# Drive-backed path so the folded structure survives runtime deaths.
# Locally, it resolves to <repo_root>/data/derived/, unchanged from before.
# Flip USE_DRIVE = False to opt out (one-off Colab testing without a Drive
# auth prompt, or to keep outputs purely on /content/).
USE_DRIVE = IS_COLAB
DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)

sns.set_theme(style="whitegrid")
print(f"DATA_ROOT: {DATA_ROOT}")
print("setup ok")
"""),

        markdown("""
## 2. Inputs — target sequence

### Background

The default example is **human ERK2 (MAPK1)**, UniProt **P28482**. Replace `SEQUENCE` with the sequence of your real target when ready. The FASTA file written below is what we hand to ColabFold.

If you have a UniProt accession but not the raw sequence, you can fetch it with one line of Python (commented-out example below). For an in-house construct, just paste the sequence as a single string.

**Sequence length matters:** AlphaFold2's memory grows roughly as O(N²) in sequence length. Up to ~1200 residues runs comfortably on a Colab T4; longer sequences may need an A100 or a careful chunking strategy. The example below (360 aa) is well within free-tier limits.
"""),

        code(title="Inputs: target name, sequence, output directory", source="""
# Default: human ERK2 (MAPK1), UniProt P28482.
TARGET_NAME = "erk2"
SEQUENCE = (
    "MAAAAAAGAGPEMVRGQVFDVGPRYTNLSYIGEGAYGMVCSAYDNLNKVRVAIKKISPFEHQTYCQRTLREIKILLRFRHENIIGINDIIRAPTI"
    "EQMKDVYIVQDLMETDLYKLLKTQHLSNDHICYFLYQILRGLKYIHSANVLHRDLKPSNLLLNTTCDLKICDFGLARVADPDHDHTGFLTEYVA"
    "TRWYRAPEIMLNSKGYTKSIDIWSVGCILAEMLSNRPIFPGKHYLDQLNHILGILGSPSQEDLNCIINLKARNYLLSLPHKNKVPWNRLFPNAD"
    "SKALDLLDKMLTFNPHKRIEVEQALAHPYLEQYYDPSDEPIAEAPFKFDMELDDLPKEKLKELIFEETARFQPGYRS"
)

# Alternative: fetch from UniProt by accession
# from urllib.request import urlopen
# SEQUENCE = urlopen("https://rest.uniprot.org/uniprotkb/P28482.fasta").read().decode().split("\\n", 1)[1].replace("\\n", "")

OUTPUT_DIR = DATA_ROOT / TARGET_NAME / "fold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Write a FASTA for ColabFold.
fasta_path = OUTPUT_DIR / f"{TARGET_NAME}.fasta"
fasta_path.write_text(f">{TARGET_NAME}\\n{SEQUENCE}\\n")

print(f"Target:    {TARGET_NAME}")
print(f"Length:    {len(SEQUENCE)} amino acids")
print(f"FASTA:     {fasta_path.relative_to(REPO_ROOT)}")
print(f"Output:    {OUTPUT_DIR.relative_to(REPO_ROOT)}")
"""),

        markdown("""
## 3. Run ColabFold

### Background

ColabFold does three things in one CLI call:

1. **MSA search.** Queries the remote MMseqs2 server with our sequence; pulls back a multiple-sequence alignment of evolutionarily related proteins. AF2 uses co-evolution signal in this MSA to predict 3-D contacts. *Takes 2–5 min, depends on server load and sequence rarity.*
2. **Structure inference.** Runs all 5 AlphaFold2 models on the MSA, producing 5 candidate structures. Each model is a slightly different neural-network weight set, trained the same way; using all 5 gives a sense of prediction stability. *Takes ~5–10 min per model on a T4 GPU; faster on an A100.*
3. **Ranking.** Sorts the 5 outputs by their mean pLDDT (or by ipTM for multimers) and writes them to disk.

The cell below uses `--num-models 5` (the standard), `--msa-mode mmseqs2_uniref_env` (the default, fast, accurate enough for most targets), and **no templates** (`--templates` would let AF use known related structures as scaffolds — useful when they exist, but for an "honest novel target" demo we leave them off).

**This is the long-running step (~30 min).** Don't navigate away from the Colab tab; Colab's idle-disconnect timeout will kill the runtime. If you have Colab Pro, set the runtime to "Background execution" for safety.
"""),

        code(title="Run ColabFold (~25–30 min on T4 — the long step)", source="""
if IS_COLAB:
    # The full command. Run from the notebook so we can capture its output.
    !colabfold_batch \\
        "{fasta_path}" \\
        "{OUTPUT_DIR}" \\
        --num-models 5 \\
        --msa-mode "mmseqs2_uniref_env" \\
        --model-type "auto"
else:
    print("Skipping ColabFold run (local mode). To exercise the parsing cells,")
    print(f"copy a prior ColabFold output into {OUTPUT_DIR}/ and re-run from §4.")
"""),

        markdown("""
### What the output looks like

ColabFold writes ~30 files to the output directory:

- `<name>_unrelaxed_rank_001_..._.pdb` … rank 5 — the predicted structures, ranked by mean pLDDT.
- `<name>_scores_rank_001_..._.json` — JSON files with per-residue pLDDT, the PAE matrix (predicted alignment error), and pTM.
- `<name>.a3m` — the multiple-sequence alignment used.
- `<name>_predicted_aligned_error_v1.json` — full PAE data.
- Various `.png` plots: pLDDT per residue, PAE heatmap.

The `parse_colabfold_ranking` helper in `aidd.folding` reads the directory and gives us a tidy table.
"""),

        code(title="Parse ColabFold output → ranking DataFrame", source="""
ranking = parse_colabfold_ranking(OUTPUT_DIR, name=TARGET_NAME)
ranking
"""),

        markdown("""
### How to read the ranking

- **`rank`** — 1 is best (highest mean pLDDT). Use rank-1 for downstream work unless you have a reason not to.
- **`mean_plddt`** — average per-residue confidence, 0–100. **> 90** indicates a very-high-confidence prediction across the whole protein; **70–90** is confident enough for docking; **< 70** suggests the fold is uncertain and you should look at the per-residue plot before trusting it.
- **`ptm`** — overall fold-quality estimate, 0–1. **> 0.8** = excellent; **> 0.7** = good; **< 0.5** = likely wrong fold.
- **`model_number`** — which of AF2's 5 neural networks produced this prediction. If models 1–5 all agree (similar pLDDT, similar fold), the prediction is stable. If only one model scores high and the rest fail, treat with caution.

For our ERK2 example we expect rank-1 mean pLDDT ~90 and pTM ~0.85 (kinases are well-represented in AF2's training data and have abundant homologs in the MSA).
"""),

        markdown("""
## 4. Per-residue confidence

### Background

A single overall pLDDT is a useful summary, but the **per-residue plot** is more informative: it shows *which parts* of the structure to trust. Typically:

- **Loops at the N- and C-terminus** of the chain dip into the low / disordered range — these tails are often disordered in real structures too, so a low pLDDT there is honest.
- **Active-site / binding-pocket residues** should be in the very-high (> 90) range — these are what docking cares about.
- **A sudden mid-chain drop in pLDDT** can indicate a flexible / mis-predicted loop, or a multi-domain protein where the relative orientation of two domains is uncertain.
"""),

        code(title="pLDDT summary stats for the best model", source="""
best_pdb = best_model_path(OUTPUT_DIR, name=TARGET_NAME)
plddt = extract_plddt(best_pdb)
summary = plddt_summary(plddt)

print(f"Best model: {best_pdb.name}")
for k, v in summary.items():
    label = k.replace('_', ' ')
    if isinstance(v, float) and 'frac_' in k:
        print(f"  {label:>20}: {v:.1%}")
    elif isinstance(v, float):
        print(f"  {label:>20}: {v:.1f}")
    else:
        print(f"  {label:>20}: {v}")
"""),

        code(title="Plot pLDDT per residue", source="""
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.fill_between(plddt.index, plddt.values, alpha=0.25, color="steelblue")
ax.plot(plddt.index, plddt.values, lw=0.9, color="steelblue")
for y, label, color in [
    (90, "very high (≥90)", "#0053D6"),
    (70, "confident (70–90)", "#65CBF3"),
    (50, "low (50–70)", "#FFDB13"),
]:
    ax.axhline(y, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Residue number")
ax.set_ylabel("pLDDT")
ax.set_ylim(0, 100)
ax.set_title(f"Per-residue confidence — {TARGET_NAME} (best model)")
plt.tight_layout()
plt.show()
"""),

        markdown("""
### Reading the plot

The dashed horizontal lines mark the AlphaFold confidence bands. Look for:

- **Two flexible tails** at residues ~1–20 and the last ~20: typical for a soluble protein. Low pLDDT here is fine.
- **A tall central plateau ≥ 90**: this is the well-folded core, which includes the ATP-binding pocket for a kinase. This is the region we will dock against.
- **No mid-protein dip** unless you know there is genuinely a flexible loop there. If there is one, check it does *not* fall in or near the binding site of interest.
"""),

        markdown("""
## 5. 3-D viewer — structure coloured by pLDDT

### Background

The 3-D viewer below uses the same colour code as the AlphaFold Database web view:

- **dark blue** — pLDDT ≥ 90 (very high)
- **light blue** — pLDDT 70–90 (confident)
- **yellow** — pLDDT 50–70 (low)
- **orange** — pLDDT < 50 (likely disordered)

Drag to rotate, scroll to zoom. The "blue core, orange tails" pattern that is characteristic of soluble proteins is the visual signature of a well-predicted single-domain fold.
"""),

        code(title="3-D viewer: structure coloured by pLDDT", source="""
view = show_structure_colored_by_plddt(best_pdb)
view.show()
"""),

        markdown("""
## 6. Compare to a reference structure (optional)

### Background

If a previously-validated structure of the same construct exists — an X-ray, an earlier AF model, or a structure of a close homolog — we compare our new prediction to it as a sanity check. For ERK2 the repo ships with `data/structures/erk2_af.pdb` (an archived AlphaFold model from the Leiden course). The two models should be near-identical if our prediction is sound, since they were generated from the same sequence by the same method.

The comparison is **Cα-RMSD** after rigid superposition. < 2 Å between two AF predictions of the same sequence is expected; > 4 Å suggests our new run had a different MSA / templates and may be picking a different conformation.

> *Note:* This comparison only works when both structures have the same number of residues. X-ray structures often omit flexible termini that AF predicts in full, in which case the comparison needs a sequence alignment first — beyond what this notebook does. We will skip the comparison gracefully when lengths differ.
"""),

        code(title="Compare to a reference structure (optional)", source="""
REF_PDB = REPO_ROOT / "data" / "structures" / "erk2_af.pdb"

if REF_PDB.exists():
    try:
        rmsd = ca_rmsd(REF_PDB, best_pdb)
        print(f"Cα-RMSD vs {REF_PDB.name}: {rmsd:.2f} Å")
        if rmsd < 2:
            print("  → near-identical fold (expected for two AF predictions of the same sequence).")
        elif rmsd < 5:
            print("  → same fold, some loop differences. Inspect visually.")
        else:
            print("  → significantly different. Compare distograms and check sequence identity.")
    except ValueError as e:
        print(f"Skipping RMSD: {e}")
        print("(The X-ray / reference may not cover the same residue range as our prediction.)")
else:
    print(f"No reference at {REF_PDB} — skipping comparison.")
"""),

        markdown("""
## 7. Save the best model for downstream use

### Background

The docking notebook (`03_dock_gnina`) expects the receptor at `data/derived/<target>/fold/<target>_best.pdb`. We copy the rank-1 prediction to that canonical location so the pipeline finds it without having to know the ColabFold-style filename. Per-residue pLDDT lives in the B-factor column and travels with the file.
"""),

        code(title="Save the best model as <target>_best.pdb for downstream notebooks", source="""
import shutil

canonical = OUTPUT_DIR / f"{TARGET_NAME}_best.pdb"
shutil.copyfile(best_pdb, canonical)
print(f"Best model copied to: {canonical.relative_to(REPO_ROOT)}")
print(f"This is what the docking notebook will load as the receptor.")
"""),

        markdown("""
## Recap

### Biomedical takeaway

We turned a protein sequence into a 3-D atomic model. For a well-behaved single-domain protein like ERK2, AlphaFold2 produces a structure essentially indistinguishable from the experimental one in the regions that matter for drug discovery — the conserved core of the ATP-binding pocket. The pLDDT colouring tells us exactly *where* to trust the prediction: dark-blue residues are dependable, orange tails are not, and a low-confidence binding-site residue is a red flag worth investigating before docking.

### Technical takeaway

ColabFold's CLI does the heavy lifting; our `src/aidd/{folding,structures,viz}.py` helpers handle the parsing and visualisation of the output. The contract with the rest of the pipeline is one file: `data/derived/<target>/fold/<target>_best.pdb` carries both 3-D coordinates and pLDDT (in the B-factor column).

### What's next in the pipeline

- **`03_dock_gnina.ipynb`** — dock the prepared ligand library (from notebook `02`) into this freshly-folded receptor.
- **`05_dock_boltz.ipynb`** — alternative fast lane: Boltz-2 co-folds the protein and the ligand together in one step. For routine targets where AF + gnina already work, this is the second-opinion / consensus method.

### Further reading

- Jumper et al., *Nature* (2021), **596**, 583 — the AlphaFold2 paper. [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- Mirdita et al., *Nat. Methods* (2022) — *ColabFold: making protein folding accessible to all*. [doi:10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)
- Tunyasuvunakool et al., *Nature* (2021) — the original release of AF predictions for the human proteome; useful context on what "well-predicted" means in practice. [doi:10.1038/s41586-021-03828-1](https://doi.org/10.1038/s41586-021-03828-1)
- AlphaFold Database, [alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk/) — pre-computed models for most well-studied proteins; often you can fetch one from here instead of running ColabFold yourself.
"""),
        accelerator="GPU",
        gpu_type="T4",
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
