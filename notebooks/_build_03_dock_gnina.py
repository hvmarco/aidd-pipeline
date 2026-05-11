"""Builder for 03_dock_gnina.ipynb.

Source of truth for the gnina docking notebook. Cells appear below in
narrative order. Never edit the .ipynb directly — see ``CLAUDE.md`` §
*Notebook workflow*.

Regenerate:
    python notebooks/_build_03_dock_gnina.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import AUTORELOAD_SNIPPET, code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "03_dock_gnina.ipynb"


def build() -> None:
    nb = notebook(
        markdown("""
# 03 — Dock a ligand library into the receptor with gnina

**aidd-pipeline · Notebook 3 of the screening workflow**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hvmarco/aidd-pipeline/blob/main/notebooks/03_dock_gnina.ipynb)

> **Upstream reference:** This notebook uses [gnina](https://github.com/gnina/gnina), a CNN-rescored fork of [AutoDock Vina](https://vina.scripps.edu/) / smina, developed by the Koes lab (Pitt). It is the "classical interpretable lane" of our pipeline. The complementary "fast lane" (Boltz-2 co-folding + affinity) is covered in notebook `05_dock_boltz`.

So far we have a receptor 3-D model (notebook `01`) and a library of drug-like, 3-D-embedded ligands (notebook `02`). This notebook brings them together: it **docks** each ligand into the receptor's binding site, scores the resulting poses, runs **PoseBusters** to filter out physically implausible poses, and writes a tidy SDF + score table for the rescoring notebook downstream.

> ⚠️ **gnina is Linux- and GPU-native.** From v1.3 (Oct 2024) the binary is linked against PyTorch + CUDA, so it needs a GPU Colab runtime — the setup cell verifies this before going further. It also runs under WSL2 + CUDA on Windows, or in a Linux+GPU container. **Local Windows and native macOS are unsupported by the docker itself** — run this notebook on Colab if you are on either of those.

## Learning objectives

After running this notebook you will be able to:

- Explain in plain language what **docking** is, and what its outputs (poses + scores) mean.
- Specify a **binding site** (center + box) and reason about its size relative to the protein.
- Run **gnina** from a Python cell, parse its SDF output, and read its score columns (`minimizedAffinity`, `CNNscore`, `CNNaffinity`).
- Perform a **redock** sanity check: redock a crystallographic ligand and verify the predicted pose is within 2 Å RMSD of the experimental one.
- Run **PoseBusters** on a docking output to flag physically implausible poses (clashes, broken bonds, bad ring geometry).
- Visualise the top hits in 3-D to spot good vs suspect poses by eye.

## Audience

- A clinician / biologist who wants to see how *in-silico* screening prioritises a library before wet-lab follow-up.
- A data / ML person who has never run docking but is comfortable with scientific Python.
- A student writing a thesis chapter on structure-based virtual screening.

## Prerequisites

- **A Colab GPU runtime** (Runtime → Change runtime type → T4 GPU). The notebook is marked GPU by default; if you opened it on a CPU runtime, switch first. gnina v1.3 links against CUDA libraries and will not load on CPU runtimes.
- The ERK2 crystal receptor (`data/structures/erk2_4fv7.pdb`) and its co-crystallised ligand (`data/ligands/erk2_4fv7_ref.pdb`) — both ship with the repo and are always available on a fresh clone. The crystal is used for the redock sanity check and as the alignment reference (and as a fallback receptor when the AF model is not yet built).
- *Optional but recommended*: notebook `01_fold_target` already run, with an AlphaFold model at `data/derived/<target>/fold/<target>_best.pdb`. When present, the cell below superposes it onto the crystal frame and uses the aligned PDB as the docking receptor (the standard pipeline path).
- *Optional*: notebook `02_prepare_ligands` already run, with a prepared SDF at `data/derived/<target>/ligands_prepared.sdf`. If absent, this notebook prepares a 10-compound smoke-test subset inline.

## Runtime

- **Setup + gnina install (Colab):** ~1 min, one-time per Colab runtime.
- **Redock sanity check:** ~30 s on a T4 GPU.
- **Library dock (25 compounds, exhaustiveness 8):** ~5–10 min on a T4 GPU. Scales roughly linearly with library size and with `--exhaustiveness`.
- **PoseBusters QC:** ~1–2 min on 25 compounds × 9 poses each.

Set `SAMPLE_N` in Section 2 to a small number (10–25) for a quick first run; bump it up once everything is wired.
"""),

        markdown("""
## Key terms used in this notebook

| Term | Meaning in one line |
|---|---|
| **Docking** | Predicting how a ligand sits inside a protein pocket. Output = one or more **poses** with **scores**. |
| **Pose** | A specific 3-D placement of the ligand inside the pocket. Docking returns several per compound (typically 9). |
| **Scoring function** | The formula that ranks poses. Classical: empirical / force-field-based (Vina, ChemPLP). Modern: CNN-rescored (gnina). |
| **Affinity** | Predicted binding free energy. Vina/smina convention: in **kcal/mol**, more *negative* = stronger binding. CNN affinity: in **pK_d** (log₁₀ of K_d), higher = stronger. |
| **gnina** | A docker that uses Vina-style sampling and a **CNN** trained on the PDBbind dataset to rescore poses. Open-source, GPU-optional. |
| **AutoDock Vina** | The classical docker gnina is built on. Empirical scoring function from 2010. |
| **Binding site / search box** | The cuboid in 3-D space inside which gnina samples poses. Smaller box → faster, more targeted; too small → misses real poses. |
| **Exhaustiveness** | Gnina's sampling-thoroughness knob. 8 is the default; 16 doubles search time for a small accuracy gain. |
| **Redock** | Take a known crystallographic ligand, dock it back into its receptor, and measure how close the predicted pose is to the experimental one. The standard methodological sanity check. |
| **RMSD** | *Root-Mean-Square Deviation* of heavy-atom positions, in Å. **< 2 Å** between predicted and experimental pose = successful redock. |
| **PoseBusters** | A pose-quality-control tool. Runs ~20 geometric / chemical checks per pose and flags ones that are physically implausible. |
| **CNN score / CNN affinity** | Outputs of gnina's pose-scoring neural network. Higher = better-looking pose / stronger predicted binding. |
| **SDF** | *Structure-Data File* — the text format gnina writes its poses + scores to. The same format we wrote the prepared library in (notebook `02`). |
"""),

        markdown("""
## Why docking matters — the biological reasoning

Wet-lab assays are expensive: every compound that gets ordered, tested, and dose-curved costs time, reagents, and freezer space. Docking is the cheapest filter in the medicinal-chemistry triage chain. It will not tell you whether a compound *is* a drug, but it will reliably remove the compounds whose 3-D shape cannot fit into the pocket at all — typically half the library on a kinase target.

The clinical analogue is informative. A binding assay measures whether a candidate compound interacts physically with the target protein in vitro. Docking is the *in-silico* version of that assay: same question (does this molecule fit?), much lower cost (CPU minutes rather than a 384-well plate), much weaker confidence in the answer (the scoring function is a rough approximation, and the protein is held rigid in standard docking). Two principles follow from this trade-off:

1. **Docking is a first-pass triage**, not a final ranking. We use it to push the bottom of the list off the bench, not to pick the winner.
2. **A single docker is rarely enough.** The downstream notebooks add an interaction-fingerprint rescorer (notebook `04`) trained on real activity data, and a second co-folding affinity predictor (notebook `05`, Boltz-2). The final shortlist (notebook `06`) requires *agreement* between these orthogonal signals.

What docking does **not** capture:

- **Protein flexibility.** Standard docking treats the receptor as rigid. Real binding often induces side-chain rotation or loop reorganisation.
- **Solvent and entropy.** The scoring function is a coarse free-energy estimate; entropy of the bulk water network is not modelled atomistically.
- **Covalent binding.** Most dockers assume non-covalent interactions only; covalent inhibitors need a specialised mode (gnina supports it; we do not turn it on here).
- **Allosteric sites.** We dock into one specified pocket. Compounds that bind elsewhere on the protein will score badly here, but might still be real binders.

These limitations are why we run docking *as one input among several*, not as a final verdict.
"""),

        markdown("""
## 1. Setup

### What this section does

Detect Colab vs local, clone the repo on Colab, install **gnina** (Linux-native — Colab downloads its static binary; local Linux / WSL2 should already have it; Windows / macOS surfaces a clear instruction to switch to Colab).

### About the gnina install (read once, then forget)

gnina ships as a Linux binary in [GitHub Releases](https://github.com/gnina/gnina/releases). From [v1.3 (Oct 2024)](https://github.com/gnina/gnina/releases/tag/v1.3) it links against PyTorch + CUDA — so a GPU runtime is **required**, not optional. We pin to **v1.3.2** with the `gnina.1.3.2` asset (the upstream release notes describe it as the *"older-CUDA, more compatible"* binary; the `cuda12.8` variant is for newer cards). To update: bump `GNINA_VERSION` + `GNINA_ASSET` below, re-run on a fresh runtime, commit on success.

The setup cell probes both `nvidia-smi` (is a GPU attached?) and `gnina --version` (does the binary load?) and bails out with a clear message if either fails, so you find out at install time rather than 5 cells later.

On Windows / macOS, the binary will not run. The local Python in those environments is still useful for the parsing / visualisation cells — but the dock itself must happen on Colab (or under WSL2 with CUDA).
"""),

        code(title="Setup: clone repo, install gnina, verify GPU + binary loads", source="""
import sys
import importlib
import shutil
import subprocess
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

# gnina pin — update via the procedure described in the markdown above.
GNINA_VERSION = "v1.3.2"
GNINA_ASSET   = "gnina.1.3.2"   # "older-CUDA, more compatible" binary; .cuda12.8 is for newer cards
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
    # 1. Fail loudly if no GPU is attached — gnina v1.3+ is CUDA-linked.
    gpu_ok = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    if not gpu_ok:
        raise RuntimeError(
            "No GPU detected. gnina v1.3+ links against CUDA and will not load "
            "on CPU runtimes. Switch to a GPU: Runtime → Change runtime type → "
            "T4 GPU, then re-run this cell."
        )

    # 2. Repo. Must be public for unauthenticated clone from Colab.
    REPO_ROOT = Path("/content/aidd-pipeline")
    if not REPO_ROOT.exists():
        !git clone https://github.com/hvmarco/aidd-pipeline.git {REPO_ROOT}
    if not (REPO_ROOT / "src" / "aidd").exists():
        raise RuntimeError(
            "Repo clone failed (likely cause: the repo is private and Colab cannot "
            "authenticate). Make github.com/hvmarco/aidd-pipeline public, or use a "
            "Personal Access Token via Colab Secrets, then re-run this cell."
        )

    # 3. gnina binary. Replace any existing one that doesn't load (e.g. a stale
    # CPU-only download from a previous broken install).
    if not _gnina_works():
        print(f"Downloading gnina {GNINA_VERSION} ({GNINA_ASSET}, verified {LAST_VERIFIED})…")
        url = f"https://github.com/gnina/gnina/releases/download/{GNINA_VERSION}/{GNINA_ASSET}"
        !wget -q -O /usr/local/bin/gnina {url}
        !chmod +x /usr/local/bin/gnina

        # Probe — fail loudly if the binary cannot load (missing CUDA libs etc.)
        probe = subprocess.run(["gnina", "--version"], capture_output=True, text=True)
        if probe.returncode != 0:
            raise RuntimeError(
                f"gnina installed but `gnina --version` exited {probe.returncode}. "
                f"stderr:\\n{probe.stderr}\\n\\nMost common cause is a missing CUDA "
                "library — check `!ldd /usr/local/bin/gnina | grep 'not found'` "
                "and confirm the runtime is GPU-backed."
            )

    # 4. PoseBusters + the rest (not in Colab's default image).
    !pip install -q posebusters rdkit datamol "prolif>=2.0" py3Dmol biopython

    # 5. Imports.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()

    # 6. Confirm everything is wired before continuing.
    !gnina --version | head -1
else:
    REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()

print(f"Repo root: {REPO_ROOT}")
print(f"Running on: {'Colab' if IS_COLAB else 'local'}")
"""),

        code(title="Imports + environment check", source=AUTORELOAD_SNIPPET + """
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
from rdkit import Chem

from aidd.docking import (
    BindingBox, box_from_center_radius,
    describe_environment, gnina_available, require_gnina,
    dock_library, redock_reference,
    parse_poses_sdf, pose_rmsd, run_posebusters,
)
from aidd.io import mount_drive_if_colab

# Set up the data/derived/ root. On Colab this mounts Google Drive (one
# OAuth prompt on first call per runtime, then silent) and resolves to a
# Drive-backed path so docked poses + scores survive runtime deaths.
# Locally, it resolves to <repo_root>/data/derived/, unchanged from before.
# Flip USE_DRIVE = False to opt out (one-off Colab testing without a Drive
# auth prompt, or to keep outputs purely on /content/).
USE_DRIVE = IS_COLAB
DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)
print(f"DATA_ROOT: {DATA_ROOT}")

sns.set_theme(style="whitegrid")

env = describe_environment()
print(f"Platform:      {env['platform']}")
print(f"gnina path:    {env['gnina_path']}")
print(f"gnina version: {env['gnina_version']}")

if not gnina_available():
    print()
    print("gnina is not on PATH. The parsing / visualisation cells below will still")
    print("work if you provide a poses.sdf from a previous Colab run, but the dock")
    print("itself cannot run on this machine. Open this notebook in Colab to run it.")
"""),

        markdown("""
## 2. Inputs — receptor, ligand library, binding site

### Background

Three things have to be set before we can dock:

1. **The receptor.** A PDB file with the protein you want to dock into. The default is the **AlphaFold model from notebook `01`** (`data/derived/<target>/fold/<target>_best.pdb`) — that is the path the pipeline assumes for any target, whether or not a crystal exists. There is one subtlety: the AlphaFold model is in *some arbitrary coordinate frame* (whatever orientation AlphaFold happened to output), whereas the binding-site coordinates in §3 are in the **4FV7 crystal coordinate frame** (they were measured on the crystal). If we docked the un-aligned AF model with the crystal-frame box, the box would point at the wrong region of space. The fix is to **superpose the AF model onto the crystal** once at the top of the notebook, and dock the aligned PDB. The cell below does that automatically when the AF model exists, using matched-by-residue-number Cα atoms (the well-resolved structural core both share). When the AF model has not been built yet (notebook 01 not run), the cell falls back to the crystal receptor directly. For a real target with no crystal you would instead run a pocket-detection tool (fpocket, P2Rank) on the AF model to derive the box in the AF frame — that path is out of scope for this teaching notebook, marked as a TODO.
2. **The ligand library.** The SDF written by notebook `02_prepare_ligands` (drug-like, PAINS-filtered, 3-D-embedded). If you have not run notebook 02 yet, the cell below falls back to preparing a small subset inline so this notebook is still runnable end-to-end as a smoke test.
3. **The search box.** A cuboid in 3-D space the docker will sample inside. For a well-studied target like ERK2 we know the binding-site coordinates (they live in the archived PLANTS config). For a new target you would derive them from a co-crystallised ligand if you have one, or from a pocket-detection tool like fpocket or P2Rank.

### About the ERK2 binding-site coordinates

For the ERK2 / 4FV7 example, the binding-site center and radius are taken directly from `_archive/configs/plants_4fv7.conf`:

- **center** = (1.343, 17.365, 40.983) Å — the geometric centre of the ATP pocket.
- **radius** = 12.9 Å — covers the ATP pocket plus a small margin around it.

gnina takes a *box* rather than a *sphere*; we convert by setting each box edge to `2 * radius + 2 * margin` Å. With `margin = 2 Å` the cube edge is about 30 Å — generous enough that the ligand can swing around inside the pocket without being clipped by the search box.

### About `SAMPLE_N` and runtime

Docking 100 prepared ligands at exhaustiveness 8 takes ~20–30 min on a Colab CPU. Set `SAMPLE_N = 10` for a quick first run while you wire everything up; bump it once the redock sanity check passes and you see the first few poses look reasonable.
"""),

        code(title="Inputs: target, receptor (AF→crystal aligned), ligand SDF, box", source="""
from aidd.structures import superpose_by_resnum

TARGET = "erk2"

# Crystal receptor + ligand for the redock sanity check.
CRYSTAL_RECEPTOR = REPO_ROOT / "data" / "structures" / "erk2_4fv7.pdb"
CRYSTAL_LIGAND   = REPO_ROOT / "data" / "ligands"    / "erk2_4fv7_ref.pdb"

# AF model from notebook 01 — gitignored, exists only after 01 has been run
# on this machine / Colab runtime.
AF_RECEPTOR = DATA_ROOT / TARGET / "fold" / f"{TARGET}_best.pdb"

# Where docking outputs land.
OUT_DIR = DATA_ROOT / TARGET / "docking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Build the receptor for the library dock. The binding-site coordinates below
# are in the 4FV7 crystal frame; if we have an AF model from notebook 01,
# align it to that frame so the same box applies. If we don't, fall back to
# the crystal receptor itself (so a fresh Colab clone can run this notebook
# without having to run 01 first; this means we're effectively redocking
# into the holo crystal, which is the optimistic / best-case scenario).
if AF_RECEPTOR.exists():
    ALIGNED_AF = AF_RECEPTOR.with_name(f"{TARGET}_best_aligned_to_4fv7.pdb")
    if not ALIGNED_AF.exists():
        info = superpose_by_resnum(AF_RECEPTOR, CRYSTAL_RECEPTOR, ALIGNED_AF)
        print(
            f"Aligned AF model onto 4FV7 crystal: "
            f"RMSD {info['rmsd']:.2f} Å over {info['n_matched_residues']} Cα atoms"
        )
    RECEPTOR = ALIGNED_AF
    RECEPTOR_KIND = "AlphaFold (notebook 01), aligned to 4FV7 crystal frame"
else:
    print("AF model from notebook 01 not found — falling back to the 4FV7 crystal.")
    print("Run notebook 01 first to dock into the AlphaFold model instead.")
    RECEPTOR = CRYSTAL_RECEPTOR
    RECEPTOR_KIND = "4FV7 crystal (AF model unavailable)"

# Ligand SDF: prefer notebook 02's output; fall back to a small inline prep
# so this notebook is end-to-end runnable on a fresh Colab clone.
LIGANDS = DATA_ROOT / TARGET / "ligands_prepared.sdf"
if not LIGANDS.exists():
    print(f"⚠ {LIGANDS.relative_to(REPO_ROOT)} not found — preparing a 10-compound")
    print("  fallback inline. For real screens, run notebook 02 first.")
    from aidd.ligands import read_smiles, prepare_library, write_sdf
    smi_path = REPO_ROOT / "data" / "compounds" / TARGET / "training_small.smi"
    df_smi = read_smiles(smi_path).head(10)
    df_prepped = prepare_library(df_smi, n_workers=1, progress=False)
    LIGANDS = DATA_ROOT / TARGET / "ligands_smoketest.sdf"
    LIGANDS.parent.mkdir(parents=True, exist_ok=True)
    write_sdf(
        df_prepped[df_prepped["ok"]],
        LIGANDS,
        props_to_write=["smiles_std", "mw", "logp", "qed"],
    )
    print(f"  wrote {sum(df_prepped['ok'])} prepared ligands → {LIGANDS.relative_to(REPO_ROOT)}")

# Binding-site geometry — from _archive/configs/plants_4fv7.conf (4FV7 frame).
BINDING_SITE_CENTER = (1.34299, 17.3648, 40.9828)
BINDING_SITE_RADIUS = 12.9007
BOX = box_from_center_radius(BINDING_SITE_CENTER, BINDING_SITE_RADIUS, margin=2.0)

# Library size for this run (None = all). Start small while wiring up.
SAMPLE_N = 25
EXHAUSTIVENESS = 8
NUM_MODES = 9

# Sanity-check the inputs.
assert RECEPTOR.exists(), f"missing receptor {RECEPTOR}"
assert LIGANDS.exists(),  f"missing ligands SDF {LIGANDS}"

print()
print(f"Target:                   {TARGET}")
print(f"Receptor:                 {RECEPTOR.relative_to(REPO_ROOT)}")
print(f"  ({RECEPTOR_KIND})")
print(f"Ligand SDF:               {LIGANDS.relative_to(REPO_ROOT)}")
print(f"Output dir:               {OUT_DIR.relative_to(REPO_ROOT)}")
print(f"Binding-site center (Å):  {BINDING_SITE_CENTER}")
print(f"Binding-site radius (Å):  {BINDING_SITE_RADIUS}")
print(f"Search box size (Å):      {BOX.size[0]:.2f} per edge")
print(f"Library sample:           {SAMPLE_N or 'all'} compounds")
print(f"Exhaustiveness:           {EXHAUSTIVENESS}  (poses per compound: {NUM_MODES})")
"""),

        markdown("""
## 3. Redock the reference ligand (methodological sanity check)

### Background

Before we trust docking on unknown ligands, we re-dock a *known* one. The crystallographic ligand E94 from the 4FV7 structure has an experimentally measured pose; if our gnina setup is sound, it should re-discover that pose to within ~2 Å heavy-atom RMSD. This is the standard "does the docker work for this target?" test, and it costs only ~30 s of CPU.

We use the **crystal receptor** (`erk2_4fv7.pdb`), not the AlphaFold model, for this check — the experimental pose was measured on the crystal, so comparing against it on a predicted structure mixes two unknowns. The library-docking step below uses the AlphaFold model as you would in a real screen.

`autobox_ligand` tells gnina to derive the search box from the reference ligand's own coordinates (plus a 4 Å margin), so we are not biasing the redock with our hand-chosen `BINDING_SITE_*` numbers. Higher exhaustiveness (16 vs 8) gives the docker more sampling rounds and tightens the confidence interval on the answer.
"""),

        code(title="Redock the reference ligand against the crystal receptor", source="""
redock = redock_reference(
    receptor=CRYSTAL_RECEPTOR,
    ref_ligand=CRYSTAL_LIGAND,
    out_dir=OUT_DIR / "redock",
    autobox_ligand=CRYSTAL_LIGAND,
    autobox_add=4.0,
    exhaustiveness=16,
    num_modes=9,
    seed=42,
)
print(f"Redock RMSD vs crystal:  {redock['rmsd_to_crystal']:.2f} Å   (target: < 2.0)")
print(f"Top-pose CNN score:      {redock['cnn_score']:.3f}")
print(f"Top-pose CNN affinity:   {redock['cnn_affinity']:.2f}  (pK_d)")
print(f"Top-pose Vina affinity:  {redock['affinity']:.2f} kcal/mol")
print(f"Poses file:              {redock['poses_sdf']}")
"""),

        markdown("""
### How to read the redock result

- **RMSD < 2 Å** → the docking workflow works on this target. Proceed.
- **2–4 Å** → the docker found a sub-pocket of the binding site but not the exact pose. Still acceptable for ranking; not great for pose-level claims.
- **> 4 Å** → the docker is not finding the correct binding mode. Common causes: the search box is in the wrong place, the receptor has a side-chain in a clashing rotamer, or the reference ligand is bulky enough that the rigid-receptor assumption breaks. Inspect by eye (next cell) before trusting library-scale results.

Be aware: 2 Å is a *workflow-validation* threshold, not a measure of pose accuracy on new ligands. New compounds may dock less accurately if their chemistry differs substantially from the reference.
"""),

        code(title="3-D viewer: crystal pose (green) vs gnina-predicted pose (cyan)", source="""
view = py3Dmol.view(width=600, height=500)
view.removeAllModels()
view.addModel(CRYSTAL_RECEPTOR.read_text(), "pdb")
view.setStyle({"cartoon": {"color": "gold", "opacity": 0.6}})

# Crystal pose (experimental ground truth)
view.addModel(CRYSTAL_LIGAND.read_text(), "pdb")
view.setStyle({"model": 1}, {"stick": {"colorscheme": "greenCarbon"}})

# gnina top-1 redocked pose
redock_poses = Path(redock["poses_sdf"])
view.addModel(redock_poses.read_text(), "sdf")
view.setStyle({"model": 2}, {"stick": {"colorscheme": "cyanCarbon"}})

view.zoomTo({"model": 1})
view.show()
"""),

        markdown("""
### Reading the overlay

Green sticks = experimental crystallographic pose. Cyan sticks = gnina's top-ranked prediction. If the workflow is sound, the two will overlap almost atom-for-atom; if not, the cyan pose will sit nearby (same pocket, wrong orientation) or in a completely different place.

The most common failure mode is **flipping**: the docker places the same atoms but rotates the ligand 180° within the pocket. This counts as a redock failure even though most of the contacts are still present. Inspect the orientation of any uniquely-identifiable atom (a chlorine, a sulfonyl group) to check.
"""),

        markdown("""
## 4. Dock the prepared library

### Background

Now the production step: each ligand in `LIGANDS` is docked into the binding site, and gnina returns up to `NUM_MODES` poses per compound. The default scoring mode is `--cnn_scoring rescore` — Vina does the conformational search (fast), gnina's CNN rescores the final pose (more accurate than the empirical Vina function on its own). Higher accuracy modes (`refinement`, `all`) are slower and pay off most on hard cases; for a first-pass triage `rescore` is the right choice.

This cell is **idempotent**: it writes `poses.sdf` and `gnina_scores.csv` under `OUT_DIR` and skips on re-run if both exist. To force a redock, pass `overwrite=True` or delete the cache. Re-running this cell after editing only the analysis code does *not* re-run gnina.

### Why we sample a subset by default

Docking 25 compounds takes ~5 min on a Colab CPU; 100 takes ~20–30 min. For the first pass — verifying the receptor + box are right and the output looks reasonable — keep `SAMPLE_N` small. Bump it up once you trust the setup. The cell below will warn if the full library is much larger than the sample.
"""),

        code(title="Dock the library (the long step — ~5 min per 25 compounds at exh=8)", source="""
ligands_to_dock = LIGANDS
if SAMPLE_N is not None:
    # Make a temporary subset SDF so gnina only sees the first SAMPLE_N compounds.
    subset_sdf = OUT_DIR / f"ligands_subset_n{SAMPLE_N}.sdf"
    sup = Chem.SDMolSupplier(str(LIGANDS), removeHs=False)
    writer = Chem.SDWriter(str(subset_sdf))
    n = 0
    for mol in sup:
        if mol is None:
            continue
        writer.write(mol)
        n += 1
        if n >= SAMPLE_N:
            break
    writer.close()
    print(f"Subset SDF: {subset_sdf.relative_to(REPO_ROOT)}  ({n} compounds)")
    ligands_to_dock = subset_sdf

scores = dock_library(
    receptor=RECEPTOR,
    ligands_sdf=ligands_to_dock,
    out_dir=OUT_DIR,
    box=BOX,
    exhaustiveness=EXHAUSTIVENESS,
    num_modes=NUM_MODES,
    cnn_scoring="rescore",
    seed=42,
)
print(f"Docked {scores['compound_id'].nunique():,} compounds → {len(scores):,} poses")
scores.head(10)
"""),

        markdown("""
### How to read the score table

One row per pose; multiple poses per compound. The score columns:

- **`affinity`** — Vina/smina-style binding free energy in kcal/mol. **More negative = stronger predicted binding.** Typical values for a kinase inhibitor: −6 to −12 kcal/mol. Anything above (less negative than) −5 is weak.
- **`cnn_score`** — gnina's CNN classification score for "this looks like a real binding pose", in 0–1. **Higher = better.** Trained on PDBbind positive / decoy pairs.
- **`cnn_affinity`** — CNN-predicted binding affinity in **pK_d** (i.e. `−log10(K_d)`). **Higher = stronger.** Typical drugs: 6–10. Sub-micromolar binders score ≥ 6.
- **`cnn_vs`** — combined virtual-screening score (`cnn_score × cnn_affinity`); good for ranking across compounds.
- **`minimized_rmsd`** — RMSD between the input pose and gnina's iteratively-refined pose. Only populated under `--cnn_scoring refinement` or `--cnn_scoring all` (the CNN-guided refinement modes). We use `rescore` here (Vina docks + CNN scores once at the end, no refinement), which is faster and standard for first-pass triage — so this column is `None` for every row. Switch to `refinement` mode if you want these values; it costs ~2–5× the wall time per compound.
- **`pose_rank`** — 1 = gnina's top pick for that compound. Subsequent ranks are alternative binding modes the docker considered, ordered by score.

A common starting view is to keep only the top-1 pose per compound:

```python
top1 = scores[scores["pose_rank"] == 1].sort_values("cnn_affinity", ascending=False)
```

Notebook `04_score_classical` will rescore these with an interaction-fingerprint + machine-learning model trained on real activity data; the gnina scores here are a useful baseline, not the final word.
"""),

        code(title="Top-1-pose-per-compound view, sorted by CNN affinity", source="""
top1 = (
    scores[scores["pose_rank"] == 1]
    .sort_values("cnn_affinity", ascending=False)
    .reset_index(drop=True)
)
print(f"{len(top1):,} compounds in top-1-pose view")
top1.head(15)
"""),

        code(title="Affinity distributions — quick visual sanity check", source="""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
sns.histplot(top1["affinity"], ax=axes[0], bins=20, color="steelblue")
axes[0].set_xlabel("Vina affinity (kcal/mol)  — lower is better")
sns.histplot(top1["cnn_score"], ax=axes[1], bins=20, color="seagreen")
axes[1].set_xlabel("CNN score (0–1)  — higher is better")
sns.histplot(top1["cnn_affinity"], ax=axes[2], bins=20, color="darkorange")
axes[2].set_xlabel("CNN affinity (pK_d)  — higher is better")
fig.suptitle("Distribution of top-1-pose scores across the docked library")
fig.tight_layout()
plt.show()
"""),

        markdown("""
### What the distributions tell you

A well-behaved screen against a kinase should show:

- **Vina affinity** centred around −7 to −9 kcal/mol, with a tail towards −10 to −12 for the strongest hits.
- **CNN score** distributed across the full 0–1 range (it is trained as a classifier; most poses get a moderate score, the best get > 0.8).
- **CNN affinity** centred around 5–7 pK_d for a generic screening library, with the strongest hits ≥ 8.

Watch out for:

- A *single peak* at −5 kcal/mol or shallower → the docker is failing to find the pocket (the search box may be wrong or off-center).
- A *bimodal* distribution → the library has two distinct chemotypes, one of which fits and one of which does not. Common with vendor catalogues. Filter further before downstream rescoring.
- CNN scores all stuck near 0 → the receptor protonation / preparation may be off; the CNN was trained on protonated PDBbind receptors.
"""),

        markdown("""
## 5. PoseBusters QC — flag physically implausible poses

### Background

Docking scoring functions tolerate cheating. A pose with a broken bond, an inside-out ring, or an atom inside a protein residue can still score well — the scoring function doesn't bake in every chemistry rule. **PoseBusters** runs about 20 explicit geometric and chemical checks on every pose and reports a per-check pass/fail:

- **Bond geometry** — bond lengths and angles match canonical ranges.
- **Stereochemistry** — chiral centres preserved relative to the input.
- **Ring planarity / aromaticity** — aromatic rings remain flat, no inverted chair conformations.
- **Internal clashes** — no overlapping atoms within the ligand.
- **Protein–ligand clashes** — the ligand does not overlap with any protein heavy atom (when `mol_cond` is provided).
- **Bound geometry** — pose makes sense given the binding site (volume overlap, no atoms outside the box, etc.).

A pose that fails PoseBusters is, almost always, not a real binding mode no matter what gnina's CNN said. In a screening shortlist you want only PoseBusters-clean poses.
"""),

        code(title="Run PoseBusters on every pose", source="""
poses_sdf = OUT_DIR / "poses.sdf"
pb_df = run_posebusters(poses_sdf, receptor=RECEPTOR, mode="dock")

print(f"PoseBusters ran on {len(pb_df):,} poses; columns = checks + pb_passes_all + first_failing")
print(f"Pass rate (all checks):  {pb_df['pb_passes_all'].mean():.1%}")

# Per-check pass rate
bool_cols = [c for c in pb_df.columns if pb_df[c].dtype == bool and c != "pb_passes_all"]
per_check = pb_df[bool_cols].mean().sort_values()
print()
print("Lowest pass-rate checks (most common failures):")
print(per_check.head(10).map("{:.1%}".format).to_string())
"""),

        markdown("""
### Reading the pass-rate table

A healthy screening run looks like:

- **Overall pass rate ≥ 80%** — most poses survive QC. Below 50% usually means the receptor was prepared badly (missing atoms, wrong protonation, or steric clashes baked into the model).
- **`bond_lengths` / `bond_angles`** — should be > 99% pass. Failures here suggest a corrupt SDF input.
- **`mol_pred_loaded` / `sanitization`** — should be 100%. Failures indicate gnina wrote a molecule RDKit cannot re-read.
- **`internal_steric_clash` / `protein-ligand_steric_clashes`** — often the lowest pass rates. A few % clash is acceptable; > 30% means the search box is positioned wrong (the docker is forcing the ligand into a steric wall) or the receptor needs cleaning.

Compounds with all top-`NUM_MODES` poses failing should be dropped before downstream rescoring.
"""),

        code(title="Merge PoseBusters flags into the gnina score table", source="""
# PoseBusters returns one row per pose in the order gnina wrote them to
# poses.sdf, which is the same order as the rows in `scores`. We align by
# index position. Length-check first so any future drift surfaces loudly.
scores_qc = scores.reset_index(drop=True).copy()
pb_aligned = pb_df.reset_index(drop=True)
assert len(scores_qc) == len(pb_aligned), (
    f"row count mismatch: gnina has {len(scores_qc)} poses, "
    f"PoseBusters has {len(pb_aligned)} rows. The merge would be wrong; "
    "check whether PoseBusters was given the same SDF the scores were parsed from."
)
assert "pb_passes_all" in pb_aligned.columns, (
    "pb_passes_all column missing from PoseBusters output. Did the bool-column "
    "detection in run_posebusters fail? Check pb_df.dtypes."
)

# Cast to numpy bool / str so dtype-flip surprises (e.g. object-dtype bool
# Series that don't AND together correctly) can't bite us downstream.
scores_qc["pb_passes_all"] = pb_aligned["pb_passes_all"].to_numpy(dtype=bool)
if "first_failing" in pb_aligned.columns:
    scores_qc["first_failing"] = pb_aligned["first_failing"].astype(str).to_numpy()

# Top-1, PoseBusters-clean. Use pb_passes_all directly, no `.get()` fallback —
# we just asserted the column exists.
top1_qc = (
    scores_qc[(scores_qc["pose_rank"] == 1) & scores_qc["pb_passes_all"]]
    .sort_values("cnn_affinity", ascending=False)
    .reset_index(drop=True)
)
n_clean = int(scores_qc["pb_passes_all"].sum())
print(
    f"{len(top1_qc):,} compounds have a PoseBusters-clean top-1 pose "
    f"(out of {scores_qc['compound_id'].nunique()} compounds, "
    f"{n_clean}/{len(scores_qc)} poses PoseBusters-clean overall)."
)
top1_qc.head(15)
"""),

        markdown("""
## 6. Visualise the top hits

### Background

Before handing a shortlist over to medicinal chemistry, look at the top poses by eye. The features to check on each compound:

- The ligand sits *inside* the binding pocket (not floating in solvent, not poking out of the back).
- Recognisable interactions are present — for a kinase, the hinge backbone hydrogen bond is the classic.
- No obvious clashes (atoms inside protein residues, distorted rings).
- The ligand shape *fits* — flat aromatic rings stacking against flat protein surfaces, polar groups facing polar residues.

A surprising number of "good"-scoring poses look pathological under visual inspection. This is the cheapest sanity check there is.
"""),

        code(title="3-D viewer: receptor + top-N CNN-affinity hits overlaid", source="""
TOP_N = 5

# Reload poses.sdf, group by compound, keep the rank-1 pose for the top-N compounds.
sup = Chem.SDMolSupplier(str(poses_sdf), removeHs=False)
all_poses = [m for m in sup if m is not None]

top_ids = top1_qc.head(TOP_N)["compound_id"].tolist()
top_poses = []
seen = set()
for mol in all_poses:
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    if name in top_ids and name not in seen:
        top_poses.append(mol)
        seen.add(name)

print(f"Showing {len(top_poses)} top compounds: {top_ids}")

view = py3Dmol.view(width=650, height=520)
view.removeAllModels()
view.addModel(RECEPTOR.read_text(), "pdb")
view.setStyle({"cartoon": {"color": "gold", "opacity": 0.55}})

palette = ["cyanCarbon", "magentaCarbon", "yellowCarbon", "greenCarbon", "orangeCarbon"]
for i, mol in enumerate(top_poses):
    view.addModel(Chem.MolToMolBlock(mol), "mol")
    view.setStyle({"model": i + 1}, {"stick": {"colorscheme": palette[i % len(palette)]}})

# Centre on the binding-site centroid we used as the docking target.
view.zoomTo({"model": 1})
view.show()
"""),

        markdown("""
## 7. Save outputs

### Background

Two artefacts are the contract between this notebook and the next:

- `data/derived/<target>/docking/poses.sdf` — all docked poses, with gnina's score tags. Notebook `04` will compute interaction fingerprints on these.
- `data/derived/<target>/docking/gnina_scores.csv` — the tidy score table, with PoseBusters flags merged in.

We rewrite the CSV with the QC-augmented columns so downstream code only has to read one file.
"""),

        code(title="Save the QC-augmented score table", source="""
scores_path = OUT_DIR / "gnina_scores.csv"
scores_qc.to_csv(scores_path, index=False)
print(f"Wrote {len(scores_qc):,} rows → {scores_path.relative_to(REPO_ROOT)}")
print(f"poses.sdf already at:  {(OUT_DIR / 'poses.sdf').relative_to(REPO_ROOT)}")
"""),

        markdown("""
## Recap

### Biomedical takeaway

We took a library of drug-like compounds and asked, for each one, "could this fit in the ATP pocket of ERK2?". The answer is a 3-D pose plus three scoring numbers (Vina affinity, CNN score, CNN affinity). The redock sanity check told us our docking workflow recovers a known crystallographic pose within ~2 Å — i.e. the workflow is sound for this target. PoseBusters then weeded out poses that look fine on paper but are physically impossible. The PoseBusters-clean, top-1-pose-per-compound subset, ordered by CNN affinity, is the input to the next round of analysis.

In a real wet-lab triage, this list would be the candidates worth interrogating with the more expensive rescorer (notebook `04`) and the orthogonal co-folding affinity check (notebook `05`); only compounds that survive *both* downstream steps will reach the shortlist.

### Technical takeaway

`src/aidd/docking.py` wraps the gnina CLI behind three composable functions: `dock_library` (batch over an SDF), `redock_reference` (single-ligand sanity check), and `run_posebusters` (per-pose QC). The module is platform-aware: it refuses loudly on Windows / macOS where gnina cannot run, and trivially extensible to a different docker (Vina, AutoDock4, smina) by swapping the CLI invocation. Idempotent caching at the disk level means re-running the notebook after an analysis edit does not re-run the (expensive) docker.

### What's next in the pipeline

- **`04_score_classical.ipynb`** — compute interaction fingerprints (ProLIF) on these poses and train a machine-learning rescorer against the ERK2 activity labels.
- **`05_dock_boltz.ipynb`** — alternative fast lane: Boltz-2 co-folds protein + ligand and outputs an affinity score, without classical docking. The orthogonal signal for the consensus shortlist.
- **`06_consensus_and_shortlist.ipynb`** — join the rescorer (notebook 04) and Boltz-2 (notebook 05) outputs; emit `shortlist.sdf` for wet-lab follow-up.

### Further reading

- McNutt et al., *J. Cheminform.* (2021), **13**, 43 — *GNINA 1.0: molecular docking with deep learning.* [doi:10.1186/s13321-021-00522-2](https://doi.org/10.1186/s13321-021-00522-2)
- Buttenschoen, Morris, Deane, *Chem. Sci.* (2024), **15**, 3130 — *PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences.* [doi:10.1039/D3SC04185A](https://doi.org/10.1039/D3SC04185A)
- Trott & Olson, *J. Comput. Chem.* (2010), **31**, 455 — the AutoDock Vina paper (the empirical scoring function gnina is built on). [doi:10.1002/jcc.21334](https://doi.org/10.1002/jcc.21334)
- Volkamer Lab **TeachOpenCADD** — *Talktorial T015: Protein-ligand docking* covers the same material in a different style. [projects.volkamerlab.org/teachopencadd](https://projects.volkamerlab.org/teachopencadd/)
"""),
        accelerator="GPU",
        gpu_type="T4",
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
