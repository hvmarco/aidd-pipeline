"""Builder for 02_prepare_ligands.ipynb.

Source of truth for the ligand-prep notebook. Cells appear below in the
order they will render. Never edit the .ipynb directly — see ``CLAUDE.md``
§ *Notebook workflow*.

Regenerate:
    python notebooks/_build_02_prepare_ligands.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import AUTORELOAD_SNIPPET, code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "02_prepare_ligands.ipynb"


def build() -> None:
    nb = notebook(
        markdown("""
# 02 — Prepare ligands: from SMILES to docking-ready 3-D molecules

**aidd-pipeline · Notebook 2 of the screening workflow**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hvmarco/aidd-pipeline/blob/main/notebooks/02_prepare_ligands.ipynb)

Before we can compare candidate molecules against a protein target by docking, we need to clean and standardise them. This notebook takes a library of compounds expressed as **SMILES strings** (a plain-text way of writing molecules), pushes each one through a series of cheminformatics checks, and writes out a file of 3-D molecules ready for the docking step.

The same workflow is used in essentially every modern virtual-screening project, from academic kinase libraries to industrial million-compound screens. We use the open-source [RDKit](https://www.rdkit.org/) and [datamol](https://datamol.io/) libraries; everything is reproducible and free.

## Learning objectives

After running this notebook you will be able to:

- Read SMILES strings and understand what they encode.
- Explain why we **standardise** molecules before computing anything on them.
- Read the classic drug-likeness descriptors (MW, LogP, HBD/HBA, TPSA, rotatable bonds, QED) and what "good" looks like for an oral drug.
- Apply Lipinski's **Rule of 5** and Veber's rules as filters.
- Recognise **PAINS** — molecules notorious for false-positive activity in screening assays — and flag them.
- Generate 3-D conformers from a SMILES string and write them to an SDF file.

## Audience

- A clinician / biologist who wants to understand the *triage* step: why some compounds get filtered out before they ever touch the wet lab.
- A data / ML person new to medicinal chemistry — the descriptors and gates here become features in later models.
- A student writing a thesis chapter on virtual screening.

## Prerequisites

- The `aidd` conda environment is created (`conda env create -f environment.yml`), or Colab installs are run.
- A SMILES library exists at `data/compounds/erk2/training_small.smi` (default, ~3.7k ERK2 compounds, ships with the repo).

## Runtime

About **1 minute for 500 compounds** in serial mode on a laptop. Scales linearly. The full 47k ERK2 library runs in **~10 minutes on Colab** (Linux + parallel workers in a single cell) or **~10–15 minutes locally** as a parallel script. See *Section 2* below on how to set `N_WORKERS` for your platform.
"""),

        markdown("""
## Key terms used in this notebook

| Term | Meaning in one line |
|---|---|
| **SMILES** | *Simplified Molecular-Input Line-Entry System* — a text string that encodes a molecular structure (e.g. `CC(=O)Oc1ccccc1C(=O)O` is aspirin). |
| **Standardise** | Convert different valid SMILES of the same compound into one canonical form (remove counter-ions, neutralise charges, fix tautomers, etc.). |
| **Tautomer** | Two structurally distinct forms of the same compound that interconvert by moving a hydrogen (e.g. keto ↔ enol). |
| **Salt / counter-ion** | An ion (often Na⁺, Cl⁻) co-crystallised with the active compound. We usually only want the active part. |
| **Descriptor** | A number computed from a molecule's structure that captures one of its physical / chemical properties (size, lipophilicity, etc.). |
| **MW** | Molecular weight, in Daltons (Da). Roughly the heaviness of the molecule. |
| **LogP** | The base-10 log of the partition coefficient between octanol and water. Positive = fat-loving (lipophilic), negative = water-loving (hydrophilic). |
| **HBD / HBA** | Hydrogen-bond donors / acceptors — atoms that can donate or accept a hydrogen bond. |
| **TPSA** | Topological polar surface area, in Å². Sum of the areas of polar atoms; correlates with membrane permeability and oral absorption. |
| **Rotatable bonds** | Single bonds that can rotate freely; high values mean the molecule is floppy. |
| **QED** | *Quantitative Estimate of Drug-likeness* — a 0–1 score combining several descriptors. Closer to 1 is more drug-like. |
| **Lipinski's Rule of 5 (Ro5)** | Classic 1997 heuristic: an orally available drug typically has MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10. |
| **Veber rules** | Added 2002: TPSA ≤ 140 Å² and rotatable bonds ≤ 10. |
| **PAINS** | *Pan-Assay Interference compoundS* — substructures that produce false-positive signals across many unrelated assays. |
| **Conformer** | A specific 3-D arrangement of a molecule's atoms (rotamers around single bonds give different conformers). |
| **ETKDGv3** | RDKit's current default conformer-generation method — *Experimental-Torsion / basic Knowledge Distance Geometry, version 3*. |
| **MMFF / UFF** | *Merck Molecular / Universal Force Field* — fast classical force fields used to relax the 3-D geometry. |
| **SDF** | *Structure-Data File* — a plain-text format storing 3-D molecules + arbitrary properties. The standard format for docking input. |
"""),

        markdown("""
## Why ligand prep matters — the biological reasoning

Public compound databases (ChEMBL, PubChem, ZINC, Enamine REAL) contain billions of molecules in heterogeneous formats. The same compound may be listed:

- As its **salt** form (e.g. "diphenhydramine hydrochloride") in one source, **free base** in another.
- In one **tautomer** in one source, another in the next.
- With or without explicit hydrogen atoms.
- With or without stereo information.
- As a SMILES string that, while valid, has redundant brackets or alternative atom ordering.

If we computed properties or docking poses on these raw strings, we would get inconsistent results that depend on *how the source happened to write the molecule*, not on the molecule itself. **Standardisation** fixes that by reducing every molecule to one canonical form. It is the cheapest, most reliable improvement you can make to a virtual-screening pipeline.

Beyond standardisation, the **drug-likeness filters** (Lipinski, Veber, QED) prune compounds that are unlikely to ever become useful oral drugs even if they were active. Spending GPU time docking molecules the size of a small peptide is rarely worth it. The **PAINS** filter goes further and removes notorious assay artefacts — molecules that *look* like they bind in many assays for chemical reasons unrelated to genuine biological activity (redox cycling, fluorescence quenching, covalent reactivity, etc.).

One caveat: these filters are heuristics, not laws. There are FDA-approved drugs that violate Ro5 (most macrocycles, some kinase inhibitors). For first-pass triage they are sensible defaults; for a hit-to-lead campaign on a specific target, you would loosen them based on what is known about the chemotype.
"""),

        markdown("""
## 1. Setup

Detect Colab vs local, set the import path, and turn on `%autoreload` so edits to `src/aidd/` propagate without restarting the kernel.
"""),

        code(title="Setup: detect Colab vs local, install pip extras, clone repo", source="""
import sys
import importlib
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    !pip install -q rdkit datamol "prolif>=2.0" posebusters meeko py3Dmol biopython scikit-learn xgboost lightgbm
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
import py3Dmol
from rdkit import Chem

from aidd.ligands import (
    read_smiles, prepare, prepare_library, write_sdf,
    LIPINSKI_RULES, VEBER_RULES,
)
from aidd.io import mount_drive_if_colab, pretty_path

# Set up the data/derived/ root. On Colab this mounts Google Drive (one
# OAuth prompt on first call per runtime, then silent) and resolves to a
# Drive-backed path so the prepared SDF survives runtime deaths.
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
## 2. Inputs — an ERK2 compound library

### Background

The repo ships with two ERK2 libraries, both from the Leiden course:

- `data/compounds/erk2/training_small.smi` — **3,703 compounds** (default for this notebook). Fast for demos and teaching. Compound IDs of the form `ERKxxxxxxxxx`.
- `data/compounds/erk2/training.smi` — **47,203 compounds** with numeric IDs that match the activity labels in `data/labels/erk2_training.tsv`. Use this when you need labelled data for training a downstream rescorer.

The library is a mix of confirmed actives and inactives from screening experiments. Real-world libraries (e.g. a vendor catalogue or your in-house collection) look exactly like this — a `.smi` file with one molecule per line.

We set two key parameters:

- **`SAMPLE_N`** — how many compounds to process in this notebook run. Default 500 keeps wall time under a minute. Set to `None` to process every compound in the file.
- **`N_WORKERS`** — number of parallel CPU processes. How to pick:

| Where you're running | Recommended `N_WORKERS` | Why |
|---|---|---|
| **Colab** (Linux) | **4–8** | Linux uses `fork` for new processes; parallel work runs cleanly in a Jupyter cell. The Colab machine usually has 2–4 cores; set `N_WORKERS = 4` and you're set. |
| **Linux Jupyter** (local) | 4–`os.cpu_count()` | Same as Colab. |
| **Windows VS Code / JupyterLab** | **1** (safe), 2–4 (usually works) | Windows uses `spawn`; our function uses `functools.partial` (picklable), so parallel often works, but some Jupyter configurations still fight it. If `N_WORKERS = 4` crashes in your Jupyter, drop to `1` or run the script footer at the bottom of this notebook. |
| **macOS Jupyter** (local) | Same as Windows | macOS uses `spawn` since Python 3.8. |
| **Plain Python script** (any OS) | `os.cpu_count()` | No Jupyter weirdness. Use the script template in the recap. |
"""),

        code(title="Inputs: SMILES file, labels, output path + run parameters", source="""
INPUT_SMI = REPO_ROOT / "data" / "compounds" / "erk2" / "training_small.smi"
LABELS    = REPO_ROOT / "data" / "labels" / "erk2_training.tsv"
OUTPUT_SDF = DATA_ROOT / "erk2" / "ligands_prepared.sdf"

SAMPLE_N = 500              # set to None to process every compound in the file
N_WORKERS = 4 if IS_COLAB else 1   # see the table above for guidance

assert INPUT_SMI.exists(), f"missing {INPUT_SMI}"
print(f"Input:     {pretty_path(INPUT_SMI, DATA_ROOT, REPO_ROOT)}")
print(f"Output:    {pretty_path(OUTPUT_SDF, DATA_ROOT, REPO_ROOT)}")
print(f"N_WORKERS: {N_WORKERS}  ({'Colab Linux' if IS_COLAB else 'local'})")
"""),

        code(title="Read SMILES file → DataFrame", source="""
df_in = read_smiles(INPUT_SMI)
print(f"Loaded {len(df_in):,} compounds")
df_in.head()
"""),

        markdown("""
### Reading a SMILES string

Each row above has two columns: a SMILES string and a compound ID. To get a feel for SMILES, here is a quick decoding key:

| SMILES fragment | Means |
|---|---|
| `C` | carbon (the `H`s are implicit) |
| `c1ccccc1` | a benzene ring (lower-case = aromatic) |
| `=O`, `=N` | double bond to O / N |
| `(...)` | branch |
| `[N+]`, `[O-]` | charged atoms |
| `/` `\\` | E/Z stereochemistry |
| `@`, `@@` | tetrahedral stereo (R/S) |

So `CC(=O)Oc1ccccc1C(=O)O` parses as *methyl–carbonyl–oxygen–benzene–carboxylic acid* = **aspirin**. The same molecule can be written many ways, hence the need for standardisation.
"""),

        markdown("""
## 3. Run the full prep pipeline

### Background

The function `prepare_library` runs six steps on every compound, with a progress bar:

1. **Parse** the SMILES into an RDKit molecule object.
2. **Standardise** with [datamol](https://datamol.io/) — uncharge, desalt, normalise functional groups, canonicalise.
3. **Compute descriptors** — MW, LogP, HBD, HBA, TPSA, rotatable bonds, ring count, heavy-atom count, QED.
4. **Check drug-likeness gates** — Lipinski Ro5, Veber, QED ≥ 0.5.
5. **Flag PAINS** — run the molecule against RDKit's combined PAINS-A/B/C catalogue.
6. **Embed in 3-D** — generate one conformer with ETKDGv3, then optimise its geometry with the MMFF force field (falling back to UFF if MMFF fails).

If any step fails (parse, standardise, embed), the row is marked `ok = False` with a `fail_reason` explaining which step. We carry the failures through to the end so we can audit them.
"""),

        code(title="Run the prep pipeline (standardise → properties → PAINS → 3-D embed)", source="""
df_subset = df_in.sample(n=min(SAMPLE_N, len(df_in)), random_state=42).reset_index(drop=True) if SAMPLE_N else df_in
print(f"Preparing {len(df_subset):,} compounds…")

df = prepare_library(df_subset, n_workers=N_WORKERS, progress=True)
print(f"Done. ok={df['ok'].sum():,}/{len(df):,}  ({df['ok'].mean():.0%} success)")
"""),

        markdown("""
### Interpreting the success rate

A success rate of **95–100%** is normal for a well-curated commercial or academic library. Lower rates suggest the source library contains unusual chemistry (organometallics, macrocycles, very large peptides) that RDKit's default embedding cannot handle. Examine `df[~df['ok']]['fail_reason'].value_counts()` to see where failures land — `standardise` failures are usually parsing issues, `embed_3d` failures are usually overly flexible or strained molecules.
"""),

        markdown("""
## 4. Property distributions

### Background

A quick visual audit of where the library sits in chemical space. The red dashed line on each plot is the Lipinski / QED threshold. Compounds to the right of the line will fail that particular rule.

What you want to see for a *general* library: a roughly bell-shaped distribution centred *below* each threshold. Many libraries are deliberately built around the Ro5 sweet spot, so the distributions can hug the lines.

What is *suspicious*: a heavy right tail in MW or LogP (probably non-drug-like chemistry; vendor catalogues sometimes contain colour pigments or polymers). Many zeros in HBD/HBA (likely fragments rather than drug-sized molecules).
"""),

        code(title="Property distributions (MW / LogP / HBD / HBA / TPSA / QED)", source="""
ok = df[df["ok"]].copy()

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
props = [
    ("mw",       "Molecular weight (Da)", 500),
    ("logp",     "LogP",                  5),
    ("hbd",      "H-bond donors",         5),
    ("hba",      "H-bond acceptors",     10),
    ("tpsa",     "Polar surface area (Å²)", 140),
    ("qed",      "QED (drug-likeness)",  0.5),
]
for ax, (col, label, threshold) in zip(axes.flat, props):
    sns.histplot(ok[col], ax=ax, bins=30, color="steelblue")
    ax.axvline(threshold, color="crimson", linestyle="--", label=f"Ro5/QED gate ≈ {threshold}")
    ax.set_xlabel(label)
    ax.legend(fontsize=8)
fig.suptitle("Property distributions (prepared library)")
fig.tight_layout()
plt.show()
"""),

        markdown("""
### Interpretation — typical values for oral drugs

For oral small-molecule drugs (the most common modality), a useful rule of thumb is:

- **MW**: 300–500 Da. Below ~200 = a fragment (different design rules); above ~700 = bioavailability problems.
- **LogP**: 1–4. Negative LogP = too hydrophilic, won't cross membranes. > 5 = too lipophilic, plasma-protein binding and metabolic clearance issues.
- **HBD**: ≤ 5. Each H donor costs energy to bury.
- **HBA**: ≤ 10. Same logic, slightly looser.
- **TPSA**: < 140 Å² for oral absorption; < 90 Å² for blood–brain-barrier penetration (relevant for CNS-targeted drugs).
- **QED**: ≥ 0.5 is a useful working threshold; the *median* approved drug scores ~0.67.

These come from analyses of marketed drugs, not from first principles. They are guidance, not law.
"""),

        markdown("""
## 5. How many compounds pass each gate?
"""),

        code(title="Pass-rate table per gate (Lipinski / Veber / QED / PAINS)", source="""
rates = {
    "Standardise + 3D embed": df["ok"].mean(),
    "Lipinski (Ro5) pass":    ok["lipinski_pass"].mean(),
    "Veber pass":             ok["veber_pass"].mean(),
    "QED ≥ 0.5":              ok["qed_pass"].mean(),
    "PAINS-clean":            (ok["pains"] == "").mean(),
}
pass_all = ok["lipinski_pass"] & ok["veber_pass"] & ok["qed_pass"] & (ok["pains"] == "")
rates["All gates combined"] = pass_all.mean()

pd.Series(rates, name="pass_rate").map("{:.1%}".format).to_frame()
"""),

        markdown("""
### Sanity check

On a real-world drug-discovery library you should see roughly:

- **Standardise + 3D embed: 95–100%** — anything lower deserves investigation.
- **Lipinski: 70–90%** — most modern libraries are designed around Ro5.
- **Veber: 85–95%** — Veber is usually less stringent than Ro5 on drug-like libraries.
- **QED ≥ 0.5: 40–70%** — QED is harder to satisfy than Lipinski; this is expected.
- **PAINS-clean: 75–95%** — PAINS catalogues are known to over-flag, but 15% PAINS hits in a screening library is normal.
- **All gates combined: 30–60%** — typical real-world triage retention.

If your pass rates are far outside these ranges, look at the library composition before trusting downstream results.
"""),

        code(title="Top PAINS substructure matches", source="""
# Most common PAINS substructures hit, if any
pains_hits = ok.loc[ok["pains"] != "", "pains"].value_counts().head(10)
if len(pains_hits):
    print("Most common PAINS matches:")
    print(pains_hits.to_string())
else:
    print("No PAINS hits in this sample.")
"""),

        markdown("""
### What "PAINS hit" really means

A PAINS match is **not** a death sentence for a compound. The original Baell & Holloway analysis (2010) flagged substructures that *frequently* gave false-positive readouts across several specific high-throughput screens, but the list has known false positives of its own — some clinically successful drugs contain PAINS substructures.

Treat PAINS as a **flag**, not a hard filter: a PAINS-positive compound that shows a strong, *reproducible*, dose-dependent signal in a counter-screen is still worth following up. For an automated triage pipeline, however, dropping PAINS hits is a reasonable default — they are more likely to waste wet-lab time than not.
"""),

        markdown("""
## 6. Visualise one prepared 3-D structure

### Background

A 3-D structure is the *output* of this notebook — it is what gets fed into docking. We pick the highest-QED compound that passes every gate and view it. Things to check by eye:

- Bond lengths look like ~1.5 Å (single C-C), ~1.4 Å (aromatic), ~1.2 Å (double bonds).
- Aromatic rings are flat.
- No atoms overlap.
- The overall shape is reasonable — molecules don't fold in on themselves like a balled-up sock.

If a conformer looks pathological, the docking pose will inherit those problems.
"""),

        code(title="3-D viewer: highest-QED PAINS-clean compound", source="""
pick = ok[(ok["pains"] == "") & ok["lipinski_pass"]].sort_values("qed", ascending=False).head(1)
mol = pick.iloc[0]["mol"]
print(f"Showing {pick.iloc[0]['name'] or 'compound'}  —  QED {pick.iloc[0]['qed']:.2f}, MW {pick.iloc[0]['mw']:.0f}")

viewer = py3Dmol.view(width=500, height=400)
viewer.addModel(Chem.MolToMolBlock(mol), "mol")
viewer.setStyle({"stick": {"colorscheme": "cyanCarbon"}})
viewer.zoomTo()
viewer.show()
"""),

        markdown("""
## 7. Cross-reference with activity labels

### Background

A common pitfall: a drug-likeness gate may filter out *more actives than inactives*, especially if the screening campaign that generated the actives used a different chemotype than the rest of the library. We sanity-check by joining our prepared library to the labels file and looking at the gate pass rate split by `Active` vs `Inactive`.

If both rates are similar, the gate is target-agnostic. If actives drop noticeably more than inactives, consider relaxing the gate or using a target-aware filter.
"""),

        code(title="Cross-reference: gate pass rate by activity label (sanity check)", source=r"""
if LABELS.exists():
    labels = pd.read_csv(LABELS, sep=r"\s+")
    labels["CPD_ID"] = labels["CPD_ID"].astype(str)
    n_match = ok["name"].isin(labels["CPD_ID"]).sum()
    if n_match == 0:
        print("No name overlap between this SMILES file and the labels — skip the cross-reference.")
        print("(This is expected for training_small.smi, whose names use the 'ERKxxx' format,")
        print(" whereas the labels file uses numeric CPD_IDs from training.smi.)")
    else:
        merged = ok.merge(labels, left_on="name", right_on="CPD_ID", how="inner")
        gate_pass = merged["lipinski_pass"] & merged["veber_pass"] & merged["qed_pass"] & (merged["pains"] == "")
        merged["gate_pass"] = gate_pass
        table = (
            merged.groupby("Active")["gate_pass"]
            .agg(["size", "sum", "mean"])
            .rename(columns={"size": "count", "sum": "passed", "mean": "pass_rate"})
        )
        table["pass_rate"] = table["pass_rate"].map("{:.1%}".format)
        print(f"Matched {n_match:,} compounds to labels.")
        display(table)
else:
    print(f"Labels file not found at {LABELS}; skipping cross-reference.")
"""),

        markdown("""
### How to read the table when it does run

- `count` — number of compounds in each class (active / inactive) that we managed to prep.
- `passed` — number that survived the combined gate.
- `pass_rate` — survival fraction.

A pass-rate gap of more than ~10 percentage points between actives and inactives is worth investigating. A smaller gap is normal and acceptable.
"""),

        markdown("""
## 8. Write the prepared library to an SDF file

### Background

**SDF** (Structure-Data File) is the lingua franca for docking inputs. It stores both the 3-D coordinates and any custom properties we want to carry through. We attach the standardised SMILES and the key descriptors so the docking notebook can use them without re-computing.

We deliberately only write the gate-passing molecules. The full DataFrame (failures included) is still in memory if you want to dig into the rejects.
"""),

        code(title="Write the prepared library to SDF", source="""
passing = ok[
    ok["lipinski_pass"] & ok["veber_pass"] & ok["qed_pass"] & (ok["pains"] == "")
].copy()

n = write_sdf(
    passing,
    OUTPUT_SDF,
    mol_col="mol",
    id_col="name",
    props_to_write=[
        "smiles_std", "mw", "logp", "hbd", "hba", "tpsa", "rotbonds", "qed",
        "lipinski_violations", "veber_violations",
    ],
)
print(f"Wrote {n:,} prepared molecules → {pretty_path(OUTPUT_SDF, DATA_ROOT, REPO_ROOT)}")
"""),

        markdown("""
## Recap

### Biomedical takeaway

We turned a heterogeneous list of candidate compounds into a clean, drug-like, PAINS-screened subset with 3-D coordinates. Roughly **half of any commercial screening library survives an honest triage** — and that is the half worth spending GPU time on for docking. The combined gate concentrates wet-lab effort on the compounds most likely to *become* drugs even if they hit, rather than the ones most likely to hit at all.

### Technical takeaway

The cheminformatics pipeline lives in `src/aidd/ligands.py` as a set of composable functions: `standardise`, `compute_properties`, `check_drug_likeness`, `flag_pains`, `embed_3d`. The library-level driver `prepare_library` runs the lot over a `pandas` DataFrame in parallel via `datamol.parallelized`. The output SDF is the input to the docking notebook.

### What's next in the pipeline

- **`03_dock_gnina.ipynb`** — actually dock the prepared molecules into the ERK2 receptor, run PoseBusters QC on the resulting poses (next step in the plan).
- **`04_score_classical.ipynb`** — compute IFPs on the docked poses and train a machine-learning rescorer against the activity labels.

### Preparing the full 47k library

The simplest path depends on where you're running:

**Colab (recommended for the full library):** in *Section 2* above, set `INPUT_SMI` to `training.smi`, `SAMPLE_N = None`, and `N_WORKERS = 4`. Re-run the notebook. Wall time ≈ 10 minutes on a standard Colab CPU instance. The runtime stays within Colab's idle-disconnect window.

**Windows / macOS Jupyter:** parallel workers are unreliable from inside Jupyter on these OSes. The two safe options are (a) run the notebook in Colab as above, or (b) save the snippet below as `prepare_full.py` next to this notebook and run it from an Anaconda Prompt with `conda activate aidd; python prepare_full.py`:

```python
import os
from pathlib import Path
from aidd.ligands import read_smiles, prepare_library, write_sdf

if __name__ == "__main__":          # the __main__ guard is essential on Windows + macOS
    df = read_smiles("data/compounds/erk2/training.smi")
    out = prepare_library(df, n_workers=os.cpu_count(), progress=True)
    write_sdf(
        out[out["ok"]],
        "data/derived/erk2/ligands_prepared_full.sdf",
        props_to_write=["smiles_std", "mw", "logp", "qed", "lipinski_pass"],
    )
```

### Further reading

- Lipinski, Lombardo, Dominy, Feeney, *Adv. Drug Deliv. Rev.* (1997) — the original Rule of 5. [doi:10.1016/S0169-409X(96)00423-1](https://doi.org/10.1016/S0169-409X(96)00423-1)
- Veber et al., *J. Med. Chem.* (2002) — TPSA + rotatable-bonds criteria. [doi:10.1021/jm020017n](https://doi.org/10.1021/jm020017n)
- Bickerton et al., *Nat. Chem.* (2012) — the QED metric. [doi:10.1038/nchem.1243](https://doi.org/10.1038/nchem.1243)
- Baell & Holloway, *J. Med. Chem.* (2010) — the original PAINS filters. [doi:10.1021/jm901137j](https://doi.org/10.1021/jm901137j)
- Riniker & Landrum, *J. Chem. Inf. Model.* (2015) — ETKDG conformer-generation method used by RDKit. [doi:10.1021/acs.jcim.5b00654](https://doi.org/10.1021/acs.jcim.5b00654)
"""),
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
