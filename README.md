# aidd-pipeline

End-to-end in-silico screening pipeline for wet-lab triage in cancer drug discovery.

**Status:** 12 of 13 planned steps complete (notebooks 00–06 done and Colab-verified end-to-end on ERK2; step 13 prune pass remaining). Next: notebook 07 (variant effect prediction — AlphaMissense + RaSP ΔΔG + gnomAD, promoted from post-13-step on 2026-05-14), notebook 08 (mutation analysis, renumbered from 07), notebook 09 (small-N mechanism-of-action), notebook 99 (production runner). See [`_planning/PROJECT_PROPOSAL.md`](_planning/PROJECT_PROPOSAL.md) for the full delivery plan and decisions log; see [`_planning/MECHANISM_OF_ACTION_SCOPE.md`](_planning/MECHANISM_OF_ACTION_SCOPE.md) for the small-N MoA scope.

## What this is

Given a **protein target** (sequence or PDB, wild-type or mutant) and a **list of candidate ligands** (SMILES), produce a ranked, defensible shortlist of compounds to prioritise in the wet lab. The pipeline runs in Jupyter notebooks (Colab for GPU steps; Windows / macOS locally for CPU steps).

The headline output of a run is `shortlist.sdf` + `shortlist.csv` for a target — ranked candidates, with per-compound docking poses, interaction fingerprints, ADMET flags, and confidence from a two-method consensus. The pipeline also supports **drug-resistance / mutation-effect studies** by running against wild-type and mutant variants side-by-side.

## What the pipeline does, step by step

A plain-language tour of each stage. The technical [stack table](#stack-at-a-glance) is below; this section answers *what each step is for*. Each step lives in a numbered notebook under [`notebooks/`](notebooks/).

### 00 — Quickstart (sanity check / demo)

Take a known protein–ligand complex (ERK2 + a co-crystal inhibitor) and compute what's called an **interaction fingerprint** — a record of *which protein residues touch the ligand and what kind of contact they make* (hydrogen bond, hydrophobic packing, π-stacking, etc.). Visualises the binding pocket in 3D. Not part of production runs; proves the building blocks work and teaches the vocabulary.

### 01 — Fold the target

**Input:** a protein sequence (amino-acid string). **Output:** a 3D structure of that protein.

Uses **AlphaFold 2** via [ColabFold](https://github.com/sokrypton/ColabFold) — the neural network that won CASP14 and changed structural biology in 2021. Computes the structure de novo from sequence alone; no crystallography needed.

*Wet-lab analogue:* X-ray crystallography or cryo-EM, but in silico in ~30 min on a Colab GPU.

### 02 — Prepare the ligand library

**Input:** a list of candidate compounds as SMILES strings. **Output:** a clean SDF file of 3D molecules ready to dock.

Per compound: clean and standardise the chemistry, compute drug-likeness descriptors (molecular weight, LogP, etc.), filter out PAINS (compounds known to cause false-positive readouts in assays), and generate a 3D conformer.

*Wet-lab analogue:* triaging a compound collection before spending money testing it — discard the obvious non-drugs first.

### 03 — Dock the ligand library

**Input:** receptor (from 01) + prepared ligands (from 02). **Output:** for each ligand, the best predicted binding pose plus several scores.

Uses **[gnina](https://github.com/gnina/gnina)** — a docker that combines classical Vina-style geometry scoring with a deep-learning model that grades each pose. Includes a *redock sanity check*: re-dock the known co-crystal ligand and confirm it lands within 2 Å of its experimental pose. **PoseBusters** validates that every pose is physically sensible.

*Wet-lab analogue:* a binding assay in silico — predicts how each compound sits in the pocket and how strong the predicted interaction is.

### 04 — Train the per-target ML rescorer

**Input:** the docking scores from 03 + interaction fingerprints + known activity labels for compounds active against this target. **Output:** a trained model + per-compound rescored shortlist.

Trains a machine-learning model (random forest / gradient boosting) to learn *which features of a docked pose actually predict real activity for this specific target*. Raw docking scores are notoriously noisy; the rescorer is custom-trained per target.

*Wet-lab analogue:* building a custom scoring rubric from past hit-finding campaigns and using it to re-rank new candidates.

### 05 — Co-folding via Boltz-2 (the "fast lane")

**Input:** protein sequence + ligand SMILES, in one shot. **Output:** predicted 3D complex *and* affinity score, jointly.

Where steps 01 + 03 fold the protein and then dock the ligand into it as two separate steps, **[Boltz-2](https://github.com/jwohlwend/boltz)** does both at once. This is the open-weights equivalent of AlphaFold 3's co-folding head. The point isn't to replace 01 + 03 — it's to get an **independent second opinion** using completely different methodology.

*Wet-lab analogue:* running the same question through two orthogonal assays (e.g. SPR and DSF) and checking they agree.

### 06 — Consensus shortlist

**Input:** outputs from 04 and 05. **Output:** the final ranked shortlist (`shortlist.sdf` + `shortlist.csv`).

Keeps only the compounds where *both* methods (gnina + ML rescorer **and** Boltz-2 affinity) rank highly. The consensus filter removes single-method false positives — a compound has to convince two independent in-silico assays before it gets on the wet-lab list.

*Wet-lab analogue:* combining two assay readouts so the wet lab focuses on the strongest dual-evidence candidates.

### 07 — Predict variant effects (computational priors)

**Input:** a variant identifier (UniProt accession + position + alt amino acid). **Output:** per-variant computational priors — pathogenicity, stability change, allele frequency.

Combines three published tools: **AlphaMissense** (DeepMind, 2023) for sequence/evolution-based pathogenicity probability, **RaSP** (eLife, 2023) for structure-based ΔΔG protein-stability prediction, and **gnomAD** for population allele frequency. These priors are independent of the binding-pose pipeline and capture mechanisms (variant destabilises the fold, variant is evolutionarily intolerable, variant is common in this ancestry) that binding-pose analysis alone misses.

Especially important for the slow-vs-rapid acetylator story: protein-stability change is the dominant mechanism behind many slow-acetylator phenotypes, and RaSP ΔΔG captures it directly.

*Wet-lab analogue:* the chart-review step before ordering a functional assay — "is this variant likely pathogenic on paper before we spend money characterising it?"

### 08 — Mutation analysis

**Input:** two completed pipeline runs — one on wild-type, one on a mutant variant. **Output:** side-by-side comparison, including the variant priors from notebook 07.

Tells you how the mutation reshaped the binding pocket, which interactions are gained / lost, which compounds drop out of the shortlist, and which new ones appear. The whole point in cancer drug discovery: **drug-resistance studies** — EGFR T790M (osimertinib resistance), BRAF V600E, KIT D816V, and so on.

*Wet-lab analogue:* testing your inhibitor panel against a known resistance-mutation construct.

### 09 — Mechanism of action / small-N

**Input:** a target + a handful (N ≤ 10) of candidate compounds + optional variant list. **Output:** one HTML report per (compound × variant) pair plus a summary CSV.

The single-compound or small-N companion to the library-screening flow. Different scientific question (mechanism-of-action investigation, pharmacogenomic-variant-effect prediction) and different output shape (per-compound clinical-decision-support card, not a ranked library shortlist). Five demos covering CRC pharmacogenomics (NAT2 + isoniazid, CYP2D6 + tamoxifen, DPYD\*2A + 5-FU, irinotecan + UGT1A1\*28) and one oncogenic driver (sotorasib + KRAS G12C). See [`_planning/MECHANISM_OF_ACTION_SCOPE.md`](_planning/MECHANISM_OF_ACTION_SCOPE.md).

*Wet-lab analogue:* the in-silico version of a focused enzyme kinetics or binding study on one drug-target pair, before ordering the wet-lab assay.

### 99 — Production runner

The single notebook for routine use once the methodology is established. Chains 01 → 06 + 07 + 08 (per mutant) — or 09 in MoA mode — into one click-and-run. Has a `RUN_MODE = library | moa` toggle (library mode runs the full 1k–10k-compound screen; moa mode runs the small-N per-compound report flow). Tighter teaching content than 00–09 — it's the audited version reviewers and funders read. Includes an optional AlphaFold-3 toggle (academic-access only, off by default; see [§9 of the proposal](_planning/PROJECT_PROPOSAL.md)).

*Wet-lab analogue:* the lab's SOP — same protocol, applied to whatever target + library you give it.

## The whole flow in one diagram

Library screening (notebooks 00–06):

```
target sequence ──► 01 fold ──┐
                              ├──► 03 dock ──► 04 ML rescore ──┐
SMILES library ──► 02 prep ───┤                                ├──► 06 consensus ──► shortlist.sdf
                              └──► 05 Boltz-2 co-fold + score ─┘
```

Variant + MoA (notebooks 07–09; build on the library flow):

```
variant ID ──► 07 variant_effect ──┬──► 08 mutation_analysis (library-side WT vs mutant diff)
              prediction           │
              (AlphaMissense       └──► 09 moa_small_n (per-compound HTML reports for N ≤ 10)
               + RaSP ΔΔG
               + gnomAD)
```

Notebook 99 ties everything into one runner with a `RUN_MODE = library | moa` toggle.

## Stack at a glance

| Stage | Tool | Where it runs |
|---|---|---|
| Ligand standardisation, drug-likeness gate | RDKit + DataMol | Local (CPU) |
| Target structure prediction | ColabFold (AlphaFold front-end) | Colab (GPU) |
| Pocket detection (if needed) | fpocket | Local |
| Classical docking (interpretable lane) | gnina (Vina-compatible + CNN rescoring) | Local (CPU) / Colab |
| Co-folding + affinity (fast lane) | Boltz-2 | Colab (GPU) |
| Pose QC | PoseBusters | Local |
| Interaction fingerprints | ProLIF | Local |
| ML rescoring (per-target) | scikit-learn / XGBoost | Local |
| Consensus rank + report | pandas | Local |
| Variant priors (pathogenicity, ΔΔG, allele frequency) | AlphaMissense + RaSP + gnomAD | Local (CPU) |
| Per-compound HTML report (MoA) | py3Dmol + ProLIF + pandas | Local |

See [`_planning/CONSULTANT_REVIEW.md`](_planning/CONSULTANT_REVIEW.md) for why these tools were chosen over alternatives.

## Repository layout

```
.
├── README.md, CLAUDE.md            # you are here / project rules for AI assistants
├── environment.yml                 # conda env (primary)
├── requirements-colab.txt          # pip extras for Colab GPU steps
├── .gitignore
│
├── src/aidd/                       # importable package (logic lives here, not in notebooks)
├── notebooks/                      # user-facing entry points (Colab-compatible)
│
├── data/                           # small reference assets + (gitignored) derived outputs
│   ├── structures/                   reference PDBs/mol2 for ERK2 and CDK2
│   ├── ligands/                      reference ligands (co-crystal)
│   ├── compounds/                    SMILES libraries per target
│   ├── labels/                       activity data for ML training
│   └── derived/                      generated outputs (NOT in git)
│
├── _planning/                      # planning docs (decisions, kill list, consultant review)
└── _archive/                       # pruned Leiden/ULLA course materials, read-only references
```

## Quick start

**Run a notebook on Colab (recommended for first time):** click the *Open in Colab* badge at the top of any notebook in [`notebooks/`](notebooks/), or follow the per-notebook links from [`notebooks/README.md`](notebooks/README.md). The setup cell handles installs and clones the repo automatically.

**Run locally (CPU-only notebooks):**

```bash
# 1. Create the conda environment (validated on Windows 11 + macOS + Linux)
mamba env create -f environment.yml
mamba activate aidd

# 2. Smoke-test the install
python -c "import rdkit, prolif, posebusters, datamol; print('ok')"

# 3. Launch Jupyter
jupyter lab
```

GPU-bound notebooks (01 fold, 03 dock, 05 co-fold) run only on Colab. CPU notebooks (00, 02, 04, 06, 07, 08, 09) run on either Colab or your laptop.

## For collaborators / for Claude

- Planning docs in [`_planning/`](_planning/) are the source of truth for decisions, design, and the open issues list.
- Project rules for AI-assisted work are in [`CLAUDE.md`](CLAUDE.md).
- The [`_archive/`](_archive/) folder is **read-only** course material kept for code-mining and inspiration; do not edit in place.

## Heavy data — where it lives

Per-screen working files (docked poses, ProLIF tables, rescorer features) **do not** live in this repository. They live in a Google Drive folder (`aidd-pipeline/targets/<target>/`) for public targets, and on local disk only for sensitive targets. `data/MANIFEST.yaml` (to be created) tracks which dataset lives where.
