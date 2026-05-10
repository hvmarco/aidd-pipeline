# aidd-pipeline

End-to-end in-silico screening pipeline for wet-lab triage in cancer drug discovery.

**Status:** scaffolding. The repository layout, planning documents, and pruned course archive are in place; pipeline code is not yet written. See [`_planning/PROJECT_PROPOSAL.md`](_planning/PROJECT_PROPOSAL.md) for the 13-step delivery plan.

## What this is

Given a **protein target** (sequence or PDB) and a **list of candidate ligands** (SMILES), produce a ranked, defensible shortlist of compounds to prioritise in the wet lab. The pipeline runs in Jupyter notebooks (Colab for GPU steps; Windows / macOS locally for CPU steps).

The headline output of a run is `shortlist.sdf` + `shortlist.csv` for a target — ranked candidates, with per-compound docking poses, interaction fingerprints, ADMET flags, and confidence from a two-method consensus.

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

## Quick start (local)

Will be filled in once `environment.yml` is validated end-to-end. Skeleton:

```bash
# 1. Create the conda environment
mamba env create -f environment.yml
mamba activate aidd

# 2. Smoke test
python -c "import rdkit, prolif, posebusters, datamol; print('ok')"

# 3. Launch Jupyter
jupyter lab
```

## For collaborators / for Claude

- Planning docs in [`_planning/`](_planning/) are the source of truth for decisions, design, and the open issues list.
- Project rules for AI-assisted work are in [`CLAUDE.md`](CLAUDE.md).
- The [`_archive/`](_archive/) folder is **read-only** course material kept for code-mining and inspiration; do not edit in place.

## Heavy data — where it lives

Per-screen working files (docked poses, ProLIF tables, rescorer features) **do not** live in this repository. They live in a Google Drive folder (`aidd-pipeline/targets/<target>/`) for public targets, and on local disk only for sensitive targets. `data/MANIFEST.yaml` (to be created) tracks which dataset lives where.
