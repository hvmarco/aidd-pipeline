# `_archive/` — read-only reference material from the Leiden/ULLA AI-in-Drug-Discovery 2025 course

These are the notebooks, configs, and small datasets we kept after pruning the course materials (see [`_planning/KILL_LIST.md`](../_planning/KILL_LIST.md) for what got deleted and why). They serve as **code-mining sources and inspiration** while we build the new pipeline; they are not meant to be executed in-place.

Do not edit files in this folder. When you need code from one of these notebooks, copy it into `src/aidd/` or a new notebook under `notebooks/` and adapt it there.

## What's here

| Path | Topic | Most useful for |
|---|---|---|
| `Week_3_Monday_Docking_and_Scoring.ipynb` | ERK2 docking + ProLIF interaction fingerprints + py3Dmol viewers. Output cells stripped. | Lifting IFP + 3D-viewer code into `src/aidd/{ifp,viz}.py`. |
| `Week_3_Tuesday_Challenge_1.ipynb` | "Best ERK2 scoring function (small dataset)" — feature engineering on docking outputs + classification + imbalanced-learn. | The ML-rescoring pattern (Step 10 of the proposal). |
| `Week_3_Tuesday_Challenge_2.ipynb` | Same challenge, larger dataset. | Same as above, more data. |
| `biopython.ipynb` | Biopython structure parsing, distograms, multi-chain examples. | Fold-QC: comparing predicted vs reference structures (Step 7). |
| `Week_6_SB-regression-extended.ipynb` | CDK2 structure-based regression on 4NJ3 + 5K4J. | The clearest end-to-end SBVS-regression pipeline in the course; closest match to our consensus rescorer. |
| `descriptors_qsar_lab.ipynb` | RDKit standardisation, fingerprints (Morgan/MACCS), descriptors, linear models. | The ligand-prep step (Step 8): SMILES → standardised → 3D embed. |
| `PCM/` | TeachOpenCADD talktorial T032 (Proteochemometrics, adenosine receptors). Includes `PCM_env.yml` if we want to recreate the env. | If we later need cross-target activity modelling (PCM). Prefer the upstream TeachOpenCADD version when available. |
| `uncertainty_active_learning/` | Bayesian NN, MC-dropout, active learning loop on AA2AR. Includes `utils.py` (seed_everything, dataloader helper) and small training data. | Round-2 triage / active learning loop. |
| `configs/plants_4fv7.conf`, `configs/plants_erk2_af.conf` | The PLANTS docking configs from the Monday tutorial. **What matters:** the binding-site coordinates `center 1.34299 17.3648 40.9828, radius 12.9007` for the ERK2 ATP pocket. | Initial seed for the gnina search box (center same, box ≈ 26 Å cube). |
| `ULLA-Leiden-README.md` | Original Leiden practicals README. | Index of what each practical covered. |

## When to delete this folder

After step 13 of the first-week deliverables in [`_planning/PROJECT_PROPOSAL.md`](../_planning/PROJECT_PROPOSAL.md): once the code we cared about has been lifted into `src/aidd/` and re-tested, the corresponding entries here can go. This folder should shrink over time and may be deleted entirely once the pipeline is self-sufficient.
