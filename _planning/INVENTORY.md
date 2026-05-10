# Inventory — AIinDD2025 course materials

Snapshot of everything currently in `C:\Users\MarcoHernandez\Projects\AIinDD2025`, grouped by topic and labelled with a reuse recommendation.

Reuse legend:
- **KEEP** — directly reusable in the new end-to-end pipeline (code, configs, reference structures).
- **REFERENCE** — useful as worked example / teaching material; keep read-only, not on the hot path.
- **DATA-OUT** — outputs of course exercises (docking features/IFPs). Heavy. Not strictly needed for the new pipeline, but useful for re-running ML scoring without re-docking.
- **DUP** — duplicate of another file already in the tree.
- **DROP** — course-only artefact (challenge submissions, locked grading cells) with no value going forward.

---

## 1. Top-level files

| Path | Topic | Reuse |
|---|---|---|
| [`Week_3_Monday_Docking_and_Scoring.ipynb`](../Week_3_Monday_Docking_and_Scoring.ipynb) (10.4 MB) | ERK2 docking walk-through: prolif IFPs, py3Dmol viewer, PLANTS results inspection, scoring. The most complete end-to-end teaching notebook. | **KEEP (reference)** — mine for reusable cells. |
| [`AF_model.pdb`](../AF_model.pdb) | AlphaFold model of ERK2/MAPK1 (used as the "AF" target across the course). | **KEEP** as `data/structures/erk2_af.pdb`. |
| [`klifs_4fv7_protein.pdb`](../klifs_4fv7_protein.pdb) | ERK2 X-ray, KLIFS-cleaned. | **KEEP** as `data/structures/erk2_4fv7.pdb`. |
| [`klifs_4fv7_ligand.pdb`](../klifs_4fv7_ligand.pdb) | Co-crystal reference ligand (PDB code E94) for redocking / binding-site definition. | **KEEP** as `data/ligands/erk2_4fv7_ref.pdb`. |
| [`training_set.txt`](../training_set.txt) | `CPD_ID Active` labels for the ERK2 compound set (≈11k rows). | **KEEP** as `data/labels/erk2_training.tsv`. |
| `AIinDD2025.code-workspace` | Empty VS Code workspace stub. | **DROP** once we have a clean repo. |

## 2. Reference structures and docked outputs (top-level)

| Path | What | Reuse |
|---|---|---|
| [`docking/`](../docking/) | The Monday lab's PLANTS run skeleton: `plants.conf`, `af_plants.conf`, `PLANTS` binary (3 MB Linux ELF), `MAPK1_AF_model.mol2`, `klifs_4fv7_protein.mol2`, `redock_ligand.mol2`, `results/`, `results_af/`. | **KEEP** the configs and mol2s as templates; **DROP** the Linux PLANTS binary (won't run on Windows/Mac); `results*/` is **REFERENCE**. |
| [`4fv7_docked/`](../4fv7_docked/) | Pre-computed `training_features.csv.gz`, `training_ifp.csv.gz` for ERK2 4FV7. ~38 MB. | **DATA-OUT** — keep zipped under `data/derived/erk2_4fv7/`. |
| [`af_docked/`](../af_docked/) | Same, for the AF model. ~39 MB. | **DATA-OUT** — keep under `data/derived/erk2_af/`. |
| [`Week_5_Monday_dataset/`](../Week_5_Monday_dataset/) | Same files as top-level (`AF_model.pdb`, `training_set.txt`, `klifs_4fv7_*`, `4fv7_docked/`, `af_docked/`, `docking/`). | **DUP** of items above. Drop after confirming hashes match. |

## 3. Course content — `AIinDD2025/`

Course taught in Python + Jupyter. Notebooks are largely self-contained (each has its own install/imports cell). Roughly in chronological order:

| Folder | Notebook | Topic | Reuse |
|---|---|---|---|
| `251014_1_Pandas` | `Pandas_nb.ipynb` (905 KB) | Pandas crash course. | **REFERENCE** (skip from pipeline). |
| `251014_2_NumPy` | `numpy.ipynb` (2.7 MB) | NumPy crash course. | **REFERENCE**. |
| `251028_1_Tuesday` | `Week_3_Tuesday_Challenge_1.ipynb` | "Best ERK2 scoring function — small" — feature engineering on docking outputs, classification, imbalanced-learn. Includes per-target docked CSVs for 4FV7/4QPA/4XJ0/AF. | **KEEP (reference)** — the scoring/ML pattern is exactly what we want to factor into a reusable module. |
| `251028_2_Tuesday` | `Week_3_Tuesday_Challenge_2.ipynb` | "Ultimate" version of the same challenge, larger dataset (155 MB tgz). | **KEEP (reference)**, but the 155 MB tgz is **DATA-OUT**. |
| `251028_Bonus` | `biopython.ipynb` (3 MB) | Biopython: structure parsing, distograms, exercises on 7VL8. | **KEEP (reference)** — distogram code is reusable for QC of unfolded/predicted structures. |
| `251117_UCPH` | `Week_6_SB-regression-extended.ipynb` (1.7 MB) | Structure-based **regression** on CDK2 (4NJ3, 5K4J): pose selection, ML, regression, evaluation. Mirrors final project. | **KEEP (reference)** — best worked example of an SB-ML regression pipeline. |
| `DeepLearning_CNN` / `Deep_Learning_CNN_v2` | `LMM_CNN_*.ipynb` + `bbbc021v1_images/`, `bbbc021v1_labels.csv` | LeNet-style CNN on cell-morphology images. | **DROP from pipeline** (off-topic for in-silico docking). Keep images out of any future repo. |
| `QSAR_and_descriptors_lab` | `descriptors_qsar_lab.ipynb` (598 KB) | RDKit descriptors, similarity search, fingerprint vs descriptor regression for solubility. | **KEEP (reference)** — fingerprint/descriptor utilities are reusable. |
| `Supervised_ML_lab` | `supervised_lab.ipynb` (44 KB) | LogReg / RF / SVM classification + regression on housing. | **REFERENCE**. |

## 4. Final project — `final_project/`

| Path | What | Reuse |
|---|---|---|
| `smiles/{chembl,prospective,xrays}.smi` | Compound libraries for CDK2 (with `ChEMBL_ID|pIC50`). | **KEEP** as `data/compounds/cdk2/*.smi`. |
| `4nj3_docked/`, `5k4j_docked/` | Pre-computed features, IFPs, RMSDs for ChEMBL / prospective / x-ray ligand sets against CDK2 PDBs 4NJ3 and 5K4J. ~22 MB total. | **DATA-OUT** — keep under `data/derived/cdk2_*/`. |
| `4nj3_docked/{ligand,protein}.pdb`, `5k4j_docked/{ligand,protein}.pdb` | Reference CDK2 receptors and ligands. | **KEEP** as `data/structures/cdk2_{4nj3,5k4j}.pdb`. |

## 5. ULLA-Leiden practicals — `ULLA-Leiden-master/`

Self-contained mini-course from Leiden:

| Folder | Topic | Reuse |
|---|---|---|
| `practicals/01_plotting_unsupervised` | UMAP / t-SNE / PCA / KMeans / hierarchical on Iris. | **REFERENCE**. |
| `practicals/02_PCM` | TeachOpenCADD T032 — Proteochemometrics on adenosine receptors. Has `PCM_env.yml`. | **KEEP (reference)** — protein descriptors (ProDEC, Z-scales) are reusable if we later need PCM. |
| `practicals/03_uncertainty_active_learning` | Bayesian NN, MC-dropout, active learning on AA2AR. Has `utils.py` with `seed_everything` + `create_dataloader`. | **KEEP (reference)** — `utils.py` is genuinely reusable. |
| `practicals/case_study` | Q9Y5N1 QSAR challenge stub. | **REFERENCE**. |
| `README.md`, `init.sh`, `.gitignore`, `.gitmodules` | Submodule pointer at `git@github.com:CDDLeiden/ULLA2023-Leiden-Assignments.git`. | **KEEP** the `.gitignore` as a starting template for the new repo. |

## 6. Shared/auxiliary

| Path | Reuse |
|---|---|
| `shared-data-AIinDD2025/solubility.csv`, `pda_exported_designs.csv`, `bbbc021v1_*` | Course datasets (ESOL solubility, protein design, cell images). | **DROP** for pipeline, **REFERENCE** otherwise. |

---

## 7. Duplicates we should collapse

When we build the new repo, the following are clearly duplicated; pick one canonical copy:

- `AF_model.pdb` appears at: top-level, `Week_5_Monday_dataset/`, `251028_1_Tuesday/`, `251028_2_Tuesday/`.
- `klifs_4fv7_protein.pdb`: top-level, `Week_5_Monday_dataset/`, `251028_1_Tuesday/`, `251028_2_Tuesday/`, `docking/`.
- `klifs_4fv7_ligand.pdb` / `reference_ligand.pdb`: top-level + `251028_*_Tuesday/`.
- `training_set.txt` (full 11k version): top-level, `Week_5_Monday_dataset/`, `251028_2_Tuesday/` (the `251028_1_Tuesday/` copy is a smaller subset — keep separately).
- `bbbc021v1_*`: `DeepLearning_CNN/`, `shared-data-AIinDD2025/`. (Plan to drop entirely.)

## 8. Gaps for the target pipeline

The course used **PLANTS** (proprietary, Linux-only binary) for docking and a **pre-baked** AlphaFold model. The user's stated goal mentions **AlphaFold + DiffDock**, neither of which appears in the materials. The new pipeline will need to introduce:

- AlphaFold inference (Colab / ColabFold notebook).
- DiffDock (or equivalent open-source alternative — Vina, gnina, DiffDock-L).
- A cross-platform PLANTS replacement, since the bundled binary won't run on Windows/Mac.

Everything **downstream** of docking (pose selection, IFP fingerprinting with prolif, scoring/QSAR ML, visualisation with py3Dmol) is fully reusable from the existing notebooks.

## 9. Approximate sizes

| Item | Size |
|---|---|
| `AIinDD2025/251028_2_Tuesday/Week_5_Tuesday_dataset_chal2.tgz` | 155 MB |
| `AIinDD2025/251028_2_Tuesday/{4fv7,4xj0,af}_docked/*` | ~150 MB total |
| `final_project/Week_6_dataset.tgz` | 23 MB |
| `4fv7_docked/`, `af_docked/` | ~77 MB |
| `shared-data-AIinDD2025/bbbc021v1_images/` | many small PNGs |

All heavy data must be excluded from git (LFS or external storage).
