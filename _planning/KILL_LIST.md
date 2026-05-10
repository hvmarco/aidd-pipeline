# Prune plan — kill list

Review this row by row. **Nothing has been deleted yet.** Mark anything you want to keep before I run the actual deletes.

Reminder of the contract: you have a separate backup of all course materials, so we're free to be aggressive. We keep only (a) small reference assets the pipeline actually consumes and (b) a curated `_archive/` of small notebooks/configs we'll either mine for code or look at for inspiration.

**Current total:** ~830 MB.
**After this prune:** ~30 MB (mostly small reference notebooks in `_archive/` and small data files).

Legend for the **Action** column:
- 🗑️ **DELETE** — gone, not recoverable from this folder (you have the off-site backup if you ever need it).
- 📦 **MOVE→data/** — relocate into the new `data/` layout; the file itself stays.
- 📚 **MOVE→_archive/** — relocate into the read-only `_archive/` folder for later reference / code mining.
- ✂️ **STRIP THEN ARCHIVE** — notebook is bloated with embedded image outputs; clear them with `jupyter nbconvert --clear-output` before moving to `_archive/`. Shrinks 10 MB notebooks to ~200 KB without losing any code or markdown.

---

## A. Heavy regenerable data — 🗑️ DELETE (≈480 MB total)

PLANTS-specific docking outputs. Our new pipeline uses gnina/Boltz-2; these CSVs would need their column schema rewritten anyway. Regeneratable if we ever care.

| Path | Size | Reason |
|---|---|---|
| `4fv7_docked/` | 37 MB | PLANTS features + IFP from ERK2 4FV7 |
| `af_docked/` | 38 MB | Same, AF model |
| `Week_5_Monday_dataset/` | 80 MB | Full duplicate of top-level course data |
| `AIinDD2025/251028_1_Tuesday/Week_5_Tuesday_dataset_chal1.tgz` | 16 MB | Course archive (extracted siblings exist) |
| `AIinDD2025/251028_1_Tuesday/{4fv7,4qpa,4xj0,af}_docked/` | 16 MB | PLANTS outputs for Tuesday challenge 1 |
| `AIinDD2025/251028_2_Tuesday/Week_5_Tuesday_dataset_chal2.tgz` | 155 MB | Course archive (largest single file) |
| `AIinDD2025/251028_2_Tuesday/{4fv7,4xj0,af}_docked/` | 150 MB | PLANTS outputs for Tuesday challenge 2 |
| `final_project/Week_6_dataset.tgz` | 23 MB | Course archive |
| `final_project/4nj3_docked/*.csv.gz` | 11 MB | PLANTS outputs (CDK2 4NJ3); the small `.pdb` files in this folder are kept — see §F |
| `final_project/5k4j_docked/*.csv.gz` | 12 MB | PLANTS outputs (CDK2 5K4J); same caveat for `.pdb` |

## B. Off-thread course content — 🗑️ DELETE (≈100 MB)

CNN cell-morphology and general Pandas/NumPy intros. Not part of the SBVS pipeline; the course used them as warm-ups.

| Path | Size | Reason |
|---|---|---|
| `AIinDD2025/DeepLearning_CNN/` | 80 MB | LeNet on cell-morphology images. Off-thread. |
| `AIinDD2025/Deep_Learning_CNN_v2/` | 1.3 MB | Same. |
| `shared-data-AIinDD2025/bbbc021v1_images/` | 57 MB | CNN training images (duplicate of CNN folder) |
| `shared-data-AIinDD2025/bbbc021v1_labels.csv` | 36 KB | CNN labels |
| `shared-data-AIinDD2025/figs/` | 4.3 MB | Course illustrations |
| `shared-data-AIinDD2025/pda_exported_designs.csv` | 3.8 MB | Protein design data, unused |
| `shared-data-AIinDD2025/solubility.csv` | 64 KB | ESOL solubility, only used by the QSAR lab. We'll fetch it from MoleculeNet/TDC if we ever want it. |
| `shared-data-AIinDD2025/QSAR Moddeling Challenge - Leiden Case Study/` | empty | Empty folder. |
| `AIinDD2025/251014_1_Pandas/` | 1.7 MB | Pandas intro notebook |
| `AIinDD2025/251014_2_NumPy/` | 5.4 MB | NumPy intro notebook |
| `AIinDD2025/Supervised_ML_lab/` | 58 KB | Generic ML on housing/cancer datasets, off-thread |
| `ULLA-Leiden-master/practicals/01_plotting_unsupervised/` | 12 MB | UMAP/t-SNE/PCA on Iris — generic |
| `ULLA-Leiden-master/practicals/case_study/` | 756 KB | Q9Y5N1 QSAR stub, never developed |
| `ULLA-Leiden-master/assignments/` | empty | Empty submodule placeholder |
| `ULLA-Leiden-master/init.sh`, `.gitmodules` | tiny | Refer to the Leiden git submodule we don't use |

## C. Duplicates of files we're keeping elsewhere — 🗑️ DELETE (≈3 MB)

Same byte content lives in another path that survives.

| Path | Survives at | Action |
|---|---|---|
| `AIinDD2025/251028_1_Tuesday/AF_model.pdb` | top-level `AF_model.pdb` | delete |
| `AIinDD2025/251028_2_Tuesday/AF_model.pdb` | same | delete |
| `AIinDD2025/251028_1_Tuesday/klifs_4fv7_protein.pdb` | top-level `klifs_4fv7_protein.pdb` | delete |
| `AIinDD2025/251028_2_Tuesday/klifs_4fv7_protein.pdb` | same | delete |
| `AIinDD2025/251028_1_Tuesday/klifs_4qpa_protein.pdb` | (not kept) | delete — 4QPA isn't a target we'll use |
| `AIinDD2025/251028_2_Tuesday/klifs_4qpa_protein.pdb` | same | delete |
| `AIinDD2025/251028_1_Tuesday/klifs_4xj0_protein.pdb` | (not kept) | delete — 4XJ0 same reason |
| `AIinDD2025/251028_2_Tuesday/klifs_4xj0_protein.pdb` | same | delete |
| `AIinDD2025/251028_1_Tuesday/reference_ligand.pdb` | top-level `klifs_4fv7_ligand.pdb` | delete (same atoms) |
| `AIinDD2025/251028_2_Tuesday/reference_ligand.pdb` | same | delete |
| `AIinDD2025/251028_1_Tuesday/training_set.txt` | (subset of top-level) | delete |
| `AIinDD2025/251028_2_Tuesday/training_set.txt` | top-level `training_set.txt` | delete |
| `AIinDD2025/251028_1_Tuesday/submission_example.csv` | n/a (course submission only) | delete |

Question: should I keep 4QPA and 4XJ0 PDBs as extra ERK2 conformations? If yes, move to `data/structures/erk2_alt/`. See §G open question 1.

## D. IDE / cache junk — 🗑️ DELETE (negligible size)

| Path | Reason |
|---|---|
| `AIinDD2025.code-workspace` (60 B) | Empty VS Code workspace stub; we'll regenerate if needed |
| All `**/.ipynb_checkpoints/` folders | Jupyter autosave; never useful in version control |
| `AIinDD2025/Supervised_ML_lab/.test.ipynb.layout`, `.Untitled.ipynb.layout`, `test.ipynb` | JupyterLab leftovers |

## E. Linux-only proprietary binary — 🗑️ DELETE

| Path | Size | Reason |
|---|---|---|
| `docking/PLANTS` | 3 MB | Linux ELF binary. Won't run on Windows/Mac. We're replacing PLANTS with gnina anyway. |
| `docking/results/`, `docking/results_af/` | ~2 MB | PLANTS run outputs from the course |

The `docking/` folder is otherwise mostly KEEP — see §F.

## F. Small reusable assets — 📦 MOVE → `data/` (≈4 MB total)

These become the new pipeline's primary inputs. Tiny files, all kept.

| From | To | Size |
|---|---|---|
| `AF_model.pdb` | `data/structures/erk2_af.pdb` | 464 KB |
| `klifs_4fv7_protein.pdb` | `data/structures/erk2_4fv7.pdb` | 448 KB |
| `klifs_4fv7_ligand.pdb` | `data/ligands/erk2_4fv7_ref.pdb` | 8 KB |
| `training_set.txt` | `data/labels/erk2_training.tsv` | 456 KB |
| `docking/MAPK1_AF_model.mol2` | `data/structures/erk2_af.mol2` | 690 KB |
| `docking/klifs_4fv7_protein.mol2` | `data/structures/erk2_4fv7.mol2` | 545 KB |
| `docking/redock_ligand.mol2` | `data/ligands/erk2_4fv7_ref.mol2` | 7 KB |
| `final_project/4nj3_docked/protein.pdb` | `data/structures/cdk2_4nj3.pdb` | 177 KB |
| `final_project/4nj3_docked/ligand.pdb` | `data/ligands/cdk2_4nj3_ref.pdb` | 4 KB |
| `final_project/5k4j_docked/protein.pdb` | `data/structures/cdk2_5k4j.pdb` | 180 KB |
| `final_project/5k4j_docked/ligand.pdb` | `data/ligands/cdk2_5k4j_ref.pdb` | 3 KB |
| `final_project/smiles/chembl.smi` | `data/compounds/cdk2/chembl.smi` | 140 KB |
| `final_project/smiles/prospective.smi` | `data/compounds/cdk2/prospective.smi` | 852 KB |
| `final_project/smiles/xrays.smi` | `data/compounds/cdk2/xrays.smi` | 15 KB |
| `AIinDD2025/251028_1_Tuesday/compounds/challenge.smi` | `data/compounds/erk2/challenge_small.smi` | 64 KB |
| `AIinDD2025/251028_1_Tuesday/compounds/training.smi` | `data/compounds/erk2/training_small.smi` | 250 KB |
| `AIinDD2025/251028_2_Tuesday/compounds/challenge.smi` | `data/compounds/erk2/challenge.smi` | 920 KB |
| `AIinDD2025/251028_2_Tuesday/compounds/training.smi` | `data/compounds/erk2/training.smi` | 2.4 MB |

## G. Small reference notebooks — 📚 MOVE → `_archive/` (≈5 MB after strip)

Kept for code-mining and inspiration. After we lift their useful code into `src/aidd/`, they go in a second prune pass.

| From | Action | Why we keep it |
|---|---|---|
| `Week_3_Monday_Docking_and_Scoring.ipynb` (11 MB) | ✂️ STRIP THEN ARCHIVE | The most complete docking + IFP + py3Dmol example. Stripped: ~200 KB. |
| `AIinDD2025/251028_1_Tuesday/Week_3_Tuesday_Challenge_1.ipynb` | 📚 MOVE | ML scoring pattern; smaller dataset |
| `AIinDD2025/251028_2_Tuesday/Week_3_Tuesday_Challenge_2.ipynb` | 📚 MOVE | Same pattern, bigger dataset |
| `AIinDD2025/251028_Bonus/biopython.ipynb` (3 MB) | ✂️ STRIP THEN ARCHIVE | Distogram code we'll reuse for fold QC |
| `AIinDD2025/251117_UCPH/Week_6_SB-regression-extended.ipynb` (1.7 MB) | 📚 MOVE | CDK2 regression pipeline — closest match to the consensus rescorer we want |
| `AIinDD2025/QSAR_and_descriptors_lab/descriptors_qsar_lab.ipynb` (598 KB) | 📚 MOVE | RDKit standardisation + descriptors recipes |
| `ULLA-Leiden-master/practicals/02_PCM/talktorial.ipynb` (515 KB) | 📚 MOVE | This *is* TeachOpenCADD T032; keep as local copy in case we want PCM later |
| `ULLA-Leiden-master/practicals/02_PCM/PCM_env.yml`, `README.md`, `images/` | 📚 MOVE | Supporting files for above |
| `ULLA-Leiden-master/practicals/03_uncertainty_active_learning/` (1.7 MB) | 📚 MOVE | Notebook + `utils.py` + small data files (AA2AR.csv 240 KB, Q99685_papyrus.tsv 565 KB). Used for the active-learning loop in round-2 triage. |
| `ULLA-Leiden-master/README.md` | 📚 MOVE | Explains what each practical covers — useful as index |
| `ULLA-Leiden-master/.gitignore` | use as template for new repo's `.gitignore` then delete |
| `docking/plants.conf`, `docking/af_plants.conf` | 📚 MOVE → `_archive/configs/` | 500 B each. The binding-site coordinates inside (`bindingsite_center 1.34299 17.3648 40.9828, radius 12.9007`) are reusable as a starting point for gnina's `--center_x` etc. |

## H. Final decisions (locked 2026-05-10)

1. **4QPA and 4XJ0 ERK2 conformations** → **DROP.** We can re-fetch from KLIFS/PDB if we ever need ensemble docking against alternative ERK2 conformers.
2. **`docking/results/` and `docking/results_af/`** → **DROP.** PLANTS↔gnina is not a meaningful per-pose comparison (different scoring functions, different units, single-ligand sample); the real sanity check is redock-RMSD against the crystal pose, which we already have.
3. **`shared-data-AIinDD2025/solubility.csv`** → **DROP.** Re-fetchable from MoleculeNet/TDC.

## I. After the prune — final shape

```
AIinDD2025/                           # to be renamed → aidd-pipeline/
├── _planning/                        # what's here now (this file + INVENTORY/PROPOSAL/CONSULTANT/CLAUDE)
├── _archive/                         # ≈5 MB of small reference material
│   ├── README.md                       (course materials origin, mapped to inventory)
│   ├── Week_3_Monday_Docking_and_Scoring.ipynb        (stripped, ~200 KB)
│   ├── Week_3_Tuesday_Challenge_1.ipynb                (~280 KB)
│   ├── Week_3_Tuesday_Challenge_2.ipynb                (~90 KB)
│   ├── biopython.ipynb                                 (stripped, ~500 KB)
│   ├── Week_6_SB-regression-extended.ipynb             (1.7 MB)
│   ├── descriptors_qsar_lab.ipynb                      (598 KB)
│   ├── PCM/                                            (TeachOpenCADD T032 copy, ~1.5 MB)
│   ├── uncertainty_active_learning/                    (1.7 MB)
│   └── configs/{plants.conf,af_plants.conf}            (binding-site coordinates worth remembering)
├── data/                             # ≈10 MB of small reusable refs (heavier stuff in Drive)
│   ├── structures/                     erk2_*.pdb, cdk2_*.pdb (+ .mol2 variants for the kinases)
│   ├── ligands/                        reference ligands
│   ├── compounds/                      erk2/, cdk2/ — SMILES libraries
│   ├── labels/                         erk2_training.tsv
│   └── derived/                        (empty for now; populated by pipeline runs, gitignored)
├── README.md                         # (to be created)
├── CLAUDE.md                         # (promoted from _planning/)
├── .gitignore                        # (seeded from ULLA template)
├── environment.yml                   # (to be created)
└── requirements-colab.txt            # (to be created)
```

## Sign-off needed

Three questions:

1. **The three "Default: drop" items in §H** — drop them? (Recommended: yes.)
2. **The bigger `compounds/training.smi`** (2.4 MB, ~12k ERK2 actives/inactives from Tuesday challenge 2). Worth keeping — that's a non-trivial labelled dataset and we'd want it to train the ERK2 rescorer in step 10 of the proposal. **Default: keep, as listed.** Confirm?
3. **Order of operations** — I propose: (a) move all KEEP items to their new paths under `data/` and `_archive/`, (b) verify the moves, (c) delete everything in §A–E in one batch, (d) strip notebook outputs on the archived ones. OK to proceed in that order?
