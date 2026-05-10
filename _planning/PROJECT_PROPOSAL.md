# Project proposal — end-to-end in-silico screening pipeline

## 0. Decisions (2026-05-10)

| Question | Status | Answer |
|---|---|---|
| Repo name | locked | `aidd-pipeline` |
| Repo destination | locked | Private GitHub repo (to be created by user) |
| v0 target | locked | ERK2 (MAPK1) as placeholder; swap to real target later |
| Environment manager | locked | conda/mamba via `environment.yml` |
| Course materials | locked | Prune **DROP** now; keep **REFERENCE** in `_archive/` until lifted (or replaced by upstream lib); second pass after extraction |
| Primary docker | locked | **gnina** (classical, ML-aware rescoring) as interpretable lane; **Boltz-2** (co-folding + affinity) as fast lane; DiffDock parked as optional/experimental. |
| Folding | locked | **ColabFold** for the interpretable lane (one-time per target); **Boltz-2** subsumes folding+docking in the fast lane. |
| Framework approach | locked | Hybrid: TeachOpenCADD-compatible conventions as spine + Boltz-2 fast lane. Course code mined only where upstream has no equivalent. |
| Pose QC | locked | **PoseBusters** mandatory gate before scoring. |
| ADMET / drug-likeness | locked | RDKit-built-in pre-filter (Lipinski/QED/SA-score/PAINS) before docking; **ADMETlab 3** API on top candidates only. |
| GPU access | locked | **Colab-only** (free or Pro). All GPU steps are Colab notebooks; local runs are CPU-only via gnina. |
| v1 throughput | locked | **1k–10k compounds per screen.** Notebook-driven with a per-stage cached-file batch driver; resumable. |
| Triage philosophy | locked | **Defensible consensus** — Boltz-2 affinity rank AND gnina+IFP-rescorer rank must both place a compound in top X% (X tunable, default 5%). |

See [CONSULTANT_REVIEW.md](CONSULTANT_REVIEW.md) for the architectural reasoning behind the reopened/new rows.

## 1. Goal (in one sentence)

Stand up a reproducible, cross-platform Jupyter pipeline that, given a target protein (sequence or PDB) and a list of candidate ligands (SMILES), produces folded structures, docked poses, interaction fingerprints, ML-scored rankings, and a shortlist of promising candidates for wet-lab follow-up.

## 2. Design principles

1. **Notebook-driven, module-backed.** User-facing entry points are notebooks (works in Colab, Windows, Mac). Heavy logic lives in a Python package (`src/aidd/…`) imported by the notebooks — keep notebook cells short.
2. **Cross-platform first.** No Linux-only binaries on the hot path. PLANTS is replaced by an open / cross-platform docker (Vina/gnina/DiffDock). Anything that requires a GPU runs in Colab.
3. **Idempotent stages with cached outputs.** Each stage writes to a deterministic path under `data/derived/<target>/<stage>/`. Re-runs short-circuit when the output already exists. Don't re-dock if features are already on disk.
4. **Data stays out of git.** Only code, configs, small reference structures, and notebooks are versioned. Heavy artefacts (`.csv.gz`, `.tgz`, `.mol2`, docking results) live under `data/` and are excluded by `.gitignore`. We'll use either a manifest file (`data/MANIFEST.yaml`) pointing to external storage, or git-lfs for selected reference PDBs.
5. **Reuse before rewrite.** Lift working code from the course notebooks (IFP construction, py3Dmol viewers, RDKit standardisation, RF/XGBoost scoring) into the package; don't rewrite from scratch.

## 3. Proposed repo layout

```
AIinDD2025/                              # repo root (rename to something project-specific later)
├── README.md                            # overview, quickstart, how to run on Colab vs local
├── CLAUDE.md                            # rules for future Claude sessions (see _planning/CLAUDE.md)
├── environment.yml                      # primary source of truth (conda/mamba)
├── requirements-colab.txt               # pip overrides for Colab where conda isn't practical
├── .gitignore
├── .gitattributes                       # if we adopt git-lfs for select PDBs
│
├── notebooks/                           # user-facing entry points
│   ├── 00_quickstart.ipynb              # smallest happy-path example, ~3 min on Colab
│   ├── 01_fold_target.ipynb             # ColabFold / AlphaFold for the target sequence
│   ├── 02_prepare_ligands.ipynb         # SMILES → standardised → 3D conformers
│   ├── 03_dock.ipynb                    # DiffDock (or Vina) on the prepared inputs
│   ├── 04_score_and_rank.ipynb          # IFPs, ML rescoring, shortlist export
│   └── 99_explore_course_examples/      # frozen copies of the most useful course nbs
│
├── src/aidd/                            # importable package
│   ├── __init__.py
│   ├── structures.py                    # PDB I/O, alignment, distograms (from biopython nb)
│   ├── ligands.py                       # SMILES standardisation, 3D embed (from QSAR lab)
│   ├── docking/
│   │   ├── diffdock.py                  # thin wrapper around DiffDock CLI / API
│   │   └── vina.py                      # fallback / CPU baseline
│   ├── ifp.py                           # prolif wrappers (lifted from Week 3 Monday)
│   ├── scoring.py                       # feature engineering, ML scorer (lifted from Tue chal)
│   ├── viz.py                           # py3Dmol helpers
│   └── io.py                            # path conventions, cached-stage helpers
│
├── data/                                # all heavy / generated; gitignored except MANIFEST
│   ├── MANIFEST.yaml                    # what lives where (URLs, hashes), if not using LFS
│   ├── structures/                      # reference PDBs (small) — possibly versioned via LFS
│   │   ├── erk2_4fv7.pdb
│   │   ├── erk2_af.pdb
│   │   ├── cdk2_4nj3.pdb
│   │   └── cdk2_5k4j.pdb
│   ├── ligands/                         # canonical reference ligands (small)
│   ├── compounds/                       # input SMILES libraries
│   ├── labels/                          # activity labels for ML
│   └── derived/                         # all generated outputs (docking, IFPs, features)
│       └── <target>/<stage>/...
│
├── _planning/                           # this folder — planning docs, not for end users
│   ├── INVENTORY.md
│   ├── PROJECT_PROPOSAL.md
│   └── CLAUDE.md
│
└── _archive/                            # raw course materials, kept read-only
    ├── ULLA-Leiden-master/
    ├── AIinDD2025/
    ├── shared-data-AIinDD2025/
    ├── final_project/
    └── ...
```

## 4. Migration plan (course materials → repo)

Concrete moves once we agree on the structure:

| From | To | Action |
|---|---|---|
| `AF_model.pdb`, `klifs_4fv7_protein.pdb`, `klifs_4fv7_ligand.pdb` | `data/structures/erk2_*.pdb`, `data/ligands/erk2_4fv7_ref.pdb` | move, rename |
| `final_project/{4nj3,5k4j}_docked/{protein,ligand}.pdb` | `data/structures/cdk2_*.pdb`, `data/ligands/cdk2_*_ref.pdb` | move |
| `training_set.txt` | `data/labels/erk2_training.tsv` | move, rename header |
| `final_project/smiles/*.smi` | `data/compounds/cdk2/*.smi` | move |
| `4fv7_docked/*.csv.gz`, `af_docked/*.csv.gz` | `data/derived/erk2_{4fv7,af}/` | move |
| `final_project/{4nj3,5k4j}_docked/*.csv.gz` | `data/derived/cdk2_{4nj3,5k4j}/` | move |
| All other course folders | `_archive/` | move as-is for reference |
| Code in `Week_3_Monday_Docking_and_Scoring.ipynb` (IFP cells, py3Dmol cells) | `src/aidd/ifp.py`, `src/aidd/viz.py` | extract and refactor |
| Code in `Week_3_Tuesday_Challenge_*.ipynb` (scoring/ML pipeline) | `src/aidd/scoring.py` | extract |
| `Week_6_SB-regression-extended.ipynb` regression pipeline | `src/aidd/scoring.py` (regression mode) | extract |
| `ULLA-Leiden-master/.gitignore` | `.gitignore` (root) | seed for our repo |
| `ULLA-Leiden-master/practicals/03_uncertainty_active_learning/utils.py` | `src/aidd/training.py` | adapt `seed_everything`, `create_dataloader` |

## 5. Discard / do-not-bring

- `AIinDD2025/DeepLearning_CNN/`, `Deep_Learning_CNN_v2/`, `shared-data-AIinDD2025/bbbc021v1_images/` — cell-morphology CNN work, off-scope.
- `AIinDD2025/Supervised_ML_lab/`, `251014_*` (Pandas/NumPy intros) — general teaching, not domain-relevant.
- `docking/PLANTS` (the 3 MB Linux ELF binary) — won't run on Windows/Mac, replaced by DiffDock/Vina.
- `Week_5_Monday_dataset/` — verified duplicate of top-level files.
- `*.tgz` archives (`Week_5_Tuesday_dataset_chal2.tgz`, `Week_6_dataset.tgz`) — only keep if their contents aren't already extracted; otherwise drop.
- `AIinDD2025.code-workspace`, `.ipynb_checkpoints/`, layout files — IDE/Jupyter junk.

## 6. Still open

1. **Heavy-data storage.** Git-LFS for the few reference PDBs + manifest-pointed external bucket (OneDrive / S3 / HuggingFace dataset) for everything generated? Or just keep everything local on your machine for now? *Default if not specified: local-only, with `data/MANIFEST.yaml` listing what *would* be uploaded later.*
2. **Real target identity.** When you're ready to swap from the ERK2 placeholder, you'll need to provide: UniProt accession or FASTA sequence, and (optionally) any known holo-PDB you'd like to dock against in parallel with the AF/ColabFold model.

## 6a. Throughput plan implied by the locked decisions

With Colab-only GPU, 1k–10k compounds, and consensus-mode triage:

| Stage | Tool | Where it runs | Order-of-magnitude time for 10k compounds |
|---|---|---|---|
| ADMET / PAINS pre-filter | RDKit | Local (CPU) | minutes |
| 3D embed + standardise | RDKit + DataMol | Local (CPU) | ~30 min |
| ColabFold target fold (one-time per target) | ColabFold | Colab GPU | 20–60 min |
| Boltz-2 fast-lane scoring | Boltz-2 | Colab GPU (A100/L4 ideal) | 5–30 GPU-hours; **the bottleneck** |
| gnina docking | gnina | Local CPU (multi-core) | 5–10 hours |
| PoseBusters | PoseBusters | Local CPU | minutes |
| ProLIF IFPs + features | ProLIF | Local CPU | <1 hour |
| Per-target rescorer (when training data exists) | sklearn/XGBoost | Local CPU | minutes |
| Consensus rank + report | pandas | Local CPU | seconds |

Implications:

- **Plan around overnight Colab Pro sessions** for any full-library Boltz-2 run. Pro+ if you can; the disconnect risk on the free tier is too high for multi-hour jobs.
- **Pre-filter aggressively before Boltz-2.** Cut the library in half (or more) with RDKit/ADMET gates so the GPU bill is on filtered compounds only.
- **Cache everything.** A per-stage `data/derived/<target>/<stage>/manifest.parquet` records which compound→pose→score is already computed; re-runs skip done work. This is mandatory at 10k scale.
- **Two-stage triage to make consensus cheap.** Run gnina+IFP on the full library (cheap, local CPU), keep top 25%, run Boltz-2 only on that subset. Consensus rule applies on the subset. This cuts Boltz-2 GPU usage by ~4x without meaningfully hurting recall on actives.

## 7. First-week deliverables

Re-sequenced for the hybrid stack. Each step has a clear "done" signal so we can pause/resume cleanly.

1. **Repo skeleton.** Rename folder to `aidd-pipeline/` (or move contents into a subfolder of that name). Seed `README.md`, `.gitignore` (from ULLA template), `environment.yml`, `requirements-colab.txt`, `CLAUDE.md` (promoted from `_planning/`). Create empty `src/aidd/`, `notebooks/`, `data/{structures,ligands,compounds,labels,derived}/`. **Done when** `mamba env create -f environment.yml` succeeds locally on Windows.
2. **Prune DROP items.** I produce a kill-list (every path that would be deleted); you sign off; I delete. Move everything else course-related into `_archive/`. **Done when** the repo root has only the new layout + `_archive/`.
3. **Migrate small reference assets.** PDBs, ligands, SMILES, labels per §4. Heavy `*_docked/*.csv.gz` into `data/derived/<target>/` (still gitignored). **Done when** `data/structures/erk2_*.pdb` exist and `_archive/` no longer has duplicates of them.
4. **First push to GitHub.** You create the empty private `aidd-pipeline` repo; I add the remote and push. **Done when** the repo is visible on github.com.
5. **`src/aidd/` v0 modules.** Lift ProLIF + py3Dmol cells from `Week_3_Monday_Docking_and_Scoring.ipynb` into `src/aidd/{ifp,viz,structures,ligands,io}.py`. **Done when** `from aidd.ifp import compute_ifp` works.
6. **`notebooks/00_quickstart.ipynb`** — reproduces the ERK2/4FV7 IFP figure using the new package on the archived course PDBs. **Done when** the figure visually matches the course's.
7. **`notebooks/01_fold_target.ipynb`** — ColabFold on ERK2 sequence; compare to archived `erk2_af.pdb` with biopython distogram from the course's `biopython.ipynb`. **Done when** RMSD to the archived AF model is <2 Å.
8. **`notebooks/02_prepare_ligands.ipynb`** — SMILES → standardised → ADMET/PAINS gate → 3D embed. **Done when** the ERK2 training SMILES survive the gate at expected rates (~70–90%).
9. **`notebooks/03_dock_gnina.ipynb`** — gnina docking of prepared ligands against the 4FV7 receptor; PoseBusters QC. Redock reference ligand as sanity check (RMSD < 2 Å). **Done when** the redock passes.
10. **`notebooks/04_score_classical.ipynb`** — ProLIF IFPs + gnina-score features → sklearn/XGBoost rescorer trained on `data/labels/erk2_training.tsv`. Lifts the Tuesday-challenge pattern. **Done when** the rescorer beats raw gnina affinity on the held-out fold (ROC-AUC or EF1%).
11. **`notebooks/05_dock_boltz.ipynb`** — Boltz-2 co-folding + affinity on the same ligand set; Colab notebook. **Done when** affinity scores correlate (Spearman ρ > 0.3) with the labels.
12. **`notebooks/06_consensus_and_shortlist.ipynb`** — joins outputs from #10 and #11, applies the consensus rule (top-X% in both), emits `shortlist.sdf` + `shortlist.csv` with per-compound poses, IFPs, scores, ADMET flags. **Done when** the shortlist on ERK2 contains a respectable fraction of the labelled actives (sanity check before going to real target).
13. **Second prune pass.** REFERENCE notebooks whose code we have fully lifted are removed from `_archive/`. **Done when** `_archive/` only contains items we haven't extracted yet.
