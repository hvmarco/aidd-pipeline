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
| Triage philosophy | locked (revised 2026-05-15) | **Defensible consensus, target-dependent filter.** Two methods implemented in `src/aidd/consensus.py`: (a) **intersection** — Boltz-2 rank AND rescorer rank both in top X% (original rule, default X=5%); (b) **rank_product_topk** — top K compounds by rank product across the joined cohort (fallback when lanes are uncorrelated). Filter choice is per-target, decided empirically via notebook 06 § 9's threshold sweep + cross-lane Spearman ρ: intersection when lanes correlate (p<0.05, ρ>0.15); rank_product_topk when they don't. See §7 step 12 for the ERK2 finding. |
| Notebook structure | locked | **8 teaching notebooks (00–07) + 1 production runner (99)**, each numbered notebook is one pipeline stage; each is independently runnable from cached intermediate outputs in `data/derived/<target>/<stage>/`. The runner (99) is built last, after 07. |
| Mutation analysis | locked | **Part of the standard workflow.** The pipeline is target-agnostic — any of notebooks 01–06 accepts either a wild-type or mutant sequence/PDB as input. A dedicated **notebook 08 (`08_mutation_analysis`, renumbered from 07 on 2026-05-14)** compares WT vs mutant outputs (structure, IFP, docking shortlist, ΔΔG) for drug-resistance / structural-impact studies. Builds on per-variant computational priors from new **notebook 07 (`07_variant_effect_prediction`)**: AlphaMissense pathogenicity + RaSP ΔΔG + gnomAD allele frequency. See §8. |
| AF3 fold provider toggle in notebook 99 | locked | Notebook 99 (production runner) gets an **optional `FOLD_PROVIDER` parameter** with values `"colabfold"` (AF2, default — open and reproducible for all users) and `"af3_server"` (AlphaFold Server API — requires per-user academic API key in Colab Secrets, owner-only path). Both branches produce the same canonical `<target>_best.pdb` so downstream cells don't care which folder was used. Single notebook with one toggle, not two parallel notebooks. See §9. |
| Research-domain focus | locked (2026-05-12) | The pipeline is **shaped for pharmacogenomics + variant-function studies in common solid tumours** (colorectal, lung, breast, GI, GU, ovarian). The central question is "*how does an amino-acid variant change enzyme function / drug binding?*" — not generic SBVS. **Notebook 08 (mutation analysis) is the headline library-side notebook** (paired with new notebook 07 `07_variant_effect_prediction` providing per-variant computational priors, and notebook 09 `09_moa_small_n` for small-N MoA), not extensions. Notebook 99 takes `(target, variants=[…])` and runs the WT-vs-variant comparison as its primary mode. The headline demos are listed in §10. |
| Post-13-step roadmap | locked (2026-05-12; updated 2026-05-14) | After step 13 closes, the build resumes with **notebook 07** (`07_variant_effect_prediction`) — AlphaMissense (pathogenicity) + gnomAD (allele frequency) + RaSP (ΔΔG stability), bundled as one new notebook (~2–3 days combined). **Promoted from post-13-step Priorities 1, 1.5, 2 into the main timeline (2026-05-14)** so notebooks 08 and 09 can show per-variant computational priors on day one rather than retrofitting them later. Then **notebook 08** (`08_mutation_analysis`, formerly nb 07 — library-side WT-vs-mutant diff), **notebook 09** (`09_moa_small_n` — small-N mechanism-of-action; see [`MECHANISM_OF_ACTION_SCOPE.md`](MECHANISM_OF_ACTION_SCOPE.md)), and **notebook 99** (production runner with `RUN_MODE = library | moa`). Remaining post-13-step items in §11: **PharmGKB / CPIC** clinical ground-truth lookup (~1–2 days), **fpocket** binding-site detection on variant folds (~1 day), **PCM** proteochemometrics (~3–5 days, optional). Resumable per-compound docking (originally priority 4.5) was pulled forward to the first commit of step 11. See §11 for the rationale and ordering. |

See [CONSULTANT_REVIEW.md](CONSULTANT_REVIEW.md) for the architectural reasoning behind the reopened/new rows.

## 1. Goal (in one sentence)

Stand up a reproducible, cross-platform Jupyter pipeline that, given a target protein (sequence or PDB, **wild-type or mutant**) and a list of candidate ligands (SMILES), produces folded structures, docked poses, interaction fingerprints, ML-scored rankings, and a shortlist of promising candidates for wet-lab follow-up — including the ability to compare WT vs mutant runs side-by-side for drug-resistance / mechanism-of-resistance studies.

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
│   ├── 01_fold_target.ipynb             # ColabFold / AlphaFold for the target sequence (WT or mutant)
│   ├── 02_prepare_ligands.ipynb         # SMILES → standardised → 3D conformers
│   ├── 03_dock_gnina.ipynb              # gnina docking + PoseBusters QC
│   ├── 04_score_classical.ipynb         # IFPs, ML rescoring
│   ├── 05_dock_boltz.ipynb              # Boltz-2 co-folding + affinity (fast lane)
│   ├── 06_consensus_and_shortlist.ipynb # consensus ranker → shortlist.sdf
│   ├── 07_variant_effect_prediction.ipynb # AlphaMissense + RaSP ΔΔG + gnomAD per-variant priors
│   ├── 08_mutation_analysis.ipynb       # diff WT vs mutant runs (structure, IFP, shortlist, ΔΔG)
│   ├── 09_moa_small_n.ipynb             # small-N mechanism-of-action: per-compound HTML reports
│   ├── 99_screen_library.ipynb          # end-to-end runner (RUN_MODE = library | moa); accepts mutations= list
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
12. **`notebooks/06_consensus_and_shortlist.ipynb`** — joins outputs from #10 and #11, applies the consensus rule (target-dependent filter; see §0 "Triage philosophy"), emits `shortlist.sdf` + `shortlist.csv` with per-compound poses, IFPs, scores, ADMET flags. Section 9 sweeps both filter methods × four thresholds + reports cross-lane Spearman ρ for empirical operating-point selection. **Done when** the shortlist on ERK2 contains a respectable fraction of the labelled actives. **Closed 2026-05-15 (commit `a846fe2`):** rank_product_topk @ TOP=0.10 → 42-compound shortlist, 25/126 actives, 19.8 % recall, 1.95× enrichment; intersection rule empirically broken on this target (Spearman ρ = 0.08 between lanes — statistically uncorrelated); rank-product alternative added to consensus module in the same step.
13. **Second prune pass — audit project artifacts.** *(Revised scope 2026-05-15.)* The original wording said "remove fully-lifted `_archive/` items" but is superseded by CLAUDE.md's "`_archive/` is the historical record, read-only" rule: `src/aidd/` modules cite archive paths as their source-of-record, so pruning would create dead citation links and erase methodology provenance. Revised scope: audit project artifacts (scratch files, stub modules, tracked build artifacts, outdated planning drafts) that WE created during the build for items that are no longer needed. **Done when** the audit completes and surfaced candidates are dispositioned. **Closed 2026-05-15 (commit `2b872f7`, empty):** audit found 0 scratch / stub / build-artifact candidates; `_archive/` untouched as the citation record; 2 untracked planning docs (DEV_CHAT_HANDOVER_2026-05-14.md, HOUSEKEEPING_BATCH.md) deleted from workspace; KILL_LIST.md + INVENTORY.md kept tracked as historical record of step 2's first prune.

*Steps 14–17 are the post-13-step substantive work added during the 2026-05-14 scoping pass (see §0 "Post-13-step roadmap" row and [`MECHANISM_OF_ACTION_SCOPE.md`](MECHANISM_OF_ACTION_SCOPE.md)). Dependency order: 14 lands before 15 and 16 — both consume nb 07's per-variant priors; 17 needs 14 + 15 + 16 done; 13 was independent of all of the above and is now closed.*

14. **`src/aidd/variants.py` + `src/aidd/stability.py` + `notebooks/07_variant_effect_prediction.ipynb`** — AlphaMissense pathogenicity lookup, RaSP ΔΔG prediction, gnomAD allele-frequency lookup; packaged as per-variant computational priors consumed by notebooks 08 and 09. **Done when** the helpers return sensible values for a known calibration set (e.g. NAT2\*5: AlphaMissense flags pathogenic + RaSP ΔΔG > 0; DPYD\*2A: similar) and the notebook smoke-tests end-to-end on Colab + locally.
15. **`notebooks/08_mutation_analysis.ipynb`** (renumbered from 07 on 2026-05-14). WT-vs-mutant pipeline diff (Cα-RMSD, IFP diff, shortlist diff, ΔΔG from nb 07). **Done when** the DPYD\*2A + 5-FU walkthrough produces a coherent comparison report including nb 07's priors; six brief follow-up demos run on the same template.
16. **`notebooks/09_moa_small_n.ipynb`** + **`src/aidd/inputs.py`** — small-N mechanism-of-action driver with per-compound HTML reports; see [`MECHANISM_OF_ACTION_SCOPE.md`](MECHANISM_OF_ACTION_SCOPE.md). **Done when** the NAT2 + isoniazid deep walkthrough produces 4 per-compound HTML reports (WT + \*5 + \*6 + \*7) + a summary CSV; four brief follow-up demos run on the same template.
17. **`notebooks/99_screen_library.ipynb`** — production runner with `FOLD_PROVIDER` and `RUN_MODE` toggles. **Done when** library mode produces a shortlist on the ERK2 + headline-demo set and `moa` mode produces per-compound HTML reports on a small smoke set; both Colab and local-CPU paths verified.

## 8. Mutation analysis — the headline workflow

**(Promoted from "extension" to "headline" on 2026-05-12 — see §10.)**

### Why it matters

In oncology, **amino-acid variants in or near the active sites of enzymes** drive the entire clinical landscape. Two mechanisms dominate:

- **Resistance mutations**: a drug works initially, the tumour evolves a single amino-acid change that reshapes the binding pocket, and the drug loses potency. Classic examples — EGFR T790M (osimertinib resistance), BRAF V600E (vemurafenib activation), ESR1 Y537S (tamoxifen resistance), ABL T315I — define the prescribing landscape for second- and third-line therapies.
- **Pharmacogenomic loss-of-function**: germline variants in drug-metabolising enzymes (DPYD, UGT1A1, NAT2, TPMT, CYP2D6) change how patients metabolise standard chemotherapy. DPYD\*2A homozygotes given full-dose 5-FU experience life-threatening toxicity; the pharmacogene structural change is the mechanism.

Any pipeline aimed at cancer drug discovery — but *especially* one shaped for common solid tumours and precision oncology — has to answer "*what happens when I run this variant against the same drugs and substrates as the wild-type?*"

The pipeline is **target-agnostic by design**, so the same notebooks (01–06) accept a mutant sequence with no code changes; the comparison is then a dedicated notebook (07) that diffs the two `data/derived/<target>[_<variant>]/` trees.

### Levels of investigation supported

| Level | Question | How the pipeline answers it | Effort beyond a normal WT run |
|---|---|---|---|
| **1. Mutant structure** | "What does the mutant look like?" | Notebook 01 with the mutant sequence | None (just paste the sequence) |
| **2. Structural difference** | "How is the mutant fold different from WT?" | `aidd.structures.ca_rmsd` + `distogram` + visual diff in notebook 08 | None (helpers already exist) |
| **3. Stability change (ΔΔG)** | "Is the mutant more / less stable?" | Per-variant RaSP ΔΔG in `aidd.stability` + a cell in **notebook 07 (`07_variant_effect_prediction`)** (promoted into the main timeline 2026-05-14, was post-13-step Priority 2) | ~1–2 days of integration work |
| **4. Effect on drug binding** | "Does my candidate library still bind the mutant?" | Run notebooks 02–06 against the mutant; notebook 08 diffs the consensus shortlists | ~2× compute (one full screen per genotype); no new code |
| **5. Atom-level dynamics** | "Did the mutation change a hinge motion, allosteric loop?" | **Out of scope** (would require MD simulation, separate field) | Not planned |

### Notebook 08 — `08_mutation_analysis.ipynb` (renumbered from 07 on 2026-05-14)

**Purpose:** consume the outputs of two complete pipeline runs (WT and mutant) and produce a clinically-meaningful comparison.

**Inputs:** two `data/derived/<target>/` directories — one labelled WT, one a mutant variant (e.g. `data/derived/erk2_wt/` and `data/derived/erk2_M106T/`).

**Outputs:**
- Cα-RMSD and pTM/pLDDT comparison between the two folds (level 1+2).
- IFP-level diff: which interactions are gained / lost in the mutant pocket.
- Side-by-side 3-D viewer of WT and mutant binding pockets.
- Diff of the two final shortlists: compounds that drop out, compounds that survive, compounds that newly appear.
- A summary table fit for a wet-lab handoff: "in WT we'd prioritise these N compounds; in the mutant the priorities reshuffle to these M, of which X are new candidates".

**When built:** after notebook 07 (`07_variant_effect_prediction`). Notebook 08 consumes nb 07's per-variant priors (AlphaMissense, RaSP ΔΔG, gnomAD) plus the consensus shortlist from nb 06, so neither can land before notebook 08 does.

**Effort estimate:** 1–2 days. Most of the work is a comparison notebook; the underlying helpers (`ca_rmsd`, `distogram`, `compute_ifp`) already exist.

### Implications for downstream architecture

- **Notebook 99 (production runner)** must accept a `mutations` parameter (list of variants to run alongside the WT). Each mutant becomes its own `data/derived/<target>_<mutation>/` directory; notebook 08 is invoked once per mutant to produce the comparison artefacts.
- **`data/derived/` naming convention** is updated to allow per-variant subdirectories: `data/derived/<target>[_<variant>]/<stage>/`. Wild-type omits the variant suffix.

## 9. Notebook 99 — optional AlphaFold 3 fold provider toggle

*Reframed 2026-05-17 following step-17 pre-implementation reading pass: the original API-client design assumed a public AF3 programmatic submission API that does not exist. See "Constraint as of 2026-05" below.*

### Background

Notebook 01 (`fold_target`) uses **AlphaFold 2 via ColabFold** for structure prediction. AF2 weights are open (CC-BY-4.0), so anyone — student, colleague, reviewer — can run the notebook and reproduce the fold. This is the right default for an open-science pipeline and is **not changing**.

In parallel, the project owner (Natallia, with academic access via her university) may want to use **AlphaFold 3** for her own runs — e.g. for grant materials where citing the newest tool is useful, or for spot-checking specific results against AF3's co-folding head. AF3's weights are **academic non-commercial only, per-user application, no redistribution**, so AF3 cannot be the default for the pipeline. It can however be an opt-in path that the notebook owner enables when they want.

### Constraint as of 2026-05 — no public AF3 programmatic submission API

AlphaFold Server ([alphafoldserver.com](https://alphafoldserver.com)) has **no public programmatic submission API** as of 2026-05. The web UI accepts visual-form input or a "Upload JSON" button for headless job specification, but submission itself is web-UI-only — no REST endpoint, no API key system, no polling endpoint for headless execution. Daily quota is ~30 jobs per user, rate-limited at the UI layer. (Confirmed via EBI AlphaFold training docs and google-deepmind/alphafold#944.)

For programmatic AF3 access, the community routes are: run the open-source `alphafold3` model locally with weights obtained by per-user academic application (Google distributes weights under an agreement; self-host on GPU), or use a paid third-party hosted re-implementation (e.g., Protenix via Neurosnap). Neither fits a Colab-first open-science notebook's design.

The honest compromise: the operator submits the fold via the alphafoldserver.com web UI under their per-user academic agreement, downloads the result zip, and points the pipeline at it. Step-17 implements this as a **parser-only helper**; if AF3 ever ships a public programmatic API, the helper can be extended (see "Forward-compat note" below).

### Design — single notebook, one toggle

`99_screen_library.ipynb` exposes a single parameter near the top:

```python
FOLD_PROVIDER = "colabfold"   # default: AF2 via ColabFold. Open weights, reproducible by anyone.
# FOLD_PROVIDER = "af3_server" # operator-only: AF3 via web UI; result zip supplied via AF3_RESULT_ZIP path.
```

The folding cell dispatches on `FOLD_PROVIDER`. Both branches produce the same canonical PDB at `data/derived/<target>/fold/<target>_best.pdb`; downstream cells (docking, scoring, consensus) don't know which folder was used.

**Two parallel notebooks (`99` and `99_af3`) are explicitly rejected** in favour of one notebook with one toggle. Two notebooks would force every other change to be made twice and would drift over time.

### Spec for the agent who builds this

1. **In `src/aidd/folding.py`** — add a new parser function alongside the existing ColabFold parser (landed at step-17 commit 1/N):
   - `fold_with_af3_server(af3_result_zip: PathLike, output_dir: PathLike, *, target_name: str = "target") -> Path`
   - **Not an API client.** The operator submits the fold via [alphafoldserver.com](https://alphafoldserver.com)'s web UI under their per-user academic agreement and downloads the result zip.
   - The helper unzips, picks the rank-1 model (prefers `*_model_0.cif`; falls back to lexically-first `.cif` with a logged warning so layout drift surfaces visibly), parses via Biopython's `MMCIFParser`, and writes the canonical `<target_name>_best.pdb` to `output_dir`. Returns the path.
   - Cross-platform stdlib `zipfile` + Biopython (already in env). No network, no auth, no GPU. Lazy Biopython import matches `prep_receptor`'s pattern in `src/aidd/inputs.py`.
   - ~150 lines including defensive validation, descriptive `raise` messages, and the rank-1 fallback.

2. **In `notebooks/_build_99_screen_library.py`** — the folding cell does:
   - If `FOLD_PROVIDER == "colabfold"`: existing ColabFold path (delegated to notebook 01's logic via the package).
   - If `FOLD_PROVIDER == "af3_server"`:
     - Read `AF3_RESULT_ZIP` (an operator-set path to the downloaded zip — typically on Drive at `data/manual_folds/<target>_af3.zip`).
     - If missing → raise with a clear message: *"Submit the fold via https://alphafoldserver.com under your academic agreement, download the result zip, and set `AF3_RESULT_ZIP` to its path."*
     - Call `fold_with_af3_server(af3_result_zip, output_dir, target_name=...)`.

3. **One markdown cell** in 99 explaining: the toggle, why the default is open, how the owner sets up the AF3 path (link to AF3 Server academic application, web-UI submission walk-through, where to drop the result zip), and an explicit caveat that anything pushed to the repo with AF3 results must include the methods citation.

4. **No copies of AF3 weights or output** in `data/derived/` ever get committed. AF3 outputs are personal data; the `.gitignore` on `data/derived/` already prevents this.

### Forward-compat note

If AF3 ever ships a public programmatic submission API, `fold_with_af3_server`'s signature can extend to support both modes: the existing parser path stays for users who prefer manual web-UI control; a new `submit_via_api=True` flag plus `api_key` kwarg adds the headless path. The on-disk contract (canonical PDB at the expected location) remains unchanged, so notebook 99's downstream cells stay agnostic to which mode was used.

### Effort estimate

~0.5 day for the parser helper + secret-handling-free notebook integration. The earlier estimate (1–2 days for API client + auth + polling + retry) was based on the API-client design that turned out not to be implementable; the parser-only design is substantially smaller.

### When this lands

The parser helper landed at step-17 commit 1/N (`src/aidd/folding.py`, the `fold_with_af3_server` function). The notebook integration lands as part of step 17's main builder commit (`_build_99_screen_library.py`). The AF3 path is an additive feature, not a prerequisite for shipping 99 in its default `FOLD_PROVIDER="colabfold"` mode.

## 10. Research-domain focus: pharmacogenomics + variant-function in common solid tumours

### What the pipeline is *for*, in one sentence

Predict how amino-acid variants in or near the active sites of cancer-relevant enzymes change drug binding, substrate metabolism, or protein function — for the common solid tumours (**colorectal, lung, breast, ovarian, GI, GU**) where amino-acid substitutions in active sites drive both **oncogenesis** and **drug response**.

This is the project owner's actual research domain. The pipeline is *not* "yet another generic SBVS pipeline"; it is shaped to answer the specific class of clinical question her work asks.

### How this shapes the existing notebooks

The pipeline code is **largely unchanged**; the framing, examples, and pedagogical emphasis shift:

| Notebook | Scale of change | What changes |
|---|---|---|
| 00 quickstart       | None     | ERK2 sanity-check stays as a *technical* demo |
| 01 fold_target      | Light    | Markdown: emphasise "fold WT + variant side-by-side"; example becomes DPYD WT + DPYD\*2A |
| 02 prepare_ligands  | Light    | Framing: examples include known substrates (5-FU, irinotecan) alongside inhibitors |
| 03 dock_gnina       | Moderate | Add **"substrate-binding vs inhibitor-binding"** markdown section; interpret scores accordingly |
| 04 score_classical  | Light    | Examples and feature framing update; mechanics unchanged |
| 05 dock_boltz       | Light    | Examples update |
| 06 consensus        | Light    | Examples update |
| **07 variant_effect_prediction** *(new)* | **Major (new notebook)** | AlphaMissense + RaSP ΔΔG + gnomAD per-variant priors; promoted from post-13-step Priorities 1, 1.5, 2 on 2026-05-14 |
| **08 mutation_analysis** | **Major (headline)** | (Was 07.) DPYD walkthrough as pedagogical centrepiece + consumes nb 07 priors (AlphaMissense + RaSP ΔΔG + gnomAD) + substrate framing + 6 brief additional demos |
| **09 moa_small_n** *(new)* | **Major (new notebook)** | Small-N mechanism-of-action; per-compound HTML reports + summary CSV; 5 demos covering CRC pharmacogenomics + KRAS; see [`MECHANISM_OF_ACTION_SCOPE.md`](MECHANISM_OF_ACTION_SCOPE.md) |
| **99 runner**       | **Major** | Re-shaped to take `(target, variants=[…])`; `RUN_MODE = library | moa` toggle; default examples are the headline-demo set; AF3 toggle stays |

### Headline demo set (notebook 08 + notebook 09 + notebook 99 defaults)

Cases covering all the named cancer types and the distinct mutation mechanisms encountered in clinical oncology. Five of these — DPYD, NAT2, CYP2D6, UGT1A1, KRAS — are shared with notebook 09 (small-N MoA; see [`MECHANISM_OF_ACTION_SCOPE.md`](MECHANISM_OF_ACTION_SCOPE.md)). The other two (ESR1, BRCA1) are library-side only: ESR1 stays library-only to keep notebook 09 focused on Natallia's pharmacogenomics / drug-metabolism enzymes, and BRCA1 is excluded from notebook 09 because olaparib binds PARP, not BRCA1, so the variant doesn't fit notebook 09's "variant alters binding" Q-shape.

| Demo | Cancer(s) | Mechanism | Drug / substrate |
|---|---|---|---|
| **DPYD\*2A + 5-FU** *(deep walkthrough)* | colorectal | pharmacogene loss-of-function (splice-variant LoF) | 5-fluorouracil (substrate) |
| **KRAS G12C + sotorasib** | colorectal, lung, pancreatic | GTPase oncogenic driver | sotorasib (covalent inhibitor) |
| **ESR1 Y537S + tamoxifen** | breast | nuclear-receptor ligand-binding-domain hot-spot (GoF / endocrine resistance) | tamoxifen / fulvestrant |
| **BRCA1 LoF + olaparib** | ovarian, breast | synthetic-lethality LoF (drug binds PARP, not BRCA — the mutation creates the vulnerability) | olaparib (PARP-bound) |
| **CYP2D6 *4 / *10 + tamoxifen** | breast (pharmacogene activation) | Phase I oxidation — poor metabolisers under-activate tamoxifen → endoxifen | tamoxifen (substrate; canonical breast-cancer pharmacogene story) |
| **NAT2 slow acetylator (\*5 / \*6 / \*7)** | colorectal, bladder (cancer risk) | Phase II acetylation — slow acetylators under-detoxify aromatic-amine carcinogens | isoniazid (textbook substrate for slow-acetylator phenotype; structural argument generalises to aromatic-amine carcinogens for the cancer-risk angle) |
| **Irinotecan + UGT1A1\*28** | colorectal | Phase II glucuronidation — UGT1A1\*28 homozygotes have reduced clearance → severe neutropenia | irinotecan (substrate; FDA-label genotype-guided dosing) |

Each illustrates a *distinct* clinical mechanism — pharmacogene LoF (DPYD), oncogenic driver (KRAS), ligand-pocket GoF (ESR1), synthetic-lethality LoF (BRCA1), drug-activation pharmacogene (CYP2D6), carcinogen-metabolism pharmacogene / cancer-risk angle (NAT2), and chemotherapy-clearance pharmacogene (UGT1A1). The deep walkthrough (DPYD) gets the full pedagogical structure; the other six get brief "*the same pattern applies here*" sections at the end of notebook 08. CYP2D6, NAT2, and UGT1A1 together cover three faces of cancer pharmacogenomics — therapy response (CYP2D6, tamoxifen activation), cancer risk (NAT2, carcinogen detoxification), and chemotherapy clearance (UGT1A1, irinotecan dosing) — all directly in the project owner's research domain.

### Why this set

- **Covers all named cancer types** (colorectal, lung, breast, ovarian; GI / GU covered by colorectal + pancreatic).
- **Covers all major mutation mechanisms** clinicians encounter — pharmacogene LoF, oncogenic driver, ligand-pocket GoF, synthetic-lethality LoF, Phase I + Phase II metabolism PGx.
- **All have clinical ground truth** (CPIC guidelines for DPYD and UGT1A1; FDA labels for sotorasib, fulvestrant, olaparib, irinotecan-UGT1A1 dosing). Predictions are falsifiable.
- **DPYD as the walkthrough specifically** because: most clinically actionable example in colorectal cancer; demonstrates *substrate* binding (the framing shift); is a *loss-of-function* variant (exercises the RaSP integration).

### TP53 R175H as a future "no-paired-drug" demo

A fifth demo, useful later: TP53 R175H is the #1 most-mutated variant across all cancers, and is structurally a "loss-of-structural-integrity" case — perfect RaSP showcase. No direct drug (TP53 isn't directly druggable in the standard sense), so it sits naturally as a "*here's what the pipeline shows when no drug exists yet*" example. Add when the four headline demos are stable.

## 11. Post-13-step roadmap — pharmacogenomics-aligned extensions

These additions extend the pipeline's coverage beyond the closed-out 13-step plan. They are **not** prerequisites for shipping the production runner (99) — they are enhancements specifically tuned to the §10 research focus. Listed in priority order by value-per-effort for this domain.

> **2026-05-14 update:** Priorities 1 (AlphaMissense), 1.5 (gnomAD), and 2 (RaSP ΔΔG) were **promoted out of post-13-step into the main timeline** as the new **notebook 07** (`07_variant_effect_prediction`). The detailed reasoning for each promoted item remains here as historical reference; the active scope lives in §0 decisions table (post-13-step row), §7 step 14, and the implementation plan in [`MECHANISM_OF_ACTION_SCOPE.md`](MECHANISM_OF_ACTION_SCOPE.md). Priorities 3 (PharmGKB / CPIC), 4 (fpocket), and 5 (PCM) remain post-13-step.

### Priority 1 — AlphaMissense lookup (~½ day) **[PROMOTED to notebook 07 on 2026-05-14]**

**What:** DeepMind's [AlphaMissense](https://www.science.org/doi/10.1126/science.adg7492) model (2023) scores every possible missense variant in the human proteome with a pathogenicity probability. Pre-computed scores are downloadable; we look up the answer rather than running the model.

**Why for this pipeline:** independent of structure-based ΔΔG, AlphaMissense gives a sequence/evolution-based pathogenicity signal. Combine the two and the pipeline can say "this variant is flagged pathogenic by AlphaMissense AND we can structurally explain why" — much stronger claim than either signal alone. Drops in as a feature column in notebook 07.

**Integration:** new helper `aidd.variants.alphamissense_score(uniprot_id, position, alt_aa)`. ~50 lines plus a one-time download of the supplementary CSV (~5 GB). Added as a column to notebook 07's variant-analysis output.

### Priority 1.5 — gnomAD allele-frequency lookup (~½ day) **[PROMOTED to notebook 07 on 2026-05-14]**

**What:** [gnomAD](https://gnomad.broadinstitute.org/) (Genome Aggregation Database) gives population-level allele frequencies for >800,000 exomes and genomes, stratified by ancestry. We query it per variant via the public GraphQL API.

**Why for this pipeline:** for pharmacogenomics specifically, allele frequency is the **triage filter** that decides which variants are worth running through the structural pipeline. A variant present in 1 person globally is not the same clinical priority as one at 2% frequency in Europeans. Combined with AlphaMissense, gnomAD gives a clean upstream filter:

- **gnomAD** answers: *is this variant common enough to study?*
- **AlphaMissense** answers: *is it likely pathogenic?*
- Their intersection (common + pathogenic) is the variant set worth investing GPU time on.

For pharmacogenes, gnomAD also surfaces **population-stratified frequencies** — DPYD\*2A is ~1.5 % in Europeans but ~0.05 % in East Asians, and that changes regional dosing recommendations. Real clinical question; trivially answered with a gnomAD lookup.

**Integration:** new helper `aidd.variants.gnomad_frequency(uniprot_id, position, alt_aa) -> dict` returning overall + per-population allele frequencies + heterozygote/homozygote counts. ~30–50 lines wrapping the gnomAD GraphQL API. Drops into the variant-selection cell of notebook 07 alongside `alphamissense_score`.

**When this lands:** ideally in the same round as AlphaMissense (Priority 1), since the two are paired conceptually. Together they form the "*should we even bother docking this variant?*" pre-filter.

### Priority 2 — RaSP ΔΔG prediction (~1–2 days) **[PROMOTED to notebook 07 on 2026-05-14]**

**What:** [RaSP](https://elifesciences.org/articles/82593) (Rapid Stability Predictions, 2023) is an open-source ML-based predictor of variant stability changes (ΔΔG, kcal/mol). Runs in seconds per mutation on CPU.

**Why for this pipeline:** for **loss-of-function variants** (DPYD, BRCA1, MMR, TP53), stability change is the *primary* mechanism — the protein doesn't fold correctly or doesn't stay folded. Binding-affinity prediction misses this entirely. ΔΔG fills the gap.

**Integration:** new module `aidd.stability` wrapping RaSP. New section of notebook 07 reports per-variant ΔΔG alongside the existing structural / IFP / docking diffs. The DPYD walkthrough specifically showcases this — DPYD\*2A is a splice variant whose stability impact is the headline.

### Priority 3 — PharmGKB / CPIC clinical-ground-truth lookup (~1–2 days)

**What:** [PharmGKB](https://www.pharmgkb.org) is the authoritative pharmacogenomics knowledge base; [CPIC](https://cpicpgx.org/) publishes formal clinical guidelines on gene-drug-phenotype interactions. Together they cover every clinically-relevant DPYD / UGT1A1 / NAT2 / CYP2D6 / TPMT variant with metabolizer phenotype and dose recommendations.

**Why for this pipeline:** gives **falsifiable ground truth** for validation. "*Pipeline predicts DPYD\*2A reduces 5-FU binding by N kcal/mol → CPIC says DPYD\*2A homozygotes need 5-FU dose reduced by 50% → pipeline prediction is consistent with clinical practice.*" Without this, the pipeline produces numbers nobody can score against reality.

**Integration:** new module `aidd.pharmacogenomics` that pulls relevant entries by gene + variant. Mostly metadata curation work, not heavy code.

### Priority 4 — fpocket binding-site detection on variant folds (~1 day)

**What:** [fpocket](https://github.com/Discngine/fpocket) is a fast pocket-detection tool. Takes a PDB, returns ranked pockets with centre + radius + volume.

**Why for this pipeline:** for *any* pharmacogene the project owner takes on beyond the demo set, hand-curating a binding-site box is the manual step that won't scale. fpocket on WT and on the variant fold ALSO gives "*pocket volume changed from N to M Å³*" as a direct readout of the variant's structural impact — independent signal beyond ΔΔG.

**Integration:** wraps in `aidd.docking` as an alternative to hand-curated binding-site coordinates. New cell in notebook 08 reporting pocket-geometry deltas WT vs variant.

### Priority 4.5 — Resumable per-compound docking (~½ day)

**What:** refactor `aidd.docking.dock_library` so each compound's docking output is written to disk before moving on, and the cache check skips already-completed compounds rather than requiring the whole library to finish in one go.

**Why:** the current implementation runs gnina as a single subprocess on the whole ligand SDF and only writes `gnina_scores.csv` after the entire run completes. If Colab disconnects or the cell is interrupted after 80 minutes of a 90-minute run, **the whole thing has to start over**. We hit this on step 10's cold cache (interrupted at 414/693 compounds; the manual harvest cell recovered usable data but the partial run wasn't reusable as a cache). For step 11's Boltz-2 co-folding the same trap is worse: per-compound inference is comparable to or slower than gnina's 52 s/compound, so 414 compounds is 4–10 hours — well past Colab's idle-disconnect window.

**Integration:** `dock_library` becomes a loop that splits the SDF into per-compound chunks, runs gnina per chunk (or in small batches), and accumulates results. Cache check walks the per-compound output directory and only docks compounds that don't yet have a saved score row. The public API stays the same; downstream callers (notebook 03, notebook 04's labelled-subset cache) don't know the difference. Step 10's existing cached output on Drive must be picked up by the new cache check exactly as before — verify with a quick re-run before moving on.

**When this lands (updated 2026-05-12):** **as the first commit of step 11**, before the Boltz-2 module work proper. Originally tracked as "after the 13-step plan closes" but the timing shifted: Boltz-2's `predict_library` needs the *same* per-compound resumability pattern, and doing both at once is cleaner than refactoring gnina later while building Boltz-2 resumable from day one. The agent's first commit in step 11 is the gnina refactor; the rest of step 11 follows.

**Why this is no longer post-13-step:** symmetry. The same caching pattern serves both `aidd.docking.dock_library` (gnina) and the new `aidd.co_folding.predict_library` (Boltz-2). Building one without the other is inconsistent.

### Priority 5 — TeachOpenCADD T032-style proteochemometrics (~3–5 days)

**What:** add **protein descriptors** alongside ligand descriptors so one model can predict binding across a *family* of related enzymes simultaneously.

**Why for this pipeline:** drug-metabolising enzymes come in families (CYP1/2/3 subfamilies; UGT1A1–10; NAT1/NAT2; SULT family). A PCM model trained on a CYP3A4 dataset can generalise to predict CYP3A5 substrate preferences. Worth adding only if the research programme actually touches multiple enzymes in the same family.

**Integration:** lifts directly from `_archive/PCM/talktorial.ipynb`. Becomes either a new notebook 10 or a section in 04.

### What's *not* on this list

For the record:

- **KLIFS kinase-pocket alignment**: useful only for the kinase subset (EGFR, BRAF, HER2, PIK3CA, CDK4/6). The pharmacogene + DDR + nuclear-receptor sides of the research focus don't benefit. Worth adding only if work shifts to majority-kinase.
- **ASAP-discovery free-energy methods (FEP)**: expensive both in compute and setup. Worth it for *publication-quality* binding-affinity claims on a small number of top compounds, not for routine triage. Keep as a "hero-number" tool, not a default.
- **Y-randomisation / applicability domain** (Pat Walters' rigour patterns): added inside notebook 04 (step 10) as standard ML hygiene; doesn't need to be in this roadmap.
