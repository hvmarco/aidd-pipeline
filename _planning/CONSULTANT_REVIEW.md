# Consultant review — pipeline architecture

A take from the outside, written before any code or files have been moved. Goal: identify the "red thread" the course is actually teaching, survey what the open-source community offers in 2026, and recommend a concrete architecture that gets to a working wet-lab triage pipeline with the least invented wheel.

## 1. What the course is actually teaching (the red thread)

Stripping the topic-by-topic chronology and the topical filler (Pandas/NumPy intros, image-CNN morphology lab) away, the course is one consistent pipeline taught twice on two targets:

```
target sequence/PDB  →  fold (AF model provided)  →  prepare receptor + ref ligand
                                                              │
SMILES library  →  standardise + 3D embed  ───────────────────┤
                                                              ▼
                                                  classical docking (PLANTS)
                                                              │
                                                              ▼
                                       per-pose features:
                                          PLANTS scores + interaction fingerprints (prolif)
                                                              │
                                                              ▼
                                       ML scorer
                                          Mon: build features, viz, IFPs
                                          Tue: classification (active/inactive) — ERK2
                                          Wk6: regression (pIC50) — CDK2
                                                              │
                                                              ▼
                                         ranked shortlist + uncertainty
                                          (uncertainty quantification + active learning practical)
```

That is the textbook pattern of **structure-based virtual screening (SBVS) with ML rescoring**. The PCM practical, the QSAR/descriptors lab, and the unsupervised plotting are bolt-on flavours of the ML step. Everything maps to one of: structure prep, ligand prep, docking, featurisation, scoring, ranking/triage.

The bits the course is light on, that wet-lab triage actually needs:

- **Pose quality control.** "Did the docker produce a physically sensible pose?" The course inspects manually with py3Dmol; there's no automated reject step.
- **ADMET / drug-likeness gating.** No PAINS, Lipinski, SA-score, solubility, hERG, etc. — none of which the course covers, but all of which a chemist will demand before ordering compounds.
- **Consensus / orthogonal validation.** The course relies on a single docker + a single ML model. Wet-lab triage usually wants two independent signals to agree.
- **Library-scale throughput.** Course datasets are ≤155 MB / ≤12k compounds. A real triage list is 10⁵–10⁶.

These four gaps are where modernising actually pays off.

## 2. The 2026 open-source landscape

A non-exhaustive map of what's mature enough to lean on. Items in **bold** are what I'd actually use; the rest are listed so we can revisit if priorities shift.

### Folding / structure prediction

- **ColabFold** — battle-tested AF2 front-end, MMseqs2 MSAs, Colab-runnable. *Default for "we need an AF model and don't have one"*.
- **Boltz-2** (MIT, 2025) — open-weights co-folding model: predicts protein + ligand + cofactor complex and an affinity score in one pass. The single biggest shift since AF2. Can replace the fold + dock + score steps for routine cases.
- **Chai-1** (Chai Discovery) — comparable to Boltz-2; good for cross-validation. Slightly more permissive licence for some commercial uses.
- AlphaFold3 — Google DeepMind's; weights restricted, open code lags. Useful as a reference but not a first-class dependency.
- ESMFold — single-sequence, fast, no MSAs needed. Worse accuracy on novel targets than ColabFold. Useful for early-stage triage of many targets.
- RoseTTAFold All-Atom — open, capable, heavier setup than ColabFold.

### Binding-site detection (often skipped, often needed)

- **fpocket** — classical, fast, OS-agnostic. Default.
- **DeepPocket / P2Rank** — ML-based, better recall on cryptic sites.
- Co-folding tools (Boltz-2, Chai-1) implicitly identify the site by placing the ligand.

### Ligand prep

- **RDKit + meeko** — RDKit for standardisation/embedding, meeko for AutoDock-family PDBQT prep. Standard.
- **OpenFF Toolkit** — modern force-field assignment if MM-based rescoring matters later.
- **DataMol** (`datamol`) — RDKit wrapper that hides most boilerplate; nicer for notebook code.

### Docking

- **gnina** — CNN-rescored docking on top of smina/Vina. Open, GPU-optional, runs on Linux/Mac/Win (WSL). Generally best classical docker for ML-aware rescoring out of the box. *Default classical docker for this project.*
- **AutoDock Vina** — universal baseline, no GPU. Fine fallback.
- **DiffDock / DiffDock-L** — diffusion-based blind docking. Still good for cryptic sites and for catching alternative binding modes; weaker than Boltz-2 / gnina for known-pocket scoring.
- **NeuralPLexer / Umol / RosettaFold-AA in docking mode** — newer, less battle-tested.
- PLANTS (course) / Glide / GOLD — proprietary; drop.

### Pose QC

- **PoseBusters** — automatic stereo/strain/clash filter for docked poses. Drop a pose if it fails. Should be a default gate.

### Interaction fingerprints

- **ProLIF** — what the course uses; still excellent. Keep.
- **PLIP** — alternative; readable text/CSV output, good for reports.

### Featurisation for ML scoring

- ProLIF IFPs (course default).
- Docking-score features (per-component PLANTS/Vina/gnina scores).
- **AEV-PLIG / SchNet / IGN** — graph/voxel learned representations if we want a learned rescorer.
- Boltz-2 affinity score itself can be used as a feature.

### ML rescoring

- **scikit-learn** RF / GBM — what the course teaches; perfectly adequate for the dataset sizes likely here.
- **XGBoost / LightGBM** — usually wins tabular benchmarks; trivial drop-in.
- **PyTorch** for active-learning / Bayesian rescoring (practical 03 from ULLA-Leiden).
- **chemprop** — message-passing NN, well-supported, good for property prediction (not pose scoring).

### ADMET / drug-likeness

- **ADMETlab 3** — web API, broad coverage, free for academic use.
- **SwissADME** — web tool, easy.
- RDKit built-ins: Lipinski, QED, SA-score, PAINS filters. No external dependency.
- **chemprop**-based custom property models if we have data.

### Benchmarks / reference data

- **Therapeutics Data Commons (TDC)** — single import for ChEMBL/BindingDB/MoleculeNet/DAVIS/etc.
- **PDBbind**, **CASF-2016**, **DUD-E** — for evaluating scorers; standard.

### Pre-built pipelines we could adopt instead of reinventing

- **TeachOpenCADD** (Volkamer lab) — pedagogical, ~25 "talktorials" covering exactly the steps above. The Leiden course's PCM practical (`02_PCM/talktorial.ipynb`) **is** TeachOpenCADD T032. *This is the closest thing to an open standard for a teaching/research SBVS pipeline. Strong recommendation to align with its conventions and reuse its modules where they fit.*
- **ASAP-discovery** (Choderalab + PostEra, ex-COVID Moonshot) — production-grade open-source SBVS framework, batteries-included, opinionated. Heavier; more "framework" than "library".
- **Practical Cheminformatics** (Pat Walters) — recipe blog + notebooks; not a framework but a goldmine of working snippets.
- **OpenFE / openmm-tools** — for downstream free-energy refinement if we get serious about top candidates.

## 3. Three architectural options

### Option A — "Lift from course, modern docker"

Stay closest to what the user already understands. Lift IFP + scoring code from the course notebooks into `src/aidd/`, swap PLANTS for gnina, swap the pre-baked AF model for ColabFold. ML scorer is the Tuesday-challenge / Week-6 pattern with sklearn/XGBoost.

- **Pros:** Smallest knowledge gap. Code reads like the course. Easy to debug because we wrote every line.
- **Cons:** Reinvents tooling that TeachOpenCADD already maintains. No pose QC, no ADMET, no consensus.

### Option B — "Adopt TeachOpenCADD as the foundation"

Treat the new repo as a TeachOpenCADD-compatible project. Use its talktorial modules where they exist (T010 for protein-ligand interactions, T015 for docking, T022 for ML, T032 for PCM). Wrap them in `notebooks/` for our own targets and add only what's missing (pose QC, ADMET gating, batch driver).

- **Pros:** Standing on the shoulders of an actively-maintained, peer-reviewed open-source project. Less code we own; less code we have to keep alive. Future-you can read the talktorials when memory fades.
- **Cons:** Some style/dependency drift between TeachOpenCADD and our own code. Their docker of record is smina; if we want gnina/DiffDock we wrap their interface.

### Option C — "Co-folding-first" (Boltz-2 as the spine)

Use Boltz-2 as the primary engine: input is protein sequence + ligand SMILES, output is co-folded complex + affinity. Use traditional docking (gnina) only as a second opinion for top candidates. ML rescoring is largely subsumed by the Boltz-2 affinity head; we keep an IFP + sklearn pipeline for explainability and as a sanity check.

- **Pros:** Closest to state-of-the-art. Drastically fewer moving parts (one model replaces fold + dock + score). Built-in affinity uncertainty.
- **Cons:** Boltz-2 is GPU-heavy (24 GB+ recommended) → essentially Colab/A100-only. Less interpretable per pose than IFP+score. If Boltz-2 is wrong, we have nothing else by default.

### Recommendation

**Hybrid B + C.** Adopt TeachOpenCADD-style modular notebooks as the structural backbone (Option B), and add a Boltz-2 notebook as the headline "fast triage" path (Option C). Use gnina + ProLIF + sklearn/XGBoost as the **interpretable** path that always runs, with PoseBusters as a default gate and RDKit-based ADMET filters as a pre-step. Course code is mined only for the few places where TeachOpenCADD doesn't have a direct equivalent (e.g. the specific PLANTS-feature engineering in the Tuesday challenges — but even there, we'd replace PLANTS-specific columns with gnina equivalents).

Concretely, the spine of the new pipeline becomes:

```
SMILES library
   │
   ├──► RDKit ADMET / PAINS / SA-score gate         (RDKit built-ins)
   │
   ▼
Standardise + 3D embed (DataMol/RDKit)
   │
   ├────────────────────────────────────────────────┐
   │                                                │
   ▼                                                ▼
Fast lane: Boltz-2 co-folding + affinity      Interpretable lane:
   (Colab / GPU)                                ColabFold once for the target
   │                                            │
   │                                            ▼
   │                                          gnina docking (+ PoseBusters QC)
   │                                            │
   │                                            ▼
   │                                          ProLIF IFPs + per-pose features
   │                                            │
   │                                            ▼
   │                                          sklearn/XGBoost rescorer
   │                                            (trained on per-target data
   │                                             with the course's ML pattern)
   │                                            │
   ▼                                            ▼
   ┌────────────────────────────────────────────────┐
   │  Consensus rank (Boltz-2 + gnina+IFP scorer)   │
   │  + uncertainty + interpretable per-pose report │
   └────────────────────────────────────────────────┘
                       │
                       ▼
            shortlist.sdf / shortlist.csv
            with reasons (poses, IFPs, scores, flags)
```

This gives the wet-lab scientist two independent signals to agree on, a physical pose to look at, an explanation of *why* each compound is on the list, and a confidence band.

## 4. Course-material mapping under the recommended stack

Updated reuse view — what each KEEP/REFERENCE item buys us if we adopt the hybrid:

| Course asset | Status under hybrid stack |
|---|---|
| `Week_3_Monday_Docking_and_Scoring.ipynb` (IFP + py3Dmol viewer) | **Reuse** the prolif + py3Dmol cells. These are unchanged regardless of docker. |
| `Week_3_Tuesday_Challenge_{1,2}.ipynb` (ML scoring pattern) | **Reuse the pattern**; rewrite feature columns to be docker-agnostic (gnina output instead of PLANTS). |
| `Week_6_SB-regression-extended.ipynb` (CDK2 regression) | **Reuse the structure** of the regression pipeline; same column-name caveat. |
| `descriptors_qsar_lab.ipynb` | **Subsume** into the RDKit/DataMol prep step. |
| `02_PCM/talktorial.ipynb` | **Use as-is** from upstream TeachOpenCADD when/if we want PCM. |
| `03_uncertainty_active_learning/` | **Reuse `utils.py`**; the AL loop becomes the "triage round 2" notebook later. |
| `biopython.ipynb` (distograms, structure parsing) | **Reuse the distogram code** for QC of folded models (compare ColabFold vs Boltz to detect bad folds). |
| `Supervised_ML_lab`, `01_plotting_unsupervised`, Pandas/NumPy intros, CNN labs | **Drop** from the active codebase; keep in `_archive/` for personal reference. |
| `docking/PLANTS` binary, `*.tgz` archives, duplicates | **Drop**. |

## 5. What this changes in the proposal

- The DiffDock-vs-Vina question is recast: **gnina** is the classical docker; **Boltz-2** is the co-folding fast lane; DiffDock becomes optional (cryptic-site exploration only).
- The "lift code from course notebooks" step becomes smaller, because TeachOpenCADD already supplies most of it.
- Environment manager: still **conda/mamba** as primary — gnina, openbabel, prolif, openff, boltz-2 all conda-installable; pip-only would be a fight.
- New mandatory components: **PoseBusters** (pose QC), **RDKit-based ADMET gate** (drug-likeness), **consensus ranker** (Boltz-2 ⊕ gnina+IFP).
- The course pre-computed CSVs (`*_features.csv.gz`, `*_ifp.csv.gz`) are **PLANTS-specific** and won't drop cleanly into a gnina-based pipeline. They stay in `_archive/` as a sanity-check dataset, not as a starting point.

## 6. Risks / things I'd want to test early

1. **Does Boltz-2 actually generalise to *our* target?** Test on ERK2 first (we have a known holo-PDB and labelled compounds), measure top-1% enrichment vs. course's PLANTS+RF baseline. If Boltz-2 doesn't beat it, demote it from the "fast lane" headline to an experimental notebook.
2. **gnina on Windows.** gnina ships clean Linux/macOS binaries; Windows users typically go via WSL or conda. We need to verify the conda-forge package works natively for you, or commit to "Windows local runs use WSL2".
3. **Colab quotas.** Boltz-2 inference + ColabFold MSAs both burn GPU time. For 10⁴-compound screens, you'll want either Colab Pro or a Lambda/RunPod instance. Plan for this.
4. **PLIP/IFP determinism across versions.** Course uses prolif==1.1.0 with custom HB angle redefinitions. The redefinitions are non-default and would surprise a reader; consider whether to keep or drop them.
5. **No wet-lab activity data for the real target yet?** Then the ML rescorer can't be trained per-target. We'd fall back to a generic learned scorer (Boltz-2's, gnina's CNN) plus IFP heuristics until experimental data flows in. This is a fine starting point but worth being explicit about.

## 7. What I'd like to know next

Three answers will let me lock the architecture and stop debating:

1. **GPU access.** Is there a non-Colab GPU available (institutional cluster, paid service, local workstation)? If yes, Boltz-2 / DiffDock become local; if no, the fast lane lives entirely on Colab and we plan accordingly.
2. **Throughput target for v1.** Are we screening ~1k compounds, ~10k, or ~100k+? This decides whether the pipeline must be batch-parallel from day one or can stay notebook-serial.
3. **Decision speed vs. rigor.** For wet-lab triage of the *first* batch, do you want a 1-day "pick 20 candidates from any plausible signal" answer, or a 1-week "pick 20 candidates we can defend with two independent in-silico methods" answer? Both are valid; they imply different default behaviour of the consensus ranker.
