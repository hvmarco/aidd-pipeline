# Notebooks

The user-facing entry points for the aidd-pipeline. Each numbered notebook covers **one pipeline stage** and is independently runnable, given the previous stage's cached outputs on disk under `data/derived/<target>/<stage>/`.

## Two files per notebook — read this first

Every notebook in this folder has **two paired files**:

| File | Purpose |
|---|---|
| `_build_<name>.py` | **Source of truth.** Contains every cell as a Python string. This is what Claude / contributors edit. |
| `<name>.ipynb` | **Generated artefact.** Produced by running the builder. This is what *you* run in Jupyter / VS Code / Colab. Cell outputs accumulate here when you run the notebook and are tracked in git so reviewers see results. |

To regenerate the notebook from the source:

```bash
python notebooks/_build_<name>.py
```

**Do not edit the `.ipynb` directly** — any change there gets overwritten on the next regen. If you spot something you want to tweak, change `_build_<name>.py` instead (and re-run the builder), or tell Claude. The full rule set is in [`../CLAUDE.md`](../CLAUDE.md) § *Notebook workflow*.

The builders share three small helpers from [`_nb_helpers.py`](_nb_helpers.py): `markdown(...)`, `code(...)`, and `notebook(...)`. Cells are emitted in the order the builder calls them — read a builder top-to-bottom and you read the notebook top-to-bottom.

## How to use this folder

**First time / learning mode** — open the notebooks in numerical order (`00` → `06`) and run them cell by cell. Each notebook has markdown blocks explaining both the biology and the technical choices, with a learning-objectives header and a recap at the end. Designed for clinicians, ML/data folks new to structural biology, and Bachelor / Master students.

**Routine screening mode / audit-ready report** — once the pipeline is dialled in for a target, `99_screen_library.ipynb` runs the whole flow end-to-end *and* is the artefact reviewers / professors / funders will read. It still teaches — just tighter than 00–06: short background per stage, methods with DOIs, a "Methods summary" cell near the top, and an executive-summary recap. Reviewers drill into 00–06 for detail.

**Mixed use** — re-run only the notebook that changes. For example, if you swap in a new candidate library, you re-run `02_prepare_ligands` and everything from `03_dock_gnina` onward; folding (`01`) and the rescorer (`04`, trained per target) stay cached.

## Pipeline overview

```
sequence / PDB ──► 01_fold_target ──┐                                            ┌──► 07_mutation_analysis (WT vs mutant diff)
(WT or mutant)                      │                                            │
                                    ├──► 03_dock_gnina ──► 04_score_classical ──┤
SMILES library ──► 02_prepare ──────┤                                            ├──► 06_consensus ──► shortlist.sdf
                                    └──► 05_dock_boltz ───────────────────────────┘
```

For **mutation studies** (e.g. drug-resistance screens), run notebooks 01–06 once for the wild-type and once per mutant variant. Notebook 07 then consumes the two `data/derived/<target>[_<variant>]/` trees and produces the side-by-side comparison: Cα-RMSD, IFP diff, side-by-side pocket viewer, and a diff of the two final shortlists.

## The notebooks

| #   | Builder                                  | Notebook                            | Stage                                        | Where it runs        | Status |
|-----|------------------------------------------|-------------------------------------|----------------------------------------------|----------------------|--------|
| 00  | `_build_00_quickstart.py`                | `00_quickstart.ipynb`               | IFP demo on ERK2 (sanity check)              | local CPU / Colab    | ✅ done |
| 01  | `_build_01_fold_target.py`               | `01_fold_target.ipynb`              | ColabFold target structure prediction         | **Colab GPU**        | planned |
| 02  | `_build_02_prepare_ligands.py`           | `02_prepare_ligands.ipynb`          | SMILES → standardised → drug-like → 3-D       | local CPU / Colab    | ✅ done |
| 03  | `_build_03_dock_gnina.py`                | `03_dock_gnina.ipynb`               | gnina docking + PoseBusters QC                | **Colab** (gnina is Linux-only) | ✅ done |
| 04  | `_build_04_score_classical.py`           | `04_score_classical.ipynb`          | IFP + ML rescorer (sklearn / XGBoost)         | **Colab** (gnina cache build), then any CPU | ✅ done (scaffold AUC: RF 0.66, XGB 0.62 vs baseline 0.48 on ERK2, 2026-05-12) |
| 05  | `_build_05_dock_boltz.py`                | `05_dock_boltz.ipynb`               | Boltz-2 co-folding + affinity (fast lane)     | **Colab GPU**        | ✅ done (scaffold AUC: Boltz-2 affinity 0.65, affinity_probability 0.70 vs baseline 0.48; Spearman vs gnina +0.27 on ERK2 413-cpd labelled subset; 412 predicted + 1 boltz_input_invalid; A100 ~124 s/compound, 2026-05-13) |
| 06  | `_build_06_consensus_and_shortlist.py`   | `06_consensus_and_shortlist.ipynb`  | consensus rank → `shortlist.sdf`              | local CPU / Colab    | planned |
| 07  | `_build_07_mutation_analysis.py`         | `07_mutation_analysis.ipynb`        | diff WT vs mutant runs (structure + IFP + shortlist) | local CPU / Colab    | planned (after 06) |
| 99  | `_build_99_screen_library.py`            | `99_screen_library.ipynb`           | end-to-end runner (accepts `mutations=` list) | Colab Pro+ recommended | planned (after 07) |

Every notebook works in Colab (the setup cell handles installs + repo clone). Colab is **mandatory** for `01` (ColabFold), `03` (gnina is Linux-native — Windows / macOS users go via Colab or WSL2), and `05` (Boltz-2). Notebook `04` is **Colab-only for its first run on a target** (it builds the labelled-subset docking cache via gnina), then runs anywhere on CPU once the cache is on Drive. The remaining CPU-only notebooks (`00`, `02`, `06`, `07`) work on Windows / macOS locally.

## Inputs / outputs at each stage

The contract between notebooks is simple: each stage reads from and writes to `data/derived/<target>/<stage>/`. The exact paths each notebook expects are stated in its first markdown cell.

| Stage              | Reads                                    | Writes                                                                |
|--------------------|------------------------------------------|------------------------------------------------------------------------|
| 00 quickstart      | `data/structures/*.pdb`, `data/ligands/*.pdb` | nothing persistent (demo)                                              |
| 01 fold_target     | target sequence (FASTA) — WT or mutant  | `data/derived/<target>[_<variant>]/fold/{pred.pdb, msa/, ranking.csv}` |
| 02 prepare_ligands | `data/compounds/<target>/*.smi`          | `data/derived/<target>/ligands/ligands_prepared.sdf`                   |
| 03 dock_gnina      | prepared SDF + receptor PDB              | `data/derived/<target>[_<variant>]/docking/{poses.sdf, gnina_scores.csv}` |
| 04 score_classical | poses SDF + receptor + activity labels   | `data/derived/<target>[_<variant>]/scoring/{rescorer.pkl, scored_poses.parquet}` |
| 05 dock_boltz      | prepared SDF + target sequence/structure | `data/derived/<target>[_<variant>]/boltz/{poses.sdf, affinity.csv}`    |
| 06 consensus       | scored_poses.parquet + affinity.csv      | `data/derived/<target>[_<variant>]/shortlist/{shortlist.sdf, shortlist.csv}` |
| 07 mutation_analysis | two completed `data/derived/<target>[_<variant>]/` trees | `data/derived/<target>/mutations/{wt_vs_<variant>.html, ...}` |
| 99 runner          | a target + a SMILES library + optional `mutations=` list | the full chain above (and notebook 07 per variant) |

## Notebook conventions

- Every notebook starts with a title cell containing **learning objectives**, **audience**, **prerequisites**, and **runtime**.
- A "key terms" table near the top defines all jargon used later.
- Each section follows: *Background* → "*what this cell does*" → code → *Interpretation*.
- Recap cell at the end with biomedical + technical takeaways and 2–3 further-reading paper DOIs.
- All notebooks include `%load_ext autoreload` so edits to `src/aidd/` propagate without kernel restarts.

See `../CLAUDE.md` § *Notebook pedagogy* for the full rule set.
