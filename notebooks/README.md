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

**First time / learning mode** — open the notebooks in numerical order (`00` → `09`) and run them cell by cell. Each notebook has markdown blocks explaining both the biology and the technical choices, with a learning-objectives header and a recap at the end. Designed for clinicians, ML/data folks new to structural biology, and Bachelor / Master students. The first 7 (`00`–`06`) cover the library-screening flow; `07` covers per-variant computational priors; `08` covers WT-vs-mutant library comparison; `09` covers small-N mechanism-of-action reports.

**Routine screening mode / audit-ready report** — once the pipeline is dialled in for a target, `99_screen_library.ipynb` runs the whole flow end-to-end *and* is the artefact reviewers / professors / funders will read. It still teaches — just tighter than 00–06: short background per stage, methods with DOIs, a "Methods summary" cell near the top, and an executive-summary recap. Reviewers drill into 00–06 for detail.

**Mixed use** — re-run only the notebook that changes. For example, if you swap in a new candidate library, you re-run `02_prepare_ligands` and everything from `03_dock_gnina` onward; folding (`01`) and the rescorer (`04`, trained per target) stay cached.

## Pipeline overview

**Library-screening flow** (notebooks 00–06):

```
sequence / PDB ──► 01_fold_target ──┐
(WT or mutant)                      │
                                    ├──► 03_dock_gnina ──► 04_score_classical ──┐
SMILES library ──► 02_prepare ──────┤                                            ├──► 06_consensus ──► shortlist.sdf
                                    └──► 05_dock_boltz ────────────────────────┘
```

**Variant + MoA flow** (notebooks 07–09; build on top of the library flow):

```
variant                                 ┌──► 08_mutation_analysis (library-side WT vs mutant diff: structure + IFP + shortlist + ΔΔG)
(e.g. NAT2 *5,                          │
 DPYD*2A)    ──► 07_variant_effect ─────┤
                  prediction            │
                  (AlphaMissense        └──► 09_moa_small_n (small-N: per-compound × per-variant HTML reports)
                   + RaSP ΔΔG
                   + gnomAD)
```

For **mutation studies** (e.g. drug-resistance screens), run notebooks 01–06 once for the wild-type and once per mutant variant. Notebook 07 produces the per-variant computational priors (pathogenicity, stability, allele frequency). Notebook 08 then consumes the two `data/derived/<target>[_<variant>]/` trees plus the 07 priors and produces the side-by-side comparison: Cα-RMSD, IFP diff, side-by-side pocket viewer, ΔΔG, and a diff of the two final shortlists.

## The notebooks

| #   | Builder                                  | Notebook                            | Stage                                        | Where it runs        | Status |
|-----|------------------------------------------|-------------------------------------|----------------------------------------------|----------------------|--------|
| 00  | `_build_00_quickstart.py`                | `00_quickstart.ipynb`               | IFP demo on ERK2 (sanity check)              | local CPU / Colab    | done |
| 01  | `_build_01_fold_target.py`               | `01_fold_target.ipynb`              | ColabFold target structure prediction         | **Colab GPU**        | planned |
| 02  | `_build_02_prepare_ligands.py`           | `02_prepare_ligands.ipynb`          | SMILES → standardised → drug-like → 3-D       | local CPU / Colab    | done |
| 03  | `_build_03_dock_gnina.py`                | `03_dock_gnina.ipynb`               | gnina docking + PoseBusters QC                | **Colab** (gnina is Linux-only) | done |
| 04  | `_build_04_score_classical.py`           | `04_score_classical.ipynb`          | IFP + ML rescorer (sklearn / XGBoost)         | **Colab** (gnina cache build), then any CPU | done (scaffold OOF AUC: RF 0.58, XGB 0.58 vs baseline 0.48 on ERK2; per-fold range 0.40–0.69; single-split realization 0.66 superseded by 5-fold OOF aggregate, 2026-05-13) |
| 05  | `_build_05_dock_boltz.py`                | `05_dock_boltz.ipynb`               | Boltz-2 co-folding + affinity (fast lane)     | **Colab GPU**        | done (scaffold AUC: Boltz-2 affinity 0.65, affinity_probability 0.70 vs baseline 0.48; Spearman vs gnina +0.27 on ERK2 413-cpd labelled subset; 412 predicted + 1 boltz_input_invalid; A100 ~124 s/compound, 2026-05-13) |
| 06  | `_build_06_consensus_and_shortlist.py`   | `06_consensus_and_shortlist.ipynb`  | consensus rank → `shortlist.sdf`              | local CPU / Colab    | done (rank_product_topk @ TOP=0.10 on ERK2 412-cpd joined cohort: shortlist 42 compounds, 25/126 actives, 19.8% recall, 1.95x enrichment vs random; intersection rule empirically broken on this target — Spearman rho=0.08 p=0.11 between rescorer-OOF and Boltz lanes; per-target operating-point sweep + cross-method rho in section 9; 2026-05-14) |
| 07  | `_build_07_variant_effect_prediction.py` | `07_variant_effect_prediction.ipynb`| AlphaMissense + RaSP ΔΔG + gnomAD: per-variant computational priors | local CPU / Colab | done (4 calibration variants run end-to-end on Colab T4; AM+gnomAD work for all 4 with bidirectional matching resolving K268R reference-flip case (UniProt has K, Ensembl has R; flip=Y in CALIBRATION SUMMARY); RaSP covered only NAT2 of 7 demo genes in 414 MB experimental-structures cache (DPYD, CYP2D6, KRAS, BRCA1, ESR1, UGT1A1 all return None — wider gap than originally documented); full proteome coverage via 9 GB AlphaFold swap deferred to step 17 per PROJECT_PROPOSAL.md § 7 done-signal update; 2026-05-16) |
| 08  | `_build_08_mutation_analysis.py`         | `08_mutation_analysis.ipynb`        | diff WT vs mutant runs (structure + IFP + shortlist + ΔΔG) | local CPU / Colab | done (DPYD I560S stub-fixture walkthrough verified on Colab: priors AM=0.850/pathogenic + gnomAD AF=6.5e-4/found + RaSP=None (DPYD outside 414 MB experimental cache; step-17 AlphaFold-RaSP swap fills); structural diff Cα-RMSD pocket+full both 0.000 Å (stub-fixture coarse I→S surgery; PDBFixer-based mutation at nb 99); IFP diff gained=2/lost=8/preserved=54 (lost set captures I560 hydrophobic contacts the surgery removes); shortlist diff (Boltz-2 + gnina-CNN consensus, rank_product_topk @ TOP=0.30) both=3 / wt_only=1 (gimeracil) / mut_only=1 (diphenhydramine) — all three diff branches fire; HTML report 1.27 MB at `data/derived/dpyd/mutations/wt_vs_i560s.html`; six follow-up template demos resolve nb 07 priors for BRCA1 C61G + CYP2D6 P34S + NAT2 I114T, partial for somatic KRAS G12C + ESR1 Y537S (germline-absent in gnomAD), UGT1A1\*28 honest out-of-scope (TATA-box promoter, not missense); real DPYD WT+I560S pipeline runs land at nb 99 production mode (step 17); 2026-05-16) |
| 09  | `_build_09_moa_small_n.py`               | `09_moa_small_n.ipynb`              | small-N mechanism-of-action: per-compound HTML reports + summary CSV | local CPU / Colab | done (NAT2 \*5/\*6/\*7 + isoniazid deep walkthrough verified on Colab T4: AM=0.086/0.100/0.081 all benign-range — known AM limitation on common functional pharmacogene variants, cross-confirms nb 07 step-14 NAT2 I114T calibration finding; RaSP ΔΔG=+2.88/+1.56/+5.07 kcal/mol all destabilising (RaSP is the load-bearing prior for this variant class, slow-acetylator stability mechanism captured); pocket-Cα-RMSD=0.000 Å (PDBFixer applyMutations leaves backbone unchanged; sidechain-level IFP diff fires gained/lost = 11/12, 12/12, 6/12); 4 brief follow-ups same template: CYP2D6 \*4 splice + \*10 P34S (AM=0.757) + tamoxifen, DPYD\*2A + 5-FU (out_of_scope_splice), UGT1A1\*28 + irinotecan (out_of_scope_promoter), KRAS G12C + sotorasib (AM=0.998, IFP gained/lost = 2/1, opposite-direction signal for covalent inhibitor at variant site); 11 (compound × variant) reports + 1 summary CSV at `data/derived/moa_reports/`; real per-(compound, variant) Boltz-2 + gnina predictions land at nb 99 step 17; `aidd.inputs.assert_wt_residue` helper added to codify pre-commit PDB-identity check (caught 6OIM=variant co-crystal class of error after-the-fact in step 16; bound into nb 09 §3 stub-fixture cell); 2026-05-17) |
| 99  | `_build_99_screen_library.py`            | `99_screen_library.ipynb`           | end-to-end runner (`RUN_MODE` = `library` or `moa`; accepts `mutations=` list) | Colab Pro+ recommended | planned (after 09) |

Every notebook works in Colab (the setup cell handles installs + repo clone). Colab is **mandatory** for `01` (ColabFold), `03` (gnina is Linux-native — Windows / macOS users go via Colab or WSL2), and `05` (Boltz-2). Notebook `04` is **Colab-only for its first run on a target** (it builds the labelled-subset docking cache via gnina), then runs anywhere on CPU once the cache is on Drive. The remaining CPU-only notebooks (`00`, `02`, `06`, `07`, `08`, `09`) work on Windows / macOS locally.

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
| 07 variant_effect_prediction | variant identifier (UniProt accession + position + alt aa) | `data/derived/<target>[_<variant>]/variant_priors/{alphamissense.json, rasp_ddg.json, gnomad.json}` |
| 08 mutation_analysis | two completed `data/derived/<target>[_<variant>]/` trees + variant priors from 07 | `data/derived/<target>/mutations/{wt_vs_<variant>.html, ...}` |
| 09 moa_small_n     | a target + a small (N≤10) SMILES set + optional `VARIANTS=` list + variant priors from 07 | `data/derived/<target>[_<variant>]/moa_reports/{summary.csv, <compound>.html}` |
| 99 runner          | a target + a SMILES library (library mode) OR a small-N MoA input (moa mode) + optional `mutations=` list | the full chain above (notebook 08 per variant; notebook 09 logic in `moa` mode) |

## Notebook conventions

- Every notebook starts with a title cell containing **learning objectives**, **audience**, **prerequisites**, and **runtime**.
- A "key terms" table near the top defines all jargon used later.
- Each section follows: *Background* → "*what this cell does*" → code → *Interpretation*.
- Recap cell at the end with biomedical + technical takeaways and 2–3 further-reading paper DOIs.
- All notebooks include `%load_ext autoreload` so edits to `src/aidd/` propagate without kernel restarts.

See `../CLAUDE.md` § *Notebook pedagogy* for the full rule set.
