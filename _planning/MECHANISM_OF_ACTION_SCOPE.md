# Mechanism-of-action / small-N / variant-toxicity scope

**Status:** scope firm 2026-05-13; renumbered + variant-effect-prediction integration 2026-05-14. Ready for dev-chat handover.

## Handover summary

The 60-second read for the dev chat. Detail below.

- **New use case:** single-compound or small-N (N ≤ 10) mechanism-of-action and pharmacogenomic-variant studies. Orthogonal to the existing library-screening flow.
- **Implementation:** new `notebook 09` (`09_moa_small_n.ipynb`) as the teaching/clinical-Q&A driver, **plus** notebook 99 absorbs the MoA flow as a toggleable mode (`RUN_MODE = "library" | "moa"`). Both build on top of new **notebook 07** (`07_variant_effect_prediction`) which provides per-variant computational priors (AlphaMissense pathogenicity + RaSP ΔΔG stability + gnomAD allele frequency); these are consumed by notebooks 08 (mutation analysis, library-side) and 09 (MoA, small-N).
- **Five headline demos** (target-fishing deferred; see "Future work"). All CRC-relevant with breast-cancer cross-over via tamoxifen. NAT2 and CYP2D6 are Natallia's actual research focus and lead the set:
  1. NAT2 \*5/\*6/\*7 + isoniazid *(deep walkthrough)*
  2. CYP2D6 \*4/\*10 + tamoxifen
  3. DPYD\*2A + 5-FU *(brief follow-up; deep walkthrough on library side in notebook 08)*
  4. Irinotecan + UGT1A1\*28
  5. Sotorasib + KRAS G12C
- **Output:** summary CSV + one HTML report per (compound × variant) pair (canonical handoff artifact). Each report shows binding-pose + IFP diff + variant priors (AlphaMissense + RaSP ΔΔG + gnomAD). No auto-interpretation, no opinionated assay suggestions.
- **Honesty:** the pipeline measures binding fit, not catalytic rate. RaSP ΔΔG (from nb 07) addresses the protein-stability dimension that drives many slow-acetylator phenotypes — meaningfully strengthens the slow/rapid story without claiming to predict k\_cat. Caveat in three places (title cell, around each enzyme demo, bottom of every HTML report).
- **New input helpers:** `fetch_pdb`, `prep_receptor`, `name_to_smiles` — additive, in a new `src/aidd/inputs.py`.

## The new use case

Beyond the original 1k–10k compound virtual-screening flow, colleagues asked whether the same pipeline can answer **small-N mechanism-of-action (MoA) questions** for one or a few compounds at a time. Two flavours of question are in v1 scope; a third is deferred to future work.

1. **"Does drug X bind protein Y?"** — a known or uncharacterized cancer drug against a candidate target. Hypothesis-generation when the MoA is partly or fully unknown.
2. **"Is binding altered by inherited variation Z?"** — pharmacogenomic question. e.g., NAT2 slow-acetylator variants (NAT2\*5, \*6, \*7); CYP2D6 poor-metaboliser variants (\*4, \*10); DPYD\*2A; UGT1A1\*28; KRAS G12C.
3. **"What's the target of compound K?"** — polypharmacology / target-fishing flavour. **Deferred to future work.** The loop direction inverts (1 compound × N targets, vs the v1 pipeline's N compounds × 1 target) and needs a target-fishing adapter not worth bundling into v1. See "Future work" below.

These are clinically motivated questions Natallia hears in oncology / cancer-genetics discussions and that the existing pipeline can address with minor adjustments.

## Which scores transfer to small-N

| Score | Absolute meaning | Useful for small-N? |
|---|---|---|
| Boltz-2 affinity (log10(IC50) µM; **lower is stronger** per feedback memory) | Yes — Boltz-2 paper validates absolute predictions | ✅ Primary signal |
| Boltz-2 binder-probability (0–1) | Yes — calibrated probabilistically | ✅ "binds at all" question |
| gnina CNN-affinity (loose pKd) | Partly — r~0.5–0.6 vs experiment | ✅ Use with PoseBusters QC |
| PoseBusters pass rate | Binary geometry sanity | ✅ Critical gate |
| AlphaMissense pathogenicity (nb 07) | Sequence/evolution-based variant prior | ✅ Variant-context column |
| RaSP ΔΔG (nb 07) | Structure-based stability prior | ✅ Variant-context column; **primary signal for slow/rapid mechanism in metabolism-enzyme demos** |
| gnomAD allele frequency (nb 07) | Population context (overall + per-ancestry) | ✅ Variant-context column |
| IFP-rescorer probability (RF/XGB) | Only for the target it was trained on (ERK2) | ❌ Skip for non-ERK2; retrain per target if needed |
| Consensus rank (shortlist) | Rank-only, requires a library | ❌ Meaningless for N≤10 |

**Rule of thumb for small-N reports:** lead with Boltz-2 affinity + binder-probability + PoseBusters; AlphaMissense + RaSP + gnomAD (nb 07) give the variant-context priors; gnina + IFP-rescorer come third as triangulation, not headline.

## Why WT-vs-variant is the best fit

The delta between WT and mutant is more robust than any single absolute score: systematic over- or under-estimation by Boltz-2 / gnina cancels when you subtract. Notebook 08 (mutation analysis) is planned for exactly this: Cα-RMSD between structures, IFP diff (which interactions changed), affinity Δ, ΔΔG. Notebook 09 reuses 08's comparison helpers at single-compound scope.

For the four enzyme demos (NAT2, CYP2D6, DPYD, UGT1A1) the clinically meaningful question often involves *catalysis* (acetylation, oxidation, reduction, glucuronidation rates), not just *binding*. Our pipeline measures whether the substrate fits the active site (a necessary condition for catalysis), not the kinetic rate of the reaction itself. **Notebook 07's RaSP ΔΔG addresses the protein-stability dimension** — for many slow-acetylator and reduced-function alleles, stability change is the dominant mechanism behind the kinetic phenotype, and ΔΔG captures it directly. See "Honesty" below for the explicit framing.

## Headline demos

Five demos, all CRC-relevant (with breast-cancer cross-over via tamoxifen). NAT2 and CYP2D6 lead because they are Natallia's actual research focus.

| # | Demo | MoA Q-shape | CRC angle |
|---|---|---|---|
| 1 | **NAT2 \*5/\*6/\*7 + isoniazid** *(deep walkthrough)* | variant alters substrate binding | Slow acetylators → reduced carcinogen detoxification → higher CRC risk |
| 2 | **CYP2D6 \*4/\*10 + tamoxifen** | variant alters substrate binding | Tamoxifen-activation pharmacogene; breast cancer cross-over |
| 3 | **DPYD\*2A + 5-FU** | variant alters substrate binding | 5-FU is backbone CRC chemo |
| 4 | **Irinotecan + UGT1A1\*28** | variant alters substrate binding | Irinotecan is CRC chemo; FDA-label genotype-guided dose |
| 5 | **Sotorasib + KRAS G12C** | drug binds target (covalent) | KRAS G12C in metastatic CRC |

**Cohesion with notebook 08.** All five demos are shared with notebook 08's library-side flow (see [`PROJECT_PROPOSAL.md`](PROJECT_PROPOSAL.md) §10). Notebook 08 carries two additional demos that stay library-side only:

- **ESR1 Y537S + tamoxifen** — breast-cancer endocrine-resistance hot-spot. Kept out of notebook 09 to preserve nb 09's focus on the pharmacogenomics / drug-metabolism enzymes (NAT2, CYP2D6, DPYD, UGT1A1) that anchor Natallia's CRC research.
- **BRCA1 LoF + olaparib** — synthetic lethality. Excluded from notebook 09 because olaparib binds PARP, not BRCA1 — the variant creates the vulnerability but doesn't itself alter drug binding, so the case doesn't fit nb 09's "variant alters binding" Q-shape.

**Substrate choice for NAT2: isoniazid.** It is the textbook NAT2 substrate — the molecule on which the slow-acetylator phenotype was originally characterised — so the demo carries the full clinical and pharmacological literature on NAT2 phenotype-genotype with it. Although isoniazid itself is a tuberculosis drug, the slow-acetylator framing transfers directly to the CRC-relevant question of carcinogen detoxification: the structural argument made on isoniazid generalises to the aromatic-amine class (PhIP, 4-aminobiphenyl, other dietary heterocyclic amines). PhIP and 4-aminobiphenyl remain plausible alternative or supplementary substrates if a second NAT2 example is wanted later.

## Notebook integration

### New notebook 09: pedagogical small-N driver

`09_moa_small_n.ipynb` slots between notebook 08 (mutation analysis) and notebook 99 (production runner). Pedagogical shape per project convention (title cell + key terms + per-section background → cell → interpretation + recap).

Input cell sketch:

```python
TARGET = {"pdb_id": "2PFR", "name": "NAT2"}
VARIANTS = ["WT", "R64Q"]                        # optional; WT-only is fine
COMPOUNDS = [
    ("isoniazid", None),                          # name only → fetch SMILES from PubChem
    ("research_compound_X", "Cc1ccc(...)..."),    # name + SMILES
]
```

Setup cell auto-fetches PDB + SMILES via the new `fetch_pdb` / `name_to_smiles` helpers, runs `prep_receptor`, applies the variant via PDBFixer mutate-in-place or notebook 01 fold. **Notebook 07** enriches each variant with AlphaMissense + RaSP ΔΔG + gnomAD context. Downstream cells call the same docking (notebook 03 logic), Boltz-2 (notebook 05 logic), and IFP (notebook 04 logic) functions. Output: per-compound × per-variant scores aggregated into a summary CSV + one HTML report per (compound × variant) pair.

Notebook 09 does NOT compete with notebooks 03 / 05 — it's a thin driver that calls the same `src/aidd/` functions with single-compound inputs and a different output format.

### Structure of notebook 09

The five demos are sections of the same notebook, not separate notebooks. One deep walkthrough (NAT2 + isoniazid) carries the full teaching markdown; the four brief sections share the pipeline calls but compress the teaching to "the same pattern applies here". This mirrors notebook 08's design in [`PROJECT_PROPOSAL.md`](PROJECT_PROPOSAL.md) §10 (one deep walkthrough + brief follow-ups).

```
┌── notebook 09 ─────────────────────────────────────────────────┐
│ Title cell + key terms + "what this notebook does NOT predict" │
│ Setup (imports, env detection, autoreload)                     │
├────────────────────────────────────────────────────────────────┤
│ DEEP WALKTHROUGH: NAT2 + isoniazid                             │
│   Background: NAT2 biology, slow-acetylator phenomenology,     │
│               CRC carcinogen-detoxification angle              │
│   Input cell:  TARGET=NAT2, VARIANTS=[WT,*5,*6,*7],            │
│                COMPOUNDS=[isoniazid]                           │
│   Run:         fetch_pdb → prep_receptor → variant handling    │
│                → variant priors (nb 07) → dock + Boltz-2 + IFP │
│   Output:      4 per-compound HTML reports (one per variant)   │
│                + 4 rows in summary CSV                         │
│   Interpretation: IFP diff, ΔΔG, binding-vs-catalysis caveat   │
├────────────────────────────────────────────────────────────────┤
│ BRIEF: CYP2D6 + tamoxifen                                      │
│   One-paragraph background, condensed input, run, interpret    │
├────────────────────────────────────────────────────────────────┤
│ BRIEF: DPYD*2A + 5-FU         (deep walkthrough lives in nb 08)│
│ BRIEF: Irinotecan + UGT1A1*28 (UGT1A1-no-crystal branch)       │
│ BRIEF: Sotorasib + KRAS G12C  (different Q-shape)              │
├────────────────────────────────────────────────────────────────┤
│ Summary CSV table: 1 row per (compound × variant)              │
│ Recap: takeaways + further reading on catalysis prediction     │
└────────────────────────────────────────────────────────────────┘
```

Each demo carries the reader through input → fetch PDB → prep receptor → apply variant → variant priors (nb 07) → dock + co-fold + IFP → per-compound HTML report. The pipeline stages are the same as notebook 99's MoA mode; the teaching density differs (deep vs brief).

### Notebook 99: second mode toggle

Notebook 99 (production runner) gets a `RUN_MODE` toggle alongside the existing `FOLD_PROVIDER`:

```python
FOLD_PROVIDER = "colabfold"   # existing (AF2 default; AF3 owner-only)
RUN_MODE      = "library"     # NEW: "library" (1k-10k screen) or "moa" (small-N MoA)
```

In `moa` mode:

- Full-library ligand prep + per-stage cache walk is skipped; inputs come from `name_to_smiles` + a hand-curated PDB ID via `fetch_pdb`.
- Notebook 07 variant priors are computed once per variant.
- The consensus-rank cell (06) is skipped — meaningless at N ≤ 10.
- Output switches from `shortlist.sdf` (rank-ordered library) to per-compound HTML reports + a summary CSV.
- Notebook 08 logic is invoked when variants are provided.

Two parallel toggles in notebook 99 (`FOLD_PROVIDER`, `RUN_MODE`) is manageable. A third toggle would be the signal that 99 needs a small `config` cell rather than scattered top-level constants.

## Implementation sequencing

Notebooks 07, 08, 09, and 99 are built in sequence — **not** together. The sequence:

1. **Done (2026-05-14)** — notebooks 00, 02, 03, 04, 05, 06. Step 12 of [`PROJECT_PROPOSAL.md`](PROJECT_PROPOSAL.md) §7 closed.
2. **Next** — step 13 second prune pass (small cleanup).
3. **Then** — `src/aidd/variants.py` + `src/aidd/stability.py` modules, then **notebook 07** (`07_variant_effect_prediction`): AlphaMissense lookup + RaSP ΔΔG + gnomAD frequency, per-variant context report. **This was previously in `PROJECT_PROPOSAL.md` §11 post-13-step (Priorities 1, 1.5, 2); promoted into the main timeline (2026-05-14) to support the slow/rapid mechanism story in nb 08 and nb 09.**
4. **Then** — **notebook 08** (`08_mutation_analysis`) — WT-vs-mutant pipeline diff (Cα-RMSD, IFP diff, shortlist diff). Library-side variant comparison; consumes nb 07's variant priors as added columns. (Was notebook 07 before the 2026-05-14 renumber.)
5. **Then** — **notebook 09** (`09_moa_small_n`, this doc) ships the small-N MoA driver plus `src/aidd/inputs.py` (and any small-N orchestration helpers).
6. **Last** — **notebook 99** is built in one round with both library mode and `RUN_MODE = "moa"`. It re-uses the `src/aidd/` functions notebooks 07–09 already shipped; no logic duplication.

Per the project rule *"notebooks stay thin, logic lives in `src/aidd/`"*, the MoA-mode behaviour (PDB fetch, SMILES lookup, receptor prep, per-compound HTML generation, single-compound WT-vs-variant diff, variant-prior lookup) lives in `src/aidd/` from day one. Notebook 09 is the pedagogical wrapper; notebook 99 in `moa` mode is the audit-shape wrapper. Both call the same functions — the nb-99 dev chat does not re-design anything from this scope, only wires the existing `src/aidd/` functions into 99's `RUN_MODE = "moa"` branch.

## Output format

Notebook 09 (and notebook 99 in `moa` mode) produces two artifacts.

### Summary comparison table

A pandas DataFrame rendered in the notebook recap, also saved as `data/derived/<target>[_<variant>]/moa_reports/summary.csv`. Columns: compound name, SMILES, variant, Boltz-2 affinity (with sign noted per `feedback_boltz_affinity_sign.md`), Boltz-2 binder probability, gnina CNN-affinity, PoseBusters pass, **AlphaMissense pathogenicity (nb 07)**, **RaSP ΔΔG (nb 07)**, **gnomAD allele frequency (nb 07)**, IFP-rescorer score (only when the rescorer was trained on this target).

At-a-glance overview for methods-paper reads and audit.

### Per-compound HTML report (canonical handover artifact)

One HTML file per (compound × variant) pair, saved to `data/derived/<target>[_<variant>]/moa_reports/<compound>.html`. Sections:

1. **Header** — compound name + SMILES + target + variant.
2. **3D pose viewer** — py3Dmol embedded into the standalone HTML; receptor cartoon + ligand sticks + key residues highlighted.
3. **Scores table** — same columns as the summary, single-row, including variant priors from notebook 07 (AlphaMissense + RaSP ΔΔG + gnomAD).
4. **Key interactions** — ProLIF residue + interaction-type list (H-bond donor/acceptor, hydrophobic, π-stacking, etc.).
5. **WT vs variant diff** (only when N\_variants > 1) — interactions gained/lost, Cα-RMSD around the binding site, affinity delta, ΔΔG (from nb 07). Notebook 08 logic invoked at single-compound scope.
6. **Bottom-line caveat** (always): *"Computational hypothesis-generation; experimental validation required before any clinical inference."*

Explicitly **NOT** included: auto-generated clinical-interpretation paragraphs, opinionated assay suggestions, per-target next-steps language. The HTML shows numbers + 3D pose + IFP; the reader interprets.

## Honesty: what the pipeline does and does NOT predict

For the enzyme demos (NAT2, CYP2D6, DPYD, UGT1A1 — four of five), the pipeline measures **whether the substrate fits the active site** (a necessary condition for catalysis), not the **catalytic rate** (k\_cat, K\_m, turnover).

For variant-vs-WT comparisons on enzymes:

- The pipeline can answer: *"Does the substrate still fit in the variant active site, and how does binding-site geometry / IFP differ from WT? Is the variant predicted to destabilise the fold (RaSP ΔΔG)? Is the variant evolutionarily flagged (AlphaMissense)? Is it common in the population (gnomAD)?"*
- The pipeline cannot answer: *"How much slower is catalysis in the variant?"*
- The pipeline definitely cannot answer: *"Will this patient have toxicity from drug X?"*

**Notebook 07's RaSP ΔΔG meaningfully strengthens the slow/rapid story for enzyme variants** — for many slow-acetylator and reduced-function alleles, protein-stability change (and consequent reduced cellular abundance) is the dominant mechanism behind the kinetic phenotype, and ΔΔG captures it directly. Combined with the binding-pose / IFP-diff signal from notebooks 08 and 09 and the AlphaMissense pathogenicity prior, the structural mechanism story is much more complete than binding-pose alone. Still circumstantial for catalytic rate — k\_cat prediction needs QM/MM (out of scope; see Future work).

Caveat language lives in three places in notebook 09:

1. **Title cell** — a "What this notebook does and does NOT predict" block before the learning objectives. One paragraph, plain language.
2. **Markdown around each enzyme demo** — one-sentence reminder where the affinity number is shown: *"For enzymes, affinity here reflects substrate-binding fit, not catalytic rate. RaSP ΔΔG (above) captures the stability dimension; together they triangulate the slow/rapid mechanism without proving catalytic rate."*
3. **Bottom of every HTML report** (see Output format) — the generic one-line caveat.

The recap includes a **"Further reading"** pointer to QM/MM and EnzyHTP-style catalysis-prediction methods as out-of-scope future work. Two DOIs is enough — it just signals to a reviewer that the gap is known.

## Wet-lab handoff

The per-compound HTML report **is** the handoff. There is no separate validation ceremony. The bottom-line caveat (see Output format) plus a reader who knows their next step is to talk to the wet-lab team.

## Input-helper contract (for the dev chat to implement)

Three new helpers in a new `src/aidd/inputs.py` module — all additive, no existing module touched. (Variant-prior helpers — AlphaMissense / RaSP / gnomAD — live in `src/aidd/variants.py` + `src/aidd/stability.py`, shipped with notebook 07.)

### `fetch_pdb(pdb_id) -> Path` — PDB by ID with caching

Download from rcsb.org by PDB ID; cache under `data/structures/<pdb_id>.pdb`; idempotent.

Clean cases for v1 demos: 2PFR (NAT2), 1H7W (DPYD with 5-iodouracil), 6OIM (KRAS G12C + sotorasib). CYP2D6 has multiple crystals (3QM4 with thioridazine, 4WNW with prinomastat); dev chat picks one — prefer a co-crystal with a ligand we can swap rather than apo.

**Edge case: UGT1A1 has no human crystal structure.** For the UGT1A1\*28 demo, the dev chat handles this gracefully — branch to an AlphaFold model (notebook 01 flow) or a UGT2B7 homology model. `fetch_pdb` surfaces "no structure available" cleanly rather than failing silently.

### `prep_receptor(pdb_path) -> Path` — receptor cleanup

PDBFixer-based:

- Remove crystallographic waters (keep selected structurally-important ones if needed — usually drop all).
- Configurably remove co-crystal ligands and cofactors. **Some cofactors are structurally required** — NADPH for CYP2D6, NAD+ for DPYD, acetyl-CoA / CoA for NAT2, UDP-glucuronic acid for UGT1A1 — and must be retained.
- Pick alternative conformation A.
- Single-chain selection (default A; configurable for dimers).
- Add missing residues / sidechains.
- Optionally assign protonation states (Reduce / PROPKA).

PDBFixer is already in our env or one pip install away. Output: cleaned PDB at the same canonical path.

### `name_to_smiles(name) -> str` — PubChem REST

Pattern from sokrypton's Boltz notebook (already vetted in `src/aidd/co_folding.py`).

- `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON`.
- Free, no auth, handles brand + generic + synonym names.
- Returns parent compound (active species after salt dissociation).

Edge cases:

- Brand vs generic: handled (e.g., "Tylenol" → acetaminophen SMILES).
- Salt forms: PubChem returns parent; don't dock the salt.
- Stereochemistry: print the SMILES, verify chirality if clinically relevant.
- No match: fall back to ChEMBL or hand-curated. For established drugs the failure rate is near zero.

### Variant handling: mutate-in-place vs fold

For variants where no crystal exists:

- **Mutate-in-place** (PDBFixer single-residue point mutation on WT crystal): faster, preserves crystal-level binding-site detail. Appropriate when the mutation is distant from the active site or local geometry isn't perturbed.
- **Fold the variant** (notebook 01 with mutant sequence): more honest if the mutation might cause structural rearrangement, but loses crystal-level detail.

User picks per-case; surface the choice in the input cell.

## What this does NOT change

- The library-screening flow (notebooks 00–08 + 99 in `library` mode) stays the canonical use case.
- No existing `src/aidd/` modules need refactoring; the new `inputs.py` helpers and the new `variants.py` / `stability.py` modules (for nb 07) are additive. Existing `co_folding.py`, `docking.py`, `ifp.py`, etc. are reused unchanged.
- No notebook 05 / Boltz-2 changes — same model, same scores, just consumed by a different driver notebook in a different output format.

## Future work

- **Target-fishing demo** (Q3-shape: 1 compound × N targets). Needs a target-fishing adapter that inverts the existing loop direction. Candidate compound: niclosamide (Wnt/β-catenin repurposing in CRC) or auranofin (thioredoxin reductase, cleaner single-target). Candidate target list: ~5–7 CRC-relevant proteins including a negative control (e.g. carbonic anhydrase II) to calibrate the ranking.
- **Catalysis prediction for enzyme demos.** The pipeline measures binding fit (+ RaSP ΔΔG for stability); reaction-rate prediction needs QM/MM or EnzyHTP-style methods. Out of scope for the in-silico screening pipeline; cited in the recap.
- **NAT2 acetylator-phenotype calibration study.** Apply the combined nb 07 + nb 08 + nb 09 signals (AlphaMissense + RaSP ΔΔG + gnomAD + IFP-diff + binding-pose delta) to a literature-curated set of well-characterized NAT2 variants (slow + rapid + intermediate). Train/fit a multi-signal slow/rapid predictor; calibrate on held-out variants; apply to novel/rare alleles seen in clinic. Real-research extension of the demo set, methodology candidate for publication in a pharmacogenomics journal.
- **PharmGKB / CPIC clinical-ground-truth lookup** — still on post-13-step roadmap (PROJECT_PROPOSAL §11). Validator for the slow/rapid prediction story.
- **fpocket binding-site detection on variant folds** — still on post-13-step roadmap (PROJECT_PROPOSAL §11).
