"""Builder for 06_consensus_and_shortlist.ipynb.

Source of truth for the consensus + shortlist notebook. Cells appear below
in narrative order. Never edit the .ipynb directly -- see ``CLAUDE.md``
section *Notebook workflow*.

Regenerate:
    python notebooks/_build_06_consensus_and_shortlist.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import AUTORELOAD_SNIPPET, code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "06_consensus_and_shortlist.ipynb"


def build() -> None:
    nb = notebook(
        markdown("""
# 06 - Consensus rank and shortlist

**aidd-pipeline - Notebook 6 of the screening workflow**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hvmarco/aidd-pipeline/blob/main/notebooks/06_consensus_and_shortlist.ipynb)

This notebook is where the two screening lanes converge. We take the per-compound rankings produced upstream:

- **Classical lane** (notebook `04_score_classical`): a per-target ML rescorer trained on gnina docking outputs + ProLIF interaction fingerprints, validated on held-out activity labels. Output: `scored_poses.parquet`.
- **Co-folding lane** (notebook `05_dock_boltz`): Boltz-2 co-folds the protein and the ligand together in one neural-network pass and predicts a binding affinity. Output: `affinity.csv`.

We join the two tables on compound identifier, keep compounds that sit in the **top 5%** of *both* lanes (the *consensus filter* locked in `_planning/PROJECT_PROPOSAL.md` section 1), and rank the survivors by **rank product** (the geometric mean of the two within-lane ranks). The result is `shortlist.sdf` and `shortlist.csv` -- the wet-lab handoff.

The shortlist is small by design. The two lanes use **different physics** and fail in **different ways**, so an intersection of their top-5%s removes most of the noise that any one method introduces on its own. The compounds that survive are the ones a chemist can defend in a meeting: each one is independently strong by both an interpretable trained model and a generative co-folding model, and the consensus rule is documented + reproducible.

## Learning objectives

After running this notebook you will be able to:

- Explain in plain language what **consensus virtual screening** is, why intersecting rankings from independent methods is more defensible than any single ranker, and where consensus actually does and does not help.
- Read a **rank-product** column and place it in context against z-score combination, Borda sum, and Pareto-front alternatives.
- Inspect a **coverage report**: how many compounds each lane scored, how big the intersection was, how many compounds got dropped because only one lane could score them.
- Read a **shortlist.csv** and a **shortlist.sdf** as a wet-lab chemist would: which compounds, in what order, with what evidence per compound.
- Adjust the `top_fraction` knob to make the shortlist tighter (more conservative) or looser (more recall, more triage work) and report the trade-off honestly.

## Audience

- A clinician or biologist who wants the one-page answer: "given the two-lane screen, which compounds should we order?"
- A medicinal chemist receiving the shortlist for triage: this notebook documents how every entry got there.
- An ML / data person learning the *evaluation* and *aggregation* end of virtual screening, not the modelling end.
- A reviewer / grant auditor: the methodology has citations and the rule is one short Python module (`src/aidd/consensus.py`) you can read in a few minutes.

## Prerequisites

- Notebook **`04_score_classical`** completed for this target. Specifically: `data/derived/<target>/scoring/scored_poses.parquet` exists and contains the cross-validated out-of-fold column `rescorer_rf_proba_oof` (shipped in commit `d84565c`). The notebook reads that column, **not** the legacy `rescorer_rf_proba`, which was a Random Forest trained on all rows including its own test fold and gave `AUC = 1.000` when filtered to held-out compounds; `compute_consensus()` raises `KeyError` if `rescorer_rf_proba_oof` is missing rather than silently falling back to the leaked column.
- Notebook **`05_dock_boltz`** completed for this target. Specifically: `data/derived/<target>/boltz/affinity.csv` exists.
- The gnina poses cache from notebook `04` (`data/derived/<target>/docking/labeled_subset/poses.sdf`) should be present too -- the shortlist SDF copies its 3-D coordinates from there. If it is missing, the notebook still produces `shortlist.csv` and a warning, just no `shortlist.sdf`.

## Runtime

- **CPU-only.** No GPU, no model loading, no MSA queries. The whole notebook runs in well under a minute on any laptop or Colab instance once the upstream caches are in place.
- Re-running with different `top_fraction` values (5%, 10%, 20%) is cheap, so feel free to scan a few values and pick the one that gives a shortlist size you can act on.
"""),

        markdown("""
## Key terms used in this notebook

| Term | Meaning in one line |
|---|---|
| **Consensus virtual screening (consensus VS)** | A ranking strategy that combines the outputs of two or more independent scoring methods so that the final ranking depends on agreement, not on any one tool. |
| **Rank product** | Geometric mean of two ranks: `sqrt(rank_a * rank_b)`. Lower = better. Scale-free across the two input lanes; the canonical aggregator in structural-biology consensus VS (Wang & Wang 2001 *J. Chem. Inf. Comput. Sci.* **41**, 1422). |
| **Borda sum** | Alternative to rank product: simply add the two ranks. Easier to read by eye, slightly less discriminating at the very top. |
| **Z-score combination** | Normalise each lane's score to mean 0 / std 1 then add. Uses score *magnitude*, not just rank, but assumes Gaussian-shaped distributions which our two lanes do not satisfy. |
| **Top-X% intersection** | A compound makes the shortlist iff it sits in the top X% by **both** lanes. Default X = 5 in this pipeline (locked in `_planning/PROJECT_PROPOSAL.md`). |
| **Inner join** | Keeping only compounds present in both input tables. The other join shapes (left, right, outer) would silently bias the ranking; we use inner-join for the shortlist and report dropped compounds in the coverage table. |
| **Anti-join** | The opposite: keep compounds in one table that are **not** in the other. We use it to surface "compounds top-X% by lane A that lane B could not score" so they are not invisibly lost. |
| **Active / Inactive** | The binary label from the labelled training subset (notebook `04`). Available only for compounds in the labelled set; any new screening library will mostly have NaN here, which is fine. |
| **Out-of-fold (OOF) prediction** | A prediction for compound *X* from a model that was **never shown** *X* during training. The cross-validated way to get an honest per-compound rescorer probability across the whole labelled set. The default rescorer column read by this notebook is `rescorer_rf_proba_oof` written by notebook `04`'s CV step. |
| **Sign convention** | Whether *higher* or *lower* values of a score mean "better". The two lanes disagree: the rescorer probability is higher-is-stronger; Boltz-2's `boltz_affinity` is `log10(IC50) µM` where **lower** is stronger. `compute_consensus` handles the negation internally via per-lane flags so on-disk files keep their upstream conventions. |
| **Shortlist** | The final per-compound list after the consensus filter. One row per compound, sorted by `consensus_rank` ascending (1 = best). |
| **Pose** | A 3-D arrangement of the ligand inside the binding pocket. Each shortlisted compound has one pose carried over from gnina's top-1 (notebook `03`/`04`). |
"""),

        markdown("""
## Why consensus -- the chemistry reasoning

Both lanes of this pipeline produce a per-compound ranking, but they come from very different places.

The **classical lane** is a small machine-learning model trained on **this target's** activity labels, looking at gnina's docking output and a ProLIF interaction fingerprint. It is *target-aware* (it knows which residue-interaction patterns correlate with measured activity for this kinase) but its accuracy is bounded by the size of the labelled set.

The **co-folding lane** is Boltz-2: a generative neural network that has learned protein-ligand structure across the whole PDB plus public affinity data. It is *target-agnostic* (it has never seen this protein's activity history) but it is much larger, sees induced-fit motions classical docking ignores, and produces an affinity prediction that is closer to a real number than a docking score.

Their failure modes are different:

- The classical lane can over-fit to scaffolds present in the labelled training set; on a new chemotype its score is less reliable.
- The co-folding lane can produce a confident-looking complex for a non-binder when the protein and the ligand are *individually* plausible but the pairing is wrong.

Two methods that agree are unlikely to be wrong **by chance** in the same direction. That is the operational logic of consensus. It is a small but real reduction in false-positive rate, paid for by a real reduction in raw recall (compounds that are only top-5% by one method get dropped).

### What consensus is **not**

It is not an **ensemble** in the ML sense. We do not average predictions, we intersect rankings. The compounds that survive are guaranteed to look strong by both methods independently; nothing about consensus assumes the two scores are calibrated or comparable in absolute terms. That is why we use rank product (scale-free) rather than z-score combination (assumes comparable scales).

It is also not a **substitute for wet-lab validation**. Consensus narrows the wet-lab budget; it does not replace it. The expectation for `shortlist.sdf` is that a chemist orders the compounds, runs a binding assay, and reports back which ones were actual binders. That feedback then improves notebook `04`'s rescorer for the next round.

### When consensus does not help

If the two lanes' rankings are **highly correlated** (Spearman rho > 0.7), the intersection is almost the same as either lane's top-5% on its own and consensus adds little. Notebook `05` reports the Spearman correlation between Boltz-2 affinity and gnina CNN-affinity in its evaluation cell. On ERK2 we expect rho in the 0.2-0.5 range -- enough agreement that the intersection is non-empty, low enough that the methods carry different information.

If one lane is much better than the other on this target (notebook `04`'s scaffold-AUC vs Boltz-2's scaffold-AUC, both reported in their own notebooks), the intersection is dominated by the strong lane. That is fine -- the weak lane is not adding noise, it is adding a sanity check.
"""),

        markdown("""
## 1 - Setup

### What this section does

Detect Colab vs local, clone the repo on Colab if needed, set up paths, mount Google Drive when on Colab so the notebook reads the upstream caches written by notebooks `04` and `05`. No GPU, no extra installs -- the dependencies (`pandas`, `numpy`, `rdkit`, `matplotlib`) are already in the conda `aidd` environment and on Colab's default Python image.
"""),

        code(title="Setup: clone repo on Colab, configure paths", source="""
import sys
import importlib
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    # CPU-only deps not in Colab's default image. pandas / numpy / matplotlib
    # / seaborn / pyarrow are pre-installed; rdkit + py3Dmol are not.
    !pip install -q rdkit py3Dmol
    REPO_ROOT = Path("/content/aidd-pipeline")
    if not REPO_ROOT.exists():
        !git clone https://github.com/hvmarco/aidd-pipeline.git {REPO_ROOT}
    if not (REPO_ROOT / "src" / "aidd").exists():
        raise RuntimeError(
            "Repo clone failed (most likely cause: the repo is private and "
            "Colab cannot authenticate). Make github.com/hvmarco/aidd-pipeline "
            "public, or use a Personal Access Token via Colab Secrets, then "
            "re-run this cell."
        )
    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()
else:
    REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    importlib.invalidate_caches()

print(f"Repo root: {REPO_ROOT}")
print(f"Running on: {'Colab' if IS_COLAB else 'local'}")
"""),

        code(title="Imports", source=AUTORELOAD_SNIPPET + """
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem

from aidd.io import mount_drive_if_colab, pretty_path
from aidd.consensus import compute_consensus, write_shortlist

sns.set_theme(style="whitegrid")
print("imports ok")
"""),

        markdown("""
### Google Drive -- read this before running the next cell

The upstream caches this notebook reads (`scored_poses.parquet` from notebook `04`, `affinity.csv` from notebook `05`) live on **Google Drive** by default when you ran the upstream notebooks on Colab. The next cell mounts Drive so we can read them.

**To opt out**, set `USE_DRIVE = False` in the cell *before* running it. The notebook will then look for the upstream caches under the local repo's `data/derived/` tree -- which works if you copied them down from Drive, or if you ran every upstream notebook locally.

If you accidentally dismiss the Drive dialog, the cell will crash. Re-run it and either authorise, or change the line to `False`.
"""),

        code(title="Drive persistence toggle  (change to False to opt out)", source="""
# ════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE AUTHORIZATION
# On Colab, the next call will prompt for Drive permission on first use.
#
# →  To OPT OUT of Drive, change the line below to:    USE_DRIVE = False
#    (notebook then reads from the local data/derived/ tree only)
# ════════════════════════════════════════════════════════════════════════
USE_DRIVE = IS_COLAB

DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)
print(f"USE_DRIVE: {USE_DRIVE}  "
      f"({'Drive-backed' if USE_DRIVE else 'local /content (or local repo)'})")
print(f"DATA_ROOT: {DATA_ROOT}")
"""),

        markdown("""
## 2 - Inputs - locate the two upstream tables

### Background

The contract between the upstream notebooks and this one is fixed by `notebooks/README.md` -- both files live under `data/derived/<target>/` in well-known sub-paths. We probe those paths, fail loudly with a useful message if either is missing, and report what we found.

We also locate the **gnina poses cache** at `data/derived/<target>/docking/labeled_subset/poses.sdf`. That file holds the 3-D coordinates that go into `shortlist.sdf`. If it is absent, the notebook still produces `shortlist.csv` (numbers and metadata only) and warns -- the csv is enough for a chemist to decide which compounds to order, the SDF is for visual inspection.
"""),

        code(title="Locate the inputs", source="""
TARGET = "erk2"

SCORING_DIR    = DATA_ROOT / TARGET / "scoring"
BOLTZ_DIR      = DATA_ROOT / TARGET / "boltz"
DOCK_DIR       = DATA_ROOT / TARGET / "docking" / "labeled_subset"
SHORTLIST_DIR  = DATA_ROOT / TARGET / "shortlist"
SHORTLIST_DIR.mkdir(parents=True, exist_ok=True)

SCORED_POSES   = SCORING_DIR / "scored_poses.parquet"
AFFINITY_CSV   = BOLTZ_DIR   / "affinity.csv"
POSES_SDF      = DOCK_DIR    / "poses.sdf"

SHORTLIST_SDF       = SHORTLIST_DIR / "shortlist.sdf"
SHORTLIST_CSV       = SHORTLIST_DIR / "shortlist.csv"
RESCORER_ONLY_CSV   = SHORTLIST_DIR / "rescorer_only_top.csv"
BOLTZ_ONLY_CSV      = SHORTLIST_DIR / "boltz_only_top.csv"
JOINED_CSV          = SHORTLIST_DIR / "joined_full.csv"

if not SCORED_POSES.exists():
    raise FileNotFoundError(
        f"Cannot find {pretty_path(SCORED_POSES, DATA_ROOT, REPO_ROOT)}. "
        "Run notebook 04_score_classical to completion first."
    )
if not AFFINITY_CSV.exists():
    raise FileNotFoundError(
        f"Cannot find {pretty_path(AFFINITY_CSV, DATA_ROOT, REPO_ROOT)}. "
        "Run notebook 05_dock_boltz to completion first."
    )

if not POSES_SDF.exists():
    print(f"⚠ poses.sdf not found at {pretty_path(POSES_SDF, DATA_ROOT, REPO_ROOT)}")
    print("  The shortlist CSV will still be written; the SDF will be empty.")
    print("  Re-run notebook 04 (or notebook 03's labelled-subset step) on Colab")
    print("  GPU once to repopulate the docking cache on Drive.")
else:
    print(f"  poses.sdf : {pretty_path(POSES_SDF, DATA_ROOT, REPO_ROOT)}")

print(f"Inputs:")
print(f"  rescorer  : {pretty_path(SCORED_POSES, DATA_ROOT, REPO_ROOT)}")
print(f"  Boltz-2   : {pretty_path(AFFINITY_CSV, DATA_ROOT, REPO_ROOT)}")
print()
print(f"Outputs (this notebook will write):")
print(f"  shortlist : {pretty_path(SHORTLIST_DIR, DATA_ROOT, REPO_ROOT)}/")
"""),

        code(title="Read the two upstream tables", source="""
# Score-column names (single source of truth -- referenced everywhere below).
# RESCORER_COL is the cross-validated out-of-fold column shipped by notebook
# 04 in commit d84565c: every compound's prediction comes from a model that
# never saw it during training. Notebook 04 also still writes the legacy
# single-fit columns (`rescorer_rf_proba`, `rescorer_xgb_proba`) for
# chemistry-inspection use, but those are data-leaked and must NEVER be
# used for ranking -- if you ever override RESCORER_COL to the legacy name,
# you will get AUC = 1.000 on labelled folds and meaningless consensus.
RESCORER_COL = "rescorer_rf_proba_oof"
BOLTZ_COL    = "boltz_affinity"

scored = pd.read_parquet(SCORED_POSES)
scored["compound_id"] = scored["compound_id"].astype(str)
print(f"scored_poses.parquet  : {len(scored):>5,} rows × {scored.shape[1]} cols")
print(f"  columns: {list(scored.columns)}")
print()

affinity = pd.read_csv(AFFINITY_CSV)
affinity["compound_id"] = affinity["compound_id"].astype(str)
print(f"affinity.csv          : {len(affinity):>5,} rows × {affinity.shape[1]} cols")
print(f"  columns: {list(affinity.columns)}")
print()

if RESCORER_COL not in scored.columns:
    raise KeyError(
        f"Expected rescorer column {RESCORER_COL!r} not found in scored_poses.parquet. "
        f"Available rescorer-like columns: "
        f"{[c for c in scored.columns if c.startswith('rescorer')]}. "
        "This notebook intentionally reads the cross-validated out-of-fold (OOF) "
        "column, not the single-fit `rescorer_rf_proba` column which is data-leaked. "
        "If your scored_poses.parquet is from before commit d84565c, re-run notebook "
        "04 to regenerate it with the OOF columns."
    )

# Quick distribution sanity check on the two score columns we will use.
# Note the sign conventions differ between the two lanes; compute_consensus
# handles the negation internally (see the "Sign convention" key term above
# and feedback_boltz_affinity_sign.md in project memory).
print("Score-column summary:")
print(f"  {RESCORER_COL:24s} (notebook 04, HIGHER = stronger): "
      f"min={scored[RESCORER_COL].min():.3f}  "
      f"median={scored[RESCORER_COL].median():.3f}  "
      f"max={scored[RESCORER_COL].max():.3f}")
print(f"  {BOLTZ_COL:24s} (notebook 05, LOWER  = stronger): "
      f"min={affinity[BOLTZ_COL].min():.3f}  "
      f"median={affinity[BOLTZ_COL].median():.3f}  "
      f"max={affinity[BOLTZ_COL].max():.3f}")
"""),

        markdown("""
### What to look for

- Both `compound_id` columns must be **strings** (we coerce above) so the join does not silently drop matches because one side is `int64` and the other is `object`.
- `rescorer_rf_proba_oof` is an **out-of-fold** probability in `[0, 1]`. *Higher* = the model thinks the compound is more likely to be active. If this column is missing, the upstream `KeyError` above tells you so explicitly -- the notebook will not silently fall back to the leaked `rescorer_rf_proba` column.
- `boltz_affinity` is `log10(IC50) µM` -- **lower** = stronger binder (a strong binder at IC50 = 10 nM scores around `-2`; a weak binder at IC50 = 10 µM scores around `+1`). This is the opposite direction from the rescorer column; `compute_consensus` negates internally so rank 1 always means "strongest binder by this lane". If `boltz_affinity` is all-NaN, notebook `05`'s parser is misaligned with the installed Boltz-2 version -- fix in `src/aidd/co_folding.py` before continuing.

The exact ranges differ per target; the point is just that both columns vary across the library and neither is degenerate.
"""),

        markdown("""
## 3 - Compute the consensus

### Background

`aidd.consensus.compute_consensus` does the actual work in one call:

1. Inner-join on `compound_id`.
2. Per-lane signed score (negate `boltz_affinity` because lower = stronger; leave the rescorer probability alone). This is what makes "rank 1" mean "strongest by this lane" regardless of the column's native direction.
3. Within-lane ranks (`1` = best after sign correction).
4. Top-`top_fraction` flags per lane (default 5%).
5. Rank product on the joined table.
6. Filter to compounds top-`top_fraction` by **both** lanes; sort ascending by rank product; assign `consensus_rank` 1-based.
7. Two anti-join side tables: compounds top-X% by rescorer that Boltz-2 could not score, and vice-versa. These are **not** in the headline shortlist (a chemist asked us for compounds with two-method agreement) but they are visible so nothing is silently lost.

### What this cell does

Calls `compute_consensus` with the locked-in defaults from `_planning/PROJECT_PROPOSAL.md`: `top_fraction=0.05` (top 5%), `rescorer_col=RESCORER_COL` (the out-of-fold rescorer column), `boltz_col=BOLTZ_COL`. The two `*_lower_is_better` flags fall back to their defaults (`False` for the rescorer, `True` for Boltz) which match what the upstream notebooks write to disk -- if you ever swap `BOLTZ_COL` to `boltz_affinity_probability`, also pass `boltz_lower_is_better=False`. The summary prints every count at every stage of the join + filter, plus the sign conventions actually applied.
"""),

        code(title="Apply the consensus rule (top 5% by both, ranked by rank product)", source="""
TOP_FRACTION = 0.05  # locked default per _planning/PROJECT_PROPOSAL.md §1

result = compute_consensus(
    scored, affinity,
    top_fraction=TOP_FRACTION,
    rescorer_col=RESCORER_COL,
    boltz_col=BOLTZ_COL,
    # Defaults match the on-disk conventions of notebooks 04 and 05; pass
    # explicitly so a future reader sees the sign discipline at the call site.
    rescorer_lower_is_better=False,
    boltz_lower_is_better=True,
)

print(f"Consensus computed at top_fraction = {TOP_FRACTION:.0%}")
print(f"  rescorer_col = {RESCORER_COL!r}  (higher = stronger)")
print(f"  boltz_col    = {BOLTZ_COL!r}     (lower  = stronger; negated internally)")
print()
for k, v in result["summary"].items():
    print(f"  {k:30s}: {v}")
"""),

        markdown("""
### How to read the summary

- `n_rescorer_total` and `n_boltz_total`: how many compounds each lane could score (after dropping NaNs).
- `n_intersection`: how many compounds appear in **both** tables. The shortlist is a subset of this number.
- `cut_rescorer` / `cut_boltz`: how many compounds count as "top X%" in each lane individually.
- `n_top_rescorer` / `n_top_boltz`: should match the cuts (modulo ties at the threshold).
- `n_shortlist`: the size of the consensus shortlist -- the headline number for this notebook.
- `n_rescorer_only_top` / `n_boltz_only_top`: side-file sizes. Compounds top-X% by one lane that the other lane never scored.

If `n_shortlist` is `0` -- the two lanes' top-X%s have empty overlap -- consider one of:

- Loosen `TOP_FRACTION` to `0.10` or `0.20` and re-run this cell. Same recipe, just more compounds survive.
- Check the Spearman correlation reported in notebook `05`. If it is near zero or negative, the two methods disagree systematically and consensus will always be small. Report this in the project notes; it is a real (but informative) result.
- Verify both lanes ran on the **same** compound cohort. If notebook `05` ran on a different SMILES set than notebook `04`, the intersection on `compound_id` will be tiny by construction.
"""),

        markdown("""
## 4 - Visualise the rank-product landscape

### Background

A scatter of (rescorer rank) vs (Boltz rank) tells you at a glance whether the two methods agree or not:

- A **diagonal cloud** running bottom-left to top-right means the two rankings are correlated -- compounds the rescorer likes also tend to score well in Boltz. Consensus then keeps the bottom-left corner (top-X% by both), which is exactly what we want.
- A **scattered cloud** with no diagonal means the two methods disagree across the library. Consensus still works: it picks compounds in the bottom-left corner of the cloud, which is the rare overlap region. The shortlist will be smaller but each entry is more informative.
- A **fully horizontal or vertical band** means one lane is degenerate (constant score). Stop and check the upstream cell.

We overlay the shortlisted compounds (in red) so you can see exactly which points the consensus rule kept.
"""),

        code(title="Scatter: rescorer rank vs Boltz rank, with shortlisted compounds highlighted", source="""
joined = result["joined"]
short  = result["shortlist"]

fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
ax.scatter(joined["rank_rescorer"], joined["rank_boltz"],
           c="lightgray", s=20, alpha=0.6, label=f"all joined (n = {len(joined)})")
if len(short) > 0:
    ax.scatter(short["rank_rescorer"], short["rank_boltz"],
               c="crimson", s=60, edgecolor="black", linewidth=0.5,
               label=f"shortlist (n = {len(short)})")

cut_r = result["summary"]["cut_rescorer"]
cut_b = result["summary"]["cut_boltz"]
ax.axvline(cut_r, color="tab:blue",   linestyle=":", alpha=0.7,
           label=f"top {TOP_FRACTION:.0%} by rescorer (rank ≤ {cut_r})")
ax.axhline(cut_b, color="tab:orange", linestyle=":", alpha=0.7,
           label=f"top {TOP_FRACTION:.0%} by Boltz (rank ≤ {cut_b})")

ax.set_xlabel("rescorer rank  (1 = best)")
ax.set_ylabel("Boltz-2 rank   (1 = best)")
ax.set_title(f"Consensus on {TARGET.upper()}: shortlist sits in the bottom-left corner")
ax.legend(loc="upper right", fontsize=9)
ax.invert_xaxis()  # so '1' is at the right edge -- bottom-LEFT is best in unflipped sense
ax.invert_yaxis()  # same for y
plt.tight_layout()
plt.show()
"""),

        markdown("""
### What the axes mean

The axes are inverted so that **rank 1 (the best compound by each lane) sits in the bottom-left**, matching the chemist's mental model of "low score = bad, high = good" even though we are plotting *ranks* (where lower is better). The shortlisted (red) points must all sit in the bottom-left rectangle defined by the two dotted threshold lines -- if any red point is outside that rectangle, the consensus filter is buggy.
"""),

        markdown("""
## 5 - Sanity check: where do the labelled actives end up?

### Background

The labelled training set has known **actives** (the binary `Active` column from the labelled subset, attached to `scored_poses.parquet` in notebook `04`). They are not what we are screening for in production -- they are already known binders -- but they are exactly the right sanity check: a healthy consensus pipeline should put a *respectable fraction* of them in the shortlist, even though the rescorer and Boltz-2 disagree on the rest.

This cell reports:

- How many shortlisted compounds carry an `Active` label (when known).
- What fraction of all known actives in the joined-on-`compound_id` set ended up in the shortlist (the **active recall** at this `top_fraction`).
- The "trivial baseline" recall you would get from picking the same number of compounds at random.

The done signal for this notebook (per `_planning/PROJECT_PROPOSAL.md` step 12) is qualitative: *"the shortlist on ERK2 contains a respectable fraction of the labelled actives."* We do not pin a numeric threshold here -- the right number depends on the lane AUCs, on `top_fraction`, and on the labelled-active count, all of which vary by target.
"""),

        code(title="Active recall in the consensus shortlist (when labels are known)", source="""
if "Active" in joined.columns and joined["Active"].notna().any():
    labelled = joined[joined["Active"].notna()].copy()
    labelled["Active"] = labelled["Active"].astype(int)

    n_actives_total       = int(labelled["Active"].sum())
    n_actives_in_shortlist = int(short["Active"].fillna(0).astype(int).sum()) if len(short) else 0
    pick_fraction          = (len(short) / len(labelled)) if len(labelled) else 0.0
    expected_random        = n_actives_total * pick_fraction
    recall                 = (n_actives_in_shortlist / n_actives_total) if n_actives_total else float("nan")

    print(f"Labelled compounds in the inner-joined set : {len(labelled):>4d}")
    print(f"  of which actives                          : {n_actives_total:>4d}")
    print()
    print(f"Shortlist size at top {TOP_FRACTION:.0%}         : {len(short):>4d}")
    print(f"  actives in the shortlist                  : {n_actives_in_shortlist:>4d}")
    print(f"  active recall                             : {recall:>5.1%}")
    print()
    print(f"Random-baseline expectation at this picksize: ≈ {expected_random:.1f} actives")
    print(f"  (i.e. you'd get ~{expected_random:.1f} actives picking {len(short)} compounds at random)")
    if n_actives_in_shortlist > expected_random:
        ratio = n_actives_in_shortlist / max(expected_random, 1e-9)
        print(f"  consensus enrichment vs random            : {ratio:.1f}x")
else:
    print("No 'Active' labels in the joined table -- skipping recall sanity check.")
    print("This is the expected case for a production library screen with no known")
    print("activity data; the shortlist is then evaluated only by wet-lab follow-up.")
"""),

        markdown("""
### How to read the recall numbers

- **Active recall** is the fraction of known actives that the consensus rule kept. At `top_fraction = 5%` on a screen where neither lane is perfect, expect single- to low-double-digit percentage recall -- the shortlist is *small and high-precision*, not exhaustive.
- **Random-baseline expectation** is what you would get picking the same number of compounds at random. If the consensus shortlist contains noticeably more actives than this baseline, the pipeline is doing what it is supposed to.
- **Consensus enrichment vs random** rolls those two numbers into a single multiplicative factor. Typical ranges on a well-curated kinase target with both lanes working: 3-10x. Above 10x means consensus is strongly enriching for actives; below ~2x means one or both lanes is weak on this target and you should look at the per-lane AUCs in notebooks `04` and `05`.

If the recall is **0** at a non-zero shortlist size, that is informative: it means the actives sit somewhere outside the bottom-left corner of the rank-vs-rank plot. Plot `Active` colouring on the scatter from section 4 to see where they land instead -- usually one lane ranks them high but the other does not.
"""),

        markdown("""
## 6 - Write the shortlist to disk

### Background

Two artefacts go on disk under `data/derived/<target>/shortlist/`:

- **`shortlist.csv`** -- one row per shortlisted compound, with `consensus_rank`, `compound_id`, both ranks, the rank product, both raw scores, the activity label (when known), and any other columns from the upstream tables. This is the wet-lab-ready row, opens in Excel, easy to e-mail.
- **`shortlist.sdf`** -- the same compounds with their **gnina top-1 3-D pose** attached, plus all the consensus columns as SD properties. Opens in PyMOL / VS Code RDKit viewer / any standard structural tool.

We also write three diagnostic files alongside:

- **`joined_full.csv`** -- the full inner-join with both ranks, top-X flags, and `made_shortlist`. Useful for rerunning with a different `top_fraction` without re-doing the join.
- **`rescorer_only_top.csv`** -- compounds top-X% by the rescorer that the Boltz lane could not score (anti-join). Surfaced so they are not invisibly lost.
- **`boltz_only_top.csv`** -- the symmetric case: compounds top-X% by Boltz that the rescorer lane could not score.

The two side files are **not** the wet-lab handoff -- they are an audit trail. A chemist asks "did we drop anything interesting because one method failed on it?" and these two files answer.
"""),

        code(title="Write shortlist.sdf, shortlist.csv, and the diagnostic side files", source="""
write_info = write_shortlist(
    short,
    poses_sdf=POSES_SDF,
    out_sdf=SHORTLIST_SDF,
    out_csv=SHORTLIST_CSV,
)
print(f"shortlist.csv : {write_info['n_csv_written']:>4d} rows  →  "
      f"{pretty_path(SHORTLIST_CSV, DATA_ROOT, REPO_ROOT)}")
print(f"shortlist.sdf : {write_info['n_sdf_written']:>4d} mols  →  "
      f"{pretty_path(SHORTLIST_SDF, DATA_ROOT, REPO_ROOT)}")
if write_info['n_csv_written'] != write_info['n_sdf_written']:
    n_missing = write_info['n_csv_written'] - write_info['n_sdf_written']
    print(f"  ({n_missing} shortlisted compound(s) had no pose in poses.sdf -- in CSV only)")

# Diagnostic side files.
result["joined"].to_csv(JOINED_CSV, index=False)
result["rescorer_only"].to_csv(RESCORER_ONLY_CSV, index=False)
result["boltz_only"].to_csv(BOLTZ_ONLY_CSV, index=False)
print()
print(f"diagnostics written:")
print(f"  joined_full.csv      : {pretty_path(JOINED_CSV,        DATA_ROOT, REPO_ROOT)}")
print(f"  rescorer_only_top.csv: {pretty_path(RESCORER_ONLY_CSV, DATA_ROOT, REPO_ROOT)}")
print(f"  boltz_only_top.csv   : {pretty_path(BOLTZ_ONLY_CSV,    DATA_ROOT, REPO_ROOT)}")
"""),

        code(title="Preview the shortlist CSV", source="""
if len(short):
    preview_cols = [
        "consensus_rank", "compound_id", "rank_product",
        "rank_rescorer", "rank_boltz",
        RESCORER_COL, BOLTZ_COL,
    ]
    if "Active" in short.columns:
        preview_cols.append("Active")
    print(short[preview_cols].head(20).to_string(index=False))
else:
    print("Shortlist is empty -- see section 3 for the diagnostic and consider")
    print("loosening TOP_FRACTION before continuing.")
"""),

        markdown("""
## 7 - Sanity check: visualise the top compound's pose

### Background

A short visual check on the top-ranked shortlisted compound confirms two things at a glance:

1. The 3-D pose actually sits in the **ATP pocket** (between the kinase N-lobe and C-lobe), not on the surface or in some loose surface cleft.
2. Standard kinase-inhibitor interactions are at least geometrically plausible -- the ligand reaches the hinge region, no atoms are clipping into the protein.

This is not a metric, it is an eyeball check that catches gross failures (a ligand placed outside the protein, a malformed pose) that no scalar score would flag. If the visual looks wrong, do not ship the compound to the wet lab without re-checking notebook `04`'s pose.

We use the **gnina top-1 pose** here (the same one that goes into `shortlist.sdf`). Notebook `05` also has co-folded `complex.cif` files per compound; if you want the Boltz-2 view of the same compound, load that file in PyMOL alongside this one.
"""),

        code(title="3-D viewer: gnina top-1 pose for the top-ranked shortlisted compound", source="""
import py3Dmol

if not SHORTLIST_SDF.exists() or write_info["n_sdf_written"] == 0:
    print("No shortlist SDF to display (either no poses cache or empty shortlist).")
else:
    sup = Chem.SDMolSupplier(str(SHORTLIST_SDF), removeHs=False)
    top_mol = next((m for m in sup if m is not None), None)
    if top_mol is None:
        print("shortlist.sdf has no readable molecules.")
    else:
        cid = top_mol.GetProp("_Name") if top_mol.HasProp("_Name") else "(no _Name)"
        rp  = top_mol.GetProp("rank_product")  if top_mol.HasProp("rank_product")  else "?"
        rr  = top_mol.GetProp("rank_rescorer") if top_mol.HasProp("rank_rescorer") else "?"
        rb  = top_mol.GetProp("rank_boltz")    if top_mol.HasProp("rank_boltz")    else "?"

        print(f"Top compound: {cid}")
        print(f"  consensus_rank = 1   rank_product = {rp}")
        print(f"  rank_rescorer  = {rr}   rank_boltz = {rb}")

        ligand_block = Chem.MolToMolBlock(top_mol)

        receptor_pdb = REPO_ROOT / "data" / "structures" / "erk2_4fv7.pdb"
        view = py3Dmol.view(width=720, height=480)
        if receptor_pdb.exists():
            view.addModel(receptor_pdb.read_text(), "pdb")
            view.setStyle({"chain": "A"}, {"cartoon": {"color": "lightgray"}})
        view.addModel(ligand_block, "mol")
        view.setStyle({"model": -1}, {"stick": {"colorscheme": "cyanCarbon"}})
        view.zoomTo({"model": -1})
        view.zoom(0.7)
        view.show()
"""),

        markdown("""
### What to look for in the viewer

- The **ligand (cyan sticks)** should sit in the cleft between the **N-lobe** (top of the kinase domain, smaller, rich in beta-sheet) and the **C-lobe** (bottom, larger, alpha-helical). For ERK2 the relevant residues are around K54 (catalytic lysine), M108 (gatekeeper), L107 / D106 (hinge).
- **If the ligand is sitting on the protein surface**, far from the kinase cleft, the gnina pose is a misdocking. Check `gnina_cnn_affinity` for that compound -- it is usually low, but not always.
- **Atoms clipping into the protein backbone** are a sign of a bad pose; treat the consensus rank with caution and re-dock with tighter exhaustiveness if needed.

For a more thorough per-compound triage, open the SDF directly in PyMOL or the VS Code RDKit viewer -- this notebook's job is the consensus computation, not interactive triage.
"""),

        markdown("""
## 8 - Step-closure summary

### Background

A single consolidated dump of every empirical number this notebook produced -- the consensus-filter counts, the active-recall sanity check, and the SDF / CSV write info. This cell exists for two reasons:

1. **Audit trail.** A reviewer (collaborator, professor, grant funder) opening the saved notebook reads the headline result without scrolling through seven sections.
2. **Pipeline closure record.** The numbers in this dump are what land in the commit message body that closes step 12 of `_planning/PROJECT_PROPOSAL.md` § 7, and in the `notebooks/README.md` status row for this notebook.

If you re-run with a different `top_fraction` (say `0.10` instead of `0.05`), every number below reflects the new setting automatically -- a chemist can scan three runs at three thresholds and pick the shortlist size that matches their wet-lab budget.
"""),

        code(title="Closure summary -- consolidated empirical findings", source="""
import json

closure_findings = {
    "target": TARGET,
    "consensus_settings": {
        "top_fraction": TOP_FRACTION,
        "rescorer_col": RESCORER_COL,
        "boltz_col": BOLTZ_COL,
        "rescorer_lower_is_better": result["summary"]["rescorer_lower_is_better"],
        "boltz_lower_is_better":    result["summary"]["boltz_lower_is_better"],
    },
    "join_coverage": {
        "n_rescorer_total":         result["summary"]["n_rescorer_total"],
        "n_boltz_total":            result["summary"]["n_boltz_total"],
        "n_intersection":           result["summary"]["n_intersection"],
        "n_rescorer_only_universe": result["summary"]["n_rescorer_only_universe"],
        "n_boltz_only_universe":    result["summary"]["n_boltz_only_universe"],
        "n_dropped_nan_rescorer":   result["summary"]["n_dropped_nan_rescorer"],
        "n_dropped_nan_boltz":      result["summary"]["n_dropped_nan_boltz"],
    },
    "consensus_filter": {
        "cut_rescorer":          result["summary"]["cut_rescorer"],
        "cut_boltz":             result["summary"]["cut_boltz"],
        "n_top_rescorer":        result["summary"]["n_top_rescorer"],
        "n_top_boltz":           result["summary"]["n_top_boltz"],
        "n_shortlist":           result["summary"]["n_shortlist"],
        "n_rescorer_only_top":   result["summary"]["n_rescorer_only_top"],
        "n_boltz_only_top":      result["summary"]["n_boltz_only_top"],
    },
    "shortlist_outputs": {
        "n_csv_written":      write_info["n_csv_written"],
        "n_sdf_written":      write_info["n_sdf_written"],
        "shortlist_csv_path": str(SHORTLIST_CSV.relative_to(REPO_ROOT))
                              if SHORTLIST_CSV.is_relative_to(REPO_ROOT)
                              else str(SHORTLIST_CSV),
        "shortlist_sdf_path": str(SHORTLIST_SDF.relative_to(REPO_ROOT))
                              if SHORTLIST_SDF.is_relative_to(REPO_ROOT)
                              else str(SHORTLIST_SDF),
    },
}

# Active-recall sanity (only meaningful when the upstream cohort carries
# Active labels -- production library screens against new chemistry will
# legitimately have None here).
if "Active" in joined.columns and joined["Active"].notna().any():
    labelled        = joined[joined["Active"].notna()]
    n_actives_total = int(labelled["Active"].astype(int).sum())
    n_actives_short = int(short["Active"].fillna(0).astype(int).sum()) if len(short) else 0
    pick_fraction   = (len(short) / len(labelled)) if len(labelled) else 0.0
    expected_random = n_actives_total * pick_fraction
    recall          = (n_actives_short / n_actives_total) if n_actives_total else None
    enrichment      = (n_actives_short / expected_random) if expected_random > 0 else None
    closure_findings["active_recall_sanity"] = {
        "n_labelled":              int(len(labelled)),
        "n_actives_total":         n_actives_total,
        "n_actives_in_shortlist":  n_actives_short,
        "active_recall":           round(recall, 4) if recall is not None else None,
        "expected_random_actives": round(expected_random, 2),
        "enrichment_vs_random":    round(enrichment, 2) if enrichment is not None else None,
    }
else:
    closure_findings["active_recall_sanity"] = None

print(json.dumps(closure_findings, indent=2))
print()

# Human-readable headline.
ar = closure_findings["active_recall_sanity"]
n_short = closure_findings["consensus_filter"]["n_shortlist"]
pct = int(round(TOP_FRACTION * 100))
if ar is not None and ar["n_actives_total"] > 0:
    rec_str = f"{ar['active_recall']:.1%}" if ar['active_recall'] is not None else "n/a"
    enr_str = f"{ar['enrichment_vs_random']:.1f}x" if ar['enrichment_vs_random'] is not None else "n/a"
    print(f"Headline: top-{pct}% intersection shortlist = {n_short} compounds, "
          f"recovers {ar['n_actives_in_shortlist']} / {ar['n_actives_total']} known actives "
          f"({rec_str}); {enr_str} enrichment vs random.")
else:
    print(f"Headline: top-{pct}% intersection shortlist = {n_short} compounds. "
          "(Cohort carries no Active labels -- active-recall sanity skipped; "
          "shortlist judged only by wet-lab follow-up.)")
"""),

        markdown("""
### How to read this cell

Three things to verify against the project's done-signal for step 12 (`_planning/PROJECT_PROPOSAL.md` § 7):

1. **`consensus_settings.rescorer_col == "rescorer_rf_proba_oof"`** and **`boltz_lower_is_better == True`** -- confirms the leak guard cleared and the sign discipline applied. If either is wrong the consensus numbers below are not trustworthy.
2. **`consensus_filter.n_shortlist`** -- the headline size of the wet-lab handoff. Tells you in one number how big the chemist's order list is.
3. **`active_recall_sanity.enrichment_vs_random`** -- the consensus enrichment factor on labelled compounds. Above ~3x means the two-lane intersection is concentrating known actives at meaningfully better than chance; near 1x means consensus is not adding signal on this target. The done-signal in the project proposal is "respectable fraction of labelled actives" without a fixed threshold, so the right framing is honest reporting of whatever number this cell prints.
"""),

        markdown("""
## Recap

### Biomedical takeaway

We took the two independent rankings from notebooks `04` (a per-target ML rescorer trained on the labelled set) and `05` (Boltz-2 co-folding) and reduced them to a small **consensus shortlist** of compounds that look strong by *both* methods. The shortlist is the wet-lab handoff -- a chemist reads `shortlist.csv` top-down, picks compounds to order, and runs a binding assay. The two-method intersection is the pipeline's defence against false positives any single tool might introduce.

For the pharmacogenomics + variant-function workflow that becomes the project's headline narrative (notebook `07`), the same `compute_consensus` call runs unchanged on a mutant variant's `data/derived/<target>_<variant>/` tree. Notebook `07` then **diffs** two consensus shortlists (wild-type vs mutant) to surface compounds whose ranking shifts when the protein changes -- the clinical question of resistance / sensitivity, in pipeline form.

### Technical takeaway

`src/aidd/consensus.py` is intentionally tiny: one ranking primitive (`rank_product`), one consensus function (`compute_consensus`), one writer (`write_shortlist`). The notebook orchestrates them around two CSV / Parquet reads and a directory of side files. Everything is CPU-only, deterministic, reproducible.

Three design decisions worth carrying forward:

- **Rank product over z-score combination.** Rank-based aggregators are scale-free across lanes whose scores live on completely different axes. The cost is losing magnitude information at the top of each lane; in exchange we get a method that does not care whether one lane outputs a 0-1 probability and the other an unbounded affinity.
- **Anti-join side files for partial coverage.** The headline `shortlist.csv` is the inner-join consensus, but the two `*_only_top.csv` files preserve the compounds that only one lane could score. A reviewer asking "what did we drop?" gets a one-file answer, and the chemist can decide whether any of those one-lane-only top hits are worth following up despite missing a consensus partner.
- **On-disk files keep upstream sign conventions; consumers negate at the boundary.** Boltz-2 writes `boltz_affinity` as `log10(IC50) µM` (lower = stronger); the rescorer writes a probability (higher = stronger). Rather than mutating the on-disk files into a single convention -- which would silently drift from Boltz's own tooling and confuse anyone re-opening the file later -- `compute_consensus` takes per-lane `*_lower_is_better` flags and handles the negation internally. The default values match what notebooks 04 and 05 actually write to disk; if you swap `boltz_col` to `boltz_affinity_probability` (already higher = better), pass `boltz_lower_is_better=False`. The same discipline lives in notebook 05's evaluation cell -- one rule, applied at every consumption point, instead of fighting Boltz's units everywhere.

### What's next in the pipeline

- **`07_mutation_analysis.ipynb`** (pending) -- runs the same consensus step against a mutant `data/derived/<target>_<variant>/` tree and **diffs** the two shortlists: which compounds drop out, which survive, which newly appear when the protein changes. This is the headline workflow for the precision-oncology framing in `_planning/PROJECT_PROPOSAL.md` section 8.
- **`99_screen_library.ipynb`** (pending) -- end-to-end runner. Notebooks `00`-`06` chained together with a `mutations=` parameter so a single command produces the consensus shortlist for a wild-type and any number of variant runs.

### Further reading

- Wang, R. & Wang, S. (2001) -- *How does consensus scoring work for virtual library screening? An idealized computer experiment.* J. Chem. Inf. Comput. Sci. **41**, 1422-1426. [doi:10.1021/ci010025x](https://doi.org/10.1021/ci010025x) (the canonical consensus-scoring paper; introduces rank-based aggregation in structural-biology virtual screening.)
- Houston, D. R. & Walkinshaw, M. D. (2013) -- *Consensus docking: improving the reliability of docking in a virtual screening context.* J. Chem. Inf. Model. **53**, 384-390. [doi:10.1021/ci300399w](https://doi.org/10.1021/ci300399w) (modern treatment; rank product compared against alternative consensus rules on a large benchmark.)
- Charifson, P. S., Corkery, J. J., Murcko, M. A., Walters, W. P. (1999) -- *Consensus scoring: a method for obtaining improved hit rates from docking databases of three-dimensional structures into proteins.* J. Med. Chem. **42**, 5100-5109. [doi:10.1021/jm990352k](https://doi.org/10.1021/jm990352k) (the foundational paper on consensus virtual screening; first showed empirically that intersecting independent scorers reduces false-positive rate.)
"""),
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
