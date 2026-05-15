"""Builder for 07_variant_effect_prediction.ipynb.

Source of truth for the per-variant computational priors notebook. Cells
appear below in narrative order. Never edit the .ipynb directly — see
``CLAUDE.md`` section *Notebook workflow*.

Regenerate:
    python notebooks/_build_07_variant_effect_prediction.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import AUTORELOAD_SNIPPET, code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "07_variant_effect_prediction.ipynb"


def build() -> None:
    nb = notebook(
        markdown("""
# 07 - Variant effect prediction (AlphaMissense + gnomAD + RaSP)

**aidd-pipeline · Notebook 7 of the screening workflow**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hvmarco/aidd-pipeline/blob/main/notebooks/07_variant_effect_prediction.ipynb)

This notebook computes three independent **per-variant priors** for any single-residue missense substitution in a human protein:

1. **AlphaMissense** — a sequence/evolution-based pathogenicity probability (0..1).
2. **gnomAD** — population-level allele frequency + per-ancestry breakdown + homozygote count.
3. **RaSP ΔΔG** — a structure-based protein-stability prediction (kcal/mol).

The three priors answer three different questions about a variant. AlphaMissense asks *"does this substitution look pathogenic given evolutionary conservation?"*. gnomAD asks *"is this variant common enough in the population to be worth studying clinically?"*. RaSP asks *"does this substitution destabilise the folded protein?"*. Combining all three gives a much fuller picture than any one alone — pathogenic variants that are also destabilising have a structural mechanism story; pathogenic variants with little stability impact (e.g. KRAS G12C) point at a non-stability mechanism (altered binding, altered catalysis); common benign-looking variants give the calibration baseline.

This notebook does NOT run AlphaMissense, RaSP, or any other model locally — all three are **lookup helpers** against precomputed public data (downloaded once and cached). The calibration walkthrough at the end of the notebook closes step 14 by checking the helpers return sensible values on a small set of well-characterised variants in NAT2, DPYD, and KRAS.

The output of this notebook (the per-variant priors) is consumed downstream by:

- **Notebook 08 (mutation analysis, library-side)** — adds the three priors as columns next to the WT-vs-mutant structural / IFP / shortlist diff for each variant in a library-screening run.
- **Notebook 09 (small-N mechanism-of-action)** — shows the three priors in each per-compound HTML report alongside the binding pose and the IFP diff.

## Learning objectives

After running this notebook you will be able to:

- Explain in plain language what **AlphaMissense pathogenicity**, **gnomAD allele frequency**, and **RaSP ΔΔG** measure, and the question each one answers.
- Read a per-variant priors table and recognise the four mechanistic archetypes the calibration set covers (loss-of-function-stability-driven, common-benign-baseline, loss-of-function in a splice-replaced position, and pathogenic-but-not-stability-driven).
- Recognise the limits of each tool: what AlphaMissense can and cannot say, why gnomAD frequencies depend on ancestry, why RaSP ΔΔG is meaningful for stability mechanism but not for catalysis rate.
- Adapt the helpers to a new variant of your own (one helper call per prior; same UniProt + position + amino-acid identifiers throughout).

## Audience

You'll get value if you are:

- A clinician or wet-lab biologist who wants computational context for a clinical variant before commissioning more expensive structural work.
- A pharmacogenomics researcher screening candidate variants by triage criteria (common + pathogenic).
- A data / ML person new to variant interpretation — this notebook introduces the three pillars (sequence evolution, population frequency, structure stability) used in modern variant analysis.
- A reviewer / grant auditor: every helper has a citation, every helper returns a number with a documented meaning, and the methodology limits are explicit.

## Prerequisites

This notebook is **independent of notebooks 00–06**. It only depends on:

- The conda environment `aidd` (`conda env create -f environment.yml`), or running on Google Colab (the setup cell installs everything).
- A working internet connection on first run (to download the AlphaMissense + RaSP caches; ~5.4 GB total). Subsequent runs hit the on-disk cache and need no network.

## Runtime

- **CPU-only.** No GPU dependency anywhere. Free Colab T4, Colab CPU, Windows / macOS local — all work identically.
- **First run on a new system: 6 – 20 minutes.** Dominated by the AlphaMissense download (~5 GB compressed → stream-decompress + filter to demo set → ~1 MB parquet) and the RaSP CSV download (~414 MB → filter → ~1 MB parquet). Network speed and Colab disk speed dominate; the actual processing is small.
- **Cached re-runs: under 30 seconds.** Both caches are on Drive (Colab) or in the repo's `data/cache/` (local), so a kernel restart re-uses them.
"""),

        markdown("""
## Key terms used in this notebook

| Term | Meaning in one line |
|---|---|
| **Missense variant** | A single-base DNA change that swaps one amino acid for another in the encoded protein (e.g. NAT2 I114T = isoleucine at position 114 changed to threonine). |
| **Reference / wild-type (WT)** | The amino acid in the canonical SwissProt sequence at a given position. |
| **Alternate (alt) amino acid** | The substituted residue; what the variant changes the position to. |
| **AlphaMissense** | DeepMind's 2023 deep-learning model that scores every possible human missense variant for pathogenicity (Cheng et al., *Science* **381**, eadg7492). |
| **Pathogenicity probability** | AlphaMissense's headline output, a number between 0 (likely benign) and 1 (likely pathogenic). |
| **AlphaMissense class** | A discrete bucket assigned by AlphaMissense from the probability: `likely_benign`, `ambiguous`, or `likely_pathogenic`. |
| **gnomAD** | Genome Aggregation Database — the largest public catalogue of human genetic variation (>800,000 exomes + genomes; Karczewski et al., *Nature* **581**, 434, 2020; v4 update 2024). |
| **Allele frequency (AF)** | Fraction of sequenced chromosomes carrying the variant. AF=0.01 means 1% of chromosomes carry the variant; AF=0 means it is unobserved in gnomAD. |
| **Allele count (ac) / Allele number (an)** | Numerator / denominator behind allele frequency. ``af = ac / an``. |
| **Homozygote count** | Number of individuals in gnomAD with two copies of the variant. Important for recessive disease genetics. |
| **Population codes** | gnomAD's per-ancestry buckets: `nfe` (non-Finnish European), `afr` (African), `eas` (East Asian), `sas` (South Asian), `amr` (admixed American), `asj` (Ashkenazi Jewish), `fin` (Finnish), `mid` (Middle Eastern), `ami` (Amish), `oth` (other). |
| **HGVSp** | Human Genome Variation Society protein-level notation (e.g. `p.Ile114Thr`). gnomAD's per-transcript annotation; we match against it to look up a variant by protein change. |
| **RaSP (Rapid Stability Predictions)** | Blaabjerg et al.'s 2023 ML model that predicts the stability change of a single-residue substitution from a protein structure (eLife **12**, e82593). |
| **ΔΔG (kcal/mol)** | "Delta-delta-G" — the difference in folding free energy between mutant and wild-type. Positive = the mutant is less stable (folds less tightly); negative = the mutant is more stable. Magnitudes around ±1 kcal/mol are mild; > +2 kcal/mol is meaningfully destabilising. |
| **Stability change vs catalysis rate** | Stability (RaSP) measures whether the protein still folds. Catalysis rate (k_cat, K_m) measures how fast the folded enzyme processes substrate. They are different quantities; this notebook predicts only the first. |
| **SwissProt canonical isoform** | The single reference protein sequence for each gene in UniProt. AlphaMissense and RaSP both score against this canonical sequence. |
| **PDB residue numbering** | Residue indexing inside a crystal structure file. Can differ from UniProt numbering when the structure lacks the N-terminal methionine, has a cleaved signal peptide, or has gaps. RaSP's predictions follow PDB numbering of the structure they were computed on. |
"""),

        markdown("""
## Why combine three priors

A single-tool answer about a variant is fragile. Each of the three tools we use here can be wrong, and they are wrong for different reasons:

- **AlphaMissense** can over-call pathogenicity in disordered regions where evolutionary conservation alone is misleading. It does not know about specific disease mechanisms.
- **gnomAD** allele frequency tells you whether a variant is *seen* in healthy populations, not whether it is pathogenic. Common variants are usually benign (selection has had time to remove pathogenic ones), but rare variants are not necessarily pathogenic — most rare variants are simply rare.
- **RaSP ΔΔG** captures the stability dimension only. A variant can be destabilising without being pathogenic (cells tolerate quite a lot of destabilisation), or pathogenic without being destabilising (e.g. a substitution at the active site that ruins catalysis but leaves the fold intact).

Combining all three gives a much richer story:

- A variant that AlphaMissense calls pathogenic, RaSP calls destabilising (`ΔΔG > 0`), and gnomAD shows is rare → strong loss-of-function-by-misfolding hypothesis.
- A variant that AlphaMissense calls pathogenic, RaSP calls neutral (`ΔΔG ≈ 0`), and gnomAD shows is rare or somatic-only → loss / gain of function not driven by stability (active-site substitution, allosteric perturbation, altered binding partner). KRAS G12C is the prototype.
- A variant that AlphaMissense calls benign, RaSP calls neutral, and gnomAD shows is common → benign polymorphism baseline.

The calibration walkthrough at the end of this notebook tests three of these archetypes head-to-head and confirms the helpers behave as expected on well-characterised variants.

### What the priors do **not** answer

- **Catalysis rate.** None of the three priors predicts k_cat or K_m. For pharmacogene metabolism (NAT2 acetylation rate, CYP2D6 oxidation rate, DPYD reduction rate, UGT1A1 glucuronidation rate), stability change captures *one* mechanism contributing to slow / poor metabolism — a meaningfully strong contributor in many alleles, but not the whole story. QM/MM and EnzyHTP-style methods are the right tools for catalysis-rate prediction; both are out of scope here.
- **Patient-level outcome.** A pathogenicity probability is not a clinical recommendation. Clinical decisions need clinical context (homozygous vs heterozygous, drug + dose, comorbidities) that none of these tools knows about.
"""),

        markdown("""
## 1 - Setup

### What this section does

Detect Colab vs local, install pip extras only where needed (this notebook has no GPU dependency and only needs `pandas` + `pyarrow` + `requests`), set up the import path for `src/aidd/`, and mount Google Drive on Colab so the (one-time) AlphaMissense and RaSP cache downloads survive runtime restarts.
"""),

        code(title="Setup: detect Colab vs local, configure paths", source="""
import sys
import importlib
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    # Conservative install. pandas / numpy / pyarrow / requests are all in
    # Colab's default image; this line is here as the safety net for the
    # rare runtime image that drops one of them.
    !pip install -q pandas pyarrow requests tqdm
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

import pandas as pd

from aidd.io import mount_drive_if_colab, pretty_path
from aidd.variants import (
    DEMO_SET_UNIPROT_IDS,
    alphamissense_score,
    alphamissense_class,
    gnomad_frequency,
)
from aidd.stability import rasp_ddg, rasp_covers

print("imports ok")
print(f"Demo-set genes covered by the priors: "
      f"{sorted(DEMO_SET_UNIPROT_IDS.values())}")
"""),

        markdown("""
### Google Drive — read this before running the next cell

The two cache files this notebook builds (AlphaMissense ~1 MB after demo-set filter, RaSP ~1 MB after demo-set filter) are tiny. The *download* that builds them is not — AlphaMissense's source is ~5 GB compressed and RaSP's is ~414 MB. Putting both caches on Google Drive when on Colab means a runtime restart re-uses them instead of re-downloading.

**To opt out**, set `USE_DRIVE = False` in the cell *before* running it. The notebook will then write the caches under the local repo's `data/cache/` tree — fine if you intend to re-run on the same Colab runtime, painful otherwise.
"""),

        code(title="Drive persistence toggle  (change to False to opt out)", source="""
# ════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE AUTHORIZATION
# On Colab, the next call will prompt for Drive permission on first use.
#
# →  To OPT OUT of Drive, change the line below to:    USE_DRIVE = False
#    (caches then live under the local data/cache/ tree only; they will
#    be re-downloaded after any Colab runtime restart)
# ════════════════════════════════════════════════════════════════════════
USE_DRIVE = IS_COLAB

DATA_ROOT = mount_drive_if_colab(REPO_ROOT, use_drive=USE_DRIVE)
# Caches are peers of data/derived/ — same Drive tree, separate sub-folder
# so they are easy to find / inspect / clear.
CACHE_ROOT = DATA_ROOT.parent / "cache"
ALPHAMISSENSE_CACHE = CACHE_ROOT / "alphamissense"
RASP_CACHE          = CACHE_ROOT / "rasp"
GNOMAD_CACHE        = CACHE_ROOT / "gnomad"

print(f"USE_DRIVE: {USE_DRIVE}  "
      f"({'Drive-backed' if USE_DRIVE else 'local /content (or local repo)'})")
print(f"Cache root: {CACHE_ROOT}")
"""),

        markdown("""
## 2 - AlphaMissense

### Background

[AlphaMissense](https://www.science.org/doi/10.1126/science.adg7492) (Cheng et al., *Science* **381**, eadg7492, 2023) is a deep-learning model from DeepMind that scores every possible single-amino-acid substitution in the human proteome for pathogenicity. It is built on top of the AlphaFold protein-structure model and trained with weak supervision from population-frequency data (very common variants are treated as benign; rare disease-associated variants as pathogenic).

The output is a single probability per (UniProt × position × alt amino acid) tuple, between 0 and 1. AlphaMissense also discretises the probability into three classes — `likely_benign`, `ambiguous`, `likely_pathogenic` — using cut-offs the authors calibrated on held-out clinical data.

The model is most useful as **one** signal of pathogenicity among several. It is much stronger than older tools (PolyPhen-2, SIFT, REVEL) on benchmark sets, but it has known weaknesses:

- It uses evolutionary conservation as the dominant signal. In **intrinsically disordered regions**, where conservation is weak by definition, AlphaMissense often returns high pathogenicity probabilities for variants that are clinically benign.
- It cannot distinguish between **mechanism of pathogenicity**. A high pathogenicity score does not tell you whether the variant destabilises the protein, ruins catalysis, abolishes a binding partner, or anything else mechanistic. That is why we pair it with RaSP (stability) and gnomAD (population context).
- It is calibrated on the **canonical SwissProt isoform** of each gene. Variants in alternative isoforms or in indel / splice variants are not in the table — the lookup returns ``None``.

We do not run the model here. We look the answer up in the precomputed table DeepMind released with the paper.

### Demo-set scope

To keep the on-disk cache small and the first-run download manageable, this notebook ships only the AlphaMissense scores for the seven demo-set genes (`NAT2`, `DPYD`, `CYP2D6`, `UGT1A1`, `KRAS`, `BRCA1`, `ESR1`). The first call to `alphamissense_score(...)` on a new system stream-downloads the upstream ~5 GB gzipped TSV and filters to the seven UniProt accessions on the fly, producing a ~1 MB parquet at `data/cache/alphamissense/demo_set.parquet`. To extend coverage to additional genes, edit `DEMO_SET_UNIPROT_IDS` in `src/aidd/variants.py` and delete the parquet to trigger a rebuild.
"""),

        markdown("""
### What this cell does

Looks up the AlphaMissense pathogenicity probability + class for the canonical NAT2 \\*5 variant (I114T at UniProt position 114) — a slow-acetylator allele in the CRC-relevant NAT2 carcinogen-detoxification pathway.

**The first call triggers the AlphaMissense download** (~5 GB stream-decompress + filter; 5 – 15 min depending on connection). Subsequent calls hit the cached parquet and are instantaneous.
"""),

        code(title="AlphaMissense lookup: NAT2 I114T (the *5 slow-acetylator marker)", source="""
am_prob  = alphamissense_score("P11245", 114, "T", cache_dir=ALPHAMISSENSE_CACHE)
am_label = alphamissense_class("P11245", 114, "T", cache_dir=ALPHAMISSENSE_CACHE)

print(f"NAT2 I114T  (UniProt P11245)")
print(f"  AlphaMissense pathogenicity = {am_prob}")
print(f"  AlphaMissense class         = {am_label}")
"""),

        markdown("""
### Interpretation

The probability is a number between 0 and 1; the class is the discretised bucket. AlphaMissense's published cut-offs are roughly: probability < 0.34 → `likely_benign`; 0.34 – 0.564 → `ambiguous`; > 0.564 → `likely_pathogenic`. NAT2 \\*5 is a well-studied slow-acetylator allele with a clear functional phenotype, so we expect AlphaMissense to lean pathogenic — but it is also relatively common in many populations, and AlphaMissense has been observed to be slightly less confident on common functionally-significant variants than on rare clinically-defined pathogenic ones. The exact number you see is the calibration data point; what matters for the calibration table at the end of the notebook is whether all four variants land in the *expected* bucket relative to each other.

If the lookup returns `None`, the most common cause is `alt_aa == wild-type at this position` (no substitution to score) or that the position falls outside the canonical isoform. The helper raises `ValueError` if you query a UniProt accession outside the demo-set scope; the message tells you how to extend.
"""),

        markdown("""
## 3 - gnomAD allele frequency

### Background

[gnomAD](https://gnomad.broadinstitute.org/) (Genome Aggregation Database; Karczewski et al., *Nature* **581**, 434, 2020; v4 update 2024) aggregates whole-exome and whole-genome sequencing from over 800,000 individuals. For every observed variant it reports the allele frequency overall and stratified by ancestry — the latter is essential for pharmacogenomics, where some variants are common in one population but rare in another (DPYD\\*2A is ~1.5% in non-Finnish Europeans but ~0.05% in East Asians, and that drives regional 5-FU dosing recommendations).

For this notebook, the question gnomAD answers is: **"how common is this variant?"**. Combined with AlphaMissense's pathogenicity call, the joint reading triages variants into actionable categories:

- *common AND pathogenic-looking* — high research priority; strong candidate for structural follow-up + clinical association studies. NAT2 \\*5 sits here for slow-acetylator pharmacogenomics.
- *common AND benign-looking* — the calibration baseline, the population-level "normal".
- *rare AND pathogenic-looking* — the long tail of monogenic-disease variants. DPYD \\*13 (I560S) sits here; rare in the population, severe phenotype in carriers.
- *unobserved (gnomAD AF = 0) AND pathogenic-looking* — typically somatic / oncogenic variants. KRAS G12C is the canonical example: oncogenic in tumour cells, near-zero germline frequency.

### How the lookup works

gnomAD does not index variants by protein position — variants live at genomic coordinates. Internally the helper queries gnomAD's public GraphQL API for *every* variant in the requested gene (one round-trip per gene), caches the response at `data/cache/gnomad/<gene>_<dataset>.json`, and filters in-memory for the (position, alt amino acid) match using the response's HGVSp annotation. The second variant in the same gene is therefore free (cache hit, no network call).

We default to **gnomAD v4** (combined exomes + genomes), the largest and most recent release.
"""),

        markdown("""
### What this cell does

Looks up gnomAD frequency information for NAT2 I114T. The returned dict has both the overall allele frequency and a per-ancestry breakdown. We print the overall numbers + the top three populations by allele count to show the ancestry stratification.
"""),

        code(title="gnomAD lookup: NAT2 I114T overall + per-ancestry breakdown", source="""
gn = gnomad_frequency("P11245", 114, "T", cache_dir=GNOMAD_CACHE)

print(f"NAT2 I114T  (UniProt P11245)")
print(f"  Found in gnomAD: {gn['found']}")
if gn['found']:
    print(f"  variant_id            = {gn['variant_id']}")
    print(f"  hgvsp                 = {gn['hgvsp']}")
    print(f"  allele frequency      = {gn['allele_freq_overall']:.4f}")
    print(f"  allele count / number = {gn['allele_count']} / {gn['allele_number']}")
    print(f"  homozygote count      = {gn['hom_count']}")
    if gn['populations']:
        print(f"\\n  Top populations by allele count:")
        top_pops = sorted(
            gn['populations'].items(),
            key=lambda kv: kv[1].get('ac', 0),
            reverse=True,
        )[:5]
        for pid, d in top_pops:
            print(f"    {pid:>4s}  ac={d['ac']:>6d}  an={d['an']:>7d}  "
                  f"af={d['af']:.4f}  hom={d['hom_count']}")
"""),

        markdown("""
### Interpretation

The overall allele frequency is the headline number — what fraction of all sequenced chromosomes carry the variant. The per-ancestry breakdown matters because pharmacogene allele frequencies vary substantially across populations (slow-acetylator NAT2 alleles are more common in some Middle Eastern and Mediterranean populations than in East Asian ones, for example). For clinical decisions in a specific patient, the per-ancestry frequency in their reported background is more informative than the global average.

A few defensive notes:

- If the variant is absent from gnomAD, `found` is `False` and the numeric fields are zero. This is the common pattern for somatic-only oncogenic drivers (KRAS G12C in germline gnomAD: AF effectively zero).
- gnomAD's HGVSp annotation is per-transcript; when multiple transcripts at the locus carry the same protein change, the helper picks the row with the highest combined exome+genome allele count (the canonical transcript in practice).
- The first call to a new gene triggers one GraphQL round-trip; subsequent calls in the same gene reuse the cached response.
"""),

        markdown("""
## 4 - RaSP (Rapid Stability Predictions) ΔΔG

### Background

[RaSP](https://elifesciences.org/articles/82593) (Blaabjerg et al., *eLife* **12**, e82593, 2023) is an ML predictor of the **stability change** caused by a single-residue substitution. Given a protein structure and a (position, wild-type, mutant) triple, it returns a ΔΔG in kcal/mol — the predicted change in folding free energy:

- ΔΔG > 0 → the mutant folds *less* tightly than the wild-type (destabilising).
- ΔΔG < 0 → the mutant folds *more* tightly (stabilising; rarer in real proteins).
- |ΔΔG| ≲ 0.5 kcal/mol → essentially neutral / within prediction noise.
- ΔΔG ≳ +2 kcal/mol → meaningfully destabilising; the mutant protein is likely to be partially unfolded at body temperature, which translates into reduced cellular abundance via faster degradation.

Why this matters for the pharmacogenes specifically: for many slow-acetylator (NAT2) and reduced-function (DPYD, UGT1A1) alleles, the kinetic phenotype (slow drug metabolism) is dominated by *reduced cellular abundance of the enzyme*, which is in turn dominated by *protein-stability change*. RaSP captures this dimension directly. Combined with the binding-pose evidence the rest of the pipeline produces, the slow-acetylator mechanism story becomes much more complete than binding alone — without claiming to predict catalytic rate, which still requires QM/MM-class methods (out of scope here; see the recap at the bottom of the notebook).

### Lookup-only — and why

The published RaSP install is currently unmaintained: it pins to Python 3.6 + PyTorch 1.2.0 + DSSP / Reduce build dependencies, and the upstream README states that *"Colab no longer supports the dependencies of RaSP and there is currently no solution in the pipeline."* Re-running RaSP locally would mean significant dependency-rescue work and would still be Linux-only.

We sidestep the install entirely by looking up against the upstream's **precomputed** saturated single-residue predictions. Source: `rasp_preds_exp_strucs_gnomad_clinvar.csv` (~414 MB) at the upstream's [share link](https://sid.erda.dk/sharelink/fFPJWflLeE) — saturated predictions on every human protein with a crystal structure.

The trade-off: we cannot run RaSP on a user-supplied custom PDB (e.g. a mutant fold from notebook 01). Predictions are fixed to the canonical experimental structures used by the upstream's saturation run.

### Coverage of the demo-set genes

`NAT2`, `DPYD`, `CYP2D6`, `KRAS`, `BRCA1`, `ESR1` — each has at least one human crystal structure, so each is covered.

`UGT1A1` is **not covered** (no human crystal structure exists). This is acknowledged in `_planning/MECHANISM_OF_ACTION_SCOPE.md`; the future notebook 09 UGT1A1\\*28 demo will branch to AlphaFold-based RaSP predictions when needed (one helper-internal data-source swap; the helper signature does not change).
"""),

        markdown("""
### What this cell does

Looks up the RaSP ΔΔG prediction for NAT2 I114T (the same variant we used for AlphaMissense and gnomAD). **The first call triggers the RaSP download** (~414 MB → demo-set filter → ~1 MB parquet; 1 – 3 min). After that, lookups are instantaneous.
"""),

        code(title="RaSP lookup: NAT2 I114T ΔΔG", source="""
ddg = rasp_ddg("P11245", 114, "I", "T", cache_dir=RASP_CACHE)
print(f"NAT2 I114T  (UniProt P11245)")
print(f"  RaSP ΔΔG = {ddg} kcal/mol  (positive = destabilising)")

# Coverage triage example: UGT1A1 has no human crystal in the upstream cache.
print(f"\\nCoverage triage:")
for uniprot, gene in sorted(DEMO_SET_UNIPROT_IDS.items(), key=lambda kv: kv[1]):
    covered = rasp_covers(uniprot, cache_dir=RASP_CACHE)
    print(f"  {gene:>8s} ({uniprot}): {'covered' if covered else 'NOT covered (no human crystal in upstream)'}")
"""),

        markdown("""
### Interpretation

The single number `RaSP ΔΔG` is the headline — positive means the mutant is less stable than wild-type. For many slow-acetylator and reduced-function pharmacogene alleles we expect a small-to-moderate positive value (the mechanism is reduced abundance via destabilisation, but not so destabilising that the protein never folds). Values around or below zero are common too — variants at surface-exposed positions far from the folding core often have little stability impact.

The coverage triage block above prints, per demo-set gene, whether the experimental-structures cache has any RaSP predictions for it. `UGT1A1` is the expected "NOT covered" entry. When notebook 09's UGT1A1\\*28 demo lands (step 16), `_build_rasp_demo_cache` in `src/aidd/stability.py` will swap to the AlphaFold-based 9 GB upstream — the lookup signature does not change, only the data source.

Two defensive points worth knowing:

- **Position-numbering drift.** RaSP predictions follow the *PDB* residue numbering of the crystal structure they were computed on. If UniProt and PDB numbering disagree at a position (cleaved signal peptide, missing N-terminal Met, gap residues), the helper's `wt_aa` cross-check returns `None` rather than silently returning the wrong residue's prediction. If you see an unexpected `None`, suspect numbering drift first.
- **Multi-PDB averaging.** When the upstream CSV has multiple rows for the same (uniprot, position, wt, mut) — typically because a protein has multiple chains in the same crystal, or multiple deposited structures — the helper returns the mean ΔΔG across rows. Spread across rows is small in practice for surface-exposed residues; somewhat larger inside the folding core.
"""),

        markdown("""
## 5 - Calibration walkthrough

### Background

The done-signal for step 14 (per `_planning/PROJECT_PROPOSAL.md` § 7) is that the helpers return sensible values on a small set of well-characterised variants. We pick four variants spanning four mechanistic archetypes:

| # | Variant | Archetype | What we expect |
|---|---|---|---|
| 1 | **NAT2 I114T (\\*5)** | LoF, stability-driven (slow acetylator) | AlphaMissense leans pathogenic; RaSP ΔΔG > 0; gnomAD common (~25–50% population-stratified) |
| 2 | **NAT2 K268R (\\*11/\\*12)** | Common, normal-function baseline | AlphaMissense leans benign; RaSP ΔΔG ≈ 0; gnomAD very common (~40–50% global) |
| 3 | **DPYD I560S (\\*13)** | LoF, stability-driven (rare; replaces \\*2A which is splice and outside protein-level prediction tools' scope) | AlphaMissense pathogenic; RaSP ΔΔG > 0; gnomAD rare |
| 4 | **KRAS G12C** | Pathogenic but NOT stability-driven (oncogenic driver) | AlphaMissense pathogenic; RaSP ΔΔG small (G12C does not destabilise the fold — the activation mechanism is altered nucleotide binding); gnomAD near zero germline |

The point of #4 is the **negative control**: if every "pathogenic" variant came back with a large positive ΔΔG, RaSP would be useless as an additional signal. KRAS G12C should show that the helpers are not just rubber-stamping pathogenic variants as destabilising. #2 is the second negative control: a genuinely common, functionally normal variant should land near AlphaMissense 0, near RaSP ΔΔG 0, and at gnomAD AF ~0.5.

### A note on DPYD\\*2A

The canonical DPYD\\*2A pharmacogene allele (`IVS14+1G>A`, splicing mutation causing exon-14 skipping) is a *splice* variant, not a missense. AlphaMissense and RaSP are both missense-only protein-level tools — neither has a meaningful answer for a splice variant per se. We use DPYD I560S (the \\*13 allele) as the calibration probe instead: it is a real, pathogenic, well-characterised DPYD missense variant with decreased enzyme activity. The methodological point — "AlphaMissense and RaSP are protein-level missense-only tools; splice variants need different methods" — is itself part of what this notebook teaches.
"""),

        markdown("""
### What this cell does

Iterates over the four calibration variants, calls all three priors per variant, assembles the result into a tidy DataFrame, prints it as a table, and emits a `CALIBRATION SUMMARY` block summarising what was queried + what came back. The block is the same shape as notebook 06's `OPERATING-POINT SUMMARY` and is the copy-pasteable artefact for the step-14 closure commit.
"""),

        code(title="Calibration walkthrough: four variants, three priors each", source="""
# Each calibration entry: (gene, uniprot_id, position, wt_aa, mut_aa, label, archetype)
CALIBRATION_VARIANTS = [
    ("NAT2", "P11245", 114, "I", "T",
     "*5 (slow acetylator marker)",
     "LoF, stability-driven"),
    ("NAT2", "P11245", 268, "K", "R",
     "*11/*12 (common, normal function)",
     "Common benign baseline"),
    ("DPYD", "Q12882", 560, "I", "S",
     "*13 (LoF; replaces *2A which is splice)",
     "LoF, stability-driven (rare)"),
    ("KRAS", "P01116",  12, "G", "C",
     "G12C (oncogenic driver)",
     "Pathogenic, NOT stability-driven"),
]

rows = []
for gene, uniprot, pos, wt, mut, label, archetype in CALIBRATION_VARIANTS:
    am_prob  = alphamissense_score(uniprot, pos, mut, cache_dir=ALPHAMISSENSE_CACHE)
    am_label = alphamissense_class(uniprot, pos, mut, cache_dir=ALPHAMISSENSE_CACHE)
    gn       = gnomad_frequency(uniprot, pos, mut, cache_dir=GNOMAD_CACHE)
    ddg      = rasp_ddg(uniprot, pos, wt, mut, cache_dir=RASP_CACHE)
    rows.append({
        "gene":               gene,
        "uniprot":            uniprot,
        "variant":            f"{wt}{pos}{mut}",
        "label":              label,
        "archetype":          archetype,
        "am_prob":            am_prob,
        "am_class":           am_label,
        "gnomad_af":          gn["allele_freq_overall"] if gn["found"] else 0.0,
        "gnomad_ac":          gn["allele_count"] if gn["found"] else 0,
        "gnomad_hom":         gn["hom_count"] if gn["found"] else 0,
        "rasp_ddg":           ddg,
    })

calibration_df = pd.DataFrame(rows)
calibration_df
"""),

        markdown("""
The next cell prints a single boxed `CALIBRATION SUMMARY` block — one row per variant, with all four prior values inline plus a one-line archetype reminder. This is the block to paste into the step-14-closure commit message after running the notebook end-to-end on Colab + locally.
"""),

        code(title="CALIBRATION SUMMARY block (copy-pasteable for step-14 closure)", source="""
def _fmt_float(x, digits=3):
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else str(x)

print("=" * 70)
print("CALIBRATION SUMMARY — variant priors (notebook 07, step 14 done-signal)")
print("=" * 70)
print(f"Cohort: {len(calibration_df)} variants across "
      f"{calibration_df['gene'].nunique()} genes "
      f"({', '.join(sorted(calibration_df['gene'].unique()))})")
print(f"Priors: AlphaMissense (Cheng 2023) + gnomAD r4 + RaSP (Blaabjerg 2023, lookup)")
print()
print(f"{'gene':>6s}  {'variant':>9s}  {'label':<42s}  "
      f"{'AM_prob':>8s}  {'AM_class':>20s}  "
      f"{'gnomAD_AF':>10s}  {'gnomAD_hom':>10s}  {'RaSP_ddG':>10s}")
print("-" * 70)
for _, r in calibration_df.iterrows():
    print(
        f"{r['gene']:>6s}  {r['variant']:>9s}  {r['label']:<42s}  "
        f"{_fmt_float(r['am_prob'], 3):>8s}  {str(r['am_class'] or '-'):>20s}  "
        f"{_fmt_float(r['gnomad_af'], 5):>10s}  {r['gnomad_hom']:>10d}  "
        f"{_fmt_float(r['rasp_ddg'], 3):>10s}"
    )
print()
print("Done-signal verdict (step 14, per PROJECT_PROPOSAL.md § 7):")
print("- LoF stability-driven   (NAT2 I114T): expect AM>0.5, RaSP>0, common gnomAD")
print("- Common benign baseline (NAT2 K268R): expect AM<0.34, RaSP~0, common gnomAD")
print("- LoF stability-driven   (DPYD I560S): expect AM>0.5, RaSP>0, rare gnomAD")
print("- Pathogenic non-stability (KRAS G12C): expect AM>0.5, RaSP small, ~0 gnomAD")
print("=" * 70)
"""),

        markdown("""
### How to read this block

Read the table top-to-bottom by archetype, not by gene. The point of the calibration is that each row matches its expected archetype pattern:

1. **NAT2 I114T (`*5`)** should look pathogenic in AlphaMissense, destabilising in RaSP, and common in gnomAD. If all three line up, the slow-acetylator stability-mechanism story is internally consistent across the three priors.
2. **NAT2 K268R (`*11/*12`)** is the population baseline. Near-zero AlphaMissense, near-zero RaSP ΔΔG, and very high gnomAD frequency together say "this is what 'common normal' looks like". Compare it side-by-side with row 1 to see the calibration: row 1 says "different from baseline in the pathogenic direction"; row 2 says "this *is* the baseline".
3. **DPYD I560S (`*13`)** stress-tests AlphaMissense + RaSP on a rare, severe loss-of-function variant. The expected pattern is similar to row 1 in pathogenicity + stability, but the gnomAD frequency should be near zero — confirming that AlphaMissense's pathogenicity call is not tautologically driven by allele frequency alone.
4. **KRAS G12C** is the negative control for RaSP. AlphaMissense should still call it pathogenic (it is — it's an oncogenic driver), and gnomAD should show essentially zero germline carriers (it's a somatic mutation), but RaSP should report a *small* ΔΔG. G12C does not destabilise the protein — the oncogenic mechanism is altered nucleotide binding, not loss of fold. If RaSP returned a large positive ΔΔG here, it would mean the helper is over-flagging pathogenic variants as destabilising; the small ΔΔG is the evidence that the three priors really do measure independent things.

If any row's numbers do not match the expected archetype, that is the surprise to investigate — could be a position-numbering issue (PDB vs UniProt drift; see RaSP section), an HGVSp-matching issue at the gnomAD layer (multiple-transcript ambiguity), or a real biological surprise worth digging into.

The block above is the artefact to paste into the step-14-closure commit message. The numerical values come straight from the helpers; the verdict lines are pre-written so future readers can see whether the calibration matched expectations.
"""),

        markdown("""
## Recap

### Biomedical takeaway

For any human missense variant in one of the seven demo-set genes (NAT2, DPYD, CYP2D6, UGT1A1, KRAS, BRCA1, ESR1) we now have three orthogonal computational priors at our fingertips: an evolution-based pathogenicity probability (AlphaMissense), a population-frequency lookup with per-ancestry breakdown (gnomAD), and a structure-based stability prediction (RaSP). The four-variant calibration walkthrough demonstrated that the three priors carry **independent information** — KRAS G12C is the proof-by-counterexample that pathogenicity does not always mean destabilisation. For the pharmacogene applications in the rest of the pipeline (notebooks 08 and 09), this triples the per-variant evidence base going into the WT-vs-mutant binding analysis without adding any new model-inference cost.

### Technical takeaway

Two short modules (`aidd.variants` for AlphaMissense + gnomAD; `aidd.stability` for RaSP) wrap each prior in a single lookup function. The architectural common denominator: **lookup-only against precomputed public data** — no local model inference, no install pain, no GPU dependency, cross-platform identical. This pattern is in turn forced by the upstream realities (AlphaMissense's full proteome ships precomputed; RaSP's official install is unmaintained but the precomputed predictions are still distributed; gnomAD has no static download for our use case but has a free public GraphQL API). Each lookup is cached on disk on first call; downstream notebooks 08 and 09 will read these caches with no additional configuration.

### What's next in the pipeline

- **`08_mutation_analysis.ipynb`** (step 15) — library-side WT-vs-mutant comparison. Reads the per-variant priors from this notebook as added columns next to the structural / IFP / shortlist diff.
- **`09_moa_small_n.ipynb`** (step 16) — small-N mechanism-of-action driver. Embeds the per-variant priors in each per-compound HTML report.
- **`99_screen_library.ipynb`** (step 17) — production runner with `RUN_MODE = library | moa` toggles. Calls these helpers in `moa` mode for each variant of interest.

### Further reading

- **AlphaMissense.** Cheng, J. *et al.* "Accurate proteome-wide missense variant effect prediction with AlphaMissense." *Science* **381**, eadg7492 (2023). [doi:10.1126/science.adg7492](https://doi.org/10.1126/science.adg7492)
- **gnomAD.** Karczewski, K.J. *et al.* "The mutational constraint spectrum quantified from variation in 141,456 humans." *Nature* **581**, 434–443 (2020). [doi:10.1038/s41586-020-2308-7](https://doi.org/10.1038/s41586-020-2308-7) — and the v4 update at [gnomad.broadinstitute.org](https://gnomad.broadinstitute.org/news/2023-11-gnomad-v4-0/).
- **RaSP.** Blaabjerg, L.M. *et al.* "Rapid protein stability prediction using deep learning representations." *eLife* **12**, e82593 (2023). [doi:10.7554/eLife.82593](https://doi.org/10.7554/eLife.82593)
- **Catalysis prediction (out of scope here, for context).** Yan, B. *et al.* "EnzyHTP: Bridging molecular dynamics and quantum mechanics for enzyme engineering." *J. Chem. Theory Comput.* (2022) — for readers who want to know what the right tool looks like for the catalysis-rate prediction this notebook explicitly does not attempt.
"""),
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
