"""Comparison helpers for WT-vs-variant pipeline output diffs.

Six helpers (five public + one private) extracted from notebook 08's inline
definitions in step 16 commit 2/4. Consumed by notebook 08 (library-side
WT-vs-mutant comparison), notebook 09 (small-N MoA driver), and notebook 99
(production runner) ``RUN_MODE = "moa"``. Same source of truth across all
three callers.

- :func:`pocket_residues_within` - residue numbers within a Cα-Cα shell of
  an anchor residue.
- :func:`ca_rmsd_pocket`         - Cα-RMSD restricted to a pocket-residue
  set.
- :func:`ifp_diff`               - gained / lost / preserved
  ``(residue, interaction_type)`` pairs between two ProLIF IFP DataFrames.
- :func:`shortlist_diff`         - both / wt_only / mut_only compound
  classification + per-compound rank deltas between two consensus
  shortlists.
- :func:`render_mutation_html`   - self-contained HTML report with embedded
  py3Dmol viewer (library-side WT-vs-variant comparison from nb 08).
- :func:`render_moa_html`        - self-contained HTML report for the
  per-(compound x variant) small-N MoA layout (nb 09). Sibling renderer
  to :func:`render_mutation_html` with a different section layout
  (no shortlist diff at N=1; out-of-scope splice / promoter branch
  produces a priors-only report).
- :func:`summarise_mutation`     - top-level aggregator: pulls nb 07 priors
  + structural + IFP + shortlist diffs into one dict ready for HTML
  rendering.

Why this module exists
----------------------
Step 15 (notebook 08 builder) defined these helpers inline because that
notebook was the first consumer. Step 16 introduced a second consumer
(notebook 09) and a third (notebook 99 ``RUN_MODE = "moa"``), so a single
source of truth replaces what would otherwise be 3x verbatim copies. The
Q1 reuse-mechanism decision (extract vs verbatim-copy vs builder-side
source-of-truth) was made in favour of extract - see commit 2/4's message.

Behaviour parity
----------------
Extraction is byte-for-byte verbatim from notebook 08's inline cells (apart
from removing the surrounding ``code(title=..., source=<triple-quoted>)``
wrapper and dedenting one indent level). The hardcoded
``compute_consensus`` parameters inside :func:`summarise_mutation`
(``top_fraction=0.30``, ``filter_method="rank_product_topk"``,
``rescorer_col="gnina_cnn_affinity"``, ``boltz_col="boltz_affinity"``,
etc.) are preserved as-is - parametrising them is deferred until a
caller actually needs different values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer

from aidd.consensus import compute_consensus
from aidd.ifp import compute_ifp
from aidd.stability import rasp_ddg
from aidd.variants import alphamissense_class, alphamissense_score, gnomad_frequency

logger = logging.getLogger("aidd.mutation")


def pocket_residues_within(pdb_path: Path, anchor_residue: int, radius_a: float = 8.0,
                           chain: str = "A") -> set[int]:
    """Residue numbers whose any-heavy-atom Cα is within ``radius_a`` Å of the anchor residue's Cα.

    Used to define the pocket-restricted shell for Cα-RMSD computation.
    The 8 Å default captures the local environment around the variant
    (the second shell of residues touching the binding pocket).
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", str(pdb_path))
    anchor_xyz = None
    ca_xyz: dict[int, np.ndarray] = {}
    for model in struct:
        for ch in model:
            if ch.id != chain:
                continue
            for residue in ch:
                if "CA" not in residue:
                    continue
                resnum = residue.id[1]
                xyz = np.array(residue["CA"].get_coord())
                ca_xyz[resnum] = xyz
                if resnum == anchor_residue:
                    anchor_xyz = xyz
        break
    if anchor_xyz is None:
        raise ValueError(f"anchor residue {anchor_residue} not in chain {chain!r} of {pdb_path}")
    return {rn for rn, xyz in ca_xyz.items() if np.linalg.norm(xyz - anchor_xyz) <= radius_a}


def ca_rmsd_pocket(pdb_wt: Path, pdb_mut: Path, pocket_residues: set[int],
                   chain: str = "A") -> dict:
    """Cα-RMSD restricted to a pocket-residue set; uses Superimposer on matched-by-resnum."""
    parser = PDBParser(QUIET=True)
    def _ca_by_resnum(path: Path) -> dict[int, "Atom"]:
        struct = parser.get_structure("x", str(path))
        for model in struct:
            for ch in model:
                if ch.id != chain:
                    continue
                return {r.id[1]: r["CA"] for r in ch if "CA" in r}
        return {}
    wt_ca = _ca_by_resnum(pdb_wt)
    mut_ca = _ca_by_resnum(pdb_mut)
    matched = sorted(r for r in pocket_residues if r in wt_ca and r in mut_ca)
    if len(matched) < 3:
        raise ValueError(
            f"Only {len(matched)} pocket residues match between {pdb_wt.name} and "
            f"{pdb_mut.name}; need >= 3 for a stable superposition."
        )
    sup = Superimposer()
    sup.set_atoms([wt_ca[r] for r in matched], [mut_ca[r] for r in matched])
    return {
        "rmsd_pocket_A":    float(sup.rms),
        "n_matched":        len(matched),
        "matched_residues": matched,
    }


def ifp_diff(ifp_wt_df: pd.DataFrame, ifp_mut_df: pd.DataFrame) -> dict:
    """Aggregate per-pose IFP to a per-(protein_residue, interaction_type) set; diff WT vs mutant.

    An interaction is "on" for a genotype if any pose in that genotype's
    IFP DataFrame has it. Returns three lists: gained (in mutant only),
    lost (in WT only), preserved. Each entry is a tuple
    ``(residue_label, interaction_type)``.
    """
    def _aggregate(df: pd.DataFrame) -> set[tuple[str, str]]:
        if df is None or df.empty:
            return set()
        # ProLIF's tidy DataFrame has a MultiIndex on columns:
        # (ligand_resid, protein_resid, interaction_type). True/False per pose row.
        any_pose = df.any(axis=0)
        return {(str(prot_resid), str(interaction))
                for (_lig, prot_resid, interaction), present in any_pose.items()
                if bool(present)}
    wt_set  = _aggregate(ifp_wt_df)
    mut_set = _aggregate(ifp_mut_df)
    return {
        "gained":    sorted(mut_set - wt_set),
        "lost":      sorted(wt_set - mut_set),
        "preserved": sorted(wt_set & mut_set),
        "n_wt":      len(wt_set),
        "n_mut":     len(mut_set),
    }


def shortlist_diff(shortlist_wt: pd.DataFrame, shortlist_mut: pd.DataFrame,
                   id_col: str = "compound_id",
                   rank_col: str = "consensus_rank") -> dict:
    """Classify shortlist compounds as both / wt_only / mut_only and report rank deltas."""
    wt_ranks  = dict(zip(shortlist_wt[id_col].astype(str), shortlist_wt[rank_col]))
    mut_ranks = dict(zip(shortlist_mut[id_col].astype(str), shortlist_mut[rank_col]))
    wt_ids  = set(wt_ranks.keys())
    mut_ids = set(mut_ranks.keys())
    both_ids = sorted(wt_ids & mut_ids, key=lambda c: wt_ranks[c])
    both = pd.DataFrame([
        {id_col: c,
          "wt_rank":     wt_ranks[c],
          "mut_rank":    mut_ranks[c],
          "rank_delta":  mut_ranks[c] - wt_ranks[c]}
        for c in both_ids
    ])
    wt_only_ids  = sorted(wt_ids  - mut_ids, key=lambda c: wt_ranks[c])
    mut_only_ids = sorted(mut_ids - wt_ids,  key=lambda c: mut_ranks[c])
    wt_only  = pd.DataFrame([
        {id_col: c, "wt_rank":  wt_ranks[c]}  for c in wt_only_ids
    ])
    mut_only = pd.DataFrame([
        {id_col: c, "mut_rank": mut_ranks[c]} for c in mut_only_ids
    ])
    return {
        "both":      both,
        "wt_only":   wt_only,
        "mut_only":  mut_only,
        "n_both":    len(both_ids),
        "n_wt_only": len(wt_only_ids),
        "n_mut_only":len(mut_only_ids),
    }


def _df_to_html_table(df: pd.DataFrame, *, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "<p><i>(empty)</i></p>"
    return df.head(max_rows).to_html(index=False, classes="aidd-table", border=0)


def render_mutation_html(summary: dict, out_path: Path, *, embed_3d: bool = True) -> Path:
    """Self-contained HTML report with embedded interactive py3Dmol.

    Sections mirror ``PROJECT_PROPOSAL.md`` §8 'Outputs':

    - Header (target, variant, date)
    - Variant priors (AlphaMissense + gnomAD + RaSP)
    - Structural diff (pocket-restricted + full-protein Cα-RMSD)
    - IFP diff (gained / lost / preserved)
    - Shortlist diff (both / wt_only / mut_only)
    - Optional embedded 3D viewer of WT + mutant overlaid
    - Bottom-line caveat
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title_line = (
        f"{summary['target'].upper()} WT vs {summary['variant']} "
        f"(UniProt {summary['uniprot']}, position {summary['position']})"
    )

    am = summary["priors"]["alphamissense"]
    gn = summary["priors"]["gnomad"]
    rp = summary["priors"]["rasp"]
    priors_html = f"""
      <table class="aidd-table"><thead>
        <tr><th>Prior</th><th>Value</th><th>Class / note</th></tr></thead>
      <tbody>
        <tr><td>AlphaMissense pathogenicity</td><td>{am.get('score')}</td>
            <td>{am.get('class', '-') or '-'}</td></tr>
        <tr><td>gnomAD allele frequency (overall)</td>
            <td>{gn.get('allele_freq_overall')}</td>
            <td>flipped: {gn.get('reference_flipped')}</td></tr>
        <tr><td>RaSP ΔΔG (kcal/mol)</td><td>{rp.get('ddg')}</td>
            <td>positive = destabilising</td></tr>
      </tbody></table>"""

    structural_html = f"""
      <table class="aidd-table"><thead>
        <tr><th>Metric</th><th>Value (Å)</th><th>Notes</th></tr></thead>
      <tbody>
        <tr><td>Pocket-restricted Cα-RMSD ({summary['structural']['radius_a']} Å shell)</td>
            <td>{summary['structural']['rmsd_pocket_A']:.3f}</td>
            <td>{summary['structural']['n_matched']} matched residues</td></tr>
        <tr><td>Full-protein Cα-RMSD</td>
            <td>{summary['structural']['rmsd_full_A']:.3f}</td>
            <td>Cα atoms matched by residue number</td></tr>
      </tbody></table>"""

    ifp = summary["ifp"]
    ifp_html = f"""
      <table class="aidd-table"><thead>
        <tr><th>Category</th><th>n</th><th>Examples</th></tr></thead>
      <tbody>
        <tr><td>Gained (mutant only)</td><td>{len(ifp['gained'])}</td>
            <td>{', '.join(str(t) for t in ifp['gained'][:5])}{'...' if len(ifp['gained']) > 5 else ''}</td></tr>
        <tr><td>Lost (WT only)</td><td>{len(ifp['lost'])}</td>
            <td>{', '.join(str(t) for t in ifp['lost'][:5])}{'...' if len(ifp['lost']) > 5 else ''}</td></tr>
        <tr><td>Preserved</td><td>{len(ifp['preserved'])}</td>
            <td>{', '.join(str(t) for t in ifp['preserved'][:5])}{'...' if len(ifp['preserved']) > 5 else ''}</td></tr>
      </tbody></table>"""

    sl = summary["shortlist"]
    shortlist_html = f"""
      <h3>Compounds in both shortlists (with rank deltas)</h3>
      {_df_to_html_table(sl['both'])}
      <h3>WT-only ({sl['n_wt_only']})</h3>
      {_df_to_html_table(sl['wt_only'])}
      <h3>Mutant-only ({sl['n_mut_only']})</h3>
      {_df_to_html_table(sl['mut_only'])}"""

    viewer_html = summary.get("viewer_html", "") if embed_3d else ""

    css = """
      body { font-family: -apple-system, system-ui, sans-serif; max-width: 1100px;
             margin: 24px auto; padding: 0 16px; color: #222; }
      h1 { font-size: 1.5em; margin-bottom: 4px; }
      h2 { font-size: 1.15em; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
      h3 { font-size: 1.0em; margin-top: 18px; }
      .aidd-table { border-collapse: collapse; margin: 8px 0; }
      .aidd-table th, .aidd-table td { padding: 6px 12px; border: 1px solid #ddd; text-align: left; }
      .aidd-table thead { background: #f4f4f6; }
      .caveat { background: #fff8dc; padding: 10px 14px; border-left: 4px solid #d4a017;
                margin-top: 28px; font-size: 0.95em; }"""

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title_line}</title>
<style>{css}</style></head>
<body>
<h1>{title_line}</h1>
<p><i>Generated by aidd-pipeline notebook 08 (mutation analysis).</i></p>

<h2>1. Variant context (notebook 07 priors)</h2>
{priors_html}

<h2>2. Structural diff</h2>
{structural_html}

<h2>3. Interaction-fingerprint diff</h2>
{ifp_html}

<h2>4. Shortlist diff</h2>
{shortlist_html}

<h2>5. 3-D viewer (WT vs mutant overlay)</h2>
{viewer_html if embed_3d else "<p><i>(3D viewer disabled; pass embed_3d=True to enable.)</i></p>"}

<div class="caveat">
<b>Computational hypothesis-generation only.</b> The numbers above describe binding-site fit
and structural change; they do not predict catalytic rate (k_cat, K_m) for enzymes, and they do
not predict patient-level clinical outcome. Wet-lab validation is required before any clinical
inference. For pharmacogene enzymes (DPYD, NAT2, CYP2D6, UGT1A1), RaSP ΔΔG captures one mechanism
contributing to reduced metabolism (protein-stability-driven abundance loss); catalysis rate per se
needs QM/MM-class methods (out of scope; see notebook 08 recap for pointers).
</div>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def summarise_mutation(target: str, variant: str, uniprot: str, position: int,
                       wt_aa: str, mut_aa: str, *,
                       wt_dir: Path, mut_dir: Path,
                       wt_name: str, mut_name: str,
                       pocket_radius_a: float = 8.0,
                       alphamissense_cache: Path | None = None,
                       gnomad_cache: Path | None = None,
                       rasp_cache: Path | None = None,
                       viewer_html: str = "") -> dict:
    """Top-level aggregator: nb 07 priors + structural + IFP + shortlist diffs.

    The IFP + shortlist diffs assume the genotype directories contain
    ``{fold, docking, scoring, boltz}`` subtrees per the per-step
    REQUIRED_ARTEFACTS convention.
    """
    am_score = alphamissense_score(uniprot, position, mut_aa, cache_dir=alphamissense_cache)
    am_class = alphamissense_class(uniprot, position, mut_aa, cache_dir=alphamissense_cache)
    gn = gnomad_frequency(uniprot, position, mut_aa, wt_aa=wt_aa, cache_dir=gnomad_cache)
    rasp = rasp_ddg(uniprot, position, wt_aa, mut_aa, cache_dir=rasp_cache)

    wt_pdb  = wt_dir  / "fold" / f"{wt_name}_best.pdb"
    mut_pdb = mut_dir / "fold" / f"{mut_name}_best.pdb"

    pocket_set = pocket_residues_within(wt_pdb, position, radius_a=pocket_radius_a)
    pocket_metric = ca_rmsd_pocket(wt_pdb, mut_pdb, pocket_set)

    # Full-protein Cα-RMSD via the matched-by-resnum Superimposer (handles
    # equal-length structures naturally since the stub mutation only edits
    # I560's sidechain atoms, leaving Cα counts identical).
    parser = PDBParser(QUIET=True)
    def _all_ca(path: Path):
        struct = parser.get_structure("x", str(path))
        for model in struct:
            for ch in model:
                if ch.id != "A":
                    continue
                return {r.id[1]: r["CA"] for r in ch if "CA" in r}
        return {}
    wt_all  = _all_ca(wt_pdb)
    mut_all = _all_ca(mut_pdb)
    common = sorted(set(wt_all) & set(mut_all))
    sup_full = Superimposer()
    sup_full.set_atoms([wt_all[r] for r in common], [mut_all[r] for r in common])
    rmsd_full = float(sup_full.rms)

    # IFPs (compute_ifp returns a DataFrame keyed by pose x interaction).
    wt_poses  = wt_dir  / "docking" / "poses.sdf"
    mut_poses = mut_dir / "docking" / "poses.sdf"
    ifp_wt  = compute_ifp(wt_pdb,  wt_poses)
    ifp_mut = compute_ifp(mut_pdb, mut_poses)
    ifp_d = ifp_diff(ifp_wt, ifp_mut)

    # Shortlists: build by running compute_consensus on each genotype's
    # scored_poses + boltz tables (non-ERK2 lane choice: gnina_cnn_affinity + boltz_affinity).
    def _shortlist(genotype_dir: Path) -> pd.DataFrame:
        scored = pd.read_parquet(genotype_dir / "scoring" / "scored_poses.parquet")
        boltz  = pd.read_csv(   genotype_dir / "boltz"   / "affinity.csv")
        result = compute_consensus(
            scored, boltz,
            top_fraction=0.30,                          # accept top-30% so the stub fixture's small
                                                        # cohort (11 compounds) produces a visible
                                                        # shortlist of 3-4 entries.
            rescorer_col="gnina_cnn_affinity",          # the "second lane" - see nb 08 §4 markdown.
            boltz_col="boltz_affinity",
            id_col="compound_id",
            rescorer_lower_is_better=False,
            boltz_lower_is_better=True,                 # log10(IC50) µM
            filter_method="rank_product_topk",
        )
        return result["shortlist"]
    shortlist_wt  = _shortlist(wt_dir)
    shortlist_mut = _shortlist(mut_dir)
    sl_d = shortlist_diff(shortlist_wt, shortlist_mut)

    return {
        "target":   target,
        "variant":  variant,
        "uniprot":  uniprot,
        "position": position,
        "priors": {
            "alphamissense": {"score": am_score, "class": am_class},
            "gnomad":        gn,
            "rasp":          {"ddg": rasp},
        },
        "structural": {
            "rmsd_pocket_A": pocket_metric["rmsd_pocket_A"],
            "rmsd_full_A":   rmsd_full,
            "n_matched":     pocket_metric["n_matched"],
            "radius_a":      pocket_radius_a,
            "pocket_size":   len(pocket_set),
        },
        "ifp":       ifp_d,
        "shortlist": sl_d,
        "viewer_html": viewer_html,
    }


def render_moa_html(record: dict, out_path: Path) -> Path:
    """Per-(compound x variant) HTML report for notebook 09 (small-N MoA).

    Sections mirror ``MECHANISM_OF_ACTION_SCOPE.md`` "Output format" /
    "Per-compound HTML report":

    1. Header (compound name + SMILES + target + variant identifier).
    2. 3D pose viewer (embedded py3Dmol HTML supplied by the caller).
    3. Scores + priors table (single-row; nb 07 priors when available).
    4. Key interactions (ProLIF (residue, interaction-type) list for the
       pose).
    5. WT vs variant diff (only when ``record["structural"]`` is present;
       absent for WT records and out-of-scope branches).
    6. Bottom-line caveat (always).

    For out-of-scope branches (splice / promoter variants the structural
    pipeline cannot model) sections 2 / 4 / 5 are replaced with a
    "structural consequence cannot be modelled" block; sections 1 / 3 / 6
    still render (priors from nb 07 work for splice / promoter variants
    at the gene level even when the protein-structural diff doesn't).

    Parameters
    ----------
    record
        Per-(compound x variant) record. Required keys: ``target_name``,
        ``uniprot``, ``variant_label``, ``variant_handling``,
        ``compound_name``, ``smiles``, ``priors``. Optional keys
        (presence triggers the corresponding section): ``scores``,
        ``ifp_per_residue``, ``structural``, ``ifp_diff``,
        ``viewer_html``, ``out_of_scope_reason``.
    out_path
        Output path for the standalone HTML file. Parent directories are
        created if missing.

    Returns
    -------
    Path
        The written HTML file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target          = record["target_name"]
    uniprot         = record["uniprot"]
    variant         = record["variant_label"]
    handling        = record.get("variant_handling", "unknown")
    compound        = record["compound_name"]
    smiles          = record.get("smiles", "") or ""

    is_wt           = variant.strip().upper() == "WT"
    is_out_of_scope = handling.startswith("out_of_scope_")
    is_boltz_failed = handling.startswith("boltz_failed_")
    # Pose viewer + IFP + structural diff are all skipped when the structural
    # pipeline didn't produce a pose -- either because the variant class is
    # out-of-scope (splice / promoter), OR because Boltz-2 failed at runtime
    # (OOM / input rejection / etc.) and there is no .cif to view.
    is_pose_unavailable = is_out_of_scope or is_boltz_failed

    if is_wt:
        title = f"{compound} bound to {target} (UniProt {uniprot}, WT baseline)"
    else:
        title = f"{compound} bound to {target} {variant} (UniProt {uniprot})"

    # --- Section 3: scores + priors table ---------------------------------
    scores = record.get("scores", {}) or {}
    priors = record.get("priors", {}) or {}
    am = priors.get("alphamissense", {}) or {}
    gn = priors.get("gnomad", {}) or {}
    rp = priors.get("rasp", {}) or {}

    def _fmt(v) -> str:
        if v is None:
            return "-"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    rasp_note = (
        "positive = destabilising"
        if rp.get("ddg") is not None
        else "not covered by RaSP cache (see nb 07)"
    )
    score_rows = [
        ("Boltz-2 affinity (log10 IC50 uM; lower=stronger)",
         scores.get("boltz_affinity"),
         "negative = predicted binder"),
        ("Boltz-2 binder probability",
         scores.get("boltz_affinity_probability"),
         "0-1; higher = more likely to bind"),
        ("gnina CNN-affinity",
         scores.get("gnina_cnn_affinity"),
         "higher = stronger predicted binding"),
        ("PoseBusters pass",
         scores.get("posebusters_pass"),
         "geometry sanity gate"),
        ("AlphaMissense pathogenicity (nb 07)",
         am.get("score"),
         (am.get("class") or "-")),
        ("gnomAD allele frequency overall (nb 07)",
         gn.get("allele_freq_overall"),
         f"found: {gn.get('found', '-')}, flipped: {gn.get('reference_flipped', '-')}"),
        ("RaSP DDG (kcal/mol; nb 07)",
         rp.get("ddg"),
         rasp_note),
    ]
    scores_html = (
        '<table class="aidd-table">'
        '<thead><tr><th>Metric</th><th>Value</th><th>Note</th></tr></thead>'
        '<tbody>'
        + ''.join(
            f"<tr><td>{label}</td><td>{_fmt(value)}</td><td>{note}</td></tr>"
            for label, value, note in score_rows
        )
        + '</tbody></table>'
    )

    # --- Pose-unavailable branch (out-of-scope OR Boltz-2 runtime failure) ---
    if is_pose_unavailable:
        if is_boltz_failed:
            error_class = handling[len("boltz_failed_"):] or "unknown"
            log_tail = record.get("error_detail") or ""
            log_tail_block = (
                f"<details><summary>Boltz log tail (last lines)</summary>"
                f"<pre>{log_tail}</pre></details>"
                if log_tail else ""
            )
            section_2_title = "2. Structural modelling: Boltz-2 prediction failed"
            banner_html = (
                f'<div class="boltz-fail">'
                f'<b>Boltz-2 affinity unavailable for this pair</b>: '
                f'<code>{error_class}</code>. The variant priors below '
                f'(AlphaMissense + gnomAD + RaSP) are independent of Boltz '
                f'and load correctly. Re-run on a larger-memory GPU (A100 / '
                f'L4) or with reduced Boltz settings to fill the Boltz '
                f'affinity row for this pair.'
                f'{log_tail_block}'
                f'</div>'
            )
        else:
            reason = (
                record.get("out_of_scope_reason")
                or "This variant class cannot be modelled by the structural pipeline."
            )
            section_2_title = "2. Structural modelling: out of scope"
            banner_html = f'<div class="oos">{reason}</div>'

        body = f"""
        <h2>1. Compound + variant</h2>
        <p><b>Target:</b> {target} (UniProt {uniprot})</p>
        <p><b>Variant:</b> {variant} (handling: {handling})</p>
        <p><b>Compound:</b> {compound}</p>
        <p><b>SMILES:</b> <code>{smiles}</code></p>

        <h2>{section_2_title}</h2>
        {banner_html}

        <h2>3. Scores + nb 07 priors</h2>
        {scores_html}
        """
    else:
        viewer_html = record.get("viewer_html") or "<p><i>(3D viewer not embedded.)</i></p>"

        ifp_pose = record.get("ifp_per_residue") or []
        if ifp_pose:
            interaction_rows = ''.join(
                f"<tr><td>{res}</td><td>{itype}</td></tr>"
                for res, itype in sorted(ifp_pose)
            )
            interactions_html = (
                '<table class="aidd-table">'
                '<thead><tr><th>Residue</th><th>Interaction type</th></tr></thead>'
                f'<tbody>{interaction_rows}</tbody></table>'
            )
        else:
            interactions_html = "<p><i>(No interactions detected for this pose.)</i></p>"

        diff_html = ""
        structural = record.get("structural")
        if not is_wt and structural:
            diff_html += f"""
            <h2>5. WT vs variant diff</h2>
            <h3>Structural</h3>
            <table class="aidd-table">
              <thead><tr><th>Metric</th><th>Value (A)</th><th>Notes</th></tr></thead>
              <tbody>
                <tr><td>Pocket-restricted Ca-RMSD ({structural.get('radius_a', 8.0)} A shell)</td>
                    <td>{_fmt(structural.get('rmsd_pocket_A'))}</td>
                    <td>{structural.get('n_matched', 0)} matched residues</td></tr>
                <tr><td>Full-protein Ca-RMSD</td>
                    <td>{_fmt(structural.get('rmsd_full_A'))}</td>
                    <td>Ca atoms matched by residue number</td></tr>
              </tbody>
            </table>"""
        ifp_diff_record = record.get("ifp_diff")
        if not is_wt and ifp_diff_record:
            gained_n = len(ifp_diff_record.get("gained", []))
            lost_n   = len(ifp_diff_record.get("lost",   []))
            pres_n   = len(ifp_diff_record.get("preserved", []))
            def _ex(pairs, n=5):
                return (', '.join(str(t) for t in pairs[:n]) +
                        ('...' if len(pairs) > n else ''))
            diff_html += f"""
            <h3>Interaction-fingerprint diff (WT vs {variant})</h3>
            <table class="aidd-table">
              <thead><tr><th>Category</th><th>n</th><th>Examples</th></tr></thead>
              <tbody>
                <tr><td>Gained (variant only)</td><td>{gained_n}</td>
                    <td>{_ex(ifp_diff_record.get('gained', []))}</td></tr>
                <tr><td>Lost (WT only)</td><td>{lost_n}</td>
                    <td>{_ex(ifp_diff_record.get('lost', []))}</td></tr>
                <tr><td>Preserved</td><td>{pres_n}</td>
                    <td>{_ex(ifp_diff_record.get('preserved', []))}</td></tr>
              </tbody>
            </table>"""

        body = f"""
        <h2>1. Compound + variant</h2>
        <p><b>Target:</b> {target} (UniProt {uniprot})</p>
        <p><b>Variant:</b> {variant} (handling: {handling})</p>
        <p><b>Compound:</b> {compound}</p>
        <p><b>SMILES:</b> <code>{smiles}</code></p>

        <h2>2. 3-D pose viewer</h2>
        {viewer_html}

        <h2>3. Scores + nb 07 priors</h2>
        {scores_html}

        <h2>4. Key interactions (this pose)</h2>
        {interactions_html}

        {diff_html}
        """

    caveat_html = """
    <div class="caveat">
    <b>Computational hypothesis-generation only.</b> Numbers above describe predicted
    binding-site fit and structural change; they do not predict catalytic rate (k_cat, K_m)
    for enzymes, nor patient-level clinical outcome. Wet-lab validation is required before
    any clinical inference. For pharmacogene enzymes (DPYD, NAT2, CYP2D6, UGT1A1), RaSP
    DDG captures one mechanism contributing to reduced metabolism (protein-stability-driven
    abundance loss); catalysis rate per se needs QM/MM-class methods (out of scope; see
    notebook 09 recap for pointers).
    </div>
    """

    css = """
      body { font-family: -apple-system, system-ui, sans-serif; max-width: 1100px;
             margin: 24px auto; padding: 0 16px; color: #222; }
      h1 { font-size: 1.5em; margin-bottom: 4px; }
      h2 { font-size: 1.15em; margin-top: 28px; border-bottom: 1px solid #ddd;
           padding-bottom: 4px; }
      h3 { font-size: 1.0em; margin-top: 18px; }
      code { background: #f6f6f8; padding: 1px 6px; border-radius: 3px; }
      .aidd-table { border-collapse: collapse; margin: 8px 0; }
      .aidd-table th, .aidd-table td { padding: 6px 12px; border: 1px solid #ddd;
                                       text-align: left; }
      .aidd-table thead { background: #f4f4f6; }
      .caveat { background: #fff8dc; padding: 10px 14px; border-left: 4px solid #d4a017;
                margin-top: 28px; font-size: 0.95em; }
      .oos { background: #f0f4ff; padding: 10px 14px; border-left: 4px solid #4060a8;
             margin-top: 8px; font-size: 0.95em; }
      .boltz-fail { background: #fff4e0; padding: 10px 14px; border-left: 4px solid #c47a00;
             margin-top: 8px; font-size: 0.95em; }
      .boltz-fail details { margin-top: 8px; }
      .boltz-fail pre { background: #fffaf0; padding: 8px; border: 1px solid #e0c890;
             border-radius: 3px; font-size: 0.85em; overflow-x: auto; }
    """

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head>
<body>
<h1>{title}</h1>
<p><i>Generated by aidd-pipeline notebook 09 (small-N mechanism of action).</i></p>

{body}

{caveat_html}
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


__all__ = [
    "pocket_residues_within",
    "ca_rmsd_pocket",
    "ifp_diff",
    "shortlist_diff",
    "render_mutation_html",
    "render_moa_html",
    "summarise_mutation",
]
