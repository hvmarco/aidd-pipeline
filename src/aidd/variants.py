"""Per-variant computational priors from public databases.

Two lookup helpers, both protein-centric (UniProt + position + alt AA), both
backed by external public data — we never run either model locally; we cache
the answers.

- :func:`alphamissense_score`  — pathogenicity probability (Cheng et al.,
  Science 2023). Resolved against a precomputed table of every possible
  missense substitution in the human SwissProt-canonical proteome.
- :func:`gnomad_frequency`     — allele frequency + per-ancestry breakdown
  + het/hom counts (Karczewski et al., Nature 2020 + the v4 update).
  Queried live against the public gnomAD GraphQL API; per-gene responses
  are cached on disk so repeated notebook runs do not re-hit the server.

These priors are consumed by notebook 07 (calibration / per-variant report)
and by notebooks 08 (mutation analysis, library-side) and 09 (small-N MoA),
which read the lookup output as added columns next to the structural / IFP
diff.

Demo-set scope
--------------
v1 ships a filtered subset for the seven demo-set genes (NAT2, DPYD,
CYP2D6, UGT1A1, KRAS, BRCA1, ESR1) — see :data:`DEMO_SET_UNIPROT_IDS`.
The AlphaMissense upstream is ~5 GB compressed; first-run on a new system
stream-downloads + filters to the demo set (typically a few thousand rows
per protein; low single-digit MB on disk). To extend to additional genes,
add the UniProt accession + gene symbol to :data:`DEMO_SET_UNIPROT_IDS`
and delete the cache to trigger a rebuild.

The gnomAD UniProt-to-gene-symbol resolution is the same dict — extending
beyond the demo set requires the same one-line edit.

Limits / honesty
----------------
- AlphaMissense scores cover SwissProt-canonical isoforms only. Variants
  in alternative isoforms, indels, or splice variants return ``None``
  (not in the table).
- gnomAD's hgvsp annotation is per-transcript; when multiple transcripts
  carry the same protein change, we pick the row with the highest combined
  exome+genome allele count (canonical transcript in practice).
- gnomAD v4 (combined exomes + genomes) is the default dataset.

Upstream
--------
- AlphaMissense — Cheng et al., Science 2023 (DOI: 10.1126/science.adg7492).
  Data: ``https://storage.googleapis.com/dm_alphamissense/``.
- gnomAD       — Karczewski et al., Nature 2020 + v4 update.
  GraphQL endpoint: ``https://gnomad.broadinstitute.org/api``.
"""

from __future__ import annotations

import functools
import gzip
import io
import json
import logging
from pathlib import Path
from typing import Union

import pandas as pd
import requests

PathLike = Union[str, Path]
logger = logging.getLogger("aidd.variants")


# ---------------------------------------------------------------------------
# Demo-set scope
# ---------------------------------------------------------------------------

DEMO_SET_UNIPROT_IDS: dict[str, str] = {
    "P11245": "NAT2",
    "Q12882": "DPYD",
    "P10635": "CYP2D6",
    "P22309": "UGT1A1",
    "P01116": "KRAS",
    "P38398": "BRCA1",
    "P03372": "ESR1",
}

GENE_TO_UNIPROT: dict[str, str] = {v: k for k, v in DEMO_SET_UNIPROT_IDS.items()}


# ---------------------------------------------------------------------------
# AlphaMissense: data location + version pin
# ---------------------------------------------------------------------------

ALPHAMISSENSE_URL = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_aa_substitutions.tsv.gz"
)
ALPHAMISSENSE_VERSION = "v1"
# Bumped after each successful demo-set cache rebuild on a fresh host.
ALPHAMISSENSE_LAST_VERIFIED: str | None = None


# ---------------------------------------------------------------------------
# gnomAD: endpoint + dataset
# ---------------------------------------------------------------------------

GNOMAD_GRAPHQL_URL = "https://gnomad.broadinstitute.org/api"
GNOMAD_DEFAULT_DATASET = "gnomad_r4"


# ---------------------------------------------------------------------------
# Amino-acid 1-letter <-> 3-letter (standard IUPAC)
# ---------------------------------------------------------------------------

_AA_1TO3: dict[str, str] = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}
_AA_3TO1: dict[str, str] = {v: k for k, v in _AA_1TO3.items()}


def _default_cache_dir(subdir: str) -> Path:
    """Default cache root under ``<cwd>/data/cache/<subdir>``.

    Notebooks override this to point at Google Drive on Colab so the
    AlphaMissense cache survives runtime restarts (re-downloading the
    upstream gzipped TSV is the slowest part of nb 07).
    """
    return Path.cwd() / "data" / "cache" / subdir


# ===========================================================================
# AlphaMissense
# ===========================================================================

def alphamissense_score(
    uniprot_id: str,
    position: int,
    alt_aa: str,
    *,
    cache_dir: PathLike | None = None,
) -> float | None:
    """Pathogenicity probability (0..1) for a missense substitution.

    Returns ``None`` when the (uniprot_id, position, alt_aa) tuple is absent
    from the AlphaMissense table. Most common reasons:

    - ``alt_aa`` matches the wild-type at that position (no substitution
      to score).
    - ``position`` is outside the SwissProt canonical isoform.
    - Wrong / unsupported UniProt accession (we ship the demo-set subset;
      see :data:`DEMO_SET_UNIPROT_IDS`).

    Parameters
    ----------
    uniprot_id
        SwissProt accession (e.g. ``"P11245"`` for human NAT2). Must be in
        :data:`DEMO_SET_UNIPROT_IDS` for the v1 cache; raises otherwise.
    position
        1-based residue position in the SwissProt canonical sequence.
    alt_aa
        Substituted amino acid as a 1-letter code (case-insensitive).
    cache_dir
        Override for the AlphaMissense cache directory. When ``None``,
        defaults to ``<cwd>/data/cache/alphamissense/``. Notebooks set
        this to a Drive-backed path on Colab so the (slow) first-run
        download survives runtime restarts.
    """
    if uniprot_id not in DEMO_SET_UNIPROT_IDS:
        raise ValueError(
            f"UniProt {uniprot_id!r} is outside the v1 demo-set cache. "
            f"Add it to DEMO_SET_UNIPROT_IDS in src/aidd/variants.py and "
            f"delete the existing cache to trigger a rebuild."
        )
    df = _load_alphamissense_cache(_resolve_cache_dir(cache_dir, "alphamissense"))
    rows = df[
        (df["uniprot_id"] == uniprot_id)
        & (df["position"] == int(position))
        & (df["alt_aa"] == alt_aa.upper())
    ]
    if rows.empty:
        return None
    return float(rows.iloc[0]["am_pathogenicity"])


def alphamissense_class(
    uniprot_id: str,
    position: int,
    alt_aa: str,
    *,
    cache_dir: PathLike | None = None,
) -> str | None:
    """AlphaMissense's discrete class for a variant.

    One of ``"likely_benign"``, ``"ambiguous"``, ``"likely_pathogenic"``,
    or ``None`` when the variant is absent from the table.
    """
    if uniprot_id not in DEMO_SET_UNIPROT_IDS:
        raise ValueError(
            f"UniProt {uniprot_id!r} is outside the v1 demo-set cache."
        )
    df = _load_alphamissense_cache(_resolve_cache_dir(cache_dir, "alphamissense"))
    rows = df[
        (df["uniprot_id"] == uniprot_id)
        & (df["position"] == int(position))
        & (df["alt_aa"] == alt_aa.upper())
    ]
    if rows.empty:
        return None
    return str(rows.iloc[0]["am_class"])


@functools.lru_cache(maxsize=4)
def _load_alphamissense_cache(cache_dir: Path) -> pd.DataFrame:
    """Load (and build-if-missing) the AlphaMissense demo-set parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / "demo_set.parquet"
    if not parquet_path.exists():
        logger.info(
            "AlphaMissense demo cache not found at %s; downloading + filtering "
            "from %s (this is a one-time ~5 GB stream-download with on-the-fly "
            "demo-set filtering; expect 5-15 min depending on connection).",
            parquet_path, ALPHAMISSENSE_URL,
        )
        _build_alphamissense_demo_cache(parquet_path)
    return pd.read_parquet(parquet_path)


def _build_alphamissense_demo_cache(parquet_path: Path) -> None:
    """Stream the AlphaMissense gzipped TSV, filter to demo-set, write parquet.

    Memory is bounded — we keep only rows whose ``uniprot_id`` is in
    :data:`DEMO_SET_UNIPROT_IDS` (a few thousand per protein). The full
    upstream is never materialised in memory or on disk.
    """
    keep_uniprots = set(DEMO_SET_UNIPROT_IDS.keys())
    rows: list[tuple[str, int, str, str, float, str]] = []

    with requests.get(ALPHAMISSENSE_URL, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) or None
        raw = _wrap_with_progress(resp.raw, total=total, desc="AlphaMissense")
        with gzip.GzipFile(fileobj=raw) as gz, io.TextIOWrapper(gz, encoding="utf-8") as fh:
            for line in fh:
                if not line or line.startswith("#"):
                    continue
                if line.startswith("uniprot_id\t"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                uniprot, variant, score_str, am_class = parts[0], parts[1], parts[2], parts[3]
                if uniprot not in keep_uniprots:
                    continue
                # protein_variant is e.g. 'V2L' (wt + position + alt).
                wt_aa = variant[:1]
                alt_aa = variant[-1:]
                try:
                    pos = int(variant[1:-1])
                    score = float(score_str)
                except ValueError:
                    continue
                rows.append((uniprot, pos, wt_aa, alt_aa, score, am_class))

    df = pd.DataFrame(
        rows,
        columns=["uniprot_id", "position", "wt_aa", "alt_aa", "am_pathogenicity", "am_class"],
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    logger.info(
        "AlphaMissense demo cache written to %s (%d rows across %d proteins).",
        parquet_path, len(df), df["uniprot_id"].nunique(),
    )


# ===========================================================================
# gnomAD
# ===========================================================================

def gnomad_frequency(
    uniprot_id: str,
    position: int,
    alt_aa: str,
    *,
    dataset: str = GNOMAD_DEFAULT_DATASET,
    cache_dir: PathLike | None = None,
) -> dict:
    """Allele frequency + per-ancestry breakdown + het/hom counts.

    Returns a dict with keys:

    - ``found``                — ``True`` iff a gnomAD record matches the
                                 protein change.
    - ``variant_id``           — gnomAD's canonical ``chrom-pos-ref-alt``
                                 ID (or ``None`` when not found).
    - ``hgvsp``                — protein-level HGVS string (e.g.
                                 ``"p.Ile114Thr"``) or ``None``.
    - ``allele_freq_overall``  — combined exome+genome allele frequency
                                 (``ac / an``); ``0.0`` when not found.
    - ``allele_count``,
      ``allele_number``        — combined exome+genome counts (``ac``, ``an``).
    - ``hom_count``            — combined homozygote count.
    - ``populations``          — dict keyed by gnomAD population code
                                 (e.g. ``"nfe"``, ``"afr"``, ``"eas"``,
                                 ``"sas"``, ``"amr"``, ``"asj"``, ``"fin"``,
                                 ``"mid"``, ``"ami"``, ``"oth"``); each value
                                 is ``{"ac": int, "an": int, "af": float,
                                 "hom_count": int}``. Empty dict when
                                 not found.

    Per-gene responses are cached at ``<cache_dir>/<gene>_<dataset>.json``
    so the second variant in the same gene resolves without a round-trip.

    Raises
    ------
    ValueError
        If ``uniprot_id`` is not in :data:`DEMO_SET_UNIPROT_IDS`. v1 ships
        the gene-symbol mapping for the seven demo-set genes only; extend
        the dict to extend coverage.
    RuntimeError
        On HTTP / GraphQL error from the gnomAD endpoint.
    """
    if uniprot_id not in DEMO_SET_UNIPROT_IDS:
        raise ValueError(
            f"UniProt {uniprot_id!r} is not in DEMO_SET_UNIPROT_IDS. "
            f"v1 ships gene-symbol mapping for the seven demo-set genes only "
            f"(NAT2, DPYD, CYP2D6, UGT1A1, KRAS, BRCA1, ESR1); to extend, add "
            f"{uniprot_id} to DEMO_SET_UNIPROT_IDS in src/aidd/variants.py."
        )
    gene = DEMO_SET_UNIPROT_IDS[uniprot_id]
    cache_path = _resolve_cache_dir(cache_dir, "gnomad")
    variants = _load_gnomad_gene_cache(gene, dataset=dataset, cache_dir=cache_path)
    selected = _select_matching_variant(variants, position=int(position), alt_aa=alt_aa.upper())
    return _summarise_gnomad_variant(selected)


def _load_gnomad_gene_cache(
    gene: str,
    *,
    dataset: str,
    cache_dir: Path,
) -> list[dict]:
    """Return the gnomAD variant list for ``gene``, fetching + caching as needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{gene}_{dataset}.json"
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text())
            return payload["variants"]
        except (OSError, json.JSONDecodeError, KeyError):
            cache_file.unlink(missing_ok=True)  # corrupt; refetch
    variants = _fetch_gnomad_variants_by_gene(gene, dataset=dataset)
    cache_file.write_text(
        json.dumps({"gene": gene, "dataset": dataset, "variants": variants}, indent=2)
    )
    return variants


def _fetch_gnomad_variants_by_gene(gene: str, *, dataset: str) -> list[dict]:
    """One GraphQL round-trip: every variant in ``gene`` for ``dataset``."""
    query = """
    query GeneVariants($gene_symbol: String!, $dataset: DatasetId!) {
      gene(gene_symbol: $gene_symbol, reference_genome: GRCh38) {
        variants(dataset: $dataset) {
          variant_id
          hgvsp
          consequence
          exome {
            ac
            an
            ac_hom
            populations { id ac an ac_hom }
          }
          genome {
            ac
            an
            ac_hom
            populations { id ac an ac_hom }
          }
        }
      }
    }
    """
    payload = {"query": query, "variables": {"gene_symbol": gene, "dataset": dataset}}
    resp = requests.post(GNOMAD_GRAPHQL_URL, json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    if "errors" in body and body["errors"]:
        raise RuntimeError(f"gnomAD GraphQL error for gene {gene!r}: {body['errors']}")
    g = (body.get("data") or {}).get("gene")
    if g is None or g.get("variants") is None:
        return []
    return list(g["variants"])


def _select_matching_variant(
    variants: list[dict],
    *,
    position: int,
    alt_aa: str,
) -> dict | None:
    """Pick the variant whose hgvsp matches (position, alt_aa).

    Variants without hgvsp (synonymous, intronic, splice, etc.) are skipped.
    When multiple transcripts carry the same protein change, prefer the row
    with the highest combined exome+genome allele count — usually the
    canonical transcript.
    """
    alt_3 = _AA_1TO3.get(alt_aa)
    if alt_3 is None:
        raise ValueError(f"alt_aa {alt_aa!r} is not a standard 1-letter AA code.")
    matches: list[tuple[int, dict]] = []
    for v in variants:
        hgvsp = v.get("hgvsp") or ""
        if not hgvsp.startswith("p.") or len(hgvsp) < 8:
            continue
        wt_3 = hgvsp[2:5]
        alt_3_v = hgvsp[-3:]
        try:
            pos_v = int(hgvsp[5:-3])
        except ValueError:
            continue
        if pos_v != position or alt_3_v != alt_3:
            continue
        if wt_3 not in _AA_3TO1:
            continue
        ex_ac = ((v.get("exome") or {}).get("ac")) or 0
        gn_ac = ((v.get("genome") or {}).get("ac")) or 0
        matches.append((ex_ac + gn_ac, v))
    if not matches:
        return None
    matches.sort(key=lambda t: t[0], reverse=True)
    return matches[0][1]


def _summarise_gnomad_variant(variant: dict | None) -> dict:
    """Reduce a gnomAD variant payload to the columns reported in nb 07."""
    if variant is None:
        return {
            "found": False,
            "variant_id": None,
            "hgvsp": None,
            "allele_freq_overall": 0.0,
            "allele_count": 0,
            "allele_number": 0,
            "hom_count": 0,
            "populations": {},
        }
    exome = variant.get("exome") or {}
    genome = variant.get("genome") or {}
    ex_ac = exome.get("ac") or 0
    gn_ac = genome.get("ac") or 0
    ex_an = exome.get("an") or 0
    gn_an = genome.get("an") or 0
    ex_hom = exome.get("ac_hom") or 0
    gn_hom = genome.get("ac_hom") or 0
    total_ac = ex_ac + gn_ac
    total_an = ex_an + gn_an
    total_hom = ex_hom + gn_hom
    af = (total_ac / total_an) if total_an else 0.0

    populations: dict[str, dict] = {}
    for source_pops in (exome.get("populations") or [], genome.get("populations") or []):
        for p in source_pops:
            pid = p.get("id")
            if not pid:
                continue
            cur = populations.setdefault(pid, {"ac": 0, "an": 0, "hom_count": 0})
            cur["ac"] += p.get("ac") or 0
            cur["an"] += p.get("an") or 0
            cur["hom_count"] += p.get("ac_hom") or 0
    for d in populations.values():
        d["af"] = (d["ac"] / d["an"]) if d["an"] else 0.0

    return {
        "found": True,
        "variant_id": variant.get("variant_id"),
        "hgvsp": variant.get("hgvsp"),
        "allele_freq_overall": af,
        "allele_count": total_ac,
        "allele_number": total_an,
        "hom_count": total_hom,
        "populations": populations,
    }


# ===========================================================================
# Helpers
# ===========================================================================

def _resolve_cache_dir(cache_dir: PathLike | None, subdir: str) -> Path:
    return Path(cache_dir) if cache_dir is not None else _default_cache_dir(subdir)


def _wrap_with_progress(raw, *, total: int | None, desc: str):
    """Wrap a file-like object with a tqdm progress bar on ``read``.

    Falls back to the unwrapped object when tqdm is unavailable so a
    minimal install (no Jupyter / no tqdm) still works.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return raw
    return tqdm.wrapattr(raw, "read", total=total, miniters=1, desc=desc, unit="B", unit_scale=True)
