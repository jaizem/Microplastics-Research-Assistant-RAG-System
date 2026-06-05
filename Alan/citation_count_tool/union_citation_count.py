"""Union citation count across free scholarly citation networks (OpenAlex + Semantic Scholar).

This module exposes a single LangChain tool, ``get_union_citation_count``, that
reports how many *distinct* papers cite a given work, taken as the UNION of the
citing sets from OpenAlex and Semantic Scholar.

WHY A UNION (and not a sum)
---------------------------
The same citation usually appears in both databases, so adding each source's
``cited_by_count`` would massively double-count. A true union is computed at the
level of *citing papers*: fetch the set of works that cite the target from each
network, reconcile their identities, and count the distinct ones. The result
sits between "the larger single source" and "the sum of both".

Citing papers are reconciled across sources by DOI (the only reliable
cross-source anchor). Citing papers without a DOI fall back to a normalized
``(title, year)`` key, which is less reliable -- the count of such fallbacks is
emitted in the audit log. Semantic Scholar is only included when the *target*
paper has a DOI; otherwise the count degrades to OpenAlex-only (also logged).

INSTALL
-------
    pip install langchain-core requests

QUICK START
-----------
    import logging
    logging.basicConfig(level=logging.INFO)   # see the per-source audit breakdown

    from union_citation_count import get_union_citation_count

    # As a LangChain tool (DOI, OpenAlex ID, or a full title):
    n = get_union_citation_count.invoke("10.1038/nature14539")
    n = get_union_citation_count.invoke("W2741809807")
    n = get_union_citation_count.invoke("Deep learning")   # title must match confidently

    # Bind it to an agent like any other tool:
    #   llm.bind_tools([get_union_citation_count])

ACCEPTED INPUTS
---------------
- DOI:           "10.1038/nature14539", "doi:10...", or a doi.org URL
- OpenAlex ID:   "W2741809807"
- Title:         free text; only accepted when ONE candidate matches above
                 ``TITLE_THRESHOLD`` similarity, otherwise a ValueError is raised
                 (titles are ambiguous; prefer a DOI to avoid a wrong-paper count).

RETURN VALUE
------------
An ``int``: the number of distinct citing papers across both sources. The
per-source breakdown (each source's count, overlap, source-only counts,
title-only fallback count, and whether Semantic Scholar was used) is written via
``logging.info`` immediately before the return, so the bare int stays clean
while the reconciliation detail remains auditable.

CONFIGURATION (module constants)
--------------------------------
- MAILTO                  Your email -> OpenAlex "polite pool" for better limits.
- S2_API_KEY              Optional Semantic Scholar key; lifts the strict
                          unauthenticated rate limit (shared across all anon users).
- TITLE_THRESHOLD         Min title similarity (0-1) to accept a free-text title.
                          Lower = more convenient but higher wrong-match risk.
- MAX_CITING_PER_SOURCE   Safety cap on citing works fetched per source
                          (None = unbounded). A union fetches EVERY citing work,
                          so highly-cited papers mean many paginated requests.

LIMITATIONS
-----------
- Cost/latency: the union pulls all citing works, so a paper with tens of
  thousands of citations means hundreds of requests per source (hence the cap).
- Semantic Scholar's citations endpoint limits pagination depth, so for very
  highly-cited papers its arm undercounts and the union leans on OpenAlex there.
- A target without a DOI cannot be reconciled against Semantic Scholar, so the
  result is OpenAlex-only (see ``semantic_scholar_used`` in the audit log).

EXTENDING
---------
Add another free source (e.g. OpenCitations, Crossref) by writing a
``_<source>_citing_keys(...) -> set`` that returns the same DOI-/title-keyed set,
then union it into ``union`` inside the tool.
"""

import re
import time
import logging
from difflib import SequenceMatcher
from typing import Optional

import requests
from langchain_core.tools import tool

# --- Configuration -----------------------------------------------------------
OPENALEX_BASE = "https://api.openalex.org"
S2_BASE = "https://api.semanticscholar.org/graph/v1"
MAILTO = ""        # OpenAlex polite pool, e.g. "you@example.com"
S2_API_KEY = ""    # optional Semantic Scholar API key
TITLE_THRESHOLD = 0.90
MAX_CITING_PER_SOURCE = 10000   # per-source safety cap; None = unbounded


# --- Helpers -----------------------------------------------------------------
def _norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()


def _key(doi: Optional[str], title: Optional[str], year=None):
    """Cross-source identity for a citing paper.

    DOI is the reliable anchor; fall back to normalized (title, year) when a DOI
    is absent (lower confidence). The leading tag lets callers count fallbacks.
    """
    if doi:
        return ("doi", doi.lower().replace("https://doi.org/", ""))
    return ("title", _norm(title), year)


def _resolve_openalex(paper: str, params: dict) -> dict:
    """Resolve a DOI / OpenAlex ID / title to a single OpenAlex work record.

    Raises ValueError when nothing is found, or when a free-text title does not
    match a single candidate confidently (to avoid a wrong-paper count).
    """
    p = paper.strip()
    if p.lower().startswith(("10.", "https://doi.org/", "doi:")):
        doi = p.replace("https://doi.org/", "").replace("doi:", "")
        r = requests.get(f"{OPENALEX_BASE}/works/doi:{doi}", params=params, timeout=30)
    elif re.fullmatch(r"[Ww]\d+", p):
        r = requests.get(f"{OPENALEX_BASE}/works/{p.upper()}", params=params, timeout=30)
    else:
        r = requests.get(
            f"{OPENALEX_BASE}/works",
            params={**params, "search": p, "per-page": 5,
                    "select": "id,title,display_name,publication_year,doi,cited_by_count"},
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            raise ValueError(f"No OpenAlex work found for: {paper!r}")
        q = _norm(p)
        best_score, best = max(
            ((SequenceMatcher(None, q, _norm(w.get("title") or w.get("display_name") or "")).ratio(), w)
             for w in results),
            key=lambda t: t[0],
        )
        if best_score < TITLE_THRESHOLD:
            raise ValueError(f"Ambiguous title {paper!r} (best match {best_score:.2f}); use a DOI.")
        return best
    if r.status_code == 404:
        raise ValueError(f"No OpenAlex work for: {paper!r}")
    r.raise_for_status()
    return r.json()


def _openalex_citing_keys(oa_id: str, params: dict) -> set:
    """Identity keys for every work citing ``oa_id`` (cursor-paginated)."""
    keys, cursor, n = set(), "*", 0
    while cursor:
        r = requests.get(
            f"{OPENALEX_BASE}/works",
            params={**params, "filter": f"cites:{oa_id}", "per-page": 200,
                    "cursor": cursor, "select": "id,doi,title,publication_year"},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        for w in data["results"]:
            keys.add(_key(w.get("doi"), w.get("title"), w.get("publication_year")))
            n += 1
        cursor = data["meta"].get("next_cursor")
        if MAX_CITING_PER_SOURCE and n >= MAX_CITING_PER_SOURCE:
            break
        time.sleep(0.1)
    return keys


def _s2_citing_keys(doi: str) -> set:
    """Identity keys for every work citing the paper with this DOI on Semantic Scholar."""
    keys, offset = set(), 0
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    while True:
        r = requests.get(
            f"{S2_BASE}/paper/DOI:{doi}/citations",
            params={"fields": "externalIds,title,year", "limit": 1000, "offset": offset},
            headers=headers, timeout=60,
        )
        if r.status_code == 404:
            return keys  # Semantic Scholar does not index this paper
        r.raise_for_status()
        data = r.json()
        rows = data.get("data", [])
        for row in rows:
            cp = row.get("citingPaper", {})
            keys.add(_key((cp.get("externalIds") or {}).get("DOI"), cp.get("title"), cp.get("year")))
        if not rows or "next" not in data:
            break
        offset = data["next"]
        if MAX_CITING_PER_SOURCE and len(keys) >= MAX_CITING_PER_SOURCE:
            break
        time.sleep(1.0)  # unauthenticated S2 is strict; supply S2_API_KEY to relax
    return keys


# --- Tool --------------------------------------------------------------------
@tool
def get_union_citation_count(paper: str) -> int:
    """Distinct citation count from the UNION of OpenAlex + Semantic Scholar.

    Counts unique papers citing the target across BOTH networks (not the sum,
    which double-counts shared citations). Citing papers are reconciled by DOI,
    falling back to title+year when a DOI is absent. Semantic Scholar is only
    included when the target has a DOI; otherwise the count is OpenAlex-only.

    Args:
        paper: DOI, OpenAlex work ID, or full title (title must match confidently).

    Returns:
        Number of distinct citing papers across both sources.
    """
    params = {"mailto": MAILTO} if MAILTO else {}
    work = _resolve_openalex(paper, params)
    oa_id = work["id"].rsplit("/", 1)[-1]
    doi = (work.get("doi") or "").replace("https://doi.org/", "")

    oa_keys = _openalex_citing_keys(oa_id, params)
    s2_keys = _s2_citing_keys(doi) if doi else set()
    union = oa_keys | s2_keys

    logging.info("union_citation_count: %s", {
        "paper": work.get("title"),
        "doi": doi or None,
        "union_citation_count": len(union),
        "openalex_count": len(oa_keys),
        "semantic_scholar_count": len(s2_keys),
        "overlap": len(oa_keys & s2_keys),
        "openalex_only": len(oa_keys - s2_keys),
        "semantic_scholar_only": len(s2_keys - oa_keys),
        "matched_by_title_only": sum(1 for k in union if k[0] == "title"),
        "semantic_scholar_used": bool(doi),
    })
    return len(union)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: "Deep learning" (LeCun, Bengio & Hinton, 2015)
    print(get_union_citation_count.invoke("10.1038/nature14539"))
