"""
recursive_faithfulness.py

Recursive, citation-aware extension of ragas' Faithfulness metric.

Idea (converged in discussion)
------------------------------
Faithfulness, as shipped, scores the fraction of a document's statements that
are inferable from a fixed context. Here we instead score a document against
the *documents it cites*, recurse into those citations to depth ``n``, and let
trust propagate upward -- "a document is only as trustworthy as its sources."

  * Verdicts stay BINARY (entailed / not), exactly as in the stock NLI prompt.

  * The unit of scoring is a LINKAGE: a (citation-marked statement, cited
    document) pair. A statement that cites two papers is two linkages. Novel
    statements carry no citation, so the source never returns them as linkages
    and they are excluded from the denominator by construction -- they neither
    help nor hurt the score.

  * A linkage's reward is NOT a flat 1. If the citing statement is entailed by
    the cited document C, the linkage earns ``trust(C)`` -- C's own recursive
    score -- instead of 1. If it is not entailed (citation distortion) it earns
    0 but still counts in the denominator.

        score(D) = sum_over_linkages( entailed ? trust(cited) : 0 )
                   / number_of_linkages

  * Base cases / guards, all returning ``leaf_trust`` (default 1.0):
      - depth horizon reached (remaining == 0);
      - a back-edge: a citation pointing to a document still open on the current
        recursion path (a cycle) -- we substitute ``leaf_trust`` to terminate
        rather than descend forever;
      - a document with no retrievable citations (foundational, or simply not in
        the corpus).
    A ROOT document with no citation-linked statements yields NaN, mirroring the
    stock metric's "no statements -> NaN".

Why the denominator is citation-marked statements (not "entailed statements
only"): with ``leaf_trust = 1.0``, normalizing over entailed statements alone is
degenerate -- no value below 1 can ever enter the system, so every document
scores 1.0. Distrust must be able to enter through cited-but-unsupported
statements, which means they have to live in the denominator.

Modeling choices that are genuinely open are exposed as constructor arguments
(``leaf_trust``) or noted inline; change them rather than editing the core loop.
"""

from __future__ import annotations

import logging
import math
import typing as t
from dataclasses import dataclass

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.callbacks import Callbacks

__all__ = [
    "Linkage",
    "DocumentSource",
    "RecursiveFaithfulness",
    "recursive_faithfulness",
]


@dataclass(frozen=True)
class Linkage:
    """A single citation act: ``statement`` is a claim in the citing document
    that is attributed to ``cited_doc_id``."""

    statement: str
    cited_doc_id: str


@t.runtime_checkable
class DocumentSource(t.Protocol):
    """The assumed 'nearly complete' source of scientific documents.

    Two async queries are required:

    * ``get_text(doc_id)`` -> the document's text, used as the NLI premise when
      this document is cited by another.
    * ``get_linkages(doc_id)`` -> one :class:`Linkage` per (citation-marked
      statement, cited document). ONLY citation-marked statements are returned;
      a document's novel claims are out of scope by construction. A statement
      citing several documents yields one linkage per cited document.

    Either call may raise (e.g. ``KeyError``) for a document outside the corpus;
    such gaps are treated as ``leaf_trust`` (benefit of the doubt).
    """

    async def get_text(self, doc_id: str) -> str: ...

    async def get_linkages(self, doc_id: str) -> t.List[Linkage]: ...


class RecursiveFaithfulness:
    def __init__(
        self,
        source: DocumentSource,
        *,
        n: int = 3,
        leaf_trust: float = 1.0,
        llm: t.Any = None,
        nli_prompt: t.Any = None,
    ) -> None:
        """
        Parameters
        ----------
        source : DocumentSource
            Resolver for document text and citation linkages.
        n : int
            Recursion depth. ``n=1`` scores a document against its direct
            citations only; larger ``n`` lets distrust propagate up from deeper
            sources.
        leaf_trust : float
            Trust assigned at the depth horizon, at back-edges, and for
            documents with no retrievable citations. Default 1.0 ("assume the
            foundation we cannot inspect is sound"); this is also what keeps a
            faithful literature review at 1.0.
        llm, nli_prompt :
            Used only by the default (LLM-backed) verdict path. Left ``None``,
            they are lazily resolved from ragas / the shipped ``faithfulness``
            instance on first use. Override :meth:`_verdicts` to supply your own
            judge (the bundled demo does this to stay dependency-free).
        """
        self.source = source
        self.n = n
        self.leaf_trust = leaf_trust
        self.llm = llm
        self.nli_prompt = nli_prompt

    # -- public API -----------------------------------------------------------

    async def ascore(self, document: str, callbacks: "Callbacks" = None) -> float:
        """Return the recursive faithfulness score for ``document``."""
        try:
            root_linkages = await self.source.get_linkages(document)
        except Exception:
            root_linkages = None
        if not root_linkages:
            logger.warning(
                "Document %r has no citation-linked statements; score is undefined.",
                document,
            )
            return math.nan

        memo: t.Dict[t.Tuple[str, int], float] = {}
        in_progress: t.Set[str] = set()
        return await self._compute(document, self.n, in_progress, memo, callbacks)

    # -- recursion ------------------------------------------------------------

    async def _compute(
        self,
        doc_id: str,
        remaining: int,
        in_progress: t.Set[str],
        memo: t.Dict[t.Tuple[str, int], float],
        callbacks: "Callbacks",
    ) -> float:
        if remaining <= 0:
            return self.leaf_trust  # depth horizon
        if doc_id in in_progress:
            return self.leaf_trust  # back-edge: cycle to an open ancestor

        key = (doc_id, remaining)
        if key in memo:
            return memo[key]

        try:
            linkages = await self.source.get_linkages(doc_id)
        except Exception:
            linkages = None
        if not linkages:
            # Foundational document, or one outside the corpus: nothing to
            # assess against the literature.
            memo[key] = self.leaf_trust
            return self.leaf_trust

        in_progress.add(doc_id)
        try:
            # Group the linkage statements by the document they cite, preserving
            # order, so each cited document is judged in a single NLI call.
            by_cite: t.Dict[str, t.List[str]] = {}
            for lk in linkages:
                by_cite.setdefault(lk.cited_doc_id, []).append(lk.statement)

            numerator = 0.0
            for cited, statements in by_cite.items():
                trust_c = await self._compute(
                    cited, remaining - 1, in_progress, memo, callbacks
                )
                try:
                    context = await self.source.get_text(cited)
                except Exception:
                    context = None
                if context is None:
                    # Cannot verify entailment; treat every linkage to this
                    # unresolvable source as supported, earning trust_c.
                    verdicts = [1] * len(statements)
                else:
                    verdicts = await self._verdicts(
                        citing_doc_id=doc_id,
                        cited_doc_id=cited,
                        context=context,
                        statements=statements,
                        callbacks=callbacks,
                    )
                numerator += sum(trust_c if v else 0.0 for v in verdicts)

            denominator = len(linkages)
            score = numerator / denominator if denominator else self.leaf_trust
        finally:
            in_progress.discard(doc_id)

        memo[key] = score
        return score

    # -- the binary NLI judge (overridable) -----------------------------------

    async def _verdicts(
        self,
        *,
        citing_doc_id: str,
        cited_doc_id: str,
        context: str,
        statements: t.List[str],
        callbacks: "Callbacks",
    ) -> t.List[int]:
        """Binary entailment of each statement against ``context`` (0/1).

        Default path reuses ragas' shipped ``NLIStatementPrompt`` unchanged.
        Override to plug in a different judge.
        """
        if not statements:
            return []

        if self.nli_prompt is None or self.llm is None:
            self._lazy_init_judge()
        assert self.llm is not None, (
            "An LLM is required for the default NLI judge. Pass llm=... or set "
            "ragas' `faithfulness.llm`, or override _verdicts()."
        )

        from _faithfulness import NLIStatementInput  # lazy: avoids ragas at import

        output = await self.nli_prompt.generate(
            data=NLIStatementInput(context=context, statements=list(statements)),
            llm=self.llm,
            callbacks=callbacks,
        )
        return self._align_verdicts(statements, output)

    def _lazy_init_judge(self) -> None:
        # Pull the NLI prompt and (optionally) the LLM from the shipped metric.
        # In a real ragas tree this import is `from ragas.metrics._faithfulness ...`.
        from _faithfulness import NLIStatementPrompt, faithfulness

        if self.nli_prompt is None:
            self.nli_prompt = NLIStatementPrompt()
        if self.llm is None:
            self.llm = faithfulness.llm

    @staticmethod
    def _align_verdicts(statements: t.List[str], output: t.Any) -> t.List[int]:
        answers = list(output.statements)
        if len(answers) == len(statements):
            return [1 if a.verdict else 0 for a in answers]
        # Defensive: the judge reordered/merged. Match on statement text.
        by_text = {a.statement.strip(): (1 if a.verdict else 0) for a in answers}
        return [by_text.get(s.strip(), 0) for s in statements]


async def recursive_faithfulness(
    document: str,
    n: int = 3,
    *,
    source: DocumentSource,
    llm: t.Any = None,
    leaf_trust: float = 1.0,
    callbacks: "Callbacks" = None,
) -> float:
    """Convenience wrapper. ``document`` is an id understood by ``source``.

    Mirrors the requested signature ``recursive_faithfulness(document, n=3)``;
    ``source`` (and, for the live judge, an LLM) must be supplied. With ``llm``
    left ``None`` the shipped ``faithfulness`` instance's LLM is reused.
    """
    metric = RecursiveFaithfulness(source, n=n, leaf_trust=leaf_trust, llm=llm)
    return await metric.ascore(document, callbacks=callbacks)


# ---------------------------------------------------------------------------
# Worked example -- reproduces the hand-computed 7/12 case, no ragas/LLM needed.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    class DictDocumentSource:
        """Trivial in-memory source for the demo."""

        def __init__(self, texts, linkages):
            self._texts = texts
            self._linkages = linkages

        async def get_text(self, doc_id):
            return self._texts[doc_id]  # KeyError -> treated as leaf upstream

        async def get_linkages(self, doc_id):
            return list(self._linkages.get(doc_id, []))

    # Graph: D->{A,B}, A->{F}, B->{F, D(back-edge)}, F is a leaf.
    linkages = {
        "D": [
            Linkage("d1 claims A", "A"),
            Linkage("d2 claims A", "A"),
            Linkage("d3 claims B", "B"),
            Linkage("d4 claims B", "B"),  # distortion of B
        ],
        "A": [
            Linkage("a1 claims F", "F"),
            Linkage("a2 claims F", "F"),
            Linkage("a3 claims F", "F"),  # distortion of F
        ],
        "B": [
            Linkage("b1 claims F", "F"),
            Linkage("b2 claims D", "D"),  # back-edge to ancestor D
        ],
        "F": [],  # leaf
    }
    texts = {k: f"text of {k}" for k in ("D", "A", "B", "F")}

    # Stub the binary NLI: 1 = entailed, 0 = not. (D and A each distort once.)
    ENTAILED = {
        "d1 claims A": 1, "d2 claims A": 1, "d3 claims B": 1, "d4 claims B": 0,
        "a1 claims F": 1, "a2 claims F": 1, "a3 claims F": 0,
        "b1 claims F": 1, "b2 claims D": 1,
    }

    class DemoRecursiveFaithfulness(RecursiveFaithfulness):
        async def _verdicts(self, *, citing_doc_id, cited_doc_id, context,
                            statements, callbacks):
            return [ENTAILED[s] for s in statements]

    async def main():
        metric = DemoRecursiveFaithfulness(
            DictDocumentSource(texts, linkages), n=5
        )
        score = await metric.ascore("D")
        print(f"trust(D) = {score:.6f}  (expected 7/12 = {7/12:.6f})")
        assert abs(score - 7 / 12) < 1e-9, score
        print("ok")

    asyncio.run(main())
