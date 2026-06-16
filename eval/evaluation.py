"""
Evaluation Support Functions
"""
import pandas as pd
from pathlib import Path
from src.config import REPO_ROOT
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
    RubricsScoreWithoutReference,
)

MAX_CONTEXTS = 4
MAX_CONTEXT_CHARS = 1200


def truncate_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    if "\n" in truncated:
        truncated = truncated.rsplit("\n", 1)[0]
    return truncated + "\n...[truncated]"


def truncate_contexts(contexts: list[str]) -> list[str]:
    return [truncate_context(c) for c in contexts[:MAX_CONTEXTS]]


def sample_reference(sample: dict) -> str:
    return (sample.get("reference") or sample.get("reference_answer") or "").strip()


def should_score_reference(sample: dict) -> bool:
    # Trap questions should not be scored against a reference answer.
    if sample.get("answerable_flag") is False:
        return False
    return bool(sample_reference(sample))


def build_ragas_dataset(samples):
    ragas_samples = []

    for sample in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=sample["question"],
                response=sample["answer"],
                retrieved_contexts=truncate_contexts(sample["contexts"]),
            )
        )

    return EvaluationDataset(samples=ragas_samples)


async def _score_reference_metrics(
    sample,
    reference: str,
    llm,
    embeddings,
    correctness_metric,
    precision_metric,
    recall_metric,
) -> dict:
    out: dict = {}

    ac = await correctness_metric.ascore(
        user_input=sample.user_input,
        response=sample.response,
        reference=reference,
    )
    out["answer_correctness"] = ac.value

    cp = await precision_metric.ascore(
        user_input=sample.user_input,
        reference=reference,
        retrieved_contexts=sample.retrieved_contexts,
    )
    out["context_precision"] = cp.value

    if sample.retrieved_contexts:
        cr = await recall_metric.ascore(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
            reference=reference,
        )
        out["context_recall"] = cr.value

    return out


# Evaluation setup
async def run_eval(dataset, llm, embeddings, question_ids=None, samples=None):
    faithfulness_metric = Faithfulness(llm=llm)
    answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_relevance_metric = ContextRelevance(llm=llm)
    scope_metric = RubricsScoreWithoutReference(
        name="scope_representation",
        rubrics={
            "score1_desc": "Answer makes claims that clearly exceed the scope (e.g. generalizes animal findings to humans, or presents a single study's results as universal consensus)",
            "score2_desc": "Answer somewhat oversteps scope in minor ways",
            "score3_desc": "Answer mostly respects scope with small ambiguities",
            "score4_desc": "Answer correctly and precisely represents the scope of the source",
        },
        llm=llm,
    )

    use_reference = samples and any(should_score_reference(s) for s in samples)
    correctness_metric = precision_metric = recall_metric = None
    if use_reference:
        correctness_metric = AnswerCorrectness(llm=llm, embeddings=embeddings)
        precision_metric = ContextPrecisionWithReference(llm=llm)
        recall_metric = ContextRecall(llm=llm)

    results = []
    sample_ids = question_ids or [None] * len(dataset.samples)
    raw_samples = samples or [{}] * len(dataset.samples)

    for sample, sample_id, raw in zip(dataset.samples, sample_ids, raw_samples):
        f_score = await faithfulness_metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )

        ar_score = await answer_relevancy_metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
        )

        cr_score = await context_relevance_metric.ascore(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
        )

        sc_score = await scope_metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
        )

        result = {
            "question_id": sample_id,
            "faithfulness": f_score.value,
            "answer_relevancy": ar_score.value,
            "context_relevance": cr_score.value,
            "scope_representation": sc_score.value,
        }

        reference = sample_reference(raw)
        if use_reference and should_score_reference(raw) and correctness_metric:
            result.update(
                await _score_reference_metrics(
                    sample,
                    reference,
                    llm,
                    embeddings,
                    correctness_metric,
                    precision_metric,
                    recall_metric,
                )
            )

        if sample_id is None:
            result["question"] = sample.user_input

        results.append(result)

    results_dir = REPO_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_json(results_dir / "eval_results.json", orient="records", indent=2)
