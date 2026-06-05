"""
Evaluation Support Functions
"""
import pandas as pd
from pathlib import Path
from rag import retriever
from ragas import EvaluationDataset, SingleTurnSample
import asyncio
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRelevance, RubricsScoreWithoutReference
from rag.ingest import get_embeddings
from rag.generator import get_instructor_llm

def retrieve_docs(question):
    return retriever.invoke(question)

def build_ragas_dataset(samples):
    ragas_samples = []

    for sample in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=sample["question"],
                response=sample["answer"],
                retrieved_contexts=sample["contexts"]
            )
        )

    return EvaluationDataset(samples=ragas_samples)

# Evaluation setup
async def run_eval(dataset, llm, embeddings):
    # metrics setup
    faithfulness_metric = Faithfulness(llm=llm)
    answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_relevance_metric = ContextRelevance(llm=llm)
    scope_metric = RubricsScoreWithoutReference(
        name="scope_representation",
        rubrics={
            "score1_desc": "Answer makes claims that clearly exceed the scope (e.g. generalizes animal findings to humans, or presents a single study's results as universal consensus)",
            "score2_desc": "Answer somewhat oversteps scope in minor ways",
            "score3_desc": "Answer mostly respects scope with small ambiguities",
            "score4_desc": "Answer correctly and precisely represents the scope of the source"
        },
        llm=llm
    )

    results = []
    
    for sample in dataset.samples:
        f_score = await faithfulness_metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts
        )

        ar_score = await answer_relevancy_metric.ascore(
            user_input=sample.user_input,
            response=sample.response
        )

        cr_score = await context_relevance_metric.ascore(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts
        )

        sc_score = await scope_metric.ascore(
            user_input=sample.user_input,
            response=sample.response
        )

        results.append({"question": sample.user_input, 
                        "answer": sample.response,
                        "retrieved_contexts": sample.retrieved_contexts,
                        "faithfulness": f_score.value, 
                        "answer_relevancy": ar_score.value,
                        "context_relevance": cr_score.value,
                        "scope_representation": sc_score.value})
    
    # save results
    results_dir = Path("../data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_json(results_dir / "eval_results.json", orient="records", indent=2)