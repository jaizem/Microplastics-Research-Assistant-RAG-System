import asyncio
import json
from pathlib import Path

from eval.evaluation import build_ragas_dataset, run_eval as run_eval_async
from rag.ingest import get_embeddings
from rag.generator import get_instructor_llm


async def evaluate_async(samples_path: str):
    samples_file = Path(samples_path)
    with samples_file.open("r", encoding="utf-8") as fh:
        samples = json.load(fh)

    sample_ids = [sample.get("id") for sample in samples]
    if any(sample_id is None for sample_id in sample_ids):
        raise ValueError("All evaluation samples must include an 'id' field.")

    dataset = build_ragas_dataset(samples)

    embeddings = get_embeddings()
    llm = get_instructor_llm()

    await run_eval_async(dataset, llm, embeddings, question_ids=sample_ids)


def evaluate(samples_path: str):
    asyncio.run(evaluate_async(samples_path))
