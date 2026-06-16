"""
CLI entrypoint for running the RAG evaluation from a terminal.
"""

import sys
from pathlib import Path
import argparse
import json
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.config  # noqa: F401
from src.config import CORPUS_DIR, QUESTIONS_FILE, ensure_openai_key


def _load_gold_questions(limit: int) -> list[dict]:
    questions = []
    with QUESTIONS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
            if len(questions) >= limit:
                break
    return questions


def _run_gold(limit: int) -> None:
    from rag.ingest import load_documents, split_documents, build_vectorstore
    from rag.pipeline import run_rag
    from rag.retriever import get_retriever
    from eval.evaluation_runner import evaluate

    questions = _load_gold_questions(limit)
    docs = load_documents(str(CORPUS_DIR))
    retriever = get_retriever(build_vectorstore(split_documents(docs)))

    samples = []
    for q in questions:
        out = run_rag(q["question"], retriever)
        samples.append({
            "id": q["question_id"],
            "question": q["question"],
            "answer": out["answer"],
            "contexts": out["contexts"],
            "reference_answer": q.get("reference_answer", ""),
            "answerable_flag": q.get("answerable_flag", True),
        })

    out_path = ROOT_DIR / "data" / "eval" / "gold_ragas_samples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    evaluate(str(out_path))
    print(f"Generated {len(samples)} gold samples and wrote to {out_path}")

    results_path = ROOT_DIR / "data" / "results" / "eval_results.json"
    if results_path.exists():
        df = pd.read_json(results_path)
        df_avg = df.mean(numeric_only=True)
        print("\nEvaluation Results (averages):")
        for k, v in df_avg.items():
            print(f"{k}: {v:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--samples",
        default=str(ROOT_DIR / "data" / "eval" / "ragas_samples.json"),
        help="Path to ragas samples json file. Each sample must include an 'id' field for result joining.",
    )
    p.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prompt for OPENAI_API_KEY if missing in environment",
    )
    p.add_argument(
        "--gold",
        action="store_true",
        help="Run RAG on data/benchmark/questions.jsonl, then evaluate with the main pipeline",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=2,
        help="With --gold, number of benchmark questions to run",
    )
    args = p.parse_args()

    ensure_openai_key(prompt_if_missing=not args.no_prompt)

    if args.gold:
        _run_gold(args.limit)
        return

    from eval.evaluation_runner import evaluate

    evaluate(args.samples)


if __name__ == "__main__":
    main()
