"""
Converted notebook: `notebooks/eval_ragas.ipynb` → script form.

This script reproduces the notebook workflow:
- load documents
- split into chunks
- build vectorstore
- generate a small set of sample Q/A pairs via `run_rag`
- write `data/eval/ragas_samples.json`
- run evaluation and print averaged metrics
"""

import sys
from pathlib import Path
import json
import types

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Apply environment fixes before importing libraries that may use OpenAI/requests
import src.config  # noqa: F401

# Temporary bug workaround from notebook (preserves original behavior)
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat
import langchain_community.llms
langchain_community.llms.VertexAI = type("VertexAI", (object,), {})

from rag.pipeline import run_rag
from rag.retriever import get_retriever
from rag.ingest import load_documents, split_documents, build_vectorstore
from eval.evaluation_runner import evaluate
import pandas as pd


def main():
    # --- Load and process documents
    # Load PDFs from data/context_corpus in the repo root
    docs = load_documents(str(ROOT / "data" / "context_corpus"))
    if not docs:
        sys.exit("Error: no documents found in data/context_corpus. Add PDF files there before running.")

    # Chunk PDFs
    split_docs = split_documents(docs)

    # Build vector store
    vector_store = build_vectorstore(split_docs)

    # Setup retriever
    retriever = get_retriever(vector_store)

    # Run experiments (same example questions as notebook)
    questions = [
        "What are microplastics doing to human health?",
        "How do microplastics enter the ocean?",
    ]

    sample_results = [run_rag(q, retriever) for q in questions]

    # Convert to sample objects with IDs and save
    samples = [
        {
            "id": f"q{i+1}",
            "question": r["question"],
            "contexts": r["contexts"],
            "answer": r["answer"],
        }
        for i, r in enumerate(sample_results)
    ]

    samples_path = Path("data") / "eval" / "ragas_samples.json"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2, ensure_ascii=False)

    # Notebook printed samples — we'll print the count instead
    print(f"Generated {len(samples)} samples and wrote to {samples_path}")

    # Run evaluation (calls existing runner)
    evaluate(str(samples_path))

    # Read results and print averaged metrics
    results_path = Path("data") / "results" / "eval_results.json"
    if results_path.exists():
        df = pd.read_json(results_path)
        df_avg = df.mean(numeric_only=True)
        print("\nEvaluation Results (averages):")
        for k, v in df_avg.items():
            print(f"{k}: {v:.4f}")
    else:
        print("No evaluation results found at", results_path)


if __name__ == "__main__":
    main()
