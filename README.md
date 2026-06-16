# Microplastics Research Assistant (RAG System)

An end-to-end retrieval-augmented generation (RAG) system designed to answer questions over a curated corpus of scientific literature on microplastics in environmental and biological systems.

This project demonstrates how LLMs can be combined with vector-based retrieval to generate accurate, context-aware responses grounded in real research papers.

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Evaluation Architecture](#evaluation-architecture)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Notes](#notes)
- [Evaluation Metrics](#evaluation-metrics)

## Overview
This repository combines PDF ingestion, chunking, semantic search, and LLM answer generation with a follow-on evaluation stage that scores outputs for faithfulness, relevance, context quality, and scope.

## Features
- PDF ingestion and preprocessing pipeline
- Text chunking and embedding generation
- Vector database retrieval (semantic search)
- Context-aware response generation using LLMs
- Tracing and evaluation with LangSmith

## How It Works
1. Research papers are ingested and parsed into clean text
2. Documents are split into chunks and converted into embeddings
3. Embeddings are stored in a vector database
4. User queries are embedded and matched against relevant chunks
5. Retrieved context is passed to an LLM to generate grounded responses
6. Generated answers are evaluated with Ragas metrics for faithfulness, relevance, context quality, and scope

## Architecture
PDFs → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer

## Evaluation Architecture
Answer + Retrieved Contexts → Ragas evaluation pipeline → Faithfulness, Answer Relevancy, Context Relevance, Scope Representation

## Repository Structure
```text
.
├── main.py
├── README.md
├── requirements.yml
├── data
│   ├── benchmark
│   ├── context_corpus
│   ├── eval
│   └── results
├── docs
│   ├── EVAL.md
│   └── GOLD_STANDARD.md
├── eval
│   ├── evaluation.py
│   ├── evaluation_runner.py
│   ├── run_eval.py
│   └── __init__.py
├── rag
│   ├── generator.py
│   ├── ingest.py
│   ├── pipeline.py
│   ├── retriever.py
│   └── __init__.py
└── src
    ├── config.py
    └── __init__.py
```

## Tech Stack
- Python
- LangChain
- LangSmith (tracing & evaluation)
- OpenAI API
- OpenAI embeddings: `text-embedding-ada-002` for both vector store and evaluation
- FAISS / Chroma (vector store)
- Pandas / NumPy

## Setup

1. Create and activate the Conda environment from `requirements.yml`:

```bash
conda env create -f requirements.yml
conda activate microplastics-research-assistant
```

2. Create a `.env` file in the project root with your OpenAI API key:

```bash
# .env
export OPENAI_API_KEY=<insert-key>
```

If you keep local-only secrets, you can use `ignored-.env` instead (it is loaded first when present).

3. Optional: if you want to use LangSmith tracing, also add these variables:

```bash
export LANGSMITH_TRACING_V2=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=<insert-key>
export LANGSMITH_PROJECT=<insert-name>
```

4. Verify the environment is active and packages are installed, then run the project.

## Usage
- `python main.py` to run the full notebook-derived workflow from ingestion through evaluation.
- `python eval/run_eval.py` to run evaluation on an existing sample file.
- `python eval/run_eval.py --gold --limit 2` for the hand-checked benchmark in `data/benchmark/questions.jsonl` (same pipeline, plus reference-based metrics when available).

See `docs/EVAL.md` and `docs/GOLD_STANDARD.md` for details.

## Key Focus Areas
- Retrieval quality and relevance
- Chunking strategy optimization
- Prompt engineering for grounded responses
- Evaluation and debugging of LLM pipelines

## Example Query & Response
**Query:**  
What do the documents say about the effect of microplastics on humans?

**Response:**  
The system synthesizes findings across multiple research papers and indicates that while adverse effects of microplastics on marine organisms are well documented, human health impacts remain less understood. Current evidence is largely based on in-vitro studies, small-scale human studies, and extrapolation from marine biology.

Key findings include:
- Evidence of genotoxicity and cytotoxicity in human cells
- Increased micronucleation and nuclear abnormalities in blood lymphocytes
- Potential links to conditions such as infertility, metabolic disorders, and cardiovascular disease
- Smaller particles (<100 μm) may penetrate biological barriers and accumulate in tissues, including the placenta
- Possible immune responses and toxicity from chemical leaching

Overall, the system highlights that while risks are increasingly recognized, current research is still insufficient to fully quantify human health effects.

## Notes
- `main.py` is the notebook-derived CLI entrypoint for the full ingestion → retrieval → evaluation workflow.
- `src/config.py` is imported early by `main.py` and `eval/run_eval.py` to load `.env` settings and clear SSL/CA bundle environment vars that can break OpenAI requests.
- Embeddings use OpenAI `text-embedding-ada-002` for both the FAISS vector store and Ragas evaluation.
- Generated evaluation samples are stored in `data/eval/ragas_samples.json`.
- If `data/context_corpus` contains no documents, `main.py` exits with an explicit error.

## Evaluation Metrics

Default metrics (used by `main.py` and `python eval/run_eval.py`):

- `faithfulness` — is the answer supported by retrieved context?
- `answer_relevancy` — does the answer address the question?
- `context_relevance` — are retrieved contexts useful for the query?
- `scope_representation` — does the answer stay within source scope?

Gold benchmark additions (used by `python eval/run_eval.py --gold`):

- `answer_correctness`
- `context_precision`
- `context_recall`

These reference-based metrics run only when `reference_answer` is present and `answerable_flag` is `true`.

## Version
- Current release: `2.0`
- Previous major update: `1.5` introduced compatibility updates for the recent LangChain community wrapper changes and LangSmith implementation revisions.

## Known Issues / Patch Notes
- This version documents the current workaround for a Ragas / LangChain integration issue affecting the import path and SSL environment.
- `src/config.py` clears `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, and `CURL_CA_BUNDLE` before OpenAI-related imports to avoid SSL errors.
- Ragas tries to import Vertex AI langchain helpers we do not use. `src/config.py` stubs those modules on import (same trick as `notebooks/eval_ragas.ipynb`).
- `main.py` imports `src.config` early so the SSL and import patches run before the RAG workflow starts.
- `eval/evaluation.py` includes context truncation to reduce prompt size and avoid Ragas max-token or incomplete-output failures during faithfulness evaluation.
- Version `1.5` specifically addressed LangChain community wrapper updates and LangSmith changes used by the Ragas evaluation path.
