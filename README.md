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
│   ├── context_corpus
│   ├── eval
│   └── results
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

### 1. Faithfulness

> The Faithfulness metric measures how factually consistent a response is with the retrieved context. It ranges from 0 to 1, with higher scores indicating better consistency.
>
> A response is considered faithful if all its claims can be supported by the retrieved context.
>
> To calculate this: 1. Identify all the claims in the response. 2. Check each claim to see if it can be inferred from the retrieved context. 3. Compute the faithfulness score using the formula:
>
> <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
>   <mtext>Faithfulness Score</mtext>
>   <mo>=</mo>
>   <mfrac>
>     <mtext>Number of claims in the response supported by the retrieved context</mtext>
>     <mtext>Total number of claims in the response</mtext>
>   </mfrac>
> </math>

- **Requires:** `answer`, `contexts`
- **Purpose:** Is the response supported by the contexts? (Reference-free, returns a score in [0,1])
- **Notes:** A similar metric in ragas is Response Groundedness which also requires: `answer`, `contexts` which returns a pass/fail value. This will not be used, since faithfulness tests this more aggressively by evaluating each of the claims within the answer, rather than the answer as a whole.

### 2. Answer Relevance

> The evaluation metric, Answer Relevancy, focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information and higher scores indicate better relevancy. This metric is computed using the question, the context and the answer.
>
> The Answer Relevancy is defined as the mean cosine similarity of the original question to a number of artifical questions, which where generated (reverse engineered) based on the answer:
>
> <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
>   <mtext>answer relevancy</mtext>
>   <mo>=</mo>
>   <mfrac>
>     <mn>1</mn>
>     <mi>N</mi>
>   </mfrac>
>   <munderover>
>     <mo data-mjx-texclass="OP">&#x2211;</mo>
>     <mrow data-mjx-texclass="ORD">
>       <mi>i</mi>
>       <mo>=</mo>
>       <mn>1</mn>
>     </mrow>
>     <mrow data-mjx-texclass="ORD">
>       <mi>N</mi>
>     </mrow>
>   </munderover>
>   <mi>c</mi>
>   <mi>o</mi>
>   <mi>s</mi>
>   <mo stretchy="false">(</mo>
>   <msub>
>     <mi>E</mi>
>     <mrow data-mjx-texclass="ORD">
>       <msub>
>         <mi>g</mi>
>         <mi>i</mi>
>       </msub>
>     </mrow>
>   </msub>
>   <mo>,</mo>
>   <msub>
>     <mi>E</mi>
>     <mi>o</mi>
>   </msub>
>   <mo stretchy="false">)</mo>
> </math>

- **Requires:** `question`, `contexts`, `answer`
- **Purpose:** Quantify how well the answer maps back to the original question via reverse-engineered question generation.

### 3. Context Relevance

> Context Relevance evaluates whether the retrieved_contexts (chunks or passages) are pertinent to the user_input. This is done via two independent "LLM-as-a-Judge" prompt calls that each rate the relevance on a scale of 0, 1, or 2. The ratings are then converted to a [0,1] scale and averaged to produce the final score. Higher scores indicate that the contexts are more closely aligned with the user's query.
>
> 0 → The retrieved contexts are not relevant to the user's query at all.
>
> 1 → The contexts are partially relevant.
>
> 2 → The contexts are completely relevant.
>
- **Requires:** `user_input`, `retrieved_contexts`
- **Purpose:** Judge whether the returned contexts are on-topic and useful for answering the query.

### 4. Rubrics-Based Criteria Scoring

> The Rubric-Based Criteria Scoring Metric is used to do evaluations based on user-defined rubrics. Each rubric defines a detailed score description, typically ranging from 1 to 5. The LLM assesses and scores responses according to these descriptions, ensuring a consistent and objective evaluation.
>
> Example use: Does the answer correctly represent the scope of the source (e.g., not saying "microplastics cause X in humans" when the paper only studied fish)?
>
- **Requires:** `response`, optional `reference` and `rubrics`
- **Purpose:** Apply task-specific rubrics for fine-grained scoring and human-aligned evaluation.

## Version
- Current release: `2.0`
- Previous major update: `1.5` introduced compatibility updates for the recent LangChain community wrapper changes and LangSmith implementation revisions.

## Known Issues / Patch Notes
- This version documents the current workaround for a Ragas / LangChain integration issue affecting the import path and SSL environment.
- `src/config.py` clears `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, and `CURL_CA_BUNDLE` before OpenAI-related imports to avoid SSL errors.
- `main.py` now imports `src.config` early so the CA/SSL patch is applied before the RAG workflow starts.
- `eval/evaluation.py` includes context truncation to reduce prompt size and avoid Ragas max-token or incomplete-output failures during faithfulness evaluation.
- Version `1.5` specifically addressed LangChain community wrapper updates and LangSmith changes used by the Ragas evaluation path.
