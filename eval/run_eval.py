import sys
from pathlib import Path
# go from /notebooks → project root
project_root = Path().resolve().parent
sys.path.append(str(project_root))


import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("CURL_CA_BUNDLE", None)


# temporary bug workaround. see link for more permament solution. 
# https://github.com/vibrantlabsai/ragas/issues/2753#issuecomment-4563590504
import types
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat
import langchain_community.llms
langchain_community.llms.VertexAI = type("VertexAI", (object,), {})


from dotenv import load_dotenv
load_dotenv()
import getpass
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


import json
from src.eval import build_ragas_dataset
with open("../notebooks/ragas_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)
dataset = build_ragas_dataset(samples)


from ragas import evaluate
from rag.ingest import get_embeddings
from rag.generator import get_instructor_llm, get_llm
#from ragas.metrics.collections import RubricsScoreWithoutReference, ContextRelevance, Faithfulness, AnswerRelevancy


embeddings = get_embeddings()
llm = get_instructor_llm()

"""
faithfulness = Faithfulness(llm=llm)
answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
context_relevance = ContextRelevance(llm=llm)

scope_score = RubricsScoreWithoutReference(
    name="scope_representation",
    rubrics={
        "score1_desc": "Answer makes claims that clearly exceed the scope (e.g. generalizes animal findings to humans, or presents a single study's results as universal consensus)",
        "score2_desc": "Answer somewhat oversteps scope in minor ways",
        "score3_desc": "Answer mostly respects scope with small ambiguities",
        "score4_desc": "Answer correctly and precisely represents the scope of the source"
    },
    llm=llm
)

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        #scope_score,
        context_relevance,
    ],
    llm = llm,
    embeddings = embeddings
)

print(result)
"""

import asyncio
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

faithfulness_metric = Faithfulness(llm=llm)
answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)

async def run_eval(dataset, llm, embeddings):
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

        print(f"Faithfulness: {f_score.value}")
        print(f"Answer Relevancy: {ar_score.value}")
        
        results.append({"faithfulness": f_score.value, "answer_relevancy": ar_score.value})
    
    return results

results = asyncio.run(run_eval(dataset, llm, embeddings))