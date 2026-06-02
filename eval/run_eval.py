import sys
from pathlib import Path

# go from /notebooks → project root
project_root = Path().resolve().parent
sys.path.append(str(project_root))

print(project_root)

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
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy,
    context_precision,
    context_recall)
from src.eval import build_ragas_dataset

with open("../notebooks/ragas_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

dataset = build_ragas_dataset(samples)


from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings

_global_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(result)