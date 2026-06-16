from dotenv import load_dotenv
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "context_corpus"
QUESTIONS_FILE = REPO_ROOT / "data" / "benchmark" / "questions.jsonl"

if (REPO_ROOT / "ignored-.env").is_file():
    load_dotenv(REPO_ROOT / "ignored-.env")
else:
    load_dotenv()

for _k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_k, None)


def ensure_openai_key(prompt_if_missing: bool = True) -> None:
    """Ensure `OPENAI_API_KEY` is present in the environment."""
    if not os.environ.get("OPENAI_API_KEY") and prompt_if_missing:
        import getpass

        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


# Ragas imports Vertex AI langchain modules we do not use. Stub them (see eval_ragas.ipynb).
import sys
import types

import langchain_community.llms

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertex_chat = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertex_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _vertex_chat
if not getattr(langchain_community.llms, "VertexAI", None):
    langchain_community.llms.VertexAI = type("VertexAI", (object,), {})
