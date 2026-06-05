import os
from dotenv import load_dotenv
load_dotenv()
import getpass
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from ragas.llms import llm_factory

prompt = ChatPromptTemplate.from_template("""
Answer using ONLY the context.

Context:
{context}

Question:
{question}
""")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
instructor_llm = llm_factory('gpt-4o-mini', client=openai_client)

def get_llm():
    return llm

def get_client():
    return openai_client

def get_instructor_llm():
    return instructor_llm

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def generate_answer(question, docs):
    context = format_docs(docs)

    chain = prompt | llm

    return chain.invoke({
        "context": context,
        "question": question
    }).content