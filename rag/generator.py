import os
from dotenv import load_dotenv
load_dotenv()
import getpass
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("""
Answer using ONLY the context.

Context:
{context}

Question:
{question}
""")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def generate_answer(question, docs):
    context = format_docs(docs)

    chain = prompt | llm

    return chain.invoke({
        "context": context,
        "question": question
    }).content



# Missing credentials.