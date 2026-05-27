from rag.retriever import retrieve_docs
from rag.generator import generate_answer, format_docs


def run_rag(question, retriever):
    docs = retrieve_docs(retriever, question)
    answer = generate_answer(question, docs)

    return {
        "question": question,
        "contexts": [d.page_content for d in docs],
        "answer": answer
    }