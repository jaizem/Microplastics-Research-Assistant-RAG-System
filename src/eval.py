"""
Evaluation Support Functions
"""

def retrieve_docs(question):
    return retriever.invoke(question)

