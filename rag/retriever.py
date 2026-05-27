def get_retriever(vector_store, k=4):
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_docs(retriever, query):
    return retriever.invoke(query)