#!/usr/bin/env python3
"""Create a FAISS vector store and retriever with LangChain"""
from langchain_community import vectorstores


def create_faiss_vectorstore(docs, embedding_model, k):
    """
    Builds a FAISS vector store from documents and a retriever on top of it.

    Args:
        docs: list of LangChain Document objects to index
        embedding_model: embedding model used to vectorize the documents
        k: number of documents the retriever returns per query

    Returns:
        tuple: (vectorstore, retriever)
            vectorstore: the FAISS vector store
            retriever: a VectorStoreRetriever returning the top k documents
    """
    vectorstore = vectorstores.FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return vectorstore, retriever
