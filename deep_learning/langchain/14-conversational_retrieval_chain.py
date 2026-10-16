#!/usr/bin/env python3
"""Create a conversational retrieval chain with LangChain"""
from langchain import chains


def create_conversational_retrieval_chain(history_aware_retriever,
                                          document_chain):
    """
    Builds a retrieval chain that fetches relevant documents using the
    conversation history and generates an answer from them.

    Args:
        history_aware_retriever: retriever using the query and chat history
            to find relevant documents
        document_chain: chain combining the retrieved documents into a
            single context for the language model

    Returns:
        a retrieval chain ready to handle user queries
    """
    return chains.create_retrieval_chain(history_aware_retriever,
                                         document_chain)
