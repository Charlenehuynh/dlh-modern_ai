#!/usr/bin/env python3
"""Load a HuggingFace embedding model with LangChain"""
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name):
    """
    Loads a HuggingFace embedding model.

    Args:
        model_name: name of the HuggingFace model to use for embeddings

    Returns:
        HuggingFaceEmbeddings: instance to compute document embeddings
    """
    return HuggingFaceEmbeddings(model_name=model_name)
