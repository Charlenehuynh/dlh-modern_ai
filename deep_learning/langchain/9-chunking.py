#!/usr/bin/env python3
"""Split LangChain documents into smaller chunks"""
from langchain import text_splitter


def split_into_chunks(docs, chunk_size, chunk_overlap):
    """
    Splits a list of documents into smaller chunks.

    Args:
        docs: list of LangChain Document objects to split
        chunk_size: maximum number of characters per chunk
        chunk_overlap: number of characters to overlap between chunks

    Returns:
        list: Document objects representing the split chunks
    """
    splitter = text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
