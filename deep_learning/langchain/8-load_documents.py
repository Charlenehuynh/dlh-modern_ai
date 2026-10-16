#!/usr/bin/env python3
"""Load documents from a folder with LangChain"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_documents(folder_path, file_pattern):
    """
    Loads documents from a folder.

    Args:
        folder_path: path to the folder containing documents
        file_pattern: glob pattern to match files

    Returns:
        list: LangChain Document objects loaded from the folder
    """
    loader = DirectoryLoader(
        folder_path,
        glob=file_pattern,
        loader_cls=TextLoader
    )
    return loader.load()
