#!/usr/bin/env python3
"""Wrap a retrieval chain with session-aware chat history"""
from langchain_core.runnables.history import RunnableWithMessageHistory


def create_session_aware_chat(retrieval_chain, get_session_history,
                              input_key, history_key, output_key):
    """
    Wraps a conversational retrieval chain so each session keeps its own
    chat history, used for retrieval and answer generation.

    Args:
        retrieval_chain: conversational retrieval chain to wrap
        get_session_history: function returning the chat history object
            for a given session id
        input_key: key of the user input
        history_key: key where the chat history is stored
        output_key: key where the model output is stored

    Returns:
        RunnableWithMessageHistory: the session-aware chain
    """
    return RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key=input_key,
        history_messages_key=history_key,
        output_messages_key=output_key
    )
