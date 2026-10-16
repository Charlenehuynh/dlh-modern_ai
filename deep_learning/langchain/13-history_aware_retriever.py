#!/usr/bin/env python3
"""Create a history-aware retriever with LangChain"""
from langchain_core import prompts
from langchain import chains


def build_history_aware_retriever(llm, retriever, human_message):
    """
    Builds a retriever that turns the conversation history and the user's
    current input into a search query before retrieving documents.

    Args:
        llm: language model used to generate the search query
        retriever: LangChain retriever performing the similarity search
        human_message: string with the user's input used to build the query

    Returns:
        tuple: (prompt, history_aware_retriever)
            prompt: the ChatPromptTemplate for generating search queries
            history_aware_retriever: retriever configured with the prompt
    """
    prompt = prompts.ChatPromptTemplate.from_messages([
        prompts.MessagesPlaceholder(variable_name="chat_history"),
        ("human", human_message)
    ])
    history_aware_retriever = chains.create_history_aware_retriever(
        llm, retriever, prompt
    )
    return prompt, history_aware_retriever
