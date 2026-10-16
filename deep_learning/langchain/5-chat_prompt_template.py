#!/usr/bin/env python3
"""Create a reusable LangChain ChatPromptTemplate"""
from langchain import prompts


def create_chat_prompt_template(system_message, human_message):
    """
    Sets up a reusable chat-style prompt.

    Args:
        system_message: string defining the assistant's instructions
        human_message: string containing the user's input with placeholders

    Returns:
        ChatPromptTemplate: object that can be formatted with variables
    """
    return prompts.ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])
