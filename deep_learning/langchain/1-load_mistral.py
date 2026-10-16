#!/usr/bin/env python3
"""Initialize a Mistral language model with LangChain"""
import langchain_mistralai


def load_mistral(model_name, temperature):
    """
    Initializes a Mistral chat model using LangChain.

    Args:
        model_name: name of the Mistral model to use
        temperature: float controlling the randomness of the output

    Returns:
        llm: an instance of ChatMistralAI
    """
    llm = langchain_mistralai.ChatMistralAI(
        model=model_name,
        temperature=temperature
    )
    return llm
