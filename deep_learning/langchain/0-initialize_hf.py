#!/usr/bin/env python3
"""Initialize a Hugging Face text-to-text LLM with LangChain"""
from langchain_community import llms


def initialize_hf_llm(model_name, max_tokens):
    """
    Sets up a Hugging Face text-to-text generation model using LangChain.

    Args:
        model_name: name of the Hugging Face model to load
        max_tokens: maximum number of tokens to generate for the output

    Returns:
        llm: an instance of HuggingFacePipeline
    """
    llm = llms.HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text2text-generation",
        model_kwargs={"dtype": "auto"},
        pipeline_kwargs={"max_new_tokens": max_tokens},
    )
    return llm
