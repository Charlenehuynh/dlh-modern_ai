#!/usr/bin/env python3
"""Reusable prompt template"""
from langchain_core.prompts import PromptTemplate


def create_prompt_template(template_str, input_variables):
    """Return a PromptTemplate built from a template string.

    template_str: prompt text with {placeholders}
    input_variables: list of the placeholder names used in template_str
    """
    return PromptTemplate(
        template=template_str,
        input_variables=input_variables
    )
