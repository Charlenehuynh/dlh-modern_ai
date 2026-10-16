#!/usr/bin/env python3
"""Prepare a sequence of system and human messages for an LLM"""

from langchain_core.messages import SystemMessage, HumanMessage


def setup_message_sequence(system_msg, human_msgs):
    """
    Combines a system message with one or more human messages.

    Args:
        system_msg: string defining the system's role or behavior
        human_msgs: list of strings representing human inputs

    Returns:
        list containing the SystemMessage followed by HumanMessage objects
    """
    messages = [SystemMessage(content=system_msg)]
    messages += [HumanMessage(content=msg) for msg in human_msgs]
    return messages
