#!/usr/bin/env python3
"""Set up a LangChain agent executor with tools"""
from langchain import agents
from langchain_community.agent_toolkits import load_tools


def load_agent_with_toolkit(llm, toolkit_names, prompt):
    """
    Sets up an agent executor capable of using multiple tools.

    Args:
        llm: a LangChain LLM instance
        toolkit_names: list of tool names to load
        prompt: a ChatPromptTemplate instance

    Returns:
        AgentExecutor: an agent executor ready to handle queries
    """
    tools = load_tools.load_tools(toolkit_names)
    agent = agents.create_tool_calling_agent(llm, tools, prompt)
    return agents.AgentExecutor(agent=agent, tools=tools, verbose=True)
