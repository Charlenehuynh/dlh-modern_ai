#!/usr/bin/env python3
"""Create a LangChain agent that can analyze a pandas DataFrame"""
from langchain_experimental.agents import agent_toolkits


def create_dataframe_agent(llm, df):
    """
    Creates an agent that answers questions about a pandas DataFrame.

    Args:
        llm: a LangChain LLM instance
        df: the pandas DataFrame to analyze

    Returns:
        AgentExecutor: an agent executor ready to handle queries
    """
    return agent_toolkits.create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
