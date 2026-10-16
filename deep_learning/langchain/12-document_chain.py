#!/usr/bin/env python3
"""Create a document chain that answers from documents and chat history"""
from langchain_core import prompts
from langchain.chains import combine_documents


def document_answer_chain(llm, system_message, human_message):
    """
    Builds a prompt and a document chain that answers questions from
    retrieved documents while using the prior conversation history.

    Args:
        llm: an instance of a language model to generate answers
        system_message: string defining instructions for the assistant
        human_message: string containing the user's input or question

    Returns:
        tuple: (prompt, document_chain)
            prompt: the ChatPromptTemplate used for generating answers
            document_chain: chain sending documents, chat history and
                user input to the LLM
    """
    prompt = prompts.ChatPromptTemplate.from_messages([
        ("system", system_message),
        prompts.MessagesPlaceholder(variable_name="chat_history"),
        ("human", human_message)
    ])
    document_chain = combine_documents.create_stuff_documents_chain(
        llm, prompt
    )
    return prompt, document_chain
