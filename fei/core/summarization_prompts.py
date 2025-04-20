"""
Summarization prompts for different summarization methods.

This module contains standalone prompts for different summarization approaches,
making it easy to customize and extend the summarization capabilities.
"""

from typing import Dict, Any, List, Tuple


def get_bullet_summary_prompt(messages: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Generate a prompt for bullet-point style summarization.
    
    Args:
        messages: List of (role, content) tuples to summarize
        
    Returns:
        List of message dictionaries for the LLM
    """
    # Format messages for summarization
    chunk_txt = "\n".join(f"{r.upper()}: {c}" for r, c in messages)
    
    return [
        {"role": "system", "content": (
            "You are a summarization engine for a software-agent chat. "
            "Create an EXTREMELY CONCISE summary in bullet-point style. "
            "Focus only on the most important information, decisions, code snippets, and key context. "
            "Be extremely brief - aim for 100-200 words maximum. "
            "The summary will replace all these messages in the conversation history, so include only what's essential."
        )},
        {"role": "user", "content": chunk_txt}
    ]


def get_narrative_summary_prompt(messages: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Generate a prompt for narrative style summarization.
    
    Args:
        messages: List of (role, content) tuples to summarize
        
    Returns:
        List of message dictionaries for the LLM
    """
    # Format messages for summarization
    chunk_txt = "\n".join(f"{r.upper()}: {c}" for r, c in messages)
    
    return [
        {"role": "system", "content": (
            "You are a summarization engine for a software-agent chat. "
            "Create an EXTREMELY CONCISE summary in narrative style. "
            "Focus only on the most important information, decisions, code snippets, and key context. "
            "Be extremely brief - aim for 100-200 words maximum. "
            "The summary will replace all these messages in the conversation history, so include only what's essential."
        )},
        {"role": "user", "content": chunk_txt}
    ]


def get_technical_summary_prompt(messages: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Generate a prompt for technical summarization with focus on code and technical details.
    
    Args:
        messages: List of (role, content) tuples to summarize
        
    Returns:
        List of message dictionaries for the LLM
    """
    # Format messages for summarization
    chunk_txt = "\n".join(f"{r.upper()}: {c}" for r, c in messages)
    
    return [
        {"role": "system", "content": (
            "You are a summarization engine for a software-agent chat. "
            "Create an EXTREMELY CONCISE technical summary. "
            "Focus primarily on code snippets, technical decisions, and implementation details. "
            "Preserve important variable names, function names, and technical terminology. "
            "Be extremely brief - aim for 100-200 words maximum. "
            "The summary will replace all these messages in the conversation history, so include only what's essential."
        )},
        {"role": "user", "content": chunk_txt}
    ]


def get_summary_prompt(messages: List[Tuple[str, str]], style: str = "bullet") -> List[Dict[str, Any]]:
    """
    Get the appropriate summary prompt based on the requested style.
    
    Args:
        messages: List of (role, content) tuples to summarize
        style: Style of summary to generate ("bullet", "narrative", or "technical")
        
    Returns:
        List of message dictionaries for the LLM
    """
    if style == "narrative":
        return get_narrative_summary_prompt(messages)
    elif style == "technical":
        return get_technical_summary_prompt(messages)
    else:  # Default to bullet
        return get_bullet_summary_prompt(messages)
