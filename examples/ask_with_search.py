#!/usr/bin/env python3
"""
Ask a question with internet search capabilities
"""

import os
import asyncio
import requests
import json
from dotenv import load_dotenv
from pathlib import Path
import argparse

# Import litellm directly for simplicity
import litellm

async def search_brave(query: str, count: int = 5, offset: int = 0) -> dict:
    """
    Search with Brave Search API
    
    Args:
        query: Search query
        count: Number of results
        offset: Pagination offset
        
    Returns:
        Search results
    """
    # Make direct API call to Brave Search
    api_key = os.environ.get("BRAVE_API_KEY", "BSABGuCvrv8TWsq-MpBTip9bnRi6JUg")
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    params = {"q": query, "count": count, "offset": offset}
    
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers=headers,
        params=params
    )
    
    response.raise_for_status()
    return response.json()

async def ask_with_search(question: str, provider: str = "anthropic", model: str = None) -> str:
    """
    Ask a question with internet search
    
    Args:
        question: Question to ask
        provider: LLM provider
        model: LLM model
        
    Returns:
        Answer to the question
    """
    # Load environment variables from .env
    load_dotenv()
    
    # Set default models
    default_models = {
        "anthropic": "claude-3-7-sonnet-20250219",
        "openai": "gpt-4o",
        "groq": "groq/llama3-70b-8192"
    }
    
    # Set model if not provided
    if not model:
        model = default_models.get(provider, default_models["anthropic"])
    
    print(f"Using provider: {provider}")
    print(f"Using model: {model}")
    
    # Print question
    print(f"\nQuestion: {question}")
    
    try:
        # Search the web
        print("Searching the web...")
        search_results = await search_brave(question, count=5)
        
        # Format search results for the LLM
        search_context = "\n\nHere are some search results that may help answer the question:\n\n"
        
        for i, result in enumerate(search_results.get("web", {}).get("results", []), 1):
            search_context += f"[{i}] {result.get('title')}\n"
            search_context += f"URL: {result.get('url')}\n"
            search_context += f"Summary: {result.get('description')}\n\n"
            
        # System prompt for search-enhanced assistance
        system_content = """You are a helpful assistant with internet search capabilities.
When asked about current information, first use the provided search results to find up-to-date information.
Always look at the search results before giving your answer, especially if the question is about current events,
technologies, or facts that might have changed recently.

When you use information from search results, cite the source in your answer."""
        
        # Add search results to the question
        enhanced_question = question + search_context
        
        # Print search results
        print("\nSearch Results:")
        for i, result in enumerate(search_results.get("web", {}).get("results", []), 1):
            print(f"{i}. {result.get('title')}")
            print(f"   URL: {result.get('url')}")
            print(f"   {result.get('description')[:100]}...")
        
        print("\nGenerating answer...")
        
        # Generate response with LiteLLM
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": enhanced_question}
        ]
        
        # Use LiteLLM directly
        completion = litellm.completion(
            model=model,
            messages=messages,
            max_tokens=1000
        )
        
        # Extract response
        answer = completion.choices[0].message.content
        
        print("\nAnswer:")
        print(answer)
        
        return answer
        
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        return error_msg

async def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Ask a question with internet search")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "groq"], help="LLM provider")
    parser.add_argument("--model", help="LLM model (defaults to provider's default model)")
    
    args = parser.parse_args()
    
    # Ask question
    await ask_with_search(args.question, args.provider, args.model)

if __name__ == "__main__":
    asyncio.run(main())