"""Search tools for the researcher agent - DuckDuckGo powered."""

from langchain_core.tools import tool

from duckduckgo_search import DDGS


@tool
def search_web(
    query: str,
    max_results: int = 8,
) -> str:
    """Search the internet using DuckDuckGo. Use this to find current information, news, facts, or research on any topic.
    
    Args:
        query: The search query - be specific and descriptive for better results
        max_results: Maximum number of results to return (default 8, max 20)
    
    Returns:
        A formatted string of search results with titles, URLs, and snippets
    """
    max_results = min(max(max_results, 1), 20)
    
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        
        if not results:
            return f"No results found for: {query}"
        
        output_parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            href = r.get("href", r.get("url", ""))
            body = r.get("body", r.get("description", ""))
            output_parts.append(f"[{i}] {title}\n    URL: {href}\n    {body}")
        
        return "\n\n".join(output_parts)
    
    except Exception as e:
        return f"Search failed: {str(e)}"
