"""Search tools for the researcher agent - DuckDuckGo powered."""

from langchain_core.tools import tool

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def _format_text_results(results: list, query: str) -> str:
    """Format text search results."""
    if not results:
        return f"No results found for: {query}"
    output_parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", r.get("url", ""))
        body = r.get("body", r.get("description", ""))
        output_parts.append(f"[{i}] {title}\n    URL: {href}\n    {body}")
    return "\n\n".join(output_parts)


def _format_news_results(results: list, query: str) -> str:
    """Format news search results."""
    if not results:
        return f"No news found for: {query}"
    output_parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", r.get("href", ""))
        body = r.get("body", r.get("description", ""))
        date = r.get("date", "")
        output_parts.append(f"[{i}] {title}\n    URL: {url}\n    Date: {date}\n    {body}")
    return "\n\n".join(output_parts)


@tool
def search_web(
    query: str,
    max_results: int = 10,
) -> str:
    """Search the internet using DuckDuckGo. Use for facts, definitions, weather, and general information.
    Always use this tool - never answer from memory. Craft specific queries like "Reston VA weather today".
    
    Args:
        query: The search query - be specific (e.g., "weather Reston Virginia", "Bitcoin price today")
        max_results: Number of results (default 10, max 20)
    
    Returns:
        Search results with titles, URLs, and snippets
    """
    max_results = min(max(max_results, 1), 20)
    try:
        ddgs = DDGS()
        results = ddgs.text(
            query,
            max_results=max_results,
            timelimit="d",  # Prefer recent results for current info
        )
        return _format_text_results(results, query)
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def search_news(
    query: str,
    max_results: int = 8,
) -> str:
    """Search recent news using DuckDuckGo. Best for weather, current events, breaking news, and today's information.
    Use this alongside search_web for weather and time-sensitive questions.
    
    Args:
        query: The search query (e.g., "Reston Virginia weather", "today's news")
        max_results: Number of results (default 8, max 20)
    
    Returns:
        News results with titles, URLs, dates, and summaries
    """
    max_results = min(max(max_results, 1), 20)
    try:
        ddgs = DDGS()
        results = ddgs.news(
            query,
            max_results=max_results,
            timelimit="d",  # Last 24 hours
        )
        return _format_news_results(results, query)
    except Exception as e:
        return f"News search failed: {str(e)}"
