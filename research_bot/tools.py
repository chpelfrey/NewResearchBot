"""Search tools for the researcher agent - DuckDuckGo powered."""

from langchain_core.tools import tool

from research_bot.research_log import get_relevant_entries

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def _format_text_results(results: list, query: str) -> str:
    """Format text search results with citation-ready URLs."""
    if not results:
        return f"No results found for: {query}"
    output_parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", r.get("url", ""))
        body = r.get("body", r.get("description", ""))
        cite = f"[{i}]({href})" if href else ""
        output_parts.append(
            f"[{i}] {title}\n    URL: {href}\n    Cite as: {cite}\n    {body}"
        )
    return "\n\n".join(output_parts)


def _format_news_results(results: list, query: str) -> str:
    """Format news search results with citation-ready URLs."""
    if not results:
        return f"No news found for: {query}"
    output_parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", r.get("href", ""))
        body = r.get("body", r.get("description", ""))
        date = r.get("date", "")
        cite = f"[{i}]({url})" if url else ""
        output_parts.append(
            f"[{i}] {title}\n    URL: {url}\n    Cite as: {cite}\n    Date: {date}\n    {body}"
        )
    return "\n\n".join(output_parts)


@tool
def search_web(
    query: str,
    max_results: int = 10,
    recent_only: bool = False,
) -> str:
    """Search the internet using DuckDuckGo. Use for facts, definitions, people, release dates, and general information.
    Always use this tool - never answer from memory. Craft specific queries (e.g. "Nintendo Switch 2 release date", "Alex Pretti nurse Minneapolis 2026").
    
    Args:
        query: The search query - be specific. For "who is X" use the exact name first, then try adding context (e.g. "Alex Pretti nurse", "Alex Pretti Minneapolis") or alternate spellings if the first search returns no useful results.
        max_results: Number of results (default 10, max 20)
        recent_only: If True, restrict to last day (use for weather, prices, today's news). Default False for broad results.
    
    Returns:
        Search results with titles, URLs, and snippets
    """
    max_results = min(max(max_results, 1), 20)
    try:
        ddgs = DDGS()
        kwargs = {"max_results": max_results}
        if recent_only:
            kwargs["timelimit"] = "d"
        raw = ddgs.text(query, **kwargs)
        results = list(raw) if raw is not None else []
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
        raw = ddgs.news(
            query,
            max_results=max_results,
            timelimit="d",  # Last 24 hours
        )
        results = list(raw) if raw is not None else []
        return _format_news_results(results, query)
    except Exception as e:
        return f"News search failed: {str(e)}"


@tool
def check_research_log(
    query: str,
    max_entries: int = 5,
) -> str:
    """Check the research log for past questions and answers that may be relevant.
    ALWAYS call this first before searching the web. If you find a relevant past answer,
    you may use it and only call search_web/search_news to fill in gaps or get newer info.

    Args:
        query: The user's question (use the same or similar phrasing).
        max_entries: Maximum number of past Q&A entries to return (default 5).

    Returns:
        Formatted past Q&A entries with query, response, timestamp, and response time; or
        a message saying no relevant past answers were found.
    """
    entries = get_relevant_entries(query, min_score=0.4, max_entries=max_entries)
    if not entries:
        return "No relevant past questions or answers found in the research log."
    parts = []
    for i, e in enumerate(entries, 1):
        q = e.get("query", "")
        r = e.get("response", "")
        ts = e.get("timestamp", "")
        sec = e.get("response_time_seconds", "")
        score = e.get("relevance_score", "")
        parts.append(
            f"[{i}] Past query: {q}\n"
            f"    Response: {r}\n"
            f"    When: {ts} (took {sec}s)\n"
            f"    Relevance: {score}"
        )
    return "\n\n".join(parts)
