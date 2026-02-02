"""Search tools for the researcher agent - DuckDuckGo, Reddit, Wikipedia, Weather."""

import json
import urllib.error
import urllib.parse
import urllib.request

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
def search_reddit(
    query: str,
    max_results: int = 10,
) -> str:
    """Search Reddit for posts and discussions. Use when the user asks about Reddit, opinions, experiences, or community discussions on a topic.
    Uses DuckDuckGo restricted to Reddit (no API key required).

    Args:
        query: The search query (e.g., "best running shoes 2024", "moving to Denver").
        max_results: Number of results (default 10, max 20).

    Returns:
        Reddit results with titles, URLs, and snippets. Cite as [N](url).
    """
    max_results = min(max(max_results, 1), 20)
    try:
        ddgs = DDGS()
        # Restrict to Reddit so results are from reddit.com
        reddit_query = f"site:reddit.com {query}"
        raw = ddgs.text(reddit_query, max_results=max_results)
        results = list(raw) if raw is not None else []
        return _format_text_results(results, query)
    except Exception as e:
        return f"Reddit search failed: {str(e)}"


def _fetch_json(url: str, timeout: float = 10.0) -> dict | list | None:
    """GET URL and parse JSON. Returns None on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchBot/1.0 (research assistant)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


# WMO weather code descriptions (Open-Meteo uses these)
_WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@tool
def get_weather(
    location: str,
) -> str:
    """Get current weather for a place. Use for questions like "weather in X", "temperature in Y", "is it raining in Z".
    Uses Open-Meteo (no API key required). Provide a city name, region, or "City, Country" for best results.

    Args:
        location: Place name (e.g., "Reston Virginia", "London UK", "Tokyo").

    Returns:
        Current weather summary with temperature, conditions, humidity, wind, and source URL for citation.
    """
    location = (location or "").strip()
    if not location:
        return "Please provide a location (e.g., city name or 'City, Country')."
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?"
        geo_url += urllib.parse.urlencode({"name": location, "count": 1, "language": "en"})
        geo_data = _fetch_json(geo_url)
        if not geo_data or "results" not in geo_data or not geo_data["results"]:
            return f"No location found for: {location}. Try a different spelling or 'City, Country'."
        res = geo_data["results"][0]
        lat = res.get("latitude")
        lon = res.get("longitude")
        name = res.get("name", location)
        admin1 = res.get("admin1", "")
        country = res.get("country", "")
        if admin1 and country:
            place_label = f"{name}, {admin1}, {country}"
        elif country:
            place_label = f"{name}, {country}"
        else:
            place_label = name
        forecast_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode({
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m",
                "timezone": "auto",
            })
        )
        forecast = _fetch_json(forecast_url)
        if not forecast or "current" not in forecast:
            return f"Weather data unavailable for: {place_label}."
        cur = forecast["current"]
        temp = cur.get("temperature_2m")
        humidity = cur.get("relative_humidity_2m")
        code = cur.get("weather_code", 0)
        wind_speed = cur.get("wind_speed_10m")
        wind_dir = cur.get("wind_direction_10m")
        conditions = _WMO_CODES.get(code, f"Weather code {code}")
        cite_url = f"https://open-meteo.com/en/weather?lat={lat}&lon={lon}"
        parts = [
            f"Weather for {place_label} (source: [Open-Meteo]({cite_url})):",
            f"- Temperature: {temp}°C" if temp is not None else "",
            f"- Conditions: {conditions}",
            f"- Humidity: {humidity}%" if humidity is not None else "",
            f"- Wind: {wind_speed} km/h from {wind_dir}°" if wind_speed is not None and wind_dir is not None else "",
        ]
        return "\n".join(p for p in parts if p).strip()
    except Exception as e:
        return f"Weather lookup failed: {str(e)}"


@tool
def search_wikipedia(
    query: str,
    max_results: int = 5,
    sentences_per_article: int = 4,
) -> str:
    """Search Wikipedia for encyclopedic articles. Use for definitions, concepts, biographies, places, and factual overviews.
    Uses the public Wikipedia API (no API key required).

    Args:
        query: The search query (e.g., "Nintendo Switch", "solar system", "Marie Curie").
        max_results: Number of articles to return (default 5, max 10).
        sentences_per_article: Summary length in sentences per article (default 4).

    Returns:
        Wikipedia results with title, URL, summary, and citation [N](url).
    """
    try:
        import wikipedia
    except ImportError:
        return "Wikipedia search unavailable: install the 'wikipedia' package (pip install wikipedia)."
    max_results = min(max(max_results, 1), 10)
    sentences_per_article = min(max(sentences_per_article, 1), 10)
    try:
        titles = wikipedia.search(query, results=max_results)
        if not titles:
            return f"No Wikipedia articles found for: {query}"
        output_parts = []
        for i, title in enumerate(titles, 1):
            try:
                summary = wikipedia.summary(title, sentences=sentences_per_article, auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                summary = f"(Disambiguation: {e.options[:5]})"
            except wikipedia.PageError:
                summary = "(Page not found)"
            page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            cite = f"[{i}]({page_url})"
            output_parts.append(
                f"[{i}] {title}\n    URL: {page_url}\n    Cite as: {cite}\n    {summary}"
            )
        return "\n\n".join(output_parts)
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"


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
