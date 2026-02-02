"""Search tools for the researcher agent - DuckDuckGo, Reddit, Wikipedia, Weather, and many APIs."""

import json
import os
import xml.etree.ElementTree as ET
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


def _fetch_json(url: str, timeout: float = 10.0, headers: dict | None = None) -> dict | list | None:
    """GET URL and parse JSON. Returns None on failure."""
    h = {"User-Agent": "ResearchBot/1.0 (research assistant; mailto:research@example.com)"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def _fetch_xml(url: str, timeout: float = 15.0) -> ET.Element | None:
    """GET URL and parse XML. Returns root element or None."""
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchBot/1.0 (research assistant)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return ET.fromstring(resp.read().decode())
    except (urllib.error.URLError, OSError, ET.ParseError):
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


# --- Academic & Scientific ---

_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


@tool
def search_arxiv(
    query: str,
    max_results: int = 10,
) -> str:
    """Search arXiv for pre-print research papers in physics, math, CS, and more. Use for academic papers and preprints.
    No API key required.

    Args:
        query: Search terms (e.g., 'transformer attention', 'quantum computing').
        max_results: Number of results (default 10, max 30).

    Returns:
        Paper titles, authors, abstracts, PDF links, and citation URLs.
    """
    max_results = min(max(max_results, 1), 30)
    try:
        url = (
            "https://export.arxiv.org/api/query?"
            + urllib.parse.urlencode({"search_query": f"all:{query}", "start": 0, "max_results": max_results})
        )
        root = _fetch_xml(url)
        if root is None:
            return f"arXiv request failed for: {query}"
        entries = root.findall("atom:entry", _ARXIV_NS)
        if not entries:
            return f"No arXiv papers found for: {query}"
        parts = []
        for i, entry in enumerate(entries, 1):
            title_el = entry.find("atom:title", _ARXIV_NS)
            title = (title_el.text or "").strip().replace("\n", " ")
            link_el = entry.find("atom:id", _ARXIV_NS)
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            pdf = ""
            for l in entry.findall("atom:link", _ARXIV_NS):
                if l.get("title") == "pdf":
                    pdf = l.get("href", "")
                    break
            summary_el = entry.find("atom:summary", _ARXIV_NS)
            summary = (summary_el.text or "").strip().replace("\n", " ")[:500] if summary_el is not None else ""
            authors = [a.find("atom:name", _ARXIV_NS) for a in entry.findall("atom:author", _ARXIV_NS)]
            author_names = [a.text.strip() for a in authors if a is not None and a.text]
            auth_str = ", ".join(author_names[:5]) + (" et al." if len(author_names) > 5 else "")
            parts.append(
                f"[{i}] {title}\n    URL: {link}\n    PDF: {pdf}\n    Authors: {auth_str}\n    {summary}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"arXiv search failed: {str(e)}"


@tool
def search_pubmed(
    query: str,
    max_results: int = 10,
) -> str:
    """Search PubMed/PMC for biomedical and life sciences literature. Use for medical and biology papers.
    No API key required.

    Args:
        query: Search terms (e.g., 'CRISPR therapy', 'COVID vaccine efficacy').
        max_results: Number of results (default 10, max 50).

    Returns:
        Article titles, authors, journal, PMID, and links to abstracts.
    """
    max_results = min(max(max_results, 1), 50)
    try:
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            + urllib.parse.urlencode({"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"})
        )
        data = _fetch_json(search_url)
        if not data or "esearchresult" not in data:
            return f"PubMed search failed for: {query}"
        esr = data["esearchresult"]
        id_list = esr.get("idlist", [])
        if not id_list:
            return f"No PubMed articles found for: {query}"
        fetch_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
            + urllib.parse.urlencode({"db": "pubmed", "id": ",".join(id_list), "retmode": "json"})
        )
        fetch_data = _fetch_json(fetch_url)
        if not fetch_data or "result" not in fetch_data:
            return f"PubMed fetch failed for: {query}"
        result = fetch_data["result"]
        parts = []
        for i, pid in enumerate(id_list, 1):
            if pid not in result:
                continue
            item = result[pid]
            title = item.get("title", "No title")
            auth_list = item.get("authors", [])
            auth_str = ", ".join(a.get("name", "") for a in auth_list[:5])
            if len(auth_list) > 5:
                auth_str += " et al."
            journal = item.get("fulljournalname", item.get("source", ""))
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            parts.append(f"[{i}] {title}\n    URL: {url}\n    Authors: {auth_str}\n    Journal: {journal}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"PubMed search failed: {str(e)}"


@tool
def search_semantic_scholar(
    query: str,
    max_results: int = 10,
) -> str:
    """Search Semantic Scholar for academic papers with AI-powered search and citations. Use for CS, ML, and general science.
    No API key required for basic use.

    Args:
        query: Search terms (e.g., 'large language models', 'graph neural networks').
        max_results: Number of results (default 10, max 25).

    Returns:
        Paper titles, authors, year, abstract snippet, and URL.
    """
    max_results = min(max(max_results, 1), 25)
    try:
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search?"
            + urllib.parse.urlencode({
                "query": query,
                "limit": max_results,
                "fields": "title,url,year,authors,abstract",
            })
        )
        data = _fetch_json(url, timeout=15.0)
        if not data or "data" not in data:
            return f"Semantic Scholar search failed or no results for: {query}"
        papers = data.get("data", [])
        if not papers:
            return f"No Semantic Scholar papers found for: {query}"
        parts = []
        for i, p in enumerate(papers, 1):
            title = p.get("title", "No title")
            paper_url = p.get("url", "")
            year = p.get("year", "")
            authors = p.get("authors", [])
            auth_str = ", ".join(a.get("name", "") for a in authors[:5])
            if len(authors) > 5:
                auth_str += " et al."
            abstract = (p.get("abstract") or "")[:400]
            parts.append(
                f"[{i}] {title}\n    URL: {paper_url}\n    Year: {year}\n    Authors: {auth_str}\n    {abstract}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"Semantic Scholar search failed: {str(e)}"


@tool
def search_crossref(
    query: str,
    max_results: int = 10,
) -> str:
    """Search CrossRef for scholarly metadata and DOIs. Use for published journal articles and DOI lookups.
    No API key required (polite pool with User-Agent).

    Args:
        query: Search terms or DOI (e.g., 'machine learning survey', '10.1038/nature').
        max_results: Number of results (default 10, max 20).

    Returns:
        Title, authors, DOI, journal, year, and URL.
    """
    max_results = min(max(max_results, 1), 20)
    try:
        url = (
            "https://api.crossref.org/works?"
            + urllib.parse.urlencode({"query": query, "rows": max_results})
        )
        data = _fetch_json(url)
        if not data or "message" not in data or "items" not in data["message"]:
            return f"CrossRef search failed or no results for: {query}"
        items = data["message"]["items"]
        if not items:
            return f"No CrossRef works found for: {query}"
        parts = []
        for i, w in enumerate(items, 1):
            title = " ".join(w.get("title", [])) or "No title"
            doi = w.get("DOI", "")
            url_doi = f"https://doi.org/{doi}" if doi else ""
            author_list = w.get("author", [])
            auth_str = ", ".join(f"{a.get('given','')} {a.get('family','')}".strip() for a in author_list[:5])
            if len(author_list) > 5:
                auth_str += " et al."
            year = (w.get("published", {}) or w.get("published-print", {}) or {}).get("date-parts", [[""]])[0][0]
            container = w.get("container-title", [])
            journal = container[0] if container else ""
            parts.append(
                f"[{i}] {title}\n    DOI: {url_doi}\n    Authors: {auth_str}\n    Year: {year}\n    Journal: {journal}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"CrossRef search failed: {str(e)}"


# --- Data & Statistics ---

# Common World Bank indicator IDs for reference
_WORLD_BANK_INDICATORS = {
    "gdp": "NY.GDP.MKTP.CD",
    "gdp growth": "NY.GDP.MKTP.KD.ZG",
    "population": "SP.POP.TOTL",
    "unemployment": "SL.UEM.TOTL.ZS",
    "life expectancy": "SP.DYN.LE00.IN",
    "co2": "EN.ATM.CO2E.KT",
    "literacy": "SE.ADT.LITR.ZS",
}


@tool
def search_world_bank(
    indicator_id: str,
    country: str = "USA",
    max_results: int = 10,
) -> str:
    """Get World Bank development data and economic indicators. Use for GDP, population, health, education stats.
    No API key required. Use indicator ID (e.g., NY.GDP.MKTP.CD for GDP, SP.POP.TOTL for population) or a known key: gdp, population, unemployment, life expectancy, co2, literacy.

    Args:
        indicator_id: Indicator code (e.g., NY.GDP.MKTP.CD) or key: gdp, population, unemployment, life expectancy, co2, literacy.
        country: Country code (e.g., USA, GBR, WLD for world). Default USA.
        max_results: Number of years of data (default 10, max 25).

    Returns:
        Indicator name, value, year, and source URL.
    """
    max_results = min(max(max_results, 1), 25)
    try:
        ind_id = _WORLD_BANK_INDICATORS.get(indicator_id.strip().lower(), indicator_id.strip())
        data_url = (
            f"https://api.worldbank.org/v2/country/{country}/indicator/{urllib.parse.quote(ind_id)}?"
            + urllib.parse.urlencode({"format": "json", "per_page": max_results, "date": "2010:2024"})
        )
        data = _fetch_json(data_url)
        if not data or not isinstance(data, list) or len(data) < 2:
            return (
                f"No World Bank data for indicator '{indicator_id}' in {country}. "
                "Try NY.GDP.MKTP.CD (GDP), SP.POP.TOTL (population), or keys: gdp, population, unemployment, life expectancy, co2, literacy. "
                "See https://data.worldbank.org/indicator"
            )
        entries = data[1]
        if not entries:
            return f"No World Bank data for {indicator_id} in {country}."
        parts = [f"World Bank data for {country} (indicator: {ind_id}):"]
        for i, e in enumerate(entries[:max_results], 1):
            val = e.get("value")
            year = e.get("date", "")
            name = e.get("indicator", {}).get("value", ind_id) if isinstance(e.get("indicator"), dict) else ind_id
            parts.append(f"  [{i}] {year}: {val} — {name}")
        parts.append("Source: https://data.worldbank.org/")
        return "\n".join(parts)
    except Exception as e:
        return f"World Bank lookup failed: {str(e)}"


@tool
def search_fred(
    series_id: str,
    api_key: str | None = None,
) -> str:
    """Get Federal Reserve Economic Data (FRED) time series. Use for US economic data (GDP, unemployment, rates).
    Requires FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html (set FRED_API_KEY env var).

    Args:
        series_id: FRED series ID (e.g., GDP, UNRATE, FEDFUNDS).
        api_key: Optional API key; otherwise uses FRED_API_KEY env var.

    Returns:
        Series description and recent observations with dates and values.
    """
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        return "FRED API requires an API key. Set FRED_API_KEY or pass api_key. Get one at https://fred.stlouisfed.org/docs/api/api_key.html"
    try:
        base = "https://api.stlouisfed.org/fred"
        series_url = f"{base}/series?series_id={series_id}&api_key={key}&file_type=json"
        meta = _fetch_json(series_url)
        if not meta or "seriess" not in meta or not meta["seriess"]:
            return f"FRED series not found: {series_id}. Try GDP, UNRATE, FEDFUNDS, etc."
        info = meta["seriess"][0]
        title = info.get("title", series_id)
        obs_url = f"{base}/series/observations?series_id={series_id}&api_key={key}&file_type=json&sort_order=desc&limit=24"
        obs = _fetch_json(obs_url)
        if not obs or "observations" not in obs:
            return f"FRED: {title} — no observations."
        parts = [f"FRED: {title}", "Recent observations:"]
        for o in obs["observations"][:20]:
            parts.append(f"  {o.get('date','')}: {o.get('value','')}")
        parts.append("Source: https://fred.stlouisfed.org/")
        return "\n".join(parts)
    except Exception as e:
        return f"FRED lookup failed: {str(e)}"


@tool
def get_openweather(
    location: str,
    api_key: str | None = None,
) -> str:
    """Get current weather from OpenWeather (alternative to Open-Meteo). Use for weather when Open-Meteo is insufficient.
    Requires OpenWeather API key (free tier): https://openweathermap.org/api (set OPENWEATHER_API_KEY env var).

    Args:
        location: City name or 'City, Country' (e.g., 'London, UK').
        api_key: Optional API key; otherwise uses OPENWEATHER_API_KEY env var.

    Returns:
        Current weather: temp, conditions, humidity, wind, and citation URL.
    """
    key = api_key or os.environ.get("OPENWEATHER_API_KEY")
    if not key:
        return "OpenWeather API requires a key. Set OPENWEATHER_API_KEY or get one at https://openweathermap.org/api"
    location = (location or "").strip()
    if not location:
        return "Please provide a location (e.g., city name or 'City, Country')."
    try:
        geo_url = (
            "https://api.openweathermap.org/geo/1.0/direct?"
            + urllib.parse.urlencode({"q": location, "limit": 1, "appid": key})
        )
        geo = _fetch_json(geo_url)
        if not geo or not isinstance(geo, list) or len(geo) == 0:
            return f"Location not found: {location}"
        lat = geo[0].get("lat")
        lon = geo[0].get("lon")
        name = geo[0].get("name", location)
        country = geo[0].get("country", "")
        weather_url = (
            "https://api.openweathermap.org/data/2.5/weather?"
            + urllib.parse.urlencode({"lat": lat, "lon": lon, "appid": key, "units": "metric"})
        )
        w = _fetch_json(weather_url)
        if not w or "main" not in w:
            return "OpenWeather data unavailable."
        main = w["main"]
        temp = main.get("temp")
        humidity = main.get("humidity")
        desc = (w.get("weather", [{}])[0].get("description", ""))
        wind = w.get("wind", {})
        wind_speed = wind.get("speed")
        cite = f"https://openweathermap.org/city/{w.get('id','')}"
        return (
            f"Weather for {name}, {country} (OpenWeather):\n"
            f"  Temperature: {temp}°C | Conditions: {desc}\n"
            f"  Humidity: {humidity}% | Wind: {wind_speed} m/s\n"
            f"  Source: {cite}"
        )
    except Exception as e:
        return f"OpenWeather failed: {str(e)}"


@tool
def search_datagov(
    query: str,
    max_results: int = 10,
) -> str:
    """Search US government datasets on data.gov. Use for official US federal datasets (census, health, climate).
    No API key required for catalog search.

    Args:
        query: Search terms (e.g., 'census population', 'climate temperature').
        max_results: Number of datasets (default 10, max 20).

    Returns:
        Dataset title, description snippet, organization, and URL.
    """
    max_results = min(max(max_results, 1), 20)
    try:
        url = (
            "https://catalog.data.gov/api/3/action/package_search?"
            + urllib.parse.urlencode({"q": query, "rows": max_results})
        )
        data = _fetch_json(url)
        if not data or "result" not in data or "results" not in data["result"]:
            return f"No data.gov results for: {query}"
        results = data["result"]["results"]
        if not results:
            return f"No data.gov datasets found for: {query}"
        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            notes = (r.get("notes") or "")[:300]
            org = (r.get("organization", {}) or {}).get("title", "") if isinstance(r.get("organization"), dict) else ""
            url_res = (r.get("resources", [{}])[0].get("url", "")) if r.get("resources") else ""
            link = r.get("url", url_res) or ""
            if link and not link.startswith("http"):
                link = "https://catalog.data.gov/dataset/" + (r.get("name", "") or "")
            parts.append(f"[{i}] {title}\n    Org: {org}\n    {notes}\n    URL: {link}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"data.gov search failed: {str(e)}"


# --- News & Media ---

@tool
def search_newsapi(
    query: str,
    max_results: int = 10,
    api_key: str | None = None,
) -> str:
    """Search news articles from various sources via NewsAPI. Use for recent news when DuckDuckGo news is insufficient.
    Free tier available; set NEWSAPI_KEY env var or pass api_key.

    Args:
        query: Search query (e.g., 'Federal Reserve interest rate').
        max_results: Number of articles (default 10, max 20).
        api_key: Optional; uses NEWSAPI_KEY env var if not set.

    Returns:
        Headlines, source, date, URL, and description.
    """
    key = api_key or os.environ.get("NEWSAPI_KEY")
    if not key:
        return "NewsAPI requires an API key (free tier). Set NEWSAPI_KEY or get one at https://newsapi.org/"
    max_results = min(max(max_results, 1), 20)
    try:
        url = (
            "https://newsapi.org/v2/everything?"
            + urllib.parse.urlencode({
                "q": query,
                "apiKey": key,
                "pageSize": max_results,
                "sortBy": "publishedAt",
                "language": "en",
            })
        )
        data = _fetch_json(url)
        if not data or data.get("status") != "ok":
            return f"NewsAPI error or no results for: {query}"
        articles = data.get("articles", [])
        if not articles:
            return f"No news articles found for: {query}"
        parts = []
        for i, a in enumerate(articles, 1):
            title = a.get("title", "No title")
            url_art = a.get("url", "")
            desc = (a.get("description") or "")[:200]
            source = a.get("source", {}).get("name", "")
            date = a.get("publishedAt", "")[:10]
            parts.append(f"[{i}] {title}\n    Source: {source} | Date: {date}\n    URL: {url_art}\n    {desc}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"NewsAPI search failed: {str(e)}"


@tool
def search_internet_archive(
    query: str,
    max_results: int = 10,
) -> str:
    """Search Internet Archive for historical web snapshots and digital collections. Use for archived pages, books, media.
    No API key required.

    Args:
        query: Search terms (e.g., 'NASA Apollo', 'Wayback Machine').
        max_results: Number of results (default 10, max 25).

    Returns:
        Title, identifier, type, date, and link to archive.
    """
    max_results = min(max(max_results, 1), 25)
    try:
        url = (
            "https://archive.org/advancedsearch.php?"
            + urllib.parse.urlencode({
                "q": query,
                "fl[]": ["identifier", "title", "type", "publicdate"],
                "sort[]": "publicdate desc",
                "output": "json",
                "rows": max_results,
            })
        )
        # advancedsearch.php returns JSON
        data = _fetch_json(url)
        if not data or "response" not in data or "docs" not in data["response"]:
            return f"No Internet Archive results for: {query}"
        docs = data["response"]["docs"]
        if not docs:
            return f"No Internet Archive items found for: {query}"
        parts = []
        for i, d in enumerate(docs, 1):
            title = d.get("title", "No title")
            if isinstance(title, list):
                title = title[0] if title else "No title"
            ident = d.get("identifier", "")
            typ = d.get("type", "unknown")
            date = d.get("publicdate", "")
            link = f"https://archive.org/details/{ident}" if ident else ""
            parts.append(f"[{i}] {title}\n    Type: {typ} | Date: {date}\n    URL: {link}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Internet Archive search failed: {str(e)}"


# --- Code & Technical ---

@tool
def search_github(
    query: str,
    max_results: int = 10,
    sort: str = "best match",
) -> str:
    """Search GitHub for code repositories, issues, and developer activity. Use for open-source projects and code.
    No API key for public search; set GITHUB_TOKEN env var for higher rate limits.

    Args:
        query: Search terms (e.g., 'langchain python', 'react hooks').
        max_results: Number of repos (default 10, max 30).
        sort: 'best match', 'stars', 'forks', 'updated'. Default 'best match'.

    Returns:
        Repo name, description, stars, language, and URL.
    """
    max_results = min(max(max_results, 1), 30)
    try:
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        params = {"q": query, "per_page": max_results, "sort": "best-match" if sort == "best match" else sort}
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(params)
        data = _fetch_json(url, headers=headers)
        if not data or "items" not in data:
            return f"GitHub search failed or no results for: {query}"
        items = data.get("items", [])
        if not items:
            return f"No GitHub repositories found for: {query}"
        parts = []
        for i, r in enumerate(items, 1):
            full_name = r.get("full_name", "")
            desc = (r.get("description") or "")[:200]
            stars = r.get("stargazers_count", 0)
            lang = r.get("language", "")
            url_repo = r.get("html_url", "")
            parts.append(f"[{i}] {full_name}\n    Stars: {stars} | Language: {lang}\n    {desc}\n    URL: {url_repo}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"GitHub search failed: {str(e)}"


@tool
def search_stackexchange(
    query: str,
    site: str = "stackoverflow",
    max_results: int = 10,
) -> str:
    """Search Stack Exchange (Stack Overflow and related Q&A sites). Use for programming Q&A and code examples.
    No API key required (quota applies).

    Args:
        query: Search terms (e.g., 'python asyncio', 'react useState').
        site: Site to search: stackoverflow, serverfault, superuser, etc. Default stackoverflow.
        max_results: Number of questions (default 10, max 20).

    Returns:
        Question title, score, answer count, tags, and link.
    """
    max_results = min(max(max_results, 1), 20)
    try:
        url = (
            "https://api.stackexchange.com/2.3/search/advanced?"
            + urllib.parse.urlencode({
                "order": "desc",
                "sort": "relevance",
                "q": query,
                "site": site,
                "pagesize": max_results,
                "filter": "!-*jN(8np*HuXGU",
            })
        )
        data = _fetch_json(url)
        if not data or "items" not in data:
            return f"Stack Exchange search failed for: {query}"
        items = data.get("items", [])
        if not items:
            return f"No Stack Exchange questions found for: {query}"
        parts = []
        for i, q in enumerate(items, 1):
            title = q.get("title", "No title")
            link = q.get("link", "")
            score = q.get("score", 0)
            answer_count = q.get("answer_count", 0)
            tags = ", ".join(q.get("tags", [])[:5])
            parts.append(f"[{i}] {title}\n    Score: {score} | Answers: {answer_count} | Tags: {tags}\n    URL: {link}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Stack Exchange search failed: {str(e)}"


@tool
def search_pypi(
    package_name: str,
) -> str:
    """Get PyPI package information for a Python package. Use for package version, summary, home page.
    No API key required. Use the package name (e.g., 'requests', 'langchain').

    Args:
        package_name: Package name (e.g., 'requests', 'numpy').

    Returns:
        Package name, version, summary, home page, and PyPI URL.
    """
    name = (package_name or "").strip()
    if not name:
        return "Please provide a PyPI package name (e.g., requests, numpy)."
    try:
        url = "https://pypi.org/pypi/" + urllib.parse.quote(name) + "/json"
        data = _fetch_json(url)
        if not data or "info" not in data:
            return f"PyPI package not found: {package_name}. Check the name at https://pypi.org/"
        info = data.get("info", {})
        pkg_name = info.get("name", name)
        version = info.get("version", "")
        summary = info.get("summary", "") or "(No summary)"
        home = info.get("home_page", "")
        return (
            f"PyPI package: {pkg_name} (version {version})\n"
            f"  Summary: {summary}\n  Home: {home}\n"
            f"  URL: https://pypi.org/project/{pkg_name}/"
        )
    except Exception as e:
        return f"PyPI lookup failed: {str(e)}"


@tool
def search_npm(
    package_name: str,
) -> str:
    """Get npm package information for JavaScript/Node packages. Use for package version, description, repo.
    No API key required.

    Args:
        package_name: Package name (e.g., 'express', 'react').

    Returns:
        Package name, version, description, repository, and npm URL.
    """
    try:
        name = (package_name or "").strip()
        if not name:
            return "Please provide an npm package name."
        url = "https://registry.npmjs.org/" + urllib.parse.quote(name)
        data = _fetch_json(url)
        if not data:
            return f"npm package not found: {package_name}"
        # Scoped packages have different structure
        dist_tags = data.get("dist-tags", {})
        latest = dist_tags.get("latest", "")
        versions = data.get("versions", {})
        info = versions.get(latest, versions.get(list(versions.keys())[-1] if versions else "", {}))
        if not info:
            info = data
        desc = data.get("description", info.get("description", ""))
        repo = (info.get("repository") or data.get("repository")) or {}
        repo_url = ""
        if isinstance(repo, dict):
            repo_url = repo.get("url", "")
        elif isinstance(repo, str):
            repo_url = repo
        home = data.get("homepage", "")
        return (
            f"npm package: {data.get('name', package_name)} (latest: {latest})\n"
            f"  Description: {desc}\n  Repository: {repo_url}\n  Homepage: {home}\n"
            f"  URL: https://www.npmjs.com/package/{name}"
        )
    except Exception as e:
        return f"npm lookup failed: {str(e)}"


# --- General Knowledge ---

@tool
def query_wikidata(
    query: str,
    max_results: int = 10,
) -> str:
    """Query Wikidata for structured knowledge (Wikipedia's knowledge base). Use for facts, entities, relations.
    No API key required. Query in English or use SPARQL keywords.

    Args:
        query: Natural language or entity name (e.g., 'Albert Einstein', 'capital of France').
        max_results: Number of results (default 10, max 20).

    Returns:
        Entity labels, descriptions, and Wikidata URLs. For complex queries use Wikidata Query Service.
    """
    max_results = min(max(max_results, 1), 20)
    try:
        # Wikidata Wbsearchentities API
        url = (
            "https://www.wikidata.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "limit": max_results,
                "format": "json",
            })
        )
        data = _fetch_json(url)
        if not data or "search" not in data:
            return f"Wikidata search failed or no results for: {query}"
        items = data.get("search", [])
        if not items:
            return f"No Wikidata entities found for: {query}"
        parts = []
        for i, e in enumerate(items, 1):
            label = e.get("label", "No label")
            desc = (e.get("description") or "")[:150]
            id_ = e.get("id", "")
            url_entity = f"https://www.wikidata.org/wiki/{id_}" if id_ else ""
            parts.append(f"[{i}] {label}\n    {desc}\n    URL: {url_entity}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Wikidata query failed: {str(e)}"


@tool
def search_openlibrary(
    query: str,
    max_results: int = 10,
) -> str:
    """Search Open Library for book metadata and availability. Use for books, authors, ISBN.
    No API key required.

    Args:
        query: Book title, author, or subject (e.g., 'Dune Frank Herbert').
        max_results: Number of results (default 10, max 20).

    Returns:
        Title, author, first publish year, cover URL, and Open Library link.
    """
    max_results = min(max(max_results, 1), 20)
    try:
        url = (
            "https://openlibrary.org/search.json?"
            + urllib.parse.urlencode({"q": query, "limit": max_results})
        )
        data = _fetch_json(url)
        if not data or "docs" not in data:
            return f"No Open Library results for: {query}"
        docs = data.get("docs", [])
        if not docs:
            return f"No books found for: {query}"
        parts = []
        for i, d in enumerate(docs, 1):
            title = d.get("title", "No title")
            author_list = d.get("author_name", [])
            author = ", ".join(author_list[:3]) if isinstance(author_list, list) else str(author_list)
            year = d.get("first_publish_year", "")
            key = d.get("key", "")
            cover = d.get("cover_i", "")
            cover_url = f"https://covers.openlibrary.org/b/id/{cover}-M.jpg" if cover else ""
            link = f"https://openlibrary.org{key}" if key else ""
            parts.append(f"[{i}] {title}\n    Author(s): {author}\n    Year: {year}\n    URL: {link}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Open Library search failed: {str(e)}"


@tool
def search_wikiquote(
    query: str,
    max_results: int = 5,
) -> str:
    """Search Wikiquote for quotations. Use for famous quotes by person or topic.
    Uses Wikipedia API to fetch quote pages. No API key required.

    Args:
        query: Person name or topic (e.g., 'Albert Einstein', 'wisdom').
        max_results: Number of quote pages (default 5, max 10).

    Returns:
        Page title and URL to Wikiquote; for full quotes open the URL.
    """
    try:
        import wikipedia
    except ImportError:
        return "Wikiquote search uses Wikipedia package. Install: pip install wikipedia"
    max_results = min(max(max_results, 1), 10)
    try:
        # Wikiquote is a separate wiki; we can search via URL
        url = (
            "https://en.wikiquote.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "opensearch",
                "search": query,
                "limit": max_results,
                "format": "json",
            })
        )
        data = _fetch_json(url)
        if not data or not isinstance(data, list) or len(data) < 2:
            return f"No Wikiquote results for: {query}"
        titles = data[1]
        urls = data[3] if len(data) > 3 else []
        if not titles:
            return f"No Wikiquote pages found for: {query}"
        parts = []
        for i, (t, u) in enumerate(zip(titles, urls or []), 1):
            parts.append(f"[{i}] {t}\n    URL: {u}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Wikiquote search failed: {str(e)}"


# --- Social / Trends ---

@tool
def search_hacker_news(
    query: str,
    max_results: int = 10,
) -> str:
    """Search Hacker News via Algolia. Use for tech news and discussions.
    No API key required.

    Args:
        query: Search terms (e.g., 'LLM', 'startup funding').
        max_results: Number of hits (default 10, max 20).

    Returns:
        Title, points, author, URL, and discussion link.
    """
    max_results = min(max(max_results, 1), 20)
    try:
        url = (
            "https://hn.algolia.com/api/v1/search?"
            + urllib.parse.urlencode({"query": query, "hitsPerPage": max_results, "tags": "story"})
        )
        data = _fetch_json(url)
        if not data or "hits" not in data:
            return f"Hacker News search failed for: {query}"
        hits = data.get("hits", [])
        if not hits:
            return f"No Hacker News stories found for: {query}"
        parts = []
        for i, h in enumerate(hits, 1):
            title = h.get("title", "No title")
            url_story = h.get("url", "")
            points = h.get("points", 0)
            author = h.get("author", "")
            object_id = h.get("objectID", "")
            discuss = f"https://news.ycombinator.com/item?id={object_id}" if object_id else ""
            parts.append(
                f"[{i}] {title}\n    Points: {points} | By: {author}\n    URL: {url_story or discuss}\n    Discuss: {discuss}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"Hacker News search failed: {str(e)}"


@tool
def search_youtube(
    query: str,
    max_results: int = 5,
    api_key: str | None = None,
) -> str:
    """Search YouTube for video metadata (titles, channel, published date). Use for video references and transcripts info.
    Requires YouTube Data API key (quota limits apply): https://developers.google.com/youtube/v3 (set YOUTUBE_API_KEY).

    Args:
        query: Search terms (e.g., 'machine learning tutorial').
        max_results: Number of videos (default 5, max 15).
        api_key: Optional; uses YOUTUBE_API_KEY env var if not set.

    Returns:
        Video title, channel, published date, and link. Does not return transcripts.
    """
    key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not key:
        return "YouTube Data API requires a key (quota limits). Set YOUTUBE_API_KEY or get one at https://developers.google.com/youtube/v3/getting-started"
    max_results = min(max(max_results, 1), 15)
    try:
        url = (
            "https://www.googleapis.com/youtube/v3/search?"
            + urllib.parse.urlencode({
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "key": key,
            })
        )
        data = _fetch_json(url)
        if not data or "items" not in data:
            return f"YouTube search failed or no results for: {query}"
        items = data.get("items", [])
        if not items:
            return f"No YouTube videos found for: {query}"
        parts = []
        for i, v in enumerate(items, 1):
            sid = v.get("id", {})
            vid = sid.get("videoId", "") if isinstance(sid, dict) else ""
            snip = v.get("snippet", {})
            title = snip.get("title", "No title")
            channel = snip.get("channelTitle", "")
            published = snip.get("publishedAt", "")[:10]
            link = f"https://www.youtube.com/watch?v={vid}" if vid else ""
            parts.append(f"[{i}] {title}\n    Channel: {channel} | Date: {published}\n    URL: {link}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"YouTube search failed: {str(e)}"


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
