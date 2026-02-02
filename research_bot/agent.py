"""Researcher agent - LangGraph + Ollama with DuckDuckGo search."""

import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from research_bot.research_log import append_entry
from research_bot.tools import check_research_log, search_news, search_web

# Default model - use a model with good tool-calling support (llama3.2, mistral, etc.)
DEFAULT_MODEL = "llama3.2"

RESEARCHER_SYSTEM_PROMPT = """You are a research assistant that finds accurate information from the internet.

CRITICAL - Check the research log FIRST, then search when needed:
1. ALWAYS call check_research_log with the user's question first.
2. If the log has no relevant answers, you MUST call search_web (and search_news when appropriate). Never answer "who is X" or factual lookups from memory—always use search.
3. For weather, news, prices, or current events: use search_news and search_web with recent_only=True so results are fresh.
4. Synthesize log and/or search results into a clear answer with specific details (names, dates, numbers).

When you search:
- Use search_web with clear, specific queries. For "who is X": first search the exact name (e.g. "Alex Pretti"), then if you get "No results" or few snippets try 2–3 alternate phrasings (e.g. "Alex Pretti nurse", "Alex Pretti Minneapolis 2026", "Alex Pretti protest"). Do not conclude "I couldn't find" until you have tried multiple queries.
- For release dates: search exact phrases like "Nintendo Switch 2 release date" or "Switch 2 launch date".
- For weather, stock prices, or breaking news: use search_news or search_web with recent_only=True.
- If the first search returns "No results found" or empty-looking results, run another search with different wording before giving up.

Rule: Do not say you couldn't find someone or something until you have called search_web at least once and, if the first result was empty or unhelpful, tried at least one more query with different phrasing."""


def create_research_agent(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    base_url: str | None = None,
):
    """
    Create and return a compiled research agent graph.
    
    Args:
        model: Ollama model name (e.g., llama3.2, mistral, llama3.1)
        temperature: Sampling temperature (lower = more deterministic)
        base_url: Optional Ollama API base URL (for remote Ollama)
    
    Returns:
        The compiled graph - use .invoke() or .stream() to run
    """
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
    )
    
    tools = [check_research_log, search_web, search_news]
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=RESEARCHER_SYSTEM_PROMPT,
    )
    
    return agent


class ResearchAgent:
    """Convenience wrapper for the research agent."""
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        base_url: str | None = None,
    ):
        self.graph = create_research_agent(model=model, temperature=temperature, base_url=base_url)
    
    def research(self, query: str, config: RunnableConfig | None = None) -> str:
        """Run a research query and return the final answer. Logs Q&A and response time to the research log."""
        start = time.perf_counter()
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config or {},
        )
        elapsed = time.perf_counter() - start
        messages = result.get("messages", [])
        answer = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break
        if answer is None:
            answer = "I couldn't generate a response. Please try again."
        try:
            append_entry(query=query, response=answer, response_time_seconds=elapsed)
        except OSError:
            pass  # Don't fail the request if logging fails
        return answer
    
    def stream(self, query: str, config: RunnableConfig | None = None):
        """Stream the research process (agent steps and final answer)."""
        return self.graph.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config or {},
            stream_mode="values",
        )
