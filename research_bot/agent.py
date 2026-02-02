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

RESEARCHER_SYSTEM_PROMPT = """You are a research assistant that finds accurate, up-to-date information from the internet.

CRITICAL - Check the research log FIRST, then search only if needed:
1. ALWAYS call check_research_log with the user's question first. Use the same or similar phrasing as the user.
2. If the log returns relevant past answers, you may use them. Only call search_web and/or search_news to fill in gaps, get newer data, or when the log has no relevant answers.
3. For weather, news, prices, or current events: prefer search_news and search_web to get fresh results even if the log had something similar.
4. Synthesize log results and/or search results into a clear answer with specific details.
5. Include key facts (temperatures, numbers, dates) when available.

When you do search:
- Use search_web with effective queries (e.g., "Reston VA weather today", "weather Reston Virginia").
- For weather, news, prices, or current events: also try search_news with similar queries.
- Run multiple searches if needed; try different phrasings.

Be thorough. If search results lack specific details, say what you found and note any gaps.
If you cannot find relevant information after checking the log and searching, say so clearly."""


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
