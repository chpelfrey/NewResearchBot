"""Researcher agent - LangGraph + Ollama with DuckDuckGo search."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from research_bot.tools import search_news, search_web

# Default model - use a model with good tool-calling support (llama3.2, mistral, etc.)
DEFAULT_MODEL = "llama3.2"

RESEARCHER_SYSTEM_PROMPT = """You are a research assistant that finds accurate, up-to-date information from the internet.

CRITICAL: You MUST call search_web (and/or search_news for current events) BEFORE answering. Never answer from memory alone. Every answer must be based on actual search results.

When a user asks a question:
1. Immediately use search_web with effective queries (e.g., "Reston VA weather today", "weather Reston Virginia")
2. For weather, news, prices, or current events: also try search_news with similar queries
3. Run multiple searches if needed - try different phrasings to get useful results
4. Synthesize the search results into a clear answer with specific details from the results
5. Include key facts (temperatures, numbers, dates) when the search returns them

Be thorough. If search results lack specific details, say what you found and note any gaps.
If you cannot find relevant information after searching, say so clearly."""


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
    
    tools = [search_web, search_news]
    
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
        """Run a research query and return the final answer."""
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config or {},
        )
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return "I couldn't generate a response. Please try again."
    
    def stream(self, query: str, config: RunnableConfig | None = None):
        """Stream the research process (agent steps and final answer)."""
        return self.graph.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config or {},
            stream_mode="values",
        )
