"""Researcher agent - LangGraph + Ollama with DuckDuckGo search."""

import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from research_bot.research_log import append_entry
from research_bot.tools import check_research_log, search_news, search_web

# Default model - use a model with good tool-calling support (llama3.2, mistral, etc.)
DEFAULT_MODEL = "llama3.2"

RESEARCHER_SYSTEM_PROMPT = """You are a research assistant that finds accurate information from the internet. Your output MUST have sentence-level citations for every factual claim.

CRITICAL - Check the research log FIRST, then search when needed:
1. ALWAYS call check_research_log with the user's question first.
2. If the log has no relevant answers, you MUST call search_web (and search_news when appropriate). Never answer "who is X" or factual lookups from memory—always use search.
3. For weather, news, prices, or current events: use search_news and search_web with recent_only=True so results are fresh.
4. Synthesize log and/or search results into a clear answer with specific details (names, dates, numbers).

MANDATORY CITATIONS - You MUST cite every factual sentence. No exceptions:
- Put a citation IMMEDIATELY after each sentence. Format: [N](url) where N matches the search result number and url is the exact URL from the results.
- One sentence = at least one citation. Two facts in one sentence = two citations: [1](url1) [2](url2).
- Use ONLY URLs from the search results. Each result shows "Cite as: [N](url)"—use that exact form in your text.
- For information taken only from the research log (no web search), write " (from prior research log)" after that sentence.
- Do NOT output any factual claim without a citation right after it. If you cannot find a source for a claim, do not include that claim.
- Before finishing, verify: every sentence that states a fact has [N](url) or " (from prior research log)" immediately after it.
- Example: "The Nintendo Switch 2 launched in March 2025. [1](https://example.com) It features an OLED display. [2](https://other.com)"

When you search:
- Use search_web with clear, specific queries. For "who is X": first search the exact name, then try 2–3 alternate phrasings if the first search returns nothing useful.
- For release dates: search exact phrases like "Nintendo Switch 2 release date".
- For weather, stock prices, or breaking news: use search_news or search_web with recent_only=True.
- If the first search returns "No results found", run another search with different wording before giving up.

Rule: Do not say you couldn't find someone or something until you have called search_web at least once and tried at least one more query if the first was unhelpful."""


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


# --- Fact-checker and formatter (no tools, single LLM calls) ---

FACT_CHECKER_SYSTEM = """You are a fact-checker. You review research drafts and flag problems.

Given the user's original question and the researcher's draft (with citations), output a structured review in this exact format:

## UNCORROBORATED
- List any factual claims that lack a citation or that cite a source that does not clearly support the claim. Quote the sentence or phrase.

## POTENTIAL BIAS
- List any sentences that appear one-sided, promotional, or from a source with clear bias (e.g. advocacy, marketing). Quote and briefly say why.

## WEAK OR UNRELIABLE SOURCES
- List any cited URLs or domains that are known to be unreliable (e.g. content farms, sensationalist sites, unverified blogs). Give the URL or citation number and a one-line reason.

## OK
- If the rest of the draft is well-cited and neutral, say "Remaining content appears well-cited and balanced." Otherwise leave this section minimal.

Be concise. If there are no issues in a category, write "None." Do not repeat the full draft."""

FORMATTER_SYSTEM = """You are a formatter. You take a research draft and fact-checker feedback and produce a clean, final report.

Rules:
1. Preserve every sentence that is well-cited and NOT flagged by the fact-checker. Keep their citations exactly: [N](url).
2. For any sentence flagged as UNCORROBORATED: either remove it or rewrite it to state uncertainty and omit the unsupported claim (e.g. "Some reports suggest X [1](url), but this could not be independently verified."). Do not keep unverified facts as facts.
3. For content flagged as POTENTIAL BIAS: soften or balance the wording, or add a brief note (e.g. "According to [source], which has a stated position on Y, ..."). Keep the citation.
4. For WEAK OR UNRELIABLE SOURCES: if a claim relies only on a flagged source, remove the claim or replace with a better-cited one. If multiple sources support it, keep the claim and cite the stronger source(s) only.
5. Output only the final report: clear, neutral, with sentence-level citations. No meta-commentary, no "I removed...". Use markdown. End with a "Sources" or "References" line only if you add new structure; otherwise in-line citations are enough."""


def _llm(model: str, temperature: float, base_url: str | None):
    """Shared LLM for fact-checker and formatter."""
    return ChatOllama(model=model, temperature=temperature, base_url=base_url)


def fact_check(
    draft: str,
    query: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    base_url: str | None = None,
) -> str:
    """Run the fact-checker on the researcher draft. Returns structured feedback."""
    llm = _llm(model, temperature, base_url)
    prompt = f"""Original question: {query}

Researcher draft:

{draft}

Review the draft and output your structured fact-check (UNCORROBORATED, POTENTIAL BIAS, WEAK OR UNRELIABLE SOURCES, OK)."""
    msg = llm.invoke([SystemMessage(content=FACT_CHECKER_SYSTEM), HumanMessage(content=prompt)])
    return msg.content if hasattr(msg, "content") and msg.content else "No feedback."


def format_report(
    draft: str,
    fact_check_feedback: str,
    query: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    base_url: str | None = None,
) -> str:
    """Produce a clean, fully-cited report from the draft and fact-check feedback."""
    llm = _llm(model, temperature, base_url)
    prompt = f"""Original question: {query}

Researcher draft:

{draft}

Fact-checker feedback:

{fact_check_feedback}

Produce the final report following the formatter rules. Output only the report."""
    msg = llm.invoke([SystemMessage(content=FORMATTER_SYSTEM), HumanMessage(content=prompt)])
    return msg.content if hasattr(msg, "content") and msg.content else draft


class ResearchAgent:
    """Convenience wrapper for the research agent."""
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        base_url: str | None = None,
    ):
        self.graph = create_research_agent(model=model, temperature=temperature, base_url=base_url)
    
    def research(
        self,
        query: str,
        config: RunnableConfig | None = None,
        *,
        log: bool = True,
    ) -> str:
        """Run a research query and return the final answer. Logs Q&A to the research log unless log=False."""
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
        if log:
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


class ResearchPipeline:
    """Three-stage pipeline: researcher → fact-checker → formatter. Use this for cited, fact-checked reports."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        base_url: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self._researcher = ResearchAgent(model=model, temperature=temperature, base_url=base_url)

    def research(self, query: str, config: RunnableConfig | None = None) -> str:
        """Run researcher → fact-checker → formatter and return the final report. Logs to research log."""
        start = time.perf_counter()
        # Stage 1: researcher draft with sentence-level citations (do not log draft)
        draft = self._researcher.research(query, config=config or {}, log=False)
        # Stage 2: fact-check for uncorroborated / biased / weak sources
        feedback = fact_check(
            draft=draft,
            query=query,
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
        )
        # Stage 3: formatter produces clean, fully-cited report
        report = format_report(
            draft=draft,
            fact_check_feedback=feedback,
            query=query,
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
        )
        elapsed = time.perf_counter() - start
        try:
            append_entry(query=query, response=report, response_time_seconds=elapsed)
        except OSError:
            pass
        return report
