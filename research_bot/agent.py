"""Researcher agent - LangGraph + Ollama with DuckDuckGo search."""

import time
from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from research_bot.research_log import append_entry
from research_bot.tools import (
    check_research_log,
    get_openweather,
    get_weather,
    query_wikidata,
    search_arxiv,
    search_crossref,
    search_datagov,
    search_fred,
    search_github,
    search_hacker_news,
    search_internet_archive,
    search_news,
    search_newsapi,
    search_npm,
    search_openlibrary,
    search_pubmed,
    search_pypi,
    search_reddit,
    search_semantic_scholar,
    search_stackexchange,
    search_web,
    search_wikipedia,
    search_wikiquote,
    search_world_bank,
    search_youtube,
)

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
- For weather: use get_weather (Open-Meteo) or get_openweather (needs OPENWEATHER_API_KEY) with a place name.
- For Reddit opinions, experiences, or discussions: use search_reddit.
- For encyclopedic facts, definitions, or overviews: use search_wikipedia.
- For stock prices or breaking news: use search_news or search_web with recent_only=True.
- Academic papers: use search_arxiv (preprints), search_pubmed (biomedical), search_semantic_scholar (AI/CS), search_crossref (DOIs/scholarly metadata).
- Economic data: use search_world_bank (development indicators), search_fred (US economic series; needs FRED_API_KEY).
- US government datasets: use search_datagov.
- News from many sources: use search_newsapi (needs NEWSAPI_KEY) or search_news (DuckDuckGo).
- Historical web/archives: use search_internet_archive.
- Code/repos: use search_github. Programming Q&A: use search_stackexchange. Package info: search_pypi (Python), search_npm (JavaScript).
- Structured knowledge: query_wikidata. Books: search_openlibrary. Quotes: search_wikiquote.
- Tech news/discussions: search_hacker_news. Video metadata: search_youtube (needs YOUTUBE_API_KEY).
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
    
    tools = [
        check_research_log,
        search_web,
        search_news,
        search_reddit,
        search_wikipedia,
        get_weather,
        # Academic & Scientific
        search_arxiv,
        search_pubmed,
        search_semantic_scholar,
        search_crossref,
        # Data & Statistics
        search_world_bank,
        search_fred,
        get_openweather,
        search_datagov,
        # News & Media
        search_newsapi,
        search_internet_archive,
        # Code & Technical
        search_github,
        search_stackexchange,
        search_pypi,
        search_npm,
        # General Knowledge
        query_wikidata,
        search_openlibrary,
        search_wikiquote,
        # Social / Trends
        search_hacker_news,
        search_youtube,
    ]
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=RESEARCHER_SYSTEM_PROMPT,
    )
    
    return agent


# --- Clarifier (no tools, single LLM call) ---

CLARIFIER_SYSTEM = """You are a clarifier. Your job is to take the user's research question and:

1. Confirm in one or two sentences what you understand they want to research (the main topic and any specific aspects).
2. Propose a short, ordered research plan so a researcher can fully answer the question. Break the question into clear steps or sub-questions if it has multiple parts (e.g. "compare X and Y" → step 1: research X, step 2: research Y, step 3: compare). For a single simple question, the plan can be one or two steps.

Output exactly this structure in markdown:

## What I'm researching
(Your 1–2 sentence confirmation of the question.)

## Suggested research plan
1. (First step or sub-question.)
2. (Second step, if needed.)
...
(Add steps as needed; keep the list concise.)

Do not answer the question yourself—only clarify and plan. Be concise."""


def clarify(
    query: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    base_url: str | None = None,
) -> str:
    """Run the clarifier on the user query. Returns confirmation + suggested research plan."""
    llm = _llm(model, temperature, base_url)
    prompt = f"""User's research question:\n\n{query}\n\nClarify what you're researching and give a suggested research plan (What I'm researching + Suggested research plan)."""
    msg = llm.invoke([SystemMessage(content=CLARIFIER_SYSTEM), HumanMessage(content=prompt)])
    return msg.content if hasattr(msg, "content") and msg.content else ""


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
1. Preserve every sentence that is well-cited and NOT flagged by the fact-checker. Keep their citations exactly and ALWAYS include the citations: [Citation Number](url). Every sentence must have a citation in the form of a numbered hyperlink.
2. For any sentence flagged as UNCORROBORATED: either remove it or rewrite it to state uncertainty and omit the unsupported claim (e.g. "Some reports suggest X [1](url), but this could not be independently verified."). Do not keep unverified facts as facts.
3. For content flagged as POTENTIAL BIAS: soften or balance the wording, or add a brief note (e.g. "According to [source], which has a stated position on Y, ..."). Keep the citation.
4. For WEAK OR UNRELIABLE SOURCES: if a claim relies only on a flagged source, remove the claim or replace with a better-cited one. If multiple sources support it, keep the claim and cite the stronger source(s) only.
5. Output only the final report: clear, neutral, with sentence-level citations. No meta-commentary, no "I removed...". Use markdown. End with a "Sources" or "References" line only if you add new structure; otherwise in-line citations are both necessary and sufficient.
6. Always include the citations in the final report. Do not omit them.
7. Writing should be in professional, but conversational tone, where information is synthesized rather than a simple list of facts."""



def _llm(model: str, temperature: float, base_url: str | None):
    """Shared LLM for fact-checker and formatter."""
    return ChatOllama(model=model, temperature=temperature, base_url=base_url)


def _tools_used_from_messages(messages: list) -> list[str]:
    """Extract tool names called during the run, in order (with duplicates preserved)."""
    names: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and "name" in tc:
                    names.append(tc["name"])
                elif hasattr(tc, "name"):
                    names.append(tc.name)
        if isinstance(msg, ToolMessage):
            names.append(getattr(msg, "name", ""))
    return names


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
        tools_used_out: list[str] | None = None,
    ) -> str:
        """Run a research query and return the final answer. Logs Q&A to the research log unless log=False.
        If tools_used_out is provided, appends the list of tool names called during the run (in order).
        """
        start = time.perf_counter()
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config or {},
        )
        elapsed = time.perf_counter() - start
        messages = result.get("messages", [])
        if tools_used_out is not None:
            tools_used_out.extend(_tools_used_from_messages(messages))
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
    """Four-stage pipeline: clarifier → researcher → fact-checker → formatter. Use this for cited, fact-checked reports."""

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

    def research(
        self,
        query: str,
        config: RunnableConfig | None = None,
        *,
        clarification_out: list[str] | None = None,
        on_plan_ready: Callable[[str], None] | None = None,
    ) -> str:
        """Run clarifier → researcher → fact-checker → formatter and return the final report. Logs to research log.

        If clarification_out is provided (e.g. a list), the clarifier's output is appended so the caller can display it.
        If on_plan_ready is provided, it is called with the clarification (scope + suggested research plan) before
        the research phase runs, so the caller can show the plan to the user first.
        The final report includes a "Tools used" section listing the tools called during research.
        """
        start = time.perf_counter()
        # Stage 0: clarifier confirms scope and suggests research plan
        clarification = clarify(
            query=query,
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
        )
        if clarification_out is not None:
            clarification_out.append(clarification)
        if on_plan_ready is not None:
            on_plan_ready(clarification)
        enhanced_query = (
            f"Original question: {query}\n\n{clarification}\n\n"
            "Follow the suggested research plan above and answer the original question fully with cited sources."
        )
        if not clarification.strip():
            enhanced_query = query  # fallback if clarifier returns nothing
        # Stage 1: researcher draft with sentence-level citations (do not log draft)
        tools_used: list[str] = []
        draft = self._researcher.research(
            enhanced_query, config=config or {}, log=False, tools_used_out=tools_used
        )
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
        # Append tools used in this run (order preserved, unique)
        seen: set[str] = set()
        unique_tools = [t for t in tools_used if t and t not in seen and not seen.add(t)]
        if unique_tools:
            report += "\n\n## Tools used\n" + ", ".join(unique_tools)
        elapsed = time.perf_counter() - start
        try:
            append_entry(query=query, response=report, response_time_seconds=elapsed)
        except OSError:
            pass
        return report
