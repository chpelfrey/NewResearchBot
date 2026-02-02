#!/usr/bin/env python3
"""CLI entry point for the Researcher AI Bot."""

import argparse
import os
import re
import sys
import time

# Allow running without pip install: add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# OSC 8: make markdown links [text](url) clickable in supported terminals (iTerm2, etc.)
_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def make_links_clickable(text: str) -> str:
    """Convert markdown links [text](url) to OSC 8 terminal hyperlinks."""
    def _replace(m):
        label, url = m.group(1), m.group(2)
        # OSC 8: \033]8;;url\033\\label\033]8;;\033\\
        return f"\033]8;;{url}\033\\[{label}]\033]8;;\033\\"
    return _LINK_PATTERN.sub(_replace, text)


def main():
    parser = argparse.ArgumentParser(
        description="Researcher AI Bot - Agentic web research using DuckDuckGo and open-source LLMs"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Research query (or leave empty for interactive mode)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        help="Ollama model name (default: llama3.2 or OLLAMA_MODEL env)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature 0-1 (default: 0.2)",
    )
    parser.add_argument(
        "-s",
        "--stream",
        action="store_true",
        help="Stream the research process (show agent steps); uses researcher only, no fact-check pipeline",
    )
    parser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="Skip fact-check and formatter (researcher only, faster)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_BASE_URL"),
        help="Ollama API base URL (for remote Ollama)",
    )
    args = parser.parse_args()

    query = " ".join(args.query) if args.query else None

    use_pipeline = not args.stream and not args.quick

    if not query:
        # Interactive mode
        print("Researcher AI Bot - Type your query and press Enter. Type 'quit' or 'exit' to stop.\n")
        if use_pipeline:
            print("Mode: researcher → fact-check → formatter (full pipeline). Use -q for quick, -s to stream.\n")
        try:
            from research_bot import ResearchAgent, ResearchPipeline

            agent = (
                ResearchPipeline(
                    model=args.model,
                    temperature=args.temperature,
                    base_url=args.base_url,
                )
                if use_pipeline
                else ResearchAgent(
                    model=args.model,
                    temperature=args.temperature,
                    base_url=args.base_url,
                )
            )
            while True:
                try:
                    query = input("Query: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not query:
                    continue
                if query.lower() in ("quit", "exit", "q"):
                    break
                if use_pipeline:
                    print("\nResearching, fact-checking, formatting...\n")
                else:
                    print("\nResearching...\n")
                if args.stream:
                    start = time.perf_counter()
                    last_answer = None
                    for chunk in agent.stream(query):
                        for msg in chunk.get("messages", []):
                            if hasattr(msg, "content") and msg.content:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    print(f"[Tool calls: {[tc.get('name') for tc in msg.tool_calls]}]\n")
                                else:
                                    last_answer = msg.content
                                    print(make_links_clickable(msg.content))
                                    print()
                    elapsed = time.perf_counter() - start
                    if last_answer:
                        try:
                            from research_bot.research_log import append_entry
                            append_entry(query=query, response=last_answer, response_time_seconds=elapsed)
                        except OSError:
                            pass
                else:
                    answer = agent.research(query)
                    print("Answer:\n")
                    print(make_links_clickable(answer))
                print("-" * 50)
        except ImportError as e:
            print("Error: Could not import research_bot. Install dependencies:", file=sys.stderr)
            print("  pip install -r requirements.txt", file=sys.stderr)
            sys.exit(1)
        return

    # Single query mode
    try:
        from research_bot import ResearchAgent, ResearchPipeline

        agent = (
            ResearchPipeline(
                model=args.model,
                temperature=args.temperature,
                base_url=args.base_url,
            )
            if use_pipeline
            else ResearchAgent(
                model=args.model,
                temperature=args.temperature,
                base_url=args.base_url,
            )
        )
        if args.stream:
            start = time.perf_counter()
            last_answer = None
            for chunk in agent.stream(query):
                for msg in chunk.get("messages", []):
                    if hasattr(msg, "content") and msg.content:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            print(f"[Tool: {[tc.get('name') for tc in msg.tool_calls]}]\n")
                        else:
                            last_answer = msg.content
                            print(make_links_clickable(msg.content))
            elapsed = time.perf_counter() - start
            if last_answer:
                try:
                    from research_bot.research_log import append_entry
                    append_entry(query=query, response=last_answer, response_time_seconds=elapsed)
                except OSError:
                    pass
        else:
            if use_pipeline:
                print("Researching, fact-checking, formatting...")
            answer = agent.research(query)
            print(make_links_clickable(answer))
    except ImportError as e:
        print("Error: Could not import research_bot. Install dependencies:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
