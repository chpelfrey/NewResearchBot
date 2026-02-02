# Researcher AI Bot

An agentic research assistant that uses **DuckDuckGo** for web search and **open-source LLMs** (via Ollama) to answer queries. The AI agent figures out what you're asking, searches the internet, and synthesizes findings into clear answers.

**100% open-source stack**: LangGraph, LangChain-Ollama, duckduckgo-search.

## Features

- **Agentic workflow**: The AI decides what to search for, runs multiple searches if needed, and synthesizes results
- **DuckDuckGo search**: No API keys required; uses the `duckduckgo-search` library
- **Local/remote LLMs**: Runs via Ollama (llama3.2, mistral, etc.)—fully local or remote
- **Streaming mode**: Watch the agent's tool calls and reasoning in real time

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running with a model that supports tool/function calling:
   - Recommended: `llama3.2`, `mistral`, `llama3.1`
   - Pull a model: `ollama pull llama3.2`
3. **Ollama running**: Start Ollama before using the bot (`ollama serve` or run the Ollama app)

## Installation

```bash
cd NewResearchBot
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Usage

### CLI

**Single query:**
```bash
python cli.py "What are the latest developments in quantum computing?"
```

**Interactive mode** (no query = enter prompts one by one):
```bash
python cli.py
```

**Stream the research process** (see agent steps and tool calls):
```bash
python cli.py -s "Current Bitcoin price"
```

**Custom model:**
```bash
python cli.py -m mistral "Explain rust programming"
```

**Remote Ollama:**
```bash
OLLAMA_BASE_URL=http://remote:11434 python cli.py "Research topic"
# or
python cli.py --base-url http://remote:11434 "Research topic"
```

### As a Python module

```python
from research_bot import ResearchAgent

agent = ResearchAgent(model="llama3.2", temperature=0.2)
answer = agent.research("What is the capital of France?")
print(answer)

# Or stream the process
for chunk in agent.stream("Latest AI news"):
    for msg in chunk.get("messages", []):
        if msg.content:
            print(msg.content)
```

### Search tool only

```python
from research_bot.tools import search_web

results = search_web.invoke({"query": "Python asyncio tutorial", "max_results": 5})
print(results)
```

## Project Structure

```
NewResearchBot/
├── cli.py              # CLI entry point
├── requirements.txt    # Dependencies
├── pyproject.toml      # Package config + script entry
├── README.md
└── research_bot/
    ├── __init__.py
    ├── agent.py        # LangGraph ReAct agent + Ollama
    └── tools.py        # DuckDuckGo search tool
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_MODEL` | Default model name (default: llama3.2) |
| `OLLAMA_BASE_URL` | Ollama API URL (default: http://localhost:11434) |

## Models

Use models that support **tool/function calling** in Ollama:

- **llama3.2** (recommended, 3B/1B)
- **llama3.1** (8B/70B)
- **mistral**
- **qwen2.5**

Check [Ollama's model library](https://ollama.ai/library) for more.

## License

MIT
