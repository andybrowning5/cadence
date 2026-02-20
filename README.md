# Cadence — Task Prioritization Agent

A [Primordial AgentStore](https://github.com/andybrowning5/AgentStore) agent. Tell Cadence your tasks and deadlines, and it suggests an optimal order. Remembers your tasks across sessions using local filesystem memory.

## Usage

```bash
pip install primordial-agentstore
primordial run https://github.com/andybrowning5/cadence
```

Or run locally:

```bash
cd cadence
pip install -e .
cadence
```

## API Keys Required

- **Anthropic** — for Claude inference ([anthropic.com](https://console.anthropic.com))

## What It Does

Talk to Cadence naturally:
- "I need to finish the report by Friday"
- "What should I work on next?"
- "I finished the report"
- "What do you remember about my deadlines?"

Cadence prioritizes based on deadlines, effort, dependencies, urgency, and quick wins. Conversation history and facts persist across sessions via the sandbox filesystem — no external services needed.

## Tech Stack

- Python 3.11+, LangChain Deep Agents + LangGraph
- Local filesystem memory (JSON)
- Rich TUI with live activity display
- Anthropic Claude (or OpenAI fallback)
