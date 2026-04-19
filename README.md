# jina-grep

A LangChain agent with a **dynamic skill system** — routes user queries to the appropriate skill at runtime, then executes the selected skill's instructions using shell tools.

## Demo

![Demo](assets/demo.png)

## How It Works

```
User Query
    │
    ▼
┌─────────────┐     skill name     ┌──────────────────┐
│ Router LLM  │ ─────────────────► │  SkillRegistry   │
│             │                    │  (skills/*.md)   │
└─────────────┘                    └────────┬─────────┘
                                            │ skill content
                                            ▼
                                   ┌─────────────────┐
                                   │  Agent LLM      │
                                   │  + ShellTool    │
                                   └────────┬────────┘
                                            │
                                 Step 1 → Step 2 → …
                                            │
                                            ▼
                                        Answer
```

1. **Router** — a lightweight LLM call selects the best skill based on skill descriptions and the user's query. Uses a separate, smaller model for speed.
2. **SkillRegistry** — loads all `skills/*.md` files at startup, parsing frontmatter (`name`, `description`) and body (system prompt)
3. **Agent** — `create_agent()` receives the selected skill's content as `system_prompt` and `ShellTool` for executing shell commands in the `data/` directory. Each step is printed live as it executes.

## Project Structure

```
jina-grep/
├── skills/
│   └── fs_search.md        # skill: exact filesystem search (find/grep/cat)
├── data/                   # files the agent is allowed to query
├── skill_registry.py       # loads and indexes skill files
├── main.py                 # router + agent loop
├── Dockerfile              # container image definition
├── .dockerignore
├── Makefile                # build and run shortcuts
└── pyproject.toml
```

## Skills

Skills are plain Markdown files with a YAML frontmatter header:

```markdown
---
name: fs-search
description: Exact search within the filesystem. Use for known filenames, exact keywords, or simple file listing via find/grep/cat.
---
You are a search agent ...
```

| Field         | Purpose                                                      |
|---------------|--------------------------------------------------------------|
| `name`        | Unique identifier used for routing and lookup                |
| `description` | One-line summary shown to the router LLM for skill selection |
| Body          | Full system prompt injected into the agent at runtime        |

**Adding a new skill** requires no code changes — drop a new `.md` file into `skills/` and it will be picked up automatically on the next run.

## Prerequisites

- [Ollama](https://ollama.com) running locally with your models pulled:
  ```bash
  ollama pull gemma4:e4b-nvfp4       # agent model
  ollama pull lfm2.5-thinking:1.2b   # router model
  ```
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Docker](https://www.docker.com) for sandboxed execution

## Installation

```bash
git clone <repo-url>
cd jina-grep
uv sync
```

## Usage

### Local

```bash
uv run main.py
```

### Docker (recommended)

Runs the agent in an isolated container with a read-only filesystem, dropped capabilities, and `data/` mounted as read-only:

```bash
make build   # build the image
make run     # start the container
```

The container connects to Ollama on the host via `OLLAMA_BASE_URL=http://host.docker.internal:11434`. To override:

```bash
OLLAMA_BASE_URL=http://your-host:11434 make run
```

The terminal shows routing and each execution step live:

```
Dynamic Skill Agent  (Ctrl+C to exit)

Query: What are the key hyperparameters for XGBoost?

  → fs-search

  Step 1  find data/ -type f -name "*.md"
          data/Grandmaster Pro Tip...md
          data/XGBoost_Tips_and_Tricks.md

  Step 2  find data/ -type f -name "XGBoost_Tips_and_Tricks.md" -exec cat {} \;
          ...
          328 more lines

──────────────────────────────────────────────────────
The key hyperparameters for XGBoost are: objective, eval_metric, ...
```

To use different models, update the `router` and `llm` parameters in `main.py`:

```python
router = ChatOllama(model="your-router-model")
llm = ChatOllama(model="your-agent-model")
```

### Security flags

| Flag | Effect |
|---|---|
| `--read-only` | Container filesystem is read-only |
| `--tmpfs /tmp`, `--tmpfs /home/agent/.cache` | Writable in-memory mounts for temp files and uv cache |
| `--cap-drop ALL` | All Linux capabilities dropped |
| `--security-opt no-new-privileges` | Prevents privilege escalation |
| `-v data:/app/data:ro` | `data/` mounted read-only |

## Dependencies

| Package                  | Version   |
|--------------------------|-----------|
| `langchain`              | ≥ 1.2.15  |
| `langchain-community`    | ≥ 0.4.1   |
| `langchain-ollama`       | ≥ 1.1.0   |
| `langchain-experimental` | ≥ 0.4.1   |
| `rich`                   | ≥ 15.0.0  |
