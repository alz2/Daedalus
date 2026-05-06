<p align="center">
  <h1 align="center">Daedalus</h1>
  <p align="center"><strong>The computer-control agent that teaches itself.</strong></p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#demo">Demo</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#contributing">Contributing</a>
</p>

---

Daedalus drives any desktop (Windows, macOS, Linux) over VNC using LLM-powered planning and a growing library of composable skills. Give it a goal in plain English — it explores the screen, builds a plan, executes it, and **learns from failure**. When something goes wrong, a learner loop analyzes the trace, proposes new skills or fixes, and retries autonomously.

No hard-coded workflows. No brittle selectors. Just vision, reasoning, and a skill library that grows with every run.

| | |
|---|---|
| **Goal-driven execution** | Describe what you want in natural language. The explorer decomposes it into a plan of typed, testable skills. |
| **Self-improving** | The learner analyzes failures, proposes new skills, and the implementor synthesizes them — all without human intervention. |
| **Vision-first** | Every decision is grounded in what's actually on screen via screenshots + a YOLO-based element detector. |
| **Composable skill library** | 18 built-in skills (click, type, scroll, vision queries, element location, etc.) plus any the agent creates at runtime. |
| **Any desktop, any OS** | Connects over VNC — works with Windows, macOS, and Linux targets from a single control plane. |
| **Full observability** | Every run produces a trace with screenshots, LLM exchanges, timing data, and optional screen recordings. |
| **Human-in-the-loop** | Plans require approval before execution. New skills require approval before adoption. Abort anytime with a hotkey. |

---

## Demo

> <p align="center"><strong>Task:</strong> <em>"Launch Firefox, navigate to nytimes.com/puzzles/sudoku, pick the hard puzzle and solve it."</em></p>

<p align="center">
  <img src="assets/demo_preview.gif" alt="Daedalus solving a NYT Hard Sudoku" width="600">
  <br>
  <em>The agent explores an unfamiliar site, learns the UI, writes a solver skill on the fly, and completes the puzzle autonomously. (4x speed)</em>
</p>

---

## Quick Start

```bash
# Install (editable)
uv sync --all-extras
uv pip install -e .

# Run unit tests (uses MockBackend, no VNC needed)
pytest

# Smoke-test against the mock backend
daedalus run --program examples/mock_smoke.yaml --backend mock

# Run a goal-driven task against a real machine
daedalus run --goal "Open Firefox and navigate to example.com" \
    --config config.local.yaml
```

### Prerequisites

- Python 3.11+
- A VNC server on the target machine (see [`scripts/setup_vnc_host.md`](scripts/setup_vnc_host.md))
- AWS credentials configured (for Bedrock LLM access)
- `uv` package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))

---

## How It Works

```
User Goal (natural language)
       │
       ▼
┌─────────────────┐    ┌─────────────────┐
│    Explorer     │◄───│  Agent Memory   │
│  (tool-loop)   │    └─────────────────┘
│                 │──► Skill retrieval (BM25)
│    plan()       │──► Program (YAML or Python)
└────────┬────────┘
         ▼
   Confirm Plan (approve / deny with comments / cancel)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Retry Loop (learner-driven, up to N attempts)         │
│                                                        │
│   Execute ──► Executor ──► Trace Recorder              │
│      │                                                 │
│      ├─ SUCCESS ──► Evaluator ──► Goal Verdict         │
│      │                                                 │
│      └─ FAILURE ──► Learner (analyzes trace)           │
│          ├─ proposes new skills ──► Implementor        │
│          ├─ proposes amendments to existing skills     │
│          └─ re-plans with suggestions ──► loop         │
└────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Explorer** | LLM tool-calling loop that decomposes goals into executable programs |
| **Planner** | Strategy decomposition, success criteria generation, and repair |
| **Learner** | Trace analysis — diagnoses failures, proposes fixes and new skills |
| **Implementor** | Synthesizes skill code with AST safety linting and sandbox testing |
| **Evaluator** | Post-execution goal verification via visual + trace criteria |
| **Executor** | Runs programs (YAML steps or Python) with timeouts and daemon management |
| **Backends** | `MockBackend` for testing, `VNCBackend` for production |
| **Memory** | Persistent cross-run fact store for learning from past runs |
| **Grounding** | YOLO-based UI element detection service |

---

## Configuration

Copy `config.example.yaml` to `config.local.yaml` and customize:

```yaml
backend:
  kind: vnc
  host_os: macos
  vnc:
    host: my-machine
    port: 5900
    password_env: DAEDALUS_VNC_PASSWORD   # reads from env var

llm:
  roles:
    planner:     bedrock/us.anthropic.claude-opus-4-6-v1
    explorer:    bedrock/us.anthropic.claude-opus-4-6-v1
    implementor: bedrock/us.anthropic.claude-sonnet-4-6
    learner:     bedrock/us.anthropic.claude-sonnet-4-6
    vision:      bedrock/us.anthropic.claude-sonnet-4-6
    cheap:       bedrock/us.anthropic.claude-sonnet-4-6
  aws_region: us-east-1
```

Credentials are never stored in config — only environment variable names are referenced.

---

## CLI Reference

```bash
daedalus run             # Execute a goal or pre-built program
daedalus plan            # Plan without executing
daedalus teach           # Run the learner on recent traces
daedalus implement       # Synthesize a skill from a spec
daedalus shell           # Interactive REPL
daedalus skills          # list | test [SKILL_ID] | sync
daedalus traces          # list | show TASK_ID
daedalus verify-litellm  # Check for supply-chain compromise
```

### Key flags for `daedalus run`

| Flag | Default | Description |
|------|---------|-------------|
| `--goal` | — | Natural-language task description |
| `--program` | — | Path to a pre-built plan YAML |
| `--config` | — | Path to YAML config |
| `--mode` / `-m` | `learn` | `learn` (explore+plan+learn), `explore` (explorer solves directly), `plan` (skip explorer) |
| `--backend` | `vnc` | `vnc` or `mock` |
| `--max-retries` / `-r` | 3 | Learner retry loops on failure |
| `--explore-steps` | 20 | Max explorer iterations |
| `--record` | off | Record screen during execution |
| `--yes` / `-y` | off | Auto-approve all prompts |

---

## Skills

Daedalus ships with 18 core skills:

`click_element` · `locate_element` · `locate_elements` · `click_all` · `type_text` · `type_shortcut` · `type_text_secret` · `scroll` · `mouse` · `wait` · `view_screen` · `vision_query` · `assert_screen_contains` · `store_query` · `populate_store_from_analysis` · `monitor_text_region` · `tick_counter` · `solve_sudoku`

Core skills are protected — the learner cannot modify them. But it can create **new** skills at runtime. For example, during the sudoku demo above, the agent autonomously created a `solve_sudoku` skill that parses the grid from a screenshot and computes the solution.

---

## Project Structure

```
src/daedalus/          Python source — agent core
  backends/            VNC + mock remote-desktop backends
  core/                Skill, registry, context, store, errors
  executor/            Program execution engine + daemon lifecycle
  explorer/            Goal → plan via LLM tool-calling
  planner/             Strategy + program generation
  learner/             Trace analysis + self-improvement
  implementor/         Skill code synthesis
  evaluator/           Goal verification
  llm/                 LiteLLM gateway with role-based routing
  tracing/             JSONL event recorder
  memory/              Persistent fact store
  ui/                  CLI confirmation + overlay
  repl/                Interactive shell
skills/                On-disk skill library (spec.yaml + skill.py)
services/grounding/    YOLO element detection microservice
frontend/              React/TypeScript trace viewer UI
scripts/               Utility scripts
examples/              Example task programs
tests/                 Unit + integration tests
```

---

## Security

- **LiteLLM supply-chain:** Versions 1.82.7–1.82.8 were compromised (March 2026 TeamPCP attack). This project pins `>=1.83.0` and includes `daedalus verify-litellm` to scan for IOCs.
- **Credentials:** All secrets are referenced by environment variable name only — never stored in config files.
- **Transparency:** Plans require user approval. New skills require user approval. An always-on-top overlay + `Ctrl+Shift+Esc` abort hotkey are active during execution.

---

## Contributing

```bash
git clone git@github.com:IcarusAICo/Daedalus.git
cd Daedalus
uv sync --all-extras
uv pip install -e ".[dev]"
pytest
```

---

## License

MIT
