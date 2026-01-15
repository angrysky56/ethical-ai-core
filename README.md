# Ethical AI Core

**Constitutional AI Dataset Generator & Judge System**

Implements the Ephemeral Judgment Layer architecture from [Bootstrapping-Core-Self-via-AI-Feedback.md](./Bootstrapping-Core-Self-via-AI-Feedback.md).

## Purpose

Generate training data for a **Judge** that evaluates AI agent behavior against hierarchical principles — detecting shortcuts, deception, hidden complexity, and Moloch patterns.

**This is NOT about controlling user requests.** It's about training a model to catch when AI agents:

- Hide issues or take shortcuts
- Claim false certainty
- Are sycophantic or deceptive
- Exhibit Moloch patterns (optimizing appearance over substance)

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e .

# Configure LLM provider
cp .env.example .env
# Edit .env with your provider (Ollama, OpenRouter, etc.)

# ⚠️  WARNING: Use LOCAL models for adversarial dataset generation!
# Red team prompts may trigger cloud provider ToS violations.

# Launch Web UI
python ui.py
# Open http://localhost:7860
```

## Web UI

```bash
python ui.py
```

**Dataset Generation Tab:**

1. **Step 1: Generate Personas** — Create user personas for diverse prompt generation
2. **Step 2: Generate Prompts** — Generate prompts with configurable min difficulty/safety
3. **Step 3: Generate Samples** — Process through Constitutional AI pipeline

**LLM Configuration (Step 2):**

- Timeout, Batch Size
- Min Difficulty (1-5) — Higher = more complex prompts
- Min Safety (1-5) — Higher = more ethically challenging

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│              Constitutional AI Data Pipeline                  │
├───────────────────────────────────────────────────────────────┤
│  1. Prompt → Base Model → Naive Response                      │
│  2. (Prompt, Response) → PrincipleEvaluator → Critique        │
│  3. Critique → Revision Model → Corrected Response            │
│                                                               │
│  Output: (prompt, naive, critique, revised) for training      │
└───────────────────────────────────────────────────────────────┘
            ↓
┌───────────────────────────────────────────────────────────────┐
│              LoReFT + DFA Training                            │
├───────────────────────────────────────────────────────────────┤
│  LoReFTBlock: Low-rank activation intervention (steering)     │
│  GlobalDFAProjector: Skip-layer error projection via B        │
│  ReFTTrainer: Short-circuit training (no backward() needed)   │
└───────────────────────────────────────────────────────────────┘
```

## Core Technologies

| Component            | File                         | Description                                   |
| -------------------- | ---------------------------- | --------------------------------------------- |
| `LoReFTBlock`        | `src/layers/delta.py`        | Low-rank activation intervention              |
| `GlobalDFAProjector` | `src/layers/dfa.py`          | Skip-layer feedback via Ethical Matrix B      |
| `ReFTPolicy`         | `src/engine/reft.py`         | Policy with interventions at specified layers |
| `ReFTTrainer`        | `src/engine/reft.py`         | Short-circuit DFA training loop               |
| `EthicalMatrix`      | `src/ethical/matrix.py`      | Seeds B from `core_principles.md` hash        |
| `PrincipleEvaluator` | `src/judgment/principles.py` | 3-tier hierarchical GenRM                     |

## Core Principles (3-Tier Hierarchy)

| Tier | Name                | Role                                   |
| ---- | ------------------- | -------------------------------------- |
| 1    | **Deontology**      | Hard constraint — "harm=harm"          |
| 2    | **Virtue Ethics**   | Character — Wisdom, Integrity, Empathy |
| 3    | **Servant Utility** | Optimization within constraints        |

Plus **Agent Self-Governance** extensions for AI-specific failures.

## Training the Judge

```bash
# Run test harness
python -m src.training.test_harness

# Or from Python
from src.training.test_harness import ConstitutionalTestHarness
harness = ConstitutionalTestHarness()
harness.train(epochs=5)
```

## Key Commands

| Command                               | Purpose                               |
| ------------------------------------- | ------------------------------------- |
| `python ui.py`                        | Launch web UI                         |
| `python -m src.training.test_harness` | Train Judge on Constitutional dataset |
| `python demo.py --prompt "..."`       | Single prompt pipeline                |
| `python run_pipeline.py`              | Full CLI pipeline                     |

## Project Structure

```
src/
├── config.py              # LLM provider config
├── llm_client.py          # Multi-provider client (Ollama, OpenRouter, etc.)
├── dataset/
│   ├── generator.py       # Constitutional data pipeline
│   └── personas.py        # Persona-based generation + SQLite
├── judgment/
│   └── principles.py      # PrincipleEvaluator (GenRM)
├── engine/
│   ├── ephemeral.py       # EphemeralEgo + SurgicalEgo (TTT)
│   └── reft.py            # ReFTPolicy + ReFTTrainer
├── layers/
│   ├── dfa.py             # DFALinear + GlobalDFAProjector
│   └── delta.py           # DeltaResidualBlock + LoReFTBlock
├── ethical/
│   └── matrix.py          # EthicalMatrix (seeds B from principles)
└── training/
    └── test_harness.py    # Constitutional training harness
```

## References

- [Design Spec](./Bootstrapping-Core-Self-via-AI-Feedback.md)
- [Constitutional AI](https://arxiv.org/pdf/2212.08073)
- [LoReFT](https://github.com/stanfordnlp/pyreft)
- [Deep Delta Learning](https://arxiv.org/abs/2601.00417)
