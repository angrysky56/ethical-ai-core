# Ethical AI Core — IDE Assistant Guide

Instructions for AI assistants working with this codebase.

---

## Project Overview

This is a **Constitutional AI dataset generator and Judge training system**. The goal is to train a Judge that evaluates AI agent outputs against hierarchical ethical principles.

**Key Insight:** This is NOT about controlling user requests — it's about catching when AI agents (like you) hide issues, take shortcuts, or exhibit Moloch patterns.

---

## Quick Reference

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e .

# Configure (edit .env)
cp .env.example .env

# Launch UI
python ui.py  # http://localhost:7860

# CLI demos
python demo.py --prompt "..."
python demo.py --personas
python run_pipeline.py

# Tests
python -m pytest tests/ -v
```

---

## Architecture Map

```
ui.py                    # Web UI (Gradio)
demo.py                  # CLI demos
run_pipeline.py          # Full pipeline script

src/
├── config.py            # Settings from .env
├── llm_client.py        # Multi-provider LLM client
├── dataset/
│   ├── generator.py     # Constitutional pipeline: Prompt → Naive → Critique → Revision
│   ├── personas.py      # Persona generation + SQLite storage
│   └── prompts.py       # Red team prompts by category
├── judgment/
│   ├── principles.py    # Hierarchical evaluator (PrincipleEvaluator)
│   └── genrm.py         # Generative Reward Model
├── engine/
│   └── ephemeral.py     # Test-Time Training (EphemeralEgo)
└── layers/
    ├── dfa.py           # Direct Feedback Alignment (fixed B matrix)
    └── delta.py         # Deep Delta Learning (surgical erasure)
```

---

## Core Principles (Hierarchical)

**Tier 1: Deontology** — Hard constraint ("harm=harm")
**Tier 2: Virtue Ethics** — Wisdom, Integrity, Empathy
**Tier 3: Servant Utility** — Maximize utility within constraints

Plus **Agent Self-Governance** addendum for AI-specific failures.

See: `core_principles.md`

---

## Key Classes & Functions

### Dataset Generation

- `ConstitutionalGenerator` (`src/dataset/generator.py`) — Main pipeline
- `PersonaGenerator` (`src/dataset/personas.py`) — Persona-based prompts
- `PersonaDB` — SQLite storage at `data/personas.db`

### Judgment

- `PrincipleEvaluator` (`src/judgment/principles.py`) — LLM-based Constitutional critique
- Returns: `{tier_violated, principle_attribution, critique, verdict}`

### Training Layers

- `DeltaResidualBlock` (`src/layers/delta.py`) — Surgical erasure
- `MultiHeadDelta` — Multiple Delta heads for different concepts
- `DFALinear` (`src/layers/dfa.py`) — Fixed feedback matrix alignment

### Inference

- `EphemeralEgo` (`src/engine/ephemeral.py`) — Test-Time Training loop

---

## Common Tasks

### Add a new red team prompt category

Edit `src/dataset/prompts.py`, add to appropriate list.

### Modify the hierarchical critique template

Edit `CRITIQUE_TEMPLATE` in `src/dataset/generator.py`.

### Change LLM provider

Edit `.env`:

```bash
LLM_PROVIDER=ollama  # or openrouter, lmstudio, openai
OLLAMA_MODEL=qwen3-vl
```

### Run the Constitutional pipeline on a prompt

```python
from src.dataset.generator import ConstitutionalGenerator
gen = ConstitutionalGenerator()
sample = gen.generate_sample("your prompt here")
```

### Query the persona database

```python
from src.dataset.personas import PersonaDB
db = PersonaDB()
print(db.get_stats())
```

---

## Testing

```bash
python -m pytest tests/ -v
```

All 14 tests should pass.

---

## Configuration

Settings in `.env`:

- `LLM_PROVIDER` — ollama, openrouter, lmstudio, openai
- `OLLAMA_MODEL` — Model name (e.g., qwen3-vl)
- `OLLAMA_BASE_URL` — API endpoint
- `ETHICAL_PRINCIPLES_PATH` — Path to core principles file

---

## References

- [Spec](./Bootstrapping-Core-Self-via-AI-Feedback.md) — Full architecture design
- [Principles](./core_principles.md) — The constitution
- [README](./README.md) — User-facing overview
