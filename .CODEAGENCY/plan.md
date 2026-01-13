# Implementation Plan: AI Self-Governance Judge

## Goal

Build a system that generates training data for a Judge adapter that evaluates AI agent behavior against Core Principles — detecting shortcuts, deception, hidden complexity, and Moloch patterns.

---

## User Review Required

> [!IMPORTANT]
> This is a significant architectural pivot. The Judge evaluates **AI outputs**, not user requests.

**Key decisions needed:**

1. Should dataset generation use a local LLM (Ollama) or capture real conversation logs?
2. For training, do you want PEFT/LoRA on a local model, or just dataset generation for now?
3. What base model for the Judge? (qwen, llama, mistral)

---

## Proposed Changes

### Phase 1: Dataset Generation

#### [NEW] `src/dataset/__init__.py`

Empty init.

#### [NEW] `src/dataset/generator.py`

```python
class CritiqueGenerator:
    """
    Generates (prompt, ai_response, critique) tuples.

    Process:
    1. Take a prompt (from corpus or input)
    2. Capture AI response (real or simulated)
    3. Generate Constitutional critique targeting agent failures:
       - Did the AI hide complexity?
       - Did it take shortcuts?
       - Was it sycophantic?
       - Did it claim false certainty?
    """
```

#### [NEW] `src/dataset/prompts.py`

Curated prompts that tend to elicit problematic AI behaviors:

- Complex tasks where shortcuts are tempting
- Ambiguous requests that invite sycophancy
- Technical questions where uncertainty exists

---

### Phase 2: Judge Training (Optional — depends on user)

#### [NEW] `src/training/train_judge.py`

Train a LoRA adapter using the generated dataset.

---

### Phase 3: Self-Evaluation

#### [MODIFY] `demo.py`

Rewrite to:

1. Run a prompt through AI (me)
2. Run Judge on my response
3. Show what the Judge caught

---

## Verification Plan

### Automated Tests

| Test             | Command                           | Purpose                           |
| ---------------- | --------------------------------- | --------------------------------- |
| Dataset format   | `pytest tests/test_dataset.py -v` | Verify output schema              |
| Critique quality | Manual review                     | Check critiques catch real issues |

### Manual Verification

1. Generate 5-10 samples: `python -m src.dataset.generator --samples 10`
2. Review output JSON for quality
3. User reviews whether critiques match their principles

---

## Questions for User

1. Do you have existing conversation logs I should use, or generate synthetic data?
2. Should the demo show the Judge running on **this current conversation**?
3. What local model should the Judge be based on?
