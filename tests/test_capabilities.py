"""Tests for the state-tracking capability sample generator.

The key correctness property: the gold trace is authored by the deterministic
reducer, so every sample's ANSWER must match an independent replay, and removed
objects must appear in 'nowhere' (never silently dropped).
"""

import re

from src.capabilities.state_tracking import (
    NOWHERE,
    OBJECTS,
    generate_tracking_samples,
)


def _final_state_from_response(response: str) -> dict:
    """Parse the last 'Step N: ...' line back into {slot: set(objs)}."""
    step_lines = [l for l in response.splitlines() if l.strip().startswith("Step ")]
    last = step_lines[-1]
    body = last.split(":", 1)[1]
    state = {}
    for chunk in body.split(";"):
        if "=" not in chunk:
            continue
        slot, items = chunk.split("=", 1)
        slot = slot.strip()
        items = items.strip()
        objs = set() if items == "empty" else {x.strip() for x in items.split(",")}
        state[slot] = objs
    return state


def test_answer_matches_final_state_in_trace():
    samples = generate_tracking_samples(count=40, seed=1)
    assert len(samples) == 40
    for s in samples:
        state = _final_state_from_response(s.revised_response)
        # The declared answer must agree with where the trace puts the object.
        if s.answer == NOWHERE:
            assert s.query_obj in state.get(NOWHERE, set())
        else:
            assert s.query_obj in state.get(s.answer, set())
        # And the ANSWER line must literally match.
        m = re.search(r"ANSWER:\s*(.+)\s*$", s.revised_response)
        assert m and m.group(1).strip() == s.answer


def test_removed_objects_go_to_nowhere_not_erased():
    # Every object that ever appears must be somewhere in the final state
    # (a real slot or nowhere) — never silently dropped.
    samples = generate_tracking_samples(count=40, seed=2, remove_prob=0.6)
    obj_vocab = set(OBJECTS)
    for s in samples:
        state = _final_state_from_response(s.revised_response)
        all_in_state = set().union(*state.values()) if state else set()
        # objects actually named in the prompt steps (from the known vocabulary)
        named = {w for w in re.findall(r"\b(\w+)\b", s.prompt) if w in obj_vocab}
        assert named <= all_in_state, f"dropped {named - all_in_state}"


def test_balance_produces_both_query_types():
    samples = generate_tracking_samples(count=30, seed=3, balance_removed_query=True)
    removed = sum(s.answer == NOWHERE for s in samples)
    assert 8 <= removed <= 22  # roughly balanced


def test_sample_dict_shape_for_training():
    s = generate_tracking_samples(count=1, seed=4)[0]
    d = s.to_sample_dict("abc123", "2026-01-01T00:00:00")
    # train_local.py reads prompt + revised_response; SampleDB needs these keys
    assert d["revised_response"] == s.revised_response
    assert d["prompt_text"] == s.prompt
    assert d["principle_attribution"] == "capability:state-tracking"
    assert "ANSWER:" in d["revised_response"]
