"""Verified state-tracking sample generator (self-contained).

Pipeline per sample:
  1. Generate a random PUT/MOVE/REMOVE task with deterministic ground truth.
  2. Compute the gold full-state trace from the deterministic reducer (NOT from
     a model) — so the training target is correct by construction.
  3. Emit a (prompt, revised_response) sample: prompt = the tracking question,
     revised_response = the complete step-by-step STATE trace + final ANSWER.

Using the reducer to author the target means we do not need an LLM in the loop
to *produce* the gold data — the gold is exact. (A model-in-the-loop verify mode
is also provided for harvesting traces a given model already gets right, but the
default and recommended path is reducer-authored gold.)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

NOWHERE = "nowhere"

OBJECTS = [
    "apple", "key", "book", "candle", "coin", "spoon", "ring", "stamp",
    "marble", "feather", "button", "thimble", "cork", "pebble", "acorn",
]
CONTAINERS = [
    "red box", "blue box", "green box", "yellow drawer", "black basket",
    "white crate", "brown bag", "gray bin",
]

_RULES = (
    "There are containers and objects. Each object is in at most one container. "
    "Taking an object out removes it from all containers (it is then nowhere). "
    "After each step, write the COMPLETE state of every container on one line, "
    "including empty containers and a 'nowhere' list for removed objects. Never "
    "drop an object from the written state; if it is removed, move it to "
    "'nowhere'. Then answer where the queried object ends up."
)


@dataclass
class _Op:
    kind: str          # PUT | MOVE | REMOVE
    obj: str
    container: str | None

    def render(self) -> str:
        if self.kind == "PUT":
            return f"Put the {self.obj} in the {self.container}."
        if self.kind == "MOVE":
            return f"Move the {self.obj} to the {self.container}."
        return f"Take the {self.obj} out."


@dataclass
class TrackingSample:
    """A training pair plus metadata, matching the constitutional SampleDB shape."""

    prompt: str
    revised_response: str
    query_obj: str
    answer: str
    n_ops: int
    n_removes: int
    meta: dict = field(default_factory=dict)

    def to_sample_dict(self, sample_id: str, timestamp: str) -> dict:
        """Shape expected by SampleDB.save_sample / train_local.py."""
        return {
            "sample_id": sample_id,
            "naive_response": "",  # no naive baseline for synthetic capability data
            "critique": "",
            "revised_response": self.revised_response,
            "principle_attribution": "capability:state-tracking",
            "tier_violated": 0,
            "timestamp": timestamp,
            "prompt_text": self.prompt,
        }


def _generate_task(rng: random.Random, n_ops: int, n_objects: int,
                   n_containers: int, remove_prob: float) -> tuple:
    """Build a random op sequence + the gold full-state trace from the reducer."""
    objects = rng.sample(OBJECTS, k=min(n_objects, len(OBJECTS)))
    containers = rng.sample(CONTAINERS, k=min(n_containers, len(CONTAINERS)))

    # state: container -> list[obj]; plus NOWHERE
    state: dict[str, list[str]] = {c: [] for c in containers}
    state[NOWHERE] = []
    ops: list[_Op] = []
    trace_lines: list[str] = []
    n_removes = 0

    def place(obj: str, dest: str) -> None:
        for ents in state.values():
            if obj in ents:
                ents.remove(obj)
        state[dest].append(obj)

    for i in range(n_ops):
        present = [o for o in objects if any(o in state[c] for c in containers)]
        roll = rng.random()
        if present and roll < remove_prob:
            obj = rng.choice(present)
            ops.append(_Op("REMOVE", obj, None))
            place(obj, NOWHERE)
            n_removes += 1
        elif present and roll < remove_prob + 0.35:
            obj = rng.choice(present)
            dest = rng.choice([c for c in containers if obj not in state[c]] or containers)
            ops.append(_Op("MOVE", obj, dest))
            place(obj, dest)
        else:
            absent = [o for o in objects if o not in present]
            obj = rng.choice(absent or objects)
            dest = rng.choice(containers)
            ops.append(_Op("PUT", obj, dest))
            place(obj, dest)
        trace_lines.append(f"Step {i + 1}: {_render_state(state, containers)}")

    # Only query objects that actually appear in the operations, so the trace
    # always contains the evidence for the answer (an object never placed would
    # be 'nowhere' by default but never mentioned in the trace — an inconsistent
    # training target).
    touched = sorted({op.obj for op in ops})
    query_obj = rng.choice(touched) if touched else rng.choice(objects)
    answer = _location_of(state, query_obj, containers)
    return ops, trace_lines, query_obj, answer, n_removes


def _render_state(state: dict, containers: list[str]) -> str:
    parts = [f"{c}={','.join(state[c]) if state[c] else 'empty'}" for c in containers]
    parts.append(f"nowhere={','.join(state[NOWHERE]) if state[NOWHERE] else 'empty'}")
    return "; ".join(parts)


def _location_of(state: dict, obj: str, containers: list[str]) -> str:
    for c in containers:
        if obj in state[c]:
            return c
    return NOWHERE


def generate_tracking_samples(
    count: int = 50,
    seed: int = 7,
    n_ops_range: tuple[int, int] = (6, 20),
    n_objects: int = 5,
    n_containers: int = 4,
    remove_prob: float = 0.4,
    balance_removed_query: bool = True,
) -> list[TrackingSample]:
    """Generate `count` verified (prompt, full-state-trace) training samples.

    Difficulty (n_ops) is sampled across n_ops_range so the model sees short and
    long traces. When balance_removed_query is set, ~half the samples query a
    removed object (answer == nowhere) so the REMOVE-tracking discipline — the
    whole point — is well represented.
    """
    rng = random.Random(seed)
    samples: list[TrackingSample] = []
    want_removed = True
    attempts = 0

    while len(samples) < count and attempts < count * 60:
        attempts += 1
        n_ops = rng.randint(*n_ops_range)
        ops, trace_lines, query_obj, answer, n_removes = _generate_task(
            rng, n_ops, n_objects, n_containers, remove_prob
        )
        is_removed = answer == NOWHERE
        if balance_removed_query and is_removed != want_removed:
            continue

        prompt = _build_prompt(ops, query_obj)
        response = _build_response(trace_lines, query_obj, answer)
        samples.append(TrackingSample(
            prompt=prompt, revised_response=response, query_obj=query_obj,
            answer=answer, n_ops=n_ops, n_removes=n_removes,
            meta={"removed_query": is_removed},
        ))
        want_removed = not want_removed

    return samples


def _build_prompt(ops: list[_Op], query_obj: str) -> str:
    steps = "\n".join(f"{i + 1}. {op.render()}" for i, op in enumerate(ops))
    return (
        f"{_RULES}\n\n"
        f"Steps:\n{steps}\n\n"
        f"Question: Where is the {query_obj} at the end?"
    )


def _build_response(trace_lines: list[str], query_obj: str, answer: str) -> str:
    trace = "\n".join(trace_lines)
    return f"{trace}\n\nANSWER: {answer}"
