"""State-tracking capability dataset generation.

Self-contained port of the verified full-state tracking agent (originally
prototyped in ../entity-tracking-externalization). Produces training samples
that teach a model the *discipline* empirically shown to eliminate silent
"ghost" errors: maintain a complete, typed world-state and re-emit ALL of it
after every operation, tracking removed items explicitly rather than dropping
them.

Why this lives alongside the ethics pipeline: the SFT trainer
(src/training/train_local.py) consumes any sample with `prompt` +
`revised_response` fields. This module emits exactly that shape, with
`revised_response` = a verified-correct step-by-step state trace. The samples
flow into the same SampleDB / JSONL the constitutional pipeline uses, so a model
can be trained on ethical alignment, state-tracking discipline, or both.

Nothing here imports from another project (per workspace rule: projects are
self-contained).
"""

__all__ = ["generate_tracking_samples", "TrackingSample"]

from src.capabilities.state_tracking import TrackingSample, generate_tracking_samples
