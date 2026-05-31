"""CLI: generate state-tracking capability samples into a run's SampleDB / JSONL.

Examples
--------
    # Add 100 verified tracking samples to a new run's dataset
    uv run python -m src.capabilities.build --count 100

    # Append to an existing run (blend with constitutional samples)
    uv run python -m src.capabilities.build --count 100 --run run_2026..._s123

    # Standalone JSONL only (train-ready: {prompt, revised_response})
    uv run python -m src.capabilities.build --count 200 --jsonl-only out.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from src.capabilities.state_tracking import generate_tracking_samples
from src.dataset.personas import RunManager, SampleDB


def _sid(text: str, i: int) -> str:
    return hashlib.sha256(f"track{i}{text}".encode()).hexdigest()[:12]


def main() -> int:
    p = argparse.ArgumentParser(description="Generate state-tracking capability samples")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-ops", type=int, default=6)
    p.add_argument("--max-ops", type=int, default=20)
    p.add_argument("--n-objects", type=int, default=5)
    p.add_argument("--n-containers", type=int, default=4)
    p.add_argument("--remove-prob", type=float, default=0.4)
    p.add_argument("--run", type=str, help="existing run id (default: new run)")
    p.add_argument("--jsonl-only", type=str, metavar="PATH",
                   help="write train-ready JSONL and skip the SampleDB")
    args = p.parse_args()

    samples = generate_tracking_samples(
        count=args.count, seed=args.seed,
        n_ops_range=(args.min_ops, args.max_ops),
        n_objects=args.n_objects, n_containers=args.n_containers,
        remove_prob=args.remove_prob,
    )
    removed = sum(s.meta.get("removed_query") for s in samples)
    print(f"Generated {len(samples)} verified tracking samples "
          f"({removed} removed-query / {len(samples) - removed} present-query).")

    if args.jsonl_only:
        out = Path(args.jsonl_only)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for s in samples:
                f.write(json.dumps(
                    {"prompt": s.prompt, "revised_response": s.revised_response},
                    ensure_ascii=False) + "\n")
        print(f"Wrote train-ready JSONL -> {out}")
        return 0

    if args.run:
        RunManager.set_run(args.run)
    else:
        RunManager.new_run()
    db = SampleDB()
    ts = datetime.now().isoformat()
    for i, s in enumerate(samples):
        db.save_sample(_sid(s.prompt, i), s.prompt, s.to_sample_dict(_sid(s.prompt, i), ts))
    print(f"Saved {len(samples)} samples to run {RunManager.get_current_run()} "
          f"({RunManager.get_run_dir()})")
    print("These now appear in the UI dataset view and JSONL export alongside "
          "any constitutional samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
