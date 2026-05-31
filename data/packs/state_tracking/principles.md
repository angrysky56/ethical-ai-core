# State-Tracking Discipline

This is a **capability pack**, not an ethics constitution. It teaches one
discipline empirically shown to eliminate silent state-tracking failures
("ghost" errors — reporting a removed entity as still present).

## The single principle

**Maintain a complete, explicit world-state and re-emit ALL of it after every
operation. Never drop an entity from the written state. Removal is a written
transition (the entity moves to `nowhere`), never an erasure.**

## Why

When a model tracks state implicitly (deciding internally what is "relevant"),
removals fail silently: the removed item is quietly carried forward and reported
as still present. Forcing the full state to be written every step makes that
failure impossible to hide — the removal must appear as a move to `nowhere`.
Tracking only the queried item (a terse "delta" scratchpad) reintroduces the
ghost, because the record that a removal happened is discarded. Completeness
beats brevity for this failure mode.

## The required output shape

After each operation, write one line with every container's full contents,
including empty ones and a `nowhere` list:

```
Step 3: red box=apple; blue box=empty; green box=key; nowhere=coin
```

Then finish with:

```
ANSWER: <container, or nowhere>
```

## Scope

This discipline generalizes to any entity-in-slot tracking (variables/values,
files/directories, inventories, balances), not just objects in containers. The
invariant is the same: complete state, explicit removals, every step.
