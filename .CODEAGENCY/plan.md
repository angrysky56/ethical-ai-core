# Implementation Plan: Fix Security & Portability

## Phase 1: Preparation
- [ ] Create/Verify branch (optional, working on main for now).
- [ ] Run existing tests to ensure baseline green.

## Phase 2: Implementation
- [ ] **[Step 1]**: Update `src/config.py`.
    - *Context*: Add `get_python_executable` and `UNSLOTH_PYTHON_PATH`.
- [ ] **[Step 2]**: Refactor `ui.py`.
    - *Context*: Remove hardcoded `/home/ty` path.
- [ ] **[Step 3]**: Refactor `src/engine/ephemeral.py`.
    - *Context*: Use `asyncio.get_running_loop()`.

## Phase 3: Verification
- [ ] Run `pytest tests/test_async.py`.
- [ ] Verify `ui.py` imports/config logic.
