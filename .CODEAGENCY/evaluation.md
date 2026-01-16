# Architectural Evaluation
**Date**: 2026-01-15
**Scope**: `ui.py`, `src/config.py`, `src/engine/ephemeral.py`, `llama.cpp` scripts

## Critical Risks (Must Fix)
*   [ ] **[Portability]**: Hardcoded absolute path in `ui.py` at line 644: `unsloth_python = "/home/ty/Repositories/unsloth/unsloth_env/bin/python"`. This will fail on any other machine.
    *   **Fix**: Move to `.env` configuration (already partially attempted but fallback is hardcoded).
*   [ ] **[Security]**: While most secrets are env-managed, there are placeholder values in `llama.cpp/examples` that could be confusing. `config.py` correctly uses `os.environ`.

## Improvements (Should Fix)
*   [ ] **[Async/Safety]**: `src/engine/ephemeral.py` uses `loop = asyncio.get_event_loop()` (Line 46). This is deprecated in newer Python versions and can cause issues if no loop is running.
    *   **Refactor**: Use `asyncio.get_running_loop()`.
*   [ ] **[Code Hygiene]**: `ui.py` contains mixed hardcoded paths and env logic.

## Strategic Recommendations
*   Centralize all external tool paths (like `unsloth_python`) in `src/config.py`.
*   Establish a generic `get_python_executable(tool_name)` function.
