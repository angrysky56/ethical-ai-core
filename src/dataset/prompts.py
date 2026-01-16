"""
Red Team Prompts for Constitutional AI Dataset Generation

These prompts are designed to stress-test the hierarchical principles:
1. Dilemma prompts (utility vs deontology conflicts)
2. Sycophancy traps (requests that invite agreement over correctness)
3. Shortcut temptations (complex tasks where cutting corners is tempting)
4. False certainty traps (questions where uncertainty exists)
"""

"""
Red Team Prompts for Constitutional AI Dataset Generation

Uses the TrainingPackLoader to fetch prompts from the active pack (e.g., data/packs/default/prompts.yaml).
"""
from src.dataset.loader import get_loader

def get_prompts_by_category() -> dict[str, list[str]]:
    """Get prompts grouped by category from the active training pack."""
    return get_loader().load_prompts()


def get_all_prompts() -> list[str]:
    """Get all prompts flattened from all categories."""
    data = get_loader().load_prompts()
    all_prompts = []
    for category_prompts in data.values():
        if isinstance(category_prompts, list):
            all_prompts.extend(category_prompts)
    return all_prompts

