import os
from pathlib import Path
from dotenv import load_dotenv
import random
from datetime import datetime

# Load .env file
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PRINCIPLES_PATH = Path(os.environ.get("ETHICAL_PRINCIPLES_PATH", PROJECT_ROOT / "core_principles.md"))

# TTT Hyperparameters
# TTT Hyperparameters
TTT_STEPS = int(os.environ.get("TTT_STEPS", "5"))
TTT_LR = float(os.environ.get("TTT_LR", "0.01"))

# External Tools Configuration
# Use UNSLOTH_PYTHON_PATH from env, or default to standard system python if not set (fallback behavior)
# Ideally, users should set this in .env to the specific venv python path if different.
UNSLOTH_PYTHON_PATH = os.environ.get("UNSLOTH_PYTHON_PATH", "python")

def get_python_executable(tool: str = "unsloth") -> str:
    """
    Get the configured Python executable path for a specific external tool.
    
    Args:
        tool (str): The name of the tool (e.g., "unsloth").
        
    Returns:
        str: The path to the python executable.
    """
    if tool == "unsloth":
        return UNSLOTH_PYTHON_PATH
    return "python"


# =============================================================================
# LLM Provider Configuration
# =============================================================================
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# Provider-specific settings
LLM_CONFIG = {
    "openrouter": {
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model": os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
    },
    "ollama": {
        "api_key": "",  # Ollama doesn't need a key
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
    },
    "lmstudio": {
        "api_key": "lm-studio",  # LM Studio accepts any key
        "base_url": os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        "model": os.environ.get("LMSTUDIO_MODEL", "local-model"),
    },
    "openai": {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    },
}

def get_llm_config(provider: str = None) -> dict:
    """Returns the active LLM configuration or specific provider config."""
    target = provider or LLM_PROVIDER
    return LLM_CONFIG.get(target, LLM_CONFIG["ollama"])


# =============================================================================
# Generation Seed (for reproducibility)
# =============================================================================

def get_global_seed() -> int:
    """Get or generate a global seed for this session."""
    env_seed = os.environ.get("GENERATION_SEED")
    if env_seed:
        return int(env_seed)
    # Generate based on timestamp if not set
    return int(datetime.now().timestamp() * 1000) % 100000

def get_run_id() -> str:
    """Generate a unique run ID for dataset tracking."""
    seed = get_global_seed()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_s{seed}"

GLOBAL_SEED = get_global_seed()
random.seed(GLOBAL_SEED)

