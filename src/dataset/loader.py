# src/dataset/loader.py

import yaml
import os
from pathlib import Path
from typing import List, Dict, Any
from src.config import PROJECT_ROOT

PACKS_DIR = PROJECT_ROOT / "data" / "packs"
DEFAULT_PACK = "default"

class TrainingPackLoader:
    _instance = None
    _current_pack = DEFAULT_PACK

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_pack(self, pack_name: str):
        if not (PACKS_DIR / pack_name).exists():
             # Fallback or error? detailed error is better
             raise ValueError(f"Training Pack '{pack_name}' not found at {PACKS_DIR / pack_name}")
        self._current_pack = pack_name

    def get_current_pack_name(self) -> str:
        return self._current_pack

    def get_pack_path(self) -> Path:
        return PACKS_DIR / self._current_pack

    def load_prompts(self) -> Dict[str, List[str]]:
        path = self.get_pack_path() / "prompts.yaml"
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def load_personas_config(self) -> Dict[str, str]:
        path = self.get_pack_path() / "personas.yaml"
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def get_principles_text(self) -> str:
        path = self.get_pack_path() / "principles.md"
        if not path.exists():
            return "No principles file found."
        return path.read_text(encoding="utf-8")

    def list_available_packs(self) -> List[str]:
        if not PACKS_DIR.exists():
            return []
        return sorted([d.name for d in PACKS_DIR.iterdir() if d.is_dir()])

# Global accessor
def get_loader() -> TrainingPackLoader:
    return TrainingPackLoader.get_instance()
