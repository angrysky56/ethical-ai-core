import torch
import hashlib
import numpy as np
from pathlib import Path

from src.config import PRINCIPLES_PATH

class EthicalMatrix:
    """
    Manages the 'Core Self' matrix B, seeded by the hash of the Core Principles text.
    This ensures that the feedback alignment target is deterministically derived from the constitution.
    """

    def __init__(self, principles_path: Path = PRINCIPLES_PATH):
        self.principles_path = principles_path
        self.seed = self._generate_seed_from_text()

    def _generate_seed_from_text(self) -> int:
        """Reads the principles file and generates a deterministic integer seed."""
        path = Path(self.principles_path)
        if not path.exists():
            # Fallback for testing if file missing
            return 42

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create a SHA256 hash of the content
        sha = hashlib.sha256(content.encode('utf-8')).hexdigest()
        # Convert first 8 bytes to int
        seed = int(sha[:16], 16) % (2**32)
        return seed

    def get_matrix(self, shape: tuple) -> torch.Tensor:
        """
        Generates the projection matrix B with the specific ethical seed.
        Args:
            shape: tuple (rows, cols) usually (in_features, out_features)
        """
        # Set torch seed temporarily to ensure determinism
        rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)

        # Create valid feedback matrix (usually Uniform or Gaussian)
        # Using Xavier-like initialization logic but fixed
        fan_in, fan_out = shape
        std = np.sqrt(2.0 / (fan_in + fan_out))
        matrix = torch.randn(*shape) * std

        # Restore RNG state so we don't mess up the rest of the training randomness
        torch.set_rng_state(rng_state)

        return matrix

def inject_ethical_matrix(model: torch.nn.Module, feedback_generator: EthicalMatrix):
    """
    Iterates through a model, finds DFALinear layers, and injects the ethically seeded B matrix.
    """
    from src.layers.dfa import DFALinear

    for name, module in model.named_modules():
        if isinstance(module, DFALinear):
            # Explicitly capture the matrix to help type inference
            mat = module.feedback_matrix
            if isinstance(mat, torch.Tensor):
                shape = tuple(mat.shape)
                ethical_B = feedback_generator.get_matrix(shape)

                with torch.no_grad():
                    mat.copy_(ethical_B)

                # Ensure it is not trainable
                mat.requires_grad = False
            print(f"Injected Ethical Matrix into layer: {name} | Seed: {feedback_generator.seed}")
