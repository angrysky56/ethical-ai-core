"""
ReFT Policy - Representation Fine-Tuning with DFA

Implements the ReFT (Representation Fine-Tuning) paradigm for ethical alignment.
Steers activations rather than modifying weights, enabling surgical interventions.

References:
- pyreft: https://github.com/stanfordnlp/pyreft
- LoReFT paper: Low-rank Linear Subspace ReFT
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from src.ethical.matrix import EthicalMatrix
from src.layers.delta import LoReFTBlock
from src.layers.dfa import GlobalDFAProjector


class ReFTPolicy(nn.Module):
    """
    A policy that applies LoReFT interventions at specified layers of a base model.

    For MVP, this wraps a simple MLP to demonstrate the concept.
    For production, this would hook into a HuggingFace transformer model.

    Architecture:
        base_model -> [intervention @ layer_i] -> output

    Training uses skip-layer DFA:
        output_error -> GlobalDFAProjector -> delta -> intervention.update_from_delta()
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        intervention_layers: Optional[list[int]] = None,
        rank: int = 4,
        num_hidden: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.intervention_layer_indices = (
            intervention_layers if intervention_layers is not None else [1]
        )

        # Build a simple MLP base (frozen in training)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.base_layers = nn.ModuleList(layers)

        # LoReFT interventions at specified layers
        self.interventions = nn.ModuleDict(
            {
                f"layer_{i}": LoReFTBlock(hidden_dim, rank=rank)
                for i in intervention_layers
            }
        )

        # Global DFA Projector for skip-layer feedback
        target_dims = {f"layer_{i}": hidden_dim for i in intervention_layers}
        self.dfa_projector = GlobalDFAProjector(output_dim, target_dims)

        # Inject ethical seed from Core Principles
        self._inject_ethical_matrix()

    def _inject_ethical_matrix(self):
        """Seed the DFA projectors with the Core Principles hash."""
        matrix = EthicalMatrix()
        self.dfa_projector.inject_ethical_seed(matrix.seed)

    def freeze_base(self):
        """Freeze the base model weights (only train interventions)."""
        for layer in self.base_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze base model weights."""
        for layer in self.base_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoReFT interventions.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Output tensor (batch, output_dim)
        """
        layer_idx = 0
        for _, layer in enumerate(self.base_layers):
            x = layer(x)

            # Check if this layer should have an intervention
            # Interventions apply AFTER the linear layer, BEFORE activation
            if (
                isinstance(layer, nn.Linear)
                and layer_idx in self.intervention_layer_indices
            ):
                intervention_name = f"layer_{layer_idx}"
                if intervention_name in self.interventions:
                    x = self.interventions[intervention_name](x)

            if isinstance(layer, nn.Linear):
                layer_idx += 1

        return x

    def dfa_update(self, output_error: torch.Tensor, lr: float = 0.01):
        """
        Perform skip-layer DFA update on all interventions.

        This bypasses the frozen base model and directly updates interventions
        using the Ethical Matrix projections.

        Args:
            output_error: Error signal at output (batch, output_dim)
            lr: Learning rate for direct update
        """
        # Project error to all intervention layers
        deltas = self.dfa_projector.project_all(output_error)

        # Update each intervention
        for name, delta in deltas.items():
            if name in self.interventions:
                # Average over batch
                delta_mean = delta.mean(dim=0)
                self.interventions[name].update_from_delta(delta_mean, lr=lr)


class ReFTTrainer:
    """
    Trainer for ReFTPolicy using skip-layer DFA.

    This implements the "short-circuit" training loop where we don't use
    standard backpropagation through the frozen base model.
    """

    def __init__(
        self,
        policy: ReFTPolicy,
        superego_fn: Callable[[torch.Tensor], torch.Tensor],
        lr: float = 0.01,
    ):
        """
        Args:
            policy: The ReFTPolicy to train
            superego_fn: Function that returns target output given input
                        (represents the GenRM/Judgment Layer)
            lr: Learning rate for DFA updates
        """
        self.policy = policy
        self.superego_fn = superego_fn
        self.lr = lr

        # Freeze base model
        self.policy.freeze_base()

    def train_step(self, x: torch.Tensor) -> dict:
        """
        Single training step using skip-layer DFA.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Dict with training metrics
        """
        # Forward pass
        output = self.policy(x)

        # Get target from Superego
        with torch.no_grad():
            target = self.superego_fn(x)

        # Compute error
        error = output - target
        loss = (error**2).mean()

        # Skip-layer DFA update (no backward() needed!)
        self.policy.dfa_update(error, lr=self.lr)

        return {"loss": loss.item(), "error_norm": error.norm().item()}

    def train_epoch(self, dataloader, verbose: bool = True) -> dict:
        """
        Train for one epoch.

        Args:
            dataloader: Yields batches of input tensors
            verbose: Print progress

        Returns:
            Dict with epoch metrics
        """
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            num_batches += 1

            if verbose and num_batches % 10 == 0:
                print(f"  Batch {num_batches}: loss={metrics['loss']:.4f}")

        return {
            "avg_loss": total_loss / max(num_batches, 1),
            "num_batches": num_batches,
        }
