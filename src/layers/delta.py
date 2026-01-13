import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaResidualBlock(nn.Module):
    """
    Implements the Deep Delta Learning (DDL) update rule (Rank-1):
    X_{l+1} = X_l + beta * k * (v.T - k.T @ X_l)

    This can be rewritten as a geometric transformation:
    X_{l+1} = (I - beta * k @ k.T)X_l + beta * k @ v.T

    Attributes:
        dim (int): The feature dimension (d).
        beta_init (float): Initial value for the gate beta [0, 2].
    """

    def __init__(self, dim, beta_init=0.1):
        super().__init__()
        self.dim = dim

        # k: The Reflection Direction (dim,)
        self.k_raw = nn.Parameter(torch.randn(dim))

        # v: The Residual Value Scalar (1,) - The target value along direction k
        # We use a scalar because we operate strictly on the 1D subspace defined by k.
        self.v = nn.Parameter(torch.zeros(1))

        # beta: The Scalar Gate [0, 2]
        self.beta_logit = nn.Parameter(torch.tensor([beta_init]))

    def get_beta(self):
        return 2.0 * torch.sigmoid(self.beta_logit)

    def get_k(self):
        return F.normalize(self.k_raw, p=2, dim=0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, dim)
        """
        beta = self.get_beta()
        k = self.get_k() # (dim,)

        # 1. Project x onto k -> scalar per example
        # x @ k -> (batch,)
        projection = torch.matmul(x, k)

        # 2. Compute the delta scalar: (Target - Current)
        # We want the component along k to become self.v
        delta_scalar = self.v - projection # (batch,)

        # 3. Apply Update: x_new = x + beta * (v - x.k) * k
        # If beta=1, output.k = v. (Perfect replacement/erasure)
        update = beta * delta_scalar.unsqueeze(1) * k.unsqueeze(0)

        return x + update


class LoReFTBlock(nn.Module):
    """
    Low-rank Linear Representation Fine-Tuning (LoReFT) Block.

    Generalizes DeltaResidualBlock from rank-1 to low-rank interventions:
    h' = h + R(h @ W_down) @ W_up

    Where R is a learnable gating function.

    This allows steering activations in a low-dimensional subspace while
    maintaining the geometric interpretability of DDL.

    Attributes:
        dim (int): The hidden dimension (d).
        rank (int): The intervention rank (r). Higher = more expressive.
        dropout (float): Dropout on the intervention for regularization.
    """

    def __init__(self, dim: int, rank: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # Down projection: (dim,) -> (rank,)
        self.W_down = nn.Linear(dim, rank, bias=False)

        # Up projection: (rank,) -> (dim,)
        self.W_up = nn.Linear(rank, dim, bias=False)

        # Learnable gate [0, 2] per rank dimension
        self.gate_logit = nn.Parameter(torch.zeros(rank))

        # Optional non-linearity in the bottleneck
        self.activation = nn.GELU()

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize for small initial intervention
        self._init_weights()

    def _init_weights(self):
        """Initialize for near-zero initial intervention."""
        nn.init.normal_(self.W_down.weight, std=0.02)
        nn.init.zeros_(self.W_up.weight)  # Start with zero output

    def get_gate(self):
        """Returns gate values in [0, 2]."""
        return 2.0 * torch.sigmoid(self.gate_logit)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, dim)
        Returns:
            Intervened activations of same shape as input.
        """
        # Project down to low-rank space
        hidden = self.W_down(x)  # (..., rank)

        # Apply non-linearity and gate
        hidden = self.activation(hidden)
        gate = self.get_gate()  # (rank,)
        hidden = hidden * gate

        # Project back up
        intervention = self.W_up(hidden)  # (..., dim)
        intervention = self.dropout(intervention)

        return x + intervention

    def update_from_delta(self, delta: torch.Tensor, lr: float = 0.01):
        """
        Direct update method for skip-layer DFA training.

        Instead of using .backward(), this allows direct parameter updates
        from a projected error signal.

        Args:
            delta: Error signal projected to this layer's dimension (dim,).
            lr: Learning rate for the direct update.
        """
        with torch.no_grad():
            # delta is (dim,), we want to update W_up which is (dim, rank)
            # Use outer product: delta.unsqueeze(1) @ ones(1, rank)
            # This steers W_up to reduce the error in direction delta
            rank = self.rank
            update = delta.unsqueeze(1) * torch.ones(1, rank, device=delta.device)
            self.W_up.weight.data -= lr * update



class MultiHeadDelta(nn.Module):
    """
    Applies multiple Delta updates in parallel (or sum).
    Allows controlling multiple concepts/dimensions.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([DeltaResidualBlock(dim) for _ in range(num_heads)])

    def forward(self, x):
        for head in self.heads:
            x = head(x)
        return x


class MultiHeadLoReFT(nn.Module):
    """
    Multiple LoReFT interventions applied sequentially.
    Each head can target different semantic subspaces.
    """
    def __init__(self, dim: int, num_heads: int = 4, rank: int = 4):
        super().__init__()
        self.heads = nn.ModuleList([
            LoReFTBlock(dim, rank=rank) for _ in range(num_heads)
        ])

    def forward(self, x):
        for head in self.heads:
            x = head(x)
        return x

