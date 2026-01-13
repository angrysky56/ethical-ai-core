import torch
import torch.nn as nn
from src.engine.ephemeral import SurgicalEgo

class SimpleOutput(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)

def test_surgical_erasure():
    """
    Verifies that SurgicalEgo can erase a specific direction from the output
    without changing the base policy weights.
    """
    dim = 5
    base_model = SimpleOutput(dim)

    # Store initial weights
    initial_weights = base_model.linear.weight.data.clone()

    # Create Surgical Ego
    # It attaches a DeltaResidualBlock to the output of base_model
    ego = SurgicalEgo(base_model, dim=dim, learning_rate=0.5, steps=20)

    # Input
    x = torch.randn(1, dim)

    # Define a "Bad Direction" that Superego hates
    bad_direction = torch.randn(dim)
    bad_direction = bad_direction / bad_direction.norm()

    # Superego Loss: maximize distance from bad_direction?
    # Or simply: minimize projection onto bad_direction.
    # We want output @ bad_direction to be 0.
    def superego_loss_fn(output):
        # output shape (1, dim)
        # project onto bad_direction
        proj = (output * bad_direction).sum()
        # minimize square of projection (erasure)
        return proj.pow(2)

    # Run Surgical TTT
    final_output, init_loss, final_loss = ego.process_request(x, superego_loss_fn)

    # 1. Verify Loss Reduction
    assert final_loss < init_loss, "Surgical TTT should reduce the projection onto bad direction"
    print(f"Initial Projection Loss: {init_loss:.5f} -> Final: {final_loss:.5f}")

    # 2. Verify Base Model Unchanged
    current_weights = base_model.linear.weight.data
    assert torch.allclose(initial_weights, current_weights), "Base model weights must remain frozen"

    # 3. Verify Erasure
    # The final output should have near-zero component along bad_direction
    # relative to the original output.
    # Note: If the base output was already orthogonal, TTT wouldn't do much.
    # But random vectors in 5D are rarely orthogonal.
    final_proj = (final_output * bad_direction).sum().abs()
    assert final_proj < 0.1, f"Final projection should be small, got {final_proj}"
