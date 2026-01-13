import torch
import pytest
from src.layers.dfa import DFALinear

def test_dfa_gradient_flow():
    """
    Verifies that the gradient w.r.t input is computed using the feedback matrix B,
    NOT the weight matrix W.
    """
    in_features = 5
    out_features = 3

    layer = DFALinear(in_features, out_features, bias=False)

    # 1. Setup Data
    x = torch.randn(1, in_features, requires_grad=True)

    # 2. Forward Pass
    y = layer(x)

    # 3. Define a Loss (simple sum)
    loss = y.sum()

    # 4. Backward Pass
    loss.backward()

    # 5. Check Gradients
    # Theoretical DFA Gradient for this simple loss (where dL/dy = ones)
    # dL/dx = dL/dy @ B.T
    # dL/dy is all ones (1, out)
    grad_output = torch.ones(1, out_features)
    expected_grad_x = grad_output.mm(layer.feedback_matrix.t())

    # Theoretical BP Gradient (if it were standard BP)
    # dL/dx = dL/dy @ W
    expected_bp_grad_x = grad_output.mm(layer.weight)

    # Assert that actual grad matches DFA expectation, NOT BP expectation
    # Note: Highly unlikely W and B.T match randomly
    assert torch.allclose(x.grad, expected_grad_x, atol=1e-6), "Gradient should follow B matrix"
    assert not torch.allclose(x.grad, expected_bp_grad_x, atol=1e-6), "Gradient should NOT follow W matrix"

def test_dfa_weight_update():
    """
    Verifies that weights still get gradients (dL/dW) even if dL/dx is messed with.
    """
    layer = DFALinear(5, 3)
    x = torch.randn(1, 5)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert torch.norm(layer.weight.grad) > 0
