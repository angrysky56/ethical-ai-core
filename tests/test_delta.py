import torch
import numpy as np
import pytest
from src.layers.delta import DeltaResidualBlock

def test_delta_spectral_properties():
    """
    Theorem 1: The eigenvalues of the operator A(X) = (I - beta k k^T)
    should be {1, ..., 1, 1 - beta}.
    """
    dim = 10
    layer = DeltaResidualBlock(dim, beta_init=0.0) # Init around beta near sig(0)*2 = 1.0?

    # Force specific beta for testing
    # If beta = 1.0, we expect Projector (eigenvalues 1, ..., 0)
    # layer.get_beta() returns 2 * sigmoid(logit).
    # To get 1.0, sigmoid -> 0.5 -> logit 0.
    layer.beta_logit.data.fill_(0.0)

    beta = layer.get_beta().item()
    assert np.isclose(beta, 1.0, atol=0.01)

    k = layer.get_k().detach()

    # Construct the Operator Matrix explicitely
    # A = I - beta * k @ k.T
    I = torch.eye(dim)
    A = I - beta * torch.outer(k, k)

    # Compute Eigenvalues
    eigvals = torch.linalg.eigvals(A)
    eigvals_real = eigvals.real.detach().numpy()

    # Sort and check
    eigvals_real.sort()

    # Expect: One eigenvalue is roughly 0.0 (1 - 1.0), others are 1.0
    assert np.isclose(eigvals_real[0], 0.0, atol=1e-5), "One eigenvalue should be 0 (Projective)"
    assert np.allclose(eigvals_real[1:], 1.0, atol=1e-5), "Other eigenvalues should be 1"

def test_erasure():
    """
    Verify that the layer actually removes information along k when beta=1.
    """
    dim = 5
    layer = DeltaResidualBlock(dim)
    layer.beta_logit.data.fill_(0.0) # beta = 1

    # Set target v to 0 for pure erasure
    layer.v.data.fill_(0.0)

    k = layer.get_k()

    # Create input strictly parallel to k
    x = 10.0 * k.unsqueeze(0) # (1, d)

    # Forward
    y = layer(x)

    # Output should be 0 vector (erased)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-5), "Input along k should be erased"

    # Create orthogonal input
    x_perp = torch.randn(1, dim)
    # orthogonalize
    x_perp = x_perp - torch.matmul(x_perp, k) * k

    y_perp = layer(x_perp)

    # Output should be identical (Identity mapping for k_perp)
    assert torch.allclose(y_perp, x_perp, atol=1e-5), "Orthogonal input should be strictly preserved"
