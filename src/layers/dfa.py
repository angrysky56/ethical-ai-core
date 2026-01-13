import torch
import torch.nn as nn
from torch.autograd import Function

class DFALinearFunction(Function):
    """
    Custom autograd function for Direct Feedback Alignment.

    Forward pass: y = x @ W.T + b
    Backward pass: grad_x = grad_y @ B (instead of W)
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None, feedback_matrix=None):
        """
        Args:
            input: (batch_size, in_features)
            weight: (out_features, in_features)
            bias: (out_features)
            feedback_matrix: (in_features, out_features) - The fixed 'B' matrix.
        """
        ctx.save_for_backward(input, weight, bias, feedback_matrix)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: (batch_size, out_features)
        """
        input, weight, bias, feedback_matrix = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = grad_feedback = None

        # 1. Compute grad_input using the Feedback Matrix B instead of Weight Transpose
        # Standard BP: grad_input = grad_output.mm(weight)
        # DFA: grad_input = grad_output.mm(feedback_matrix.t())
        # Wait, feedback_matrix is usually (in, out). So B * e -> (out, in) * (batch, out)^T ?
        # Let's align dimensions.
        # input: (batch, in)
        # weight: (out, in)
        # grad_output: (batch, out)
        # feedback_matrix (B): (in, out) - conceptually maps error back to input space.

        if feedback_matrix is not None:
            # DFA Update: Propagate error backwards through B
            # grad_input = grad_output @ B.T
            # (batch, out) @ (out, in) -> (batch, in)
            grad_input = grad_output.mm(feedback_matrix.t())
        else:
            # Fallback to standard BP if no B provided (should ideally not happen in this usage)
            grad_input = grad_output.mm(weight)

        # 2. Compute grad_weight (Standard update rule for weights)
        # grad_weight = grad_output.T @ input
        # (out, batch) @ (batch, in) -> (out, in)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        # 3. Compute grad_bias
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        # feedback_matrix has no gradient
        return grad_input, grad_weight, grad_bias, None

class DFALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(DFALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # The Fixed Feedback Matrix 'B'
        # shape: (in_features, out_features) to map (batch, out) -> (batch, in) during backward?
        # Let's re-verify dimension.
        # grad_output is (batch, out). We need (batch, in).
        # We need to multiply (batch, out) by (out, in).
        # So B should be (out, in) effectively replacing weight (out, in).
        # But often in literature B is described as connecting error to upstream.
        # Let's stick to B having same shape as W for simplicity of replacement, or W.t()
        # If we use W.t() shape, that is (in, out).
        # grad_output (batch, out) @ B (out, in) -> NO.
        # grad_output (batch, out) @ B_transposed (out, in)??

        # Let's define B as (in_features, out_features).
        # grad_input = grad_output @ B.T -> (batch, out) @ (out, in) -> (batch, in). Correct.
        self.register_buffer('feedback_matrix', torch.randn(in_features, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize B (randomly, but fixed)
        # In our full implementation, this will be overwritten by the "Core Principles" seeder.
        nn.init.uniform_(self.feedback_matrix, -0.1, 0.1)

    def forward(self, input):
        return DFALinearFunction.apply(input, self.weight, self.bias, self.feedback_matrix)


class GlobalDFAProjector(nn.Module):
    """
    Projects output error signals directly to intervention layer dimensions.

    This enables "skip-layer" feedback alignment where we bypass the frozen
    model layers and project the output error directly to the LoReFT intervention
    layer using the Ethical Matrix B.

    The key insight: Instead of loss.backward() through all layers, we compute:
        delta_intervention = output_error @ B

    Where B is seeded from the Core Principles (immutable conscience).

    Attributes:
        output_dim (int): Dimension of the output/logit space.
        target_dims (dict): Mapping of layer_name -> dimension for each intervention.
    """

    def __init__(self, output_dim: int, target_dims: dict[str, int]):
        super().__init__()
        self.output_dim = output_dim
        self.target_dims = target_dims

        # Create a B matrix for each target layer
        # These project from output_dim -> layer_dim
        self.projectors = nn.ModuleDict()
        for layer_name, layer_dim in target_dims.items():
            # Register as buffer (not trainable)
            proj = nn.Linear(output_dim, layer_dim, bias=False)
            proj.weight.requires_grad = False
            self.projectors[layer_name] = proj

    def inject_ethical_seed(self, seed: int):
        """
        Reinitialize all projection matrices with a deterministic seed.
        This ties the feedback alignment to the Core Principles hash.
        """
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

        for proj in self.projectors.values():
            nn.init.orthogonal_(proj.weight)
            proj.weight.mul_(0.1)  # Scale down for stability

        torch.set_rng_state(rng_state)

    def project(self, output_error: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Project output error to the target intervention layer.

        Args:
            output_error: The error at the output layer (batch, output_dim)
            layer_name: Which intervention layer to project to

        Returns:
            Projected error signal (batch, layer_dim) ready for direct update.
        """
        if layer_name not in self.projectors:
            raise ValueError(f"Unknown layer: {layer_name}. Available: {list(self.projectors.keys())}")

        return self.projectors[layer_name](output_error)

    def project_all(self, output_error: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Project output error to all intervention layers at once.

        Args:
            output_error: The error at the output layer (batch, output_dim)

        Returns:
            Dict mapping layer_name -> projected error signal
        """
        return {
            name: proj(output_error)
            for name, proj in self.projectors.items()
        }

