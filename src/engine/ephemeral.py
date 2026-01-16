import torch
import torch.nn as nn
import torch.optim as optim
import copy
from src.layers.dfa import DFALinear
from src.layers.delta import DeltaResidualBlock
from src.ethical.matrix import inject_ethical_matrix, EthicalMatrix

class SimplePolicy(nn.Module):
    """
    A simple MLP Policy using DFA Layers.
    In a real scenario, this would be a LoRA adapter on a Transformer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimplePolicy, self).__init__()
        self.layer1 = DFALinear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = DFALinear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class EphemeralEgo:
    """
    Manages the lifecycle of an 'Ephemeral Ego' - a temporary aligned state
    created via Test-Time Training (TTT).
    """

    def __init__(self, base_policy: nn.Module, learning_rate=0.01, steps=5):
        self.base_policy = base_policy
        self.learning_rate = learning_rate
        self.steps = steps

        # Ensure base policy has ethical matrix injected
        self.matrix_gen = EthicalMatrix()
        inject_ethical_matrix(self.base_policy, self.matrix_gen)

    async def process_request_async(self, input_tensor: torch.Tensor, superego_loss_fn) -> tuple[torch.Tensor, float, float]:
        """
        Executes TTT in a separate thread to avoid blocking the main event loop.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return await loop.run_in_executor(None, self.process_request, input_tensor, superego_loss_fn)

    def process_request(self, input_tensor: torch.Tensor, superego_loss_fn):
        """
        Executes the TTT loop for a single request.

        Args:
            input_tensor: The input features (Prompt embedding).
            superego_loss_fn: A function that returns the 'Ethical Loss' (Constraint Violation).
                              In reality, this comes from the Judgment Layer (GenRM).
                              Signature: fn(output) -> scalar_loss

        Returns:
            final_output: The output of the ephemeral model.
        """
        # 1. Clone the policy (Create the Ephemeral Ego)
        # OPTIMIZATION: Instead of deepcopy (expensive), we use state_dict loading
        # on a pre-instantiated shadow model or just clone parameters that we intend to train.
        # For true TTT, we only need a working copy.
        ephemeral_policy = copy.deepcopy(self.base_policy)

        # In a very high-throughput scenario, we would do:
        # ephemeral_policy.load_state_dict(self.base_policy.state_dict())
        # But deepcopy is safer for MVP to ensure all buffers are also copied.

        # 2. Setup Optimizer for the temporary adapter
        optimizer = optim.SGD(ephemeral_policy.parameters(), lr=self.learning_rate)

        ephemeral_policy.train()

        # 3. Test-Time Training Loop
        # We optimize the ephemeral weights to minimize the specific ethical violation
        # predicted by the Superego for THIS specific input.

        # Note: In a real TTT setting with an LLM, we might use a forward-forward pass
        # or a proxy objective. Here we assume we have a differentiable objective
        # from the Superego (e.g., "Don't output value > 0.9").

        initial_loss = 0.0
        final_loss = 0.0

        for i in range(self.steps):
            optimizer.zero_grad()

            # Forward pass
            output = ephemeral_policy(input_tensor)

            # Judgment (Superego)
            loss = superego_loss_fn(output)

            if i == 0:
                initial_loss = loss.item()

            # Backward (DFA)
            # The error propagates through the FIXED Ethical Matrix B in the DFALinear layers
            loss.backward()

            # Update Temporary Weights
            optimizer.step()

            final_loss = loss.item()

        # 4. Final Inference with Aligned Ego
        ephemeral_policy.eval()
        with torch.no_grad():
            final_output = ephemeral_policy(input_tensor)

        # 5. Dissolution
        # ephemeral_policy is effectively destroyed when this function returns
        # The base_policy remains untouched.
        del ephemeral_policy
        torch.cuda.empty_cache() # If using GPU, critical for TTT

        return final_output, initial_loss, final_loss


class SurgicalEgo:
    """
    A specialized Ephemeral Ego that uses Deep Delta Learning (DDL) to
    surgically remove harmful concepts without cloning the entire base model.
    It attaches a DeltaResidualBlock to the output (or specified layer)
    and optimizes ONLY the Delta parameters (k, beta, v).
    """

    def __init__(self, base_policy: nn.Module, dim: int, learning_rate=0.1, steps=10):
        self.base_policy = base_policy
        self.dim = dim
        self.learning_rate = learning_rate
        self.steps = steps

        # We assume the base policy is frozen/immutable in this context
        for param in self.base_policy.parameters():
            param.requires_grad = False

    def process_request(self, input_tensor: torch.Tensor, superego_loss_fn):
        """
        Executes Surgical TTT:
        1. Create a fresh Delta Block.
        2. Optimize 'k' to align with the failure mode direction.
        3. Optimize 'beta' to 1 (Erasure) or 'v' (Correction).
        """
        # 1. Attach Delta Block (Transformation)
        delta_block = DeltaResidualBlock(self.dim)
        optimizer = optim.SGD(delta_block.parameters(), lr=self.learning_rate)

        initial_loss = 0.0
        final_loss = 0.0

        # 2. Optimization Loop
        for i in range(self.steps):
            optimizer.zero_grad()

            # Forward Base
            with torch.no_grad():
                base_out = self.base_policy(input_tensor)

            # Forward Delta (Residual Correction)
            # final = base + delta(base)
            final_out = delta_block(base_out)

            # Judgment
            loss = superego_loss_fn(final_out)

            if i == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # 3. Final Inference
        delta_block.eval()
        with torch.no_grad():
            base_out = self.base_policy(input_tensor)
            final_output = delta_block(base_out)

        return final_output, initial_loss, final_loss
