import torch
import copy
from src.engine.ephemeral import EphemeralEgo, SimplePolicy

def test_ephemeral_isolation():
    """
    Verifies that TTT updates do not affect the base model.
    """
    # 1. Setup Base Model
    base_model = SimplePolicy(10, 20, 5)
    initial_weights = copy.deepcopy(base_model.layer1.weight.data)

    # 2. Setup TTT Engine
    ego = EphemeralEgo(base_model, steps=10, learning_rate=0.1)

    # 3. Create a constraint-violating input
    # Superego wants output to be all zeros
    input_tensor = torch.randn(1, 10)

    def superego_loss(output):
        # penalize magnitude
        return output.pow(2).mean()

    # 4. Run TTT
    output, initial_loss, final_loss = ego.process_request(input_tensor, superego_loss)

    # 5. Check Improvement
    assert final_loss < initial_loss, "TTT should reduce the specific loss"

    # 6. Check Isolation
    # The base model weights should be IDENTICAL to initial_weights
    # The 'ephemeral' modifications should have evaporated.
    current_base_weights = base_model.layer1.weight.data
    assert torch.allclose(initial_weights, current_base_weights), "Base model should be immutable"

def test_ethical_determinism():
    """
    Verifies that the Ethical Matrix B is deterministic given the same principles.
    """
    from src.ethical.matrix import EthicalMatrix

    e1 = EthicalMatrix("Core Principles.txt")
    e2 = EthicalMatrix("Core Principles.txt") # Same file

    m1 = e1.get_matrix((10, 10))
    m2 = e2.get_matrix((10, 10))

    assert torch.allclose(m1, m2), "Matrix B must be deterministic for the same principles"
