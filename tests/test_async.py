import pytest
import asyncio
import torch
from src.engine.ephemeral import EphemeralEgo, SimplePolicy

@pytest.mark.asyncio
async def test_async_isolation():
    """
    Verifies that async TTT works without blocking and maintains isolation.
    """
    base_model = SimplePolicy(10, 20, 5)
    ego = EphemeralEgo(base_model, steps=5)

    input_tensor = torch.randn(1, 10)

    def superego_loss(output):
        return output.mean().pow(2)

    # Launch async task
    future = ego.process_request_async(input_tensor, superego_loss)

    # Do something else concurrently (sleep)
    await asyncio.sleep(0.1)

    output, init_loss, final_loss = await future

    assert final_loss < init_loss
    assert output is not None
