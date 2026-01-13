import torch
from src.judgment.genrm import GenRM

def test_genrm_output_schema():
    """Verify GenRM outputs valid JudgmentSchema mock."""
    model = GenRM()

    judgment = model.judge("prompt", "response")

    assert "analysis" in judgment
    assert "virtue_assessment" in judgment
    assert "verdict" in judgment
    assert "score" in judgment

def test_genrm_forward_shape():
    """Verify scalar score output shape."""
    model = GenRM()
    input_ids = torch.randint(0, 100, (1, 10))

    score = model(input_ids)

    assert score.shape == (1, 1)
    assert 0.0 <= score.item() <= 1.0
