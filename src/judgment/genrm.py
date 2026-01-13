import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from src.judgment import JudgmentSchema

class GenRM(nn.Module):
    """
    Generative Reward Model (Superego).
    Outputs structured ethical judgment logic, then a score.
    """
    def __init__(self, model_name="gpt2"):
        super().__init__()
        # For MVP, we use a small generic backbone
        # In prod, this would be a molecular transformer or LoRA adapter
        self.config = GPT2Config(vocab_size=50257, n_embd=768, n_layer=4, n_head=4)
        self.backbone = GPT2Model(self.config)

        # Head for the scalar score (0-1)
        self.score_head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, -1, :] # CLS token equivalent

        score_logit = self.score_head(hidden)
        score = torch.sigmoid(score_logit)

        return score

    def judge(self, prompt_text: str, response_text: str) -> JudgmentSchema:
        """
        Full inference loop generating the JSON schema.
        MVP: Returns a mock schema based on random score for demonstration,
        unless wired to PrincipleEvaluator.
        """
        # In a real implementation, this would generate tokens "{"Analysis": ...}"
        # For this MVP, we return a structural placeholder.
        return {
            "analysis": "GenRM Analysis Placeholder",
            "virtue_assessment": "Standard",
            "verdict": "approve",
            "score": 0.9
        }
