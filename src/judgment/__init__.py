from typing import TypedDict, Literal

class JudgmentSchema(TypedDict):
    analysis: str          # "The user is asking for X..."
    virtue_assessment: str # "The tone should be Empathetic..."
    verdict: Literal["approve", "refuse_redirect", "refuse_hard"]
    score: float           # 0.0 (violation) to 1.0 (aligned)
