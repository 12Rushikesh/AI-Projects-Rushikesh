from typing import TypedDict

class AgentThought(TypedDict):
    action: str
    damage_type: str
    confidence: str
    reason: str
    confidence_score: float
