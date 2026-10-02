"""State definition of the triage graph: input state + output schema."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

Urgency = Literal["critical", "high", "medium", "low"]
NextAction = Literal["auto-respond", "route to specialist", "escalate to human"]


class TriageResult(BaseModel):
    urgency: Urgency
    product: str = ""
    issue_types: list[str] = Field(default_factory=list)
    sentiment: str = ""
    sentiment_trajectory: str = ""
    kb_refs: list[str] = Field(default_factory=list)
    next_action: NextAction = "route to specialist"
    reasoning: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return self.model_dump()


class TicketState(TypedDict, total=False):
    """Graph state: ticket in, TriageResult layers out."""

    ticket: dict
    ticket_text: str
    analysis: dict
    tool_outputs: dict
    result: TriageResult
