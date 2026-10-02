from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

Urgency = Literal["critical", "high", "medium", "low"]
NextAction = Literal["auto-respond", "route to specialist", "escalate to human"]
Sentiment = Literal["positive", "neutral", "negative", "unknown"]


class TriageResult(BaseModel):
    urgency: Urgency
    product: str = ""
    issue_types: list[str] = Field(default_factory=list)
    sentiment: Sentiment = "neutral"
    kb_refs: list[str] = Field(default_factory=list)
    next_action: NextAction = "route to specialist"
    reasoning: str = ""
    confidence: float = 0.0

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v: str) -> str:
        s = str(v).lower().strip()
        if s in ("pos", "positive"):
            return "positive"
        if s in ("neg", "negative"):
            return "negative"
        if s in ("neu", "nue", "neutral"):
            return "neutral"
        if s in ("unknown", ""):
            return "unknown"
        # If open-ended adjective received, map to 3-class fallback
        if any(w in s for w in ("furious", "angry", "frustrated", "bad", "terrible", "urgent")):
            return "negative"
        if any(w in s for w in ("friendly", "good", "happy", "great", "positive")):
            return "positive"
        return "neutral"

    def to_dict(self) -> dict:
        return self.model_dump()


class TicketState(TypedDict, total=False):
    """Graph state for ReAct triage agent."""

    ticket: dict
    ticket_text: str
    messages: Annotated[list[BaseMessage], add_messages]
    analysis: dict
    tool_outputs: dict
    result: TriageResult
    fallback: bool

