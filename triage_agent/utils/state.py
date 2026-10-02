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
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("product", mode="before")
    @classmethod
    def normalize_product(cls, v: object) -> str:
        return str(v or "").strip().lower() or "general"

    @field_validator("issue_types", mode="before")
    @classmethod
    def normalize_issue_types(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(t).strip().lower() for t in v if str(t).strip()]

    @field_validator("kb_refs", mode="before")
    @classmethod
    def normalize_kb_refs(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        seen: list[str] = []
        for r in v:
            s = str(r).strip()
            if s and s not in seen:
                seen.append(s)
        return seen[:2]

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: object) -> float:
        try:
            f = float(v)  # type: ignore[arg-type]
        except Exception:
            return 0.0
        # Accept 0-100 percent style from drifting LLMs, then clamp.
        if f > 1.0 and f <= 100.0:
            f = f / 100.0
        return max(0.0, min(1.0, f))

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v: str) -> str:
        # Only exact labels + common abbreviations plus clearly valenced
        # adjectives. Anything unrecognized falls back to neutral instead
        # of guessing, so LLM schema drift stays visible.
        s = str(v).lower().strip()
        if s in ("pos", "positive"):
            return "positive"
        if s in ("neg", "negative"):
            return "negative"
        if s in ("neu", "nue", "neutral"):
            return "neutral"
        if s in ("unknown", ""):
            return "unknown"
        if s in ("furious", "angry", "irate", "upset", "frustrated", "unhappy", "disappointed"):
            return "negative"
        if s in ("happy", "pleased", "satisfied", "delighted"):
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

