from enum import Enum

from pydantic import BaseModel, Field


class ChatIntent(str, Enum):
    """Supported routes for an incoming chat message."""

    POLICY_QUESTION = "POLICY_QUESTION"
    CASUAL = "CASUAL"
    AMBIGUOUS = "AMBIGUOUS"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class IntentResult(BaseModel):
    """Structured result returned by the chat intent classifier."""

    intent: ChatIntent = Field(
        description="The route that best matches the user's current message."
    )
    reply: str | None = Field(
        default=None,
        description=(
            "A user-facing response for non-policy routes. "
            "Policy questions leave this empty and continue to RAG."
        ),
    )
