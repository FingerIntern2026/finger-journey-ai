from enum import Enum

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def validate_reply_for_route(self) -> "IntentResult":
        if self.intent == ChatIntent.POLICY_QUESTION:
            if self.reply is not None:
                raise ValueError("POLICY_QUESTION must not include a reply.")
            return self

        if self.reply is None or not self.reply.strip():
            raise ValueError(f"{self.intent.value} must include a reply.")

        self.reply = self.reply.strip()
        return self
