# schemas.py
# 챗봇 요청/응답의 데이터 모양을 정의하는 파일 (Java의 DTO 역할)

from pydantic import BaseModel, Field
from typing import Literal


# 대화 내역 한 턴을 담는 틀
# role은 "user" 아니면 "assistant" 둘 중 하나만 허용 (Literal이 그 역할)
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str


# 프론트에서 챗봇한테 질문 보낼 때 쓰는 틀 (요청)
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)  # 빈 문자열 금지
    history: list[ChatTurn] = []        # 기본값: 빈 리스트 (히스토리 없어도 됨)


# 챗봇이 답변 줄 때 쓰는 틀 (응답)
class ChatResponse(BaseModel):
    reply: str