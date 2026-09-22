# schemas.py
# 챗봇/리포트 요청·응답의 데이터 모양을 정의하는 파일 (Java의 DTO 역할)

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


# ===== AI 완주 리포트 =====
# 필드명을 Spring의 AiReportRequestDto/AiReportResponseDto(Jackson, camelCase 직렬화)와
# 그대로 맞춰서, 양쪽에서 별도 변환 없이 바로 주고받을 수 있게 함

# 퀴즈 응답 한 문항 (Spring의 AiQuizResponseDto와 동일)
class AiQuizResponse(BaseModel):
    question: str
    selectedOption: str


# 3행시 한 줄 (Spring의 AiAcrosticLineDto와 동일)
class AiAcrosticLine(BaseModel):
    letter: str
    text: str


# Spring → AI 서버 요청 (Spring의 AiReportRequestDto와 동일)
class ReportRequest(BaseModel):
    employeeName: str = Field(min_length=1)
    quizResponses: list[AiQuizResponse] = Field(min_length=1)
    acrosticLines: list[AiAcrosticLine] = []


# AI 서버 → Spring 응답 (Spring의 AiReportResponseDto와 동일)
# headline/quote/quoteDescription : 리포트 화면 상단에 쓰이는 한 줄 타이틀 + 인용구 박스
# (9/22 팀 아티팩트 시안대로 리포트 결과 화면을 복원하면서 추가)
class ReportResponse(BaseModel):
    headline: str
    reportContent: str
    keywords: list[str]
    quote: str
    quoteDescription: str
