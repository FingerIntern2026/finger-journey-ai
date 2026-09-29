# main.py
# FastAPI 서버의 시작점. 앱을 생성하고 챗봇 API 엔드포인트를 정의합니다.

import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

from .schemas import (
    ChatRequest,
    ChatResponse,
    IngestionResponse,
    RagSearchRequest,
    RagSearchResponse,
    ReportRequest,
    ReportResponse,
)
from .report_prompts import REPORT_SYSTEM_PROMPT, build_report_prompt
from .rag.chain import get_rag_answer_chain
from .rag.ingestion_service import get_ingestion_service
from .rag.retriever import PgVectorRetriever
from .rag.warmup import warm_up_rag


# .env 파일의 환경변수(GEMINI_API_KEY 등)를 읽어옵니다.
load_dotenv()


# Gemini API 클라이언트를 생성합니다.
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("RAG embedding model warm-up started.")
    warm_up_rag()
    logger.info("RAG embedding model warm-up completed.")
    yield


# FastAPI 앱을 생성합니다.
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/rag/documents/ingest", response_model=IngestionResponse)
def ingest_documents():
    return get_ingestion_service().ingest_directory()


@app.post("/api/rag/search", response_model=RagSearchResponse)
def search_documents(request: RagSearchRequest):
    documents = PgVectorRetriever(
        search_limit=request.limit,
        category=request.category,
    ).invoke(request.query)
    return RagSearchResponse(
        query=request.query,
        results=[
            {
                "content": document.page_content,
                **document.metadata,
            }
            for document in documents
        ],
    )


# POST /api/chat
# 프론트에서 현재 질문(message)과 이전 대화(history)를 받아
# 사내 규정과 함께 Gemini에 전달합니다.
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        answer = get_rag_answer_chain().answer(
            question=request.message,
            history=[
                (turn.role, turn.text)
                for turn in request.history
            ],
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="RAG 답변 생성에 실패했습니다."
        )

    return ChatResponse(
        reply=answer.reply,
        sources=[
            {
                "file_name": source.file_name,
                "category": source.category,
                "heading": source.heading,
                "similarity": source.similarity,
            }
            for source in answer.sources
        ],
    )


# POST /report/generate
# Spring 백엔드가 내부적으로 호출하는 전용 엔드포인트 (프론트가 직접 호출하지 않음)
# 사원 이름 + 퀴즈 응답 9개 + 3행시(가변 2~4행)를 받아서 reportContent + keywords(4개)를 반환합니다.
@app.post("/report/generate", response_model=ReportResponse)
def generate_report(request: ReportRequest):
    quiz_responses = [r.model_dump() for r in request.quizResponses]
    acrostic_lines = [l.model_dump() for l in request.acrosticLines]
    prompt = build_report_prompt(request.employeeName, quiz_responses, acrostic_lines)

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=REPORT_SYSTEM_PROMPT
            )
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Gemini API 호출에 실패했습니다."
        )

    # Gemini가 ```json ... ``` 코드블록으로 감싸서 줄 때가 있어서 벗겨내고 파싱합니다.
    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        llm_result = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI 응답을 리포트 형식으로 해석하지 못했습니다."
        )

    return ReportResponse(**llm_result)
