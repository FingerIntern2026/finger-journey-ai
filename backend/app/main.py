# main.py
# FastAPI 서버의 시작점. 여기서 앱을 만들고, 엔드포인트(주소)를 정의합니다.

import os
from google.genai import types
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, HTTPException

from .schemas import ChatRequest, ChatResponse
from .prompts import SYSTEM_PROMPT

# .env 파일에 적어둔 값들(GEMINI_API_KEY 등)을 읽어옵니다
load_dotenv()

# Gemini API에게 요청을 보낼 때 쓸 클라이언트(창구 역할)를 하나 만들어둡니다
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# FastAPI 앱 생성 - 이게 우리 서버 전체를 대표하는 존재입니다
app = FastAPI()


# POST 방식으로 /api/chat 주소에 요청이 오면 이 함수가 실행됩니다
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    # history 리스트를 "user: ...\nassistant: ..." 형태의 긴 텍스트로 합칩니다
    history_text = "\n".join(
        f"{turn.role}: {turn.text}"
        for turn in request.history
    )

    # 시스템 프롬프트 + 이전 대화 + 이번 질문을 하나의 프롬프트로 합칩니다
    prompt = f"""
{SYSTEM_PROMPT}

[이전 대화]
{history_text}

[현재 질문]
user: {request.message}
"""

    # Gemini API 호출 - 실패하면 500 에러로 처리
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"[이전 대화]\n{history_text}\n\n[현재 질문]\nuser: {request.message}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT  # 시스템 프롬프트를 여기 별도로 전달
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Gemini API 호출에 실패했습니다."
        )
        
    # 응답 텍스트만 뽑아서 ChatResponse 형태로 반환
    return ChatResponse(
        reply=response.text
    )