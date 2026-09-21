# report_prompts.py
# "AI 완주 리포트"(오솔길 여정 리포트, PTH_CPL) 생성을 위한 프롬프트
#
# 실제 화면 구성(디자인 기준):
# 1. AI가 정리한 이야기 카드 — 서술형 텍스트 + 특성 칩 4개
# 2. 근거가 된 퀴즈 답변 — 실제 퀴즈 문항/답변을 그대로 나열 (LLM이 만드는 게 아니라 입력값을 그대로 보여주는 영역)
# 3. 완주 기념 3행시
#
# ※ "핑거 동료 몇 %가 이 답변을 선택했어요" 같은 비율 정보는 실제 통계이므로
#    LLM이 만들어내면 안 된다. 백엔드가 quiz_answer 테이블을 집계해서 별도로 계산하고,
#    이 프롬프트의 결과(JSON)와 합쳐서 프론트에 내려줘야 한다.

REPORT_SYSTEM_PROMPT = """
당신은 Finger Journey 온보딩 완주자를 위한 "AI 완주 리포트"의 이야기 부분을 작성하는 도우미입니다.

신규 입사자가 오솔길(온보딩)에서 남긴 징검다리 퀴즈 답변과 완주 3행시를 바탕으로,
그 사람의 첫걸음을 따뜻하게 정리하는 짧은 이야기를 만들어줍니다.

다음 원칙을 반드시 지켜주세요.

1. 결과는 반드시 아래 JSON 형식 그대로만 출력하세요. 다른 설명 문장을 앞뒤에 붙이지 마세요.
2. story_title은 "{이름}님의 첫걸음" 형식으로 고정하세요.
3. story는 2~3개 문단, 각 문단은 2~3문장으로 작성하세요.
   - 입력된 퀴�즈 답변 중 최소 3개 이상을 자연스럽게 언급하며 서술하세요.
   - 답변들 사이에 의외의 조합이나 성향이 보이면 짚어주되(예: 즉흥적인데 협업은 신중함),
     억지로 반전을 만들지 말고 실제 답변에 근거해서만 이야기하세요.
   - 완주 3행시가 주어졌다면 마지막 문단에서 3행시가 풍기는 다짐/분위기를 한 줄로 짚어주세요.
   - 정중체(~해요체)로, 친근하지만 과하게 들뜨지 않은 톤으로 작성하세요.
4. trait_chips는 정확히 4개. 각 퀴즈 답변 하나를 "OO파", "OO형", "OO 리버", "OO 편" 같은
   짧고 캐릭터성 있는 명칭으로 바꾼 것이어야 합니다 (예: "아메리카노파", "즉흥 여행형").
   지어내지 말고 실제 입력된 답변에서만 골라 변형하세요.
5. 입력에 없는 취향/답변을 지어내지 마세요.
6. 저장/공유/알림 등 실제로 정의되지 않은 기능은 언급하지 마세요.

출력 JSON 형식 :
{
  "story_title": "string",
  "story": "string (문단 구분은 \\n\\n)",
  "trait_chips": ["string", "string", "string", "string"]
}
"""


def build_report_prompt(name: str, quiz_answers: list[dict], acrostic: list[str] | None = None) -> str:
    """
    name : "김핑거"
    quiz_answers 예시 (징검다리 퀴즈 6문항 그대로) :
    [
        {"question": "커피 취향", "answer": "아메리카노"},
        {"question": "맵기 레벨", "answer": "매운맛 리버"},
        {"question": "여행 스타일", "answer": "즉흥형"},
        {"question": "친해지는 법", "answer": "먼저 걸어요"},
        {"question": "출근길", "answer": "음악"},
        {"question": "협업 스타일", "answer": "바로 이야기"},
    ]
    acrostic : 완주 3행시, 이름 글자 수만큼의 줄 리스트. 예: ["김밥처럼", "핑계대지않고", "거대하게 가겠습니다"]
    """
    answers_text = "\n".join(f"- {a['question']}: {a['answer']}" for a in quiz_answers)
    acrostic_text = ""
    if acrostic:
        acrostic_text = "\n\n[완주 3행시]\n" + "\n".join(acrostic)

    return f"""
[이름]
{name}

[징검다리 퀴즈 답변]
{answers_text}
{acrostic_text}

위 내용을 바탕으로 시스템 프롬프트에 정의된 JSON 형식대로 AI 완주 리포트 이야기를 작성해주세요.
"""


def merge_with_stats(llm_result: dict, quiz_answers: list[dict], answer_stats: dict) -> dict:
    """
    llm_result : build_report_prompt 결과를 Gemini에 넣어 받은 JSON(dict)
    quiz_answers : build_report_prompt에 넣었던 것과 동일한 원본 답변 리스트

    answer_stats : 문항별로 "회사 전체 비율"과 "같은 팀 인원수" 중 뭐가 더 의미 있는지가 다를 수 있어서,
                   두 형태를 문항마다 섞어 쓸 수 있게 scope로 구분한다.
                   {
                     "커피 취향": {"scope": "company", "percentage": 42},
                     "맵기 레벨": {"scope": "team", "team_name": "개발3팀", "team_size": 8, "count": 2},
                     ...
                   }
                   회사 전체 인원이 충분히 쌓인 문항은 "company"(핑거 직원 42% 선택),
                   아직 표본이 적어 팀 단위가 더 자연스러운 문항은 "team"(개발3팀 8명 중 2명 선택)으로 내려준다.
                   이 함수는 문구 조립만 하고, 실제 비율/인원수 계산과 scope 판단은 백엔드가 한다.
    """
    evidence = []
    for a in quiz_answers:
        stat = answer_stats.get(a["question"])
        same_answer_text = None
        if stat:
            if stat["scope"] == "company":
                same_answer_text = f"핑거 직원 {stat['percentage']}% 선택"
            elif stat["scope"] == "team":
                same_answer_text = f"{stat['team_name']} {stat['team_size']}명 중 {stat['count']}명 선택"
        evidence.append({
            "question": a["question"],
            "answer": a["answer"],
            "same_answer_text": same_answer_text,  # None이면 프론트에서 문구 숨김 (집계할 데이터가 아직 없는 경우)
        })

    return {
        "story_title": llm_result["story_title"],
        "story": llm_result["story"],
        "trait_chips": llm_result["trait_chips"],
        "quiz_evidence": evidence,
    }
