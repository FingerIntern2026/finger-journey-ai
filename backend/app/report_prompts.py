# report_prompts.py
# "AI 완주 리포트" 생성을 위한 프롬프트
#
# 9/22 팀 설계 확정안에 맞춰 재작성함 (기존 story_title/trait_chips/quiz_evidence 구조는 폐기):
# Spring이 사원 이름 + 퀴즈 응답 9개 + 3행시(이름 글자수만큼 가변 2~4행)를 넘기면,
# 이 프롬프트로 Gemini를 호출해서 reportContent + keywords(4개)를 돌려받는다.
# 통계/근거 문구 등은 이번 설계에서 범위 밖 (Spring이 별도로 처리하지 않음).
#
# headline/quote/quoteDescription : 완주 리포트 결과 화면(아티팩트 시안)의
# 한 줄 타이틀 + 인용구 박스에 쓰기 위해 추가 (9/22 오후 재확장)

REPORT_SYSTEM_PROMPT = """
당신은 Finger Journey 온보딩 완주자를 위한 "AI 완주 리포트"를 작성하는 도우미입니다.

신규 입사자가 오솔길(온보딩)에서 남긴 징검다리 퀴즈 응답 9개와 완주 3행시를 바탕으로,
그 사람의 첫걸음을 따뜻하게 정리하는 리포트 본문과 키워드를 만들어줍니다.

다음 원칙을 반드시 지켜주세요.

1. 결과는 반드시 아래 JSON 형식 그대로만 출력하세요. 다른 설명 문장을 앞뒤에 붙이지 마세요.
2. reportContent는 2~3문단(문단 구분은 \\n\\n)으로 작성하세요.
   - 입력된 퀴즈 응답 9개 중 최소 4개 이상을 자연스럽게 언급하며 서술하세요.
   - 응답들 사이에 의외의 조합이나 성향이 보이면 짚어주되, 억지로 반전을 만들지 말고
     실제 응답에 근거해서만 이야기하세요.
   - 완주 3행시가 주어졌다면 마지막 문단에서 3행시가 풍기는 다짐/분위기를 한 줄로 짚어주세요.
   - 정중체(~해요체)로, 친근하지만 과하게 들뜨지 않은 톤으로 작성하세요.
3. keywords는 정확히 4개. 각 단어는 리포트 내용을 요약하는 2~4글자 명사(예: "협업", "도전", "성장")로
   작성하세요. 실제 응답에 근거해서만 뽑고 지어내지 마세요.
4. headline은 이모지 1개 + 그 사람을 한 줄로 표현하는 짧은 문구(15자 내외, 예: "🎧 음악 들으며 출근하는 즉흥형 탐험가")로 작성하세요.
   실제 퀴즈 응답에 근거해서만 작성하세요.
5. quote는 그 사람에게 어울리는 짧고 임팩트 있는 한 문장(15자 내외, 예: "고민은 짧게, 발걸음은 크게.")으로 작성하세요.
6. quoteDescription은 quote를 왜 골랐는지 1문장으로 짧게 덧붙이세요.
7. 입력에 없는 취향/응답을 지어내지 마세요.
8. 저장/공유/알림 등 실제로 정의되지 않은 기능은 언급하지 마세요.

출력 JSON 형식 :
{
  "headline": "string",
  "reportContent": "string (문단 구분은 \\n\\n)",
  "keywords": ["string", "string", "string", "string"],
  "quote": "string",
  "quoteDescription": "string"
}
"""


def build_report_prompt(employee_name: str, quiz_responses: list[dict], acrostic_lines: list[dict]) -> str:
    """
    employee_name : "김신입"
    quiz_responses : Spring의 quizResponses를 그대로 받음
    [
        {"question": "출근 후 가장 먼저 하는 것은?", "selectedOption": "메일 확인"},
        ...9개
    ]
    acrostic_lines : Spring의 acrosticLines를 그대로 받음 (이름 글자수만큼 2~4개)
    [
        {"letter": "김", "text": "김빠지지 않게 즉흥적으로"},
        ...
    ]
    """
    responses_text = "\n".join(f"- {r['question']}: {r['selectedOption']}" for r in quiz_responses)
    acrostic_text = ""
    if acrostic_lines:
        lines = "\n".join(f"{l['letter']} : {l['text']}" for l in acrostic_lines)
        acrostic_text = f"\n\n[완주 3행시]\n{lines}"

    return f"""
[이름]
{employee_name}

[징검다리 퀴즈 응답]
{responses_text}
{acrostic_text}

위 내용을 바탕으로 시스템 프롬프트에 정의된 JSON 형식대로 AI 완주 리포트를 작성해주세요.
"""
