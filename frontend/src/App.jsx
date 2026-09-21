import { useState } from "react";
import "./App.css";

// 진입 시 채팅바 위에 보여줄 추천 질문 칩
// company_context.py의 데모 규정 항목(근무시간/연차/재택/복장)에 맞춰 구성
const SUGGESTED_QUESTIONS = [
  "근무시간이 어떻게 되나요?",
  "연차는 어떻게 신청하나요?",
  "재택근무 가능한가요?",
  "복장 규정이 있나요?",
];

function App() {
  // 지금까지의 대화 내역
  const [messages, setMessages] = useState([]);

  // 현재 입력창 내용
  const [input, setInput] = useState("");

  // AI 답변 대기 여부
  const [loading, setLoading] = useState(false);

  // 메시지 전송
  // overrideText가 있으면(추천 질문 칩 클릭) 그 문구를 바로 보내고,
  // 없으면 입력창(input)에 있는 내용을 보낸다
  const handleSend = async (overrideText) => {
    const textToSend = overrideText ?? input;

    // 빈 메시지이거나 이미 답변을 기다리는 중이면 전송하지 않음
    if (!textToSend.trim() || loading) {
      return;
    }

    // 현재 질문
    const currentInput = textToSend.trim();

    // 사용자 메시지 생성
    const userMessage = {
      role: "user",
      text: currentInput,
    };

    // 사용자 메시지를 화면에 먼저 추가
    setMessages((prevMessages) => [
      ...prevMessages,
      userMessage,
    ]);

    // 입력창 비우기
    setInput("");

    // AI 답변 대기 시작
    setLoading(true);

    try {
      // FastAPI 호출
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },

        // 현재 질문 + 이전 대화 내역 전달
        body: JSON.stringify({
          message: currentInput,
          history: messages,
        }),
      });

      // 서버 오류 처리
      if (!response.ok) {
        throw new Error("챗봇 요청에 실패했습니다.");
      }

      // FastAPI 응답
      const data = await response.json();

      // AI 메시지 생성
      const assistantMessage = {
        role: "assistant",
        text: data.reply,
      };

      // AI 답변을 화면에 추가
      setMessages((prevMessages) => [
        ...prevMessages,
        assistantMessage,
      ]);
    } catch (error) {
      console.error("챗봇 오류:", error);
    } finally {
      // AI 답변 대기 종료
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">

      {/* 상단 헤더 */}
      <h1>Finger Journey</h1>
      <p>신규 입사자 온보딩 챗봇</p>

      {/* 채팅 메시지 영역 */}
      <div className="chat-messages">

        {/* 대화가 없을 때 */}
        {messages.length === 0 && !loading && (
          <p>궁금한 사내 규정을 물어보세요.</p>
        )}

        {/* 대화 목록 */}
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message-row ${
              message.role === "user"
                ? "user"
                : "assistant"
            }`}
          >
            <div className="message-bubble">
              {message.text}
            </div>
          </div>
        ))}

        {/* AI 답변 대기 */}
        {loading && (
          <div className="message-row assistant">
            <div className="message-bubble">
              답변을 작성하고 있습니다...
            </div>
          </div>
        )}
      </div>

      {/* 추천 질문 칩 — 대화 중에도 계속 노출 */}
      {!loading && (
        <div className="suggestion-chips">
          {SUGGESTED_QUESTIONS.map((question) => (
            <button
              key={question}
              type="button"
              className="suggestion-chip"
              onClick={() => handleSend(question)}
            >
              {question}
            </button>
          ))}
        </div>
      )}

      {/* 입력 영역 */}
      <div className="chat-input">

        <textarea
          value={input}
          placeholder="질문을 입력하세요."
          rows={1}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            // Enter → 전송
            // Shift + Enter → 줄바꿈
            // 한글 조합 중 Enter → 전송하지 않음
            if (
              e.key === "Enter" &&
              !e.shiftKey &&
              !e.nativeEvent.isComposing
            ) {
              e.preventDefault();
              handleSend();
            }
          }}
        />

        <button
          type="button"
          onClick={() => handleSend()}
          disabled={loading || !input.trim()}
          aria-label="메시지 전송"
        >
          전송
        </button>

      </div>
    </div>
  );
}

export default App;