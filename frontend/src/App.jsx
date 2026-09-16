import { useState } from "react";
import "./App.css";

function App() {
  // 지금까지의 대화 내역
  const [messages, setMessages] = useState([]);

  // 현재 입력창 내용
  const [input, setInput] = useState("");

  // AI 답변 대기 여부
  const [loading, setLoading] = useState(false);

  // 메시지 전송
  const handleSend = async () => {
    // 빈 메시지이거나 이미 답변을 기다리는 중이면 전송하지 않음
    if (!input.trim() || loading) {
      return;
    }

    // 현재 질문
    const currentInput = input.trim();

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
          onClick={handleSend}
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