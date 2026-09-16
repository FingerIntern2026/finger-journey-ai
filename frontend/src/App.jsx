import { useState } from "react";
import "./App.css";

function App() {
  // 지금까지의 대화 내역
  const [messages, setMessages] = useState([]);

  // 현재 입력창에 작성 중인 내용
  const [input, setInput] = useState("");

  // Gemini 답변을 기다리고 있는지 여부
  const [loading, setLoading] = useState(false);

  // 메시지 전송
  const handleSend = async () => {
    // 빈 메시지이거나 이미 답변을 기다리는 중이면 전송하지 않음
    if (!input.trim() || loading) {
      return;
    }

    // 현재 사용자가 입력한 메시지
    const userMessage = {
      role: "user",
      text: input,
    };

    // input을 비우기 전에 현재 질문 저장
    const currentInput = input;

    // 사용자 메시지를 화면에 먼저 추가
    setMessages((prevMessages) => [
      ...prevMessages,
      userMessage,
    ]);

    // 입력창 비우기
    setInput("");

    // 답변 대기 시작
    setLoading(true);

    try {
      // FastAPI 챗봇 API 호출
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },

        // 현재 질문 + 이전 대화 내역 전송
        body: JSON.stringify({
          message: currentInput,
          history: messages,
        }),
      });

      // HTTP 에러 처리
      if (!response.ok) {
        throw new Error("챗봇 요청에 실패했습니다.");
      }

      // FastAPI 응답을 JSON으로 변환
      const data = await response.json();

      // Gemini 답변을 메시지 형태로 생성
      const assistantMessage = {
        role: "assistant",
        text: data.reply,
      };

      // Gemini 답변을 화면에 추가
      setMessages((prevMessages) => [
        ...prevMessages,
        assistantMessage,
      ]);
    } catch (error) {
      console.error("챗봇 오류:", error);
    } finally {
      // 답변 대기 종료
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <h1>Finger Journey</h1>
      <p>신규 입사자 온보딩 챗봇</p>

      {/* 대화 영역 */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <p>궁금한 사내 규정을 물어보세요.</p>
        )}

        {messages.map((message, index) => (
          <div key={index}>
            <strong>
              {message.role === "user" ? "나" : "AI"}
            </strong>

            <p>{message.text}</p>
          </div>
        ))}

        {/* Gemini 답변 대기 표시 */}
        {loading && <p>AI가 답변을 작성하고 있습니다...</p>}
      </div>

      {/* 입력 영역 */}
      <div className="chat-input">
        <input
          type="text"
          value={input}
          placeholder="질문을 입력하세요."
          onChange={(e) => setInput(e.target.value)}
        />

        <button
          onClick={handleSend}
          disabled={loading}
        >
          {loading ? "답변 중..." : "전송"}
        </button>
      </div>
    </div>
  );
}

export default App;