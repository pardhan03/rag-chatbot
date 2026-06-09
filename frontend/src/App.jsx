import { useState } from 'react'
import './App.css'
import axios from "axios";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input) return;

    const userMessage = {
      role: "user",
      content: input
    };

    setMessages(prev => [...prev, userMessage]);

    const response = await axios.post(
      "http://localhost:8000/chat",
      {
        question: input
      }
    );

    const botMessage = {
      role: "assistant",
      content: response.data.answer
    };

    setMessages(prev => [...prev, botMessage]);

    setInput("");
  };
  return (
    <div className="container">
      <div className="chat-window">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={msg.role}
          >
            {msg.content}
          </div>
        ))}
      </div>

      <div className="input-area">
        <input
          value={input}
          onChange={(e) =>
            setInput(e.target.value)
          }
        />

        <button onClick={sendMessage}>
          Send
        </button>
      </div>
    </div>
  )
}

export default App
