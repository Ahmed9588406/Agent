import React, { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Sparkles } from "lucide-react";

const ChatBox = ({ token = "demo-token", userName: userNameProp }) => {
  const [userName, setUserName] = useState(userNameProp || null);

  // Egyptian Arabic greeting with username fallback
  const getInitialMessage = (name) =>
    name
      ? `أهلاً يا ${name}! 👋 أنا مساعدك الذكي. إزاي أقدر أساعدك النهارده؟`
      : "أهلاً! 👋 أنا مساعدك الذكي. إزاي أقدر أساعدك النهارده؟";

  const [messages, setMessages] = useState([
    { text: getInitialMessage(userNameProp || null), isUser: false, timestamp: new Date() }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Add this helper to send a message programmatically
  const sendMessageWithText = async (text) => {
    if (!text.trim()) return;

    const userMessage = { text, isUser: true, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInput("");

    try {
      setIsTyping(true);

      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: text }),
        credentials: "include"
      });

      const data = await res.json();
      setIsTyping(false);

      const llmMessage = {
        text: data.response || "Sorry, I couldn't get a response from the server.",
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, llmMessage]);
    } catch (error) {
      setIsTyping(false);
      const errorMessage = {
        text: "Sorry, I'm having trouble connecting right now. Please try again.",
        isUser: false,
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const sendMessage = async () => {
    await sendMessageWithText(input);
  };

  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Fetch username only if not provided as prop
  useEffect(() => {
    // Only fetch if userNameProp is not provided and user is likely authenticated
    if (!userNameProp) {
      fetch("http://127.0.0.1:8000/user/info", {
        credentials: "include"
      })
        .then(res => {
          if (res.status === 401) return null; // Not authenticated, don't update userName
          return res.json();
        })
        .then(data => {
          if (data && data.name) setUserName(data.name);
        })
        .catch(() => setUserName(null));
    }
  }, [userNameProp]);

  // Update only the first message with the username, preserving chat history
  useEffect(() => {
    if (userName) {
      setMessages(prev => {
        // Only update if the greeting doesn't already contain the username
        if (
          prev.length > 0 &&
          !prev[0].text.includes(userName)
        ) {
          return [
            { ...prev[0], text: getInitialMessage(userName) },
            ...prev.slice(1)
          ];
        }
        return prev;
      });
    }
  }, [userName]);

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto bg-gradient-to-br from-slate-50 to-white">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-xl border-b border-slate-200/60 px-6 py-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="relative">
            <img
              src="/bumblebee.png"
              alt="Bumblebee"
              className="w-10 h-10 rounded-full border border-slate-200 shadow-lg object-cover"
            />
            <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-white animate-pulse"></div>
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">
              Bumblebee
            </h1>
            {/* Optionally keep the small image here if you want */}
          </div>
          <p className="text-sm text-slate-500">Always ready to help</p>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-transparent">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex gap-3 animate-in slide-in-from-bottom-2 duration-300 ${
              msg.isUser ? "justify-end" : "justify-start"
            }`}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            {!msg.isUser && (
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-md">
                  <Bot className="w-4 h-4 text-white" />
                </div>
              </div>
            )}
            
            <div className={`max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg xl:max-w-xl group ${msg.isUser ? "order-first" : ""}`}>
              <div
                className={`px-4 py-3 rounded-2xl shadow-sm transition-all duration-200 hover:shadow-md ${
                  msg.isUser
                    ? "bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-br-md"
                    : msg.isError
                    ? "bg-red-50 text-red-700 border border-red-200 rounded-bl-md"
                    : "bg-white text-slate-800 border border-slate-200 rounded-bl-md hover:bg-slate-50"
                }`}
              >
                <p className="text-sm leading-relaxed">{msg.text}</p>
              </div>
              <div className={`text-xs text-slate-400 mt-1 px-1 ${msg.isUser ? "text-right" : "text-left"}`}>
                {formatTime(msg.timestamp)}
              </div>
            </div>

            {msg.isUser && (
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-full flex items-center justify-center shadow-md">
                  <User className="w-4 h-4 text-white" />
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex gap-3 animate-in slide-in-from-bottom-2 duration-300">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-md">
                <Bot className="w-4 h-4 text-white" />
              </div>
            </div>
            <div className="bg-white text-slate-800 border border-slate-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Container */}
      <div className="bg-white/80 backdrop-blur-xl border-t border-slate-200/60 px-6 py-4">
        <div className="flex gap-3 items-end">
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
              placeholder="Type your message..."
              disabled={isTyping}
              className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 placeholder-slate-400 text-slate-800 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-100 focus:bg-white"
            />
          </div>
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isTyping}
            className="p-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-2xl hover:from-blue-600 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-md group"
          >
            <Send className="w-5 h-5 transform group-hover:translate-x-0.5 transition-transform duration-200" />
          </button>
        </div>
        
        
        </div>
      </div>
    
  );
};

export default ChatBox;