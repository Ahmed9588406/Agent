import React from "react";
import ChatBox from "../components/ChatBox";

const ChatPage = ({ token, user }) => {
  return (
    <div>
      <h1>Customer Support Chat</h1>
      <ChatBox token={token} userName={user?.name} />
    </div>
  );
};

export default ChatPage;