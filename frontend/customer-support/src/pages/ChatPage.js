import React from "react";
import ChatBox from "../components/ChatBox";

const ChatPage = ({ token, user }) => {
  return (
    <div>
      <ChatBox token={token} userName={user?.name} />
    </div>
  );
};

export default ChatPage;