import React from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import ChatPage from "./pages/ChatPage";

const App = () => {
  const location = useLocation();

  const handleLogin = (newToken) => {
    localStorage.setItem("token", newToken);
  };

  return (
    <>
      <Routes>
        <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route
          path="/chat"
          element={
            <ChatPage
              token={location.state?.token}
              user={location.state?.user}
            />
          }
        />
        <Route path="/" element={<LoginPage onLogin={handleLogin} />} />
      </Routes>
    </>
  );
};

export default App;