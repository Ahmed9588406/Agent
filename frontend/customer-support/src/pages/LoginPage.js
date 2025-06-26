import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import AuthForm from "../components/AuthForm";

const LoginPage = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async (email, password) => {
    setLoading(true);
    setError("");
    try {
      // 1. Call your backend login endpoint
      const loginRes = await fetch("/user/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, password }),
      });
      const loginData = await loginRes.json();
      if (loginData.error) throw new Error(loginData.error);

      // 2. Fetch user info (including name)
      const infoRes = await fetch("/user/info", {
        method: "GET",
        credentials: "include",
      });
      const infoData = await infoRes.json();
      if (infoData.detail || infoData.error) throw new Error(infoData.detail || infoData.error);

      // 3. Navigate to ChatPage with user info
      navigate("/chat", {
        state: {
          token: loginData.user_id, // or session token if needed
          user: { name: infoData.name, email },
        },
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Login</h1>
      <AuthForm type="login" onLogin={handleLogin} loading={loading} error={error} />
    </div>
  );
};

export default LoginPage;