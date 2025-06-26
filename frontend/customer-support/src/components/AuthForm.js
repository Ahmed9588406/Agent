import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const AuthForm = ({ type }) => {
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    // If session token exists, redirect to chat
    const token = localStorage.getItem("session_token");
    if (token) {
      navigate("/chat");
    }
  }, [navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const endpoint =
        type === "signup"
          ? "http://127.0.0.1:8000/user/signup"
          : "http://127.0.0.1:8000/user/login";
      const response = await axios.post(endpoint, formData, {
        withCredentials: true,
      });
      setMessage(response.data.message);
      // Store session token if present in response or cookie
      if (type === "login" && response.data.message === "Login successful") {
        // Wait a moment to ensure cookie is set before fetching user info
        setTimeout(async () => {
          try {
            // Always send withCredentials: true to include cookies
            const infoRes = await axios.get("http://127.0.0.1:8000/user/info", {
              withCredentials: true
            });
            const infoData = infoRes.data;
            console.log("User info fetched:", infoData); // Debug log
            navigate("/chat", {
              state: {
                token: response.data.user_id,
                userName: infoData.name,
                email: infoData.email
              }
            });
          } catch (infoErr) {
            console.error("Failed to fetch user info:", infoErr); // Debug log
            setMessage("Login succeeded but failed to fetch user info. Please check your backend CORS settings, cookie settings, and ensure cookies are allowed.");
          }
        }, 1000); // Increase delay to 1000ms (1 second)
        return;
      }
    } catch (error) {
      setMessage(error.response?.data?.detail || "An error occurred.");
    }
  };

  return (
    <div>
      <h2>{type === "signup" ? "Sign Up" : "Log In"}</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          value={formData.email}
          onChange={(e) =>
            setFormData({ ...formData, email: e.target.value })
          }
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={formData.password}
          onChange={(e) =>
            setFormData({ ...formData, password: e.target.value })
          }
          required
        />
        <button type="submit">{type === "signup" ? "Sign Up" : "Log In"}</button>
      </form>
      <p>{message}</p>
    </div>
  );
};

export default AuthForm;