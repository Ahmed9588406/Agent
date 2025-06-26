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
        // Try to get token from response (if backend returns it)
        if (response.data.session_token) {
          localStorage.setItem("session_token", response.data.session_token);
        } else {
          // Or try to get from cookie (browser will send it automatically if withCredentials is true)
          // Optionally, you can call a /me or /profile endpoint to verify and get user info
          localStorage.setItem("session_token", "1"); // Dummy value to indicate logged in
        }
        navigate("/chat");
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