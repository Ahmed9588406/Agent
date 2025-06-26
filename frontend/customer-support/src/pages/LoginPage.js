import React from "react";
import AuthForm from "../components/AuthForm";

const LoginPage = () => {
  return (
    <div>
      <h1>Login</h1>
      <AuthForm type="login" />
    </div>
  );
};

export default LoginPage;