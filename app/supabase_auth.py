import os
from supabase import create_client
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class User(BaseModel):
    email: str
    password: str
    name: str

class UserLogin(BaseModel):
    email: str
    password: str

user_router = APIRouter(
    prefix="/user",
    tags=["user"]
)

@user_router.post("/signup")
async def signup(user: User):
    try:
        response = supabase.auth.sign_up({
            "email": user.email,
            "password": user.password  # send plain password
        })

        # Get the user id from the auth response
        user_id = None
        if hasattr(response, "user") and response.user:
            user_id = getattr(response.user, "id", None)
        elif isinstance(response, dict) and "user" in response:
            user_id = response["user"].get("id")
        if not user_id:
            return {"error": "Failed to retrieve user id from Supabase Auth"}

        user_Data = {
            "id": user_id,
            "email": user.email,
            "name": user.name,
            "profile_image_url": None
        }
        user_response = supabase.table("users").insert(user_Data).execute()
        print(user_response)
        return {"message": "User created successfully", "user_id": user_id}

    except Exception as e:
        return {"error": str(e)}

@user_router.post("/login")
async def login(user: UserLogin, response: Response):
    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password
        })
        if hasattr(auth_response, "error") and auth_response.error:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        print("Auth response session:", getattr(auth_response, "session", None))
        print("Auth response user:", getattr(auth_response, "user", None))
        session_token = getattr(auth_response.session, 'access_token', None)
        if session_token:
            response.set_cookie(
                key="session_token",
                value=session_token,
                httponly=True,
                max_age=60*60*24,
                samesite="none",
                secure=True  # Required for SameSite=none
            )
        return {
            "message": "Login successful",
            "user_id": getattr(auth_response.user, 'id', None),
            "email": getattr(auth_response.user, 'email', None)
        }
    except Exception as e:
        return {"error": str(e)}

@user_router.get("/info")
async def get_user_info(request: Request):
    session_token = request.cookies.get("session_token")
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        headers = {
            "Authorization": f"Bearer {session_token}",
            "apikey": SUPABASE_KEY
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or expired session token")
        user_data = resp.json()
        user_id = user_data.get("id")
        email = user_data.get("email")
        if not user_id:
            raise HTTPException(status_code=401, detail="User not found in auth")
        user_row = supabase.table("users").select("name,profile_image_url").eq("id", user_id).single().execute()
        name = None
        profile_image_url = None
        if hasattr(user_row, "data") and user_row.data:
            if user_row.data.get("name") is not None:
                name = user_row.data.get("name")
                profile_image_url = user_row.data.get("profile_image_url")
        elif isinstance(user_row, dict) and "data" in user_row and user_row["data"]:
            if user_row["data"].get("name") is not None:
                name = user_row["data"].get("name")
                profile_image_url = user_row["data"].get("profile_image_url")
        if not name:
            raise HTTPException(status_code=404, detail="User name not found in users table")
        return {"name": name, "profile_image_url": profile_image_url, "email": email}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user info: {str(e)}")

# Add this block at the top-level where your FastAPI app is created (not inside the router)
from fastapi import FastAPI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change to your frontend's URL/port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)