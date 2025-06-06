from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os

from .agent_main import process_user_input, get_system_info, memory
import asyncio

# JWT settings
SECRET_KEY = os.getenv('AGENT_JWT_SECRET', 'supersecretkey123')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Demo user (change password after first login!)
FAKE_USER = {
    'username': 'admin',
    'password': "ChangeMe123!"
}

app = FastAPI(title="Agent Remote API", docs_url="/api/docs", openapi_url="/api/openapi.json")

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

class Token(BaseModel):
    access_token: str
    token_type: str

class CommandRequest(BaseModel):
    command: str
    is_voice_mode: Optional[bool] = False

class CommandResponse(BaseModel):
    response: str

class StatsResponse(BaseModel):
    cpu: float
    memory: float
    disk: float
    os: str
    os_version: str

class MemoryResponse(BaseModel):
    facts: dict
    preferences: dict
    conversations: list

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username != FAKE_USER['username']:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception

@app.post("/api/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != FAKE_USER['username'] or form_data.password != FAKE_USER['password']:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/command", response_model=CommandResponse)
async def run_command(req: CommandRequest, username: str = Depends(verify_token)):
    resp = await process_user_input(req.command, is_voice_mode=req.is_voice_mode)
    if isinstance(resp, tuple):
        resp = resp[0]
    return {"response": resp}

@app.get("/api/stats", response_model=StatsResponse)
def get_stats(username: str = Depends(verify_token)):
    # Use get_system_info from agent_main.py
    import platform, psutil
    return {
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent,
        "os": platform.system(),
        "os_version": platform.version(),
    }

@app.get("/api/memory", response_model=MemoryResponse)
def get_memory(username: str = Depends(verify_token)):
    facts = memory.get_facts()
    preferences = memory.get_preferences()
    conversations = memory.get_recent_conversations(10)
    return {"facts": facts, "preferences": preferences, "conversations": conversations} 