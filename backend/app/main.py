from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv

from app.routers import auth, users, players, predictions
from app.database import init_database, close_database

load_dotenv()

app = FastAPI(title="Trade Analyzer API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    await init_database()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    await close_database()

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(players.router, prefix="/api/players", tags=["players"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])

@app.get("/")
async def root():
    return {"message": "Welcome to Trade Analyzer API"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

