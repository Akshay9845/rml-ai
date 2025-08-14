"""
FastAPI Server for RML System
"""

import time
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .core import RMLSystem, RMLResponse
from .config import RMLConfig


# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    response_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: float

class ReadyResponse(BaseModel):
    ready: bool
    entries: int
    device: str


# Initialize FastAPI app
app = FastAPI(
    title="RML-AI API",
    description="Resonant Memory Learning AI System API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RML system instance
rml_system: RMLSystem = None


@app.on_event("startup")
async def startup_event():
    """Initialize RML system on startup"""
    global rml_system
    
    print("Initializing RML system...")
    config = RMLConfig()
    print(f"Configuration: {config}")
    
    try:
        rml_system = RMLSystem(config)
        print("RML system initialized successfully!")
    except Exception as e:
        print(f"Error initializing RML system: {e}")
        raise e


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "RML-AI API",
        "description": "Resonant Memory Learning AI System",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time()
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready_check():
    """Ready check endpoint"""
    if rml_system is None:
        raise HTTPException(status_code=503, detail="RML system not initialized")
    
    stats = rml_system.memory.get_stats()
    return ReadyResponse(
        ready=True,
        entries=stats['total_entries'],
        device=rml_system.config.device
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for RML queries"""
    if rml_system is None:
        raise HTTPException(status_code=503, detail="RML system not initialized")
    
    try:
        response = rml_system.query(request.message)
        return ChatResponse(
            answer=response.answer,
            response_ms=response.response_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/config")
async def get_config():
    """Get current configuration"""
    if rml_system is None:
        raise HTTPException(status_code=503, detail="RML system not initialized")
    
    return rml_system.config.to_dict()


def main():
    """Main function to run the server"""
    uvicorn.run(
        "rml_ai.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main() 