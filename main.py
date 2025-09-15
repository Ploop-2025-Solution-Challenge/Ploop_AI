"""
Ploop AI Server
A FastAPI-based AI server for the Ploop 2025 Solution Challenge
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import platform
import psutil
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("config.json not found, using default configuration")
        return {
            "server": {
                "name": "Ploop AI Server",
                "version": "1.0.0",
                "description": "AI Server for Ploop 2025 Solution Challenge"
            },
            "ai": {
                "default_model": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.7
            },
            "features": ["AI Chat", "Ploop Analysis", "Health Monitoring"]
        }

config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title=config["server"]["name"],
    description=config["server"]["description"],
    version=config["server"]["version"]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    timestamp: str
    system_info: Optional[Dict[str, Any]] = None

class AIRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class AIResponse(BaseModel):
    response: str
    ai_model_used: str
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[float] = None

class PloopAnalysisRequest(BaseModel):
    data: Dict[str, Any]
    analysis_type: str = "general"

class PloopAnalysisResponse(BaseModel):
    analysis: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    processing_time_ms: Optional[float] = None

# Health check endpoint
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    system_info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "disk_usage_percent": psutil.disk_usage('/').percent
    }
    
    return HealthResponse(
        status="healthy",
        message=f"{config['server']['name']} is running",
        version=config["server"]["version"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        system_info=system_info
    )

# AI chat endpoint
@app.post("/ai/chat", response_model=AIResponse)
async def ai_chat(request: AIRequest):
    """
    Process AI chat requests
    """
    start_time = datetime.now(timezone.utc)
    
    try:
        # Use defaults from config if not provided
        model = request.model or config["ai"]["default_model"]
        max_tokens = request.max_tokens or config["ai"]["max_tokens"]
        temperature = request.temperature or config["ai"]["temperature"]
        
        # For now, return a mock response
        # In a real implementation, this would integrate with OpenAI or other AI services
        response_text = f"AI Response to: {request.prompt}"
        
        logger.info(f"Processing AI request with prompt: {request.prompt[:50]}...")
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        return AIResponse(
            response=response_text,
            ai_model_used=model,
            tokens_used=len(request.prompt.split()) + len(response_text.split()),
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        logger.error(f"Error processing AI request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")

# Ploop-specific analysis endpoint
@app.post("/ploop/analyze", response_model=PloopAnalysisResponse)
async def ploop_analyze(request: PloopAnalysisRequest):
    """
    Perform Ploop-specific data analysis
    """
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(f"Processing Ploop analysis of type: {request.analysis_type}")
        
        # Mock analysis - in real implementation, this would process the data
        analysis_result = {
            "data_points_analyzed": len(request.data),
            "analysis_type": request.analysis_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence_score": 0.85
        }
        
        insights = [
            "Data shows positive trends in user engagement",
            "Peak activity occurs during specific time windows",
            "User behavior patterns suggest optimization opportunities"
        ]
        
        recommendations = [
            "Consider implementing targeted notifications during peak hours",
            "Optimize user interface based on usage patterns",
            "Implement predictive analytics for better user experience"
        ]
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        return PloopAnalysisResponse(
            analysis=analysis_result,
            insights=insights,
            recommendations=recommendations,
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        logger.error(f"Error processing Ploop analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get server configuration (non-sensitive)"""
    return config

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)